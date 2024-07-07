# local imports
from utils.logger import Logger
from db.db import DB
from configs.config import MainConfig
from confz import BaseConfig, FileSource
from registration_separation import process_data, calculate_minutes_difference
# other imports
from pathlib import Path
import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import repeat
from ultralytics import YOLO
from ultralytics.engine.results import Results
from torchvision.transforms.functional import normalize
import cv2
from PIL import Image

L = Logger()

class ImageProcessor:
    def __init__(self, config_path: str) -> "ImageProcessor":
        self.db = DB()
        self.config = MainConfig(config_sources=FileSource(file=config_path))
        self.device = self.config.device
        self.mean = [123.675, 116.28, 103.535]
        self.std = [58.395, 57.12, 57.375]

        self.detector = YOLO(self.config.detector.weights).to(self.device)
        self.classificator = torch.load(self.config.classificator.weights).to(self.device)

        self.mapping = self.open_mapping(self.config.mapping)
        self.pathes_to_imgs = [i for i in Path(self.config.src_dir).glob("*")
                               if i.suffix.lower() in [".jpeg", ".jpg", ".png"]]
        self.folder_name = Path(self.config.src_dir).name.split("/")[-1]

    # function to read a file
    def open_mapping(self, path_mapping: str) -> dict[int, str]:
        with open(path_mapping, 'r') as txt_file:
            lines = txt_file.readlines()
            lines = [i.strip() for i in lines]
            return {k: v for k, v in enumerate(lines)}

    # extracting crops from images
    def extract_crops(self, results: list[Results], config: BaseConfig) -> dict[str, torch.Tensor]:
        dict_crops = {}
        for res_per_img in results:
            if len(res_per_img) > 0:
                crops_per_img = []
                for box in res_per_img.boxes:
                    x0, y0, x1, y1 = box.xyxy.cpu().numpy().ravel().astype(np.int32)
                    crop = res_per_img.orig_img[y0: y1, x0: x1]
                    crop = cv2.resize(crop, config.imgsz, interpolation=cv2.INTER_LINEAR)
                    crop = torch.from_numpy(crop.transpose(2, 0, 1))
                    crop = crop.unsqueeze(0)
                    crop = normalize(crop.float(), mean=self.mean, std=self.std)
                    crops_per_img.append(crop)
                dict_crops[Path(res_per_img.path).name] = torch.cat(crops_per_img)
        return dict_crops

    # doing predictions, call to start processing
    def process_images(self) -> None:
        if not self.pathes_to_imgs:
            self.logger.info("No images to process.")
            return

        list_predictions = []
        num_packages_det = np.ceil(len(self.pathes_to_imgs) /
                                   self.config.detector.batch_size).astype(np.int32)
        exifs = self.extract_exifs()

        with torch.no_grad():
            for i in tqdm(range(num_packages_det), colour="green"):
                batch_images_det = self.pathes_to_imgs[self.config.detector.batch_size * i:
                                                       self.config.detector.batch_size * (1 + i)]
                results_det = self.detector(batch_images_det,
                                            iou=self.config.detector.iou,
                                            conf=self.config.detector.conf,
                                            imgsz=self.config.detector.imgsz,
                                            verbose=False,
                                            device=self.device)
                if len(results_det) > 0:
                    dict_crops = self.extract_crops(results_det, config=self.config.classificator)
                    list_predictions.extend(self.classify_crops(dict_crops, exifs))

        self.save_predictions(list_predictions)

    # extracting metadata from images
    def extract_exifs(self) -> list:
        exifs = []
        for img in self.pathes_to_imgs:
            im = Image.open(img)
            exif = im.getexif()
            creation_time = exif.get(306)
            exifs.append(creation_time)
        return exifs

    # detailed classification
    def classify_crops(self,
                       dict_crops: dict[str, torch.Tensor],
                       exifs: list) -> list:
        list_predictions = []
        for img_name, batch_images_cls in dict_crops.items():
            exif = exifs[self.pathes_to_imgs.index(Path(self.config.src_dir) / img_name)]
            num_packages_cls = np.ceil(len(batch_images_cls) /
                                       self.config.classificator.batch_size).astype(np.int32)
            for j in range(num_packages_cls):
                batch_images_cls = batch_images_cls[self.config.classificator.batch_size * j:
                                                    self.config.classificator.batch_size * (1 + j)]
                logits = self.classificator(batch_images_cls.to(self.device))
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                top_p, top_class_idx = probabilities.topk(1, dim=1)
                top_p = top_p.cpu().numpy().ravel()
                top_class_idx = top_class_idx.cpu().numpy().ravel()
                class_names = [self.mapping[top_class_idx[idx]] for idx, _
                               in enumerate(batch_images_cls)]
                list_predictions.extend([[self.folder_name,
                                          name, cls, prob, '--', exif]
                                         for name, cls, prob
                                         in zip(repeat(img_name, len(class_names)), class_names, top_p)])
        return list_predictions

    # saving data to local database
    def save_predictions(self, list_predictions: list) -> None:
        table = pd.DataFrame(list_predictions, columns=["folder_name",
                                                        "image_name",
                                                        "class_predict",
                                                        "confidence",
                                                        "registration_class",
                                                        "registration_date"])
        grouped_df = table.groupby(["image_name", "class_predict"]).agg(
            count=('confidence', 'size'),
            confidence=('confidence', 'mean'))
        merged_df = table.merge(grouped_df, on=["image_name", "class_predict"],
                                how="left").drop_duplicates(subset=["image_name", "class_predict"])
        merged_df = merged_df.drop(columns=["confidence_x"]).rename(columns={"confidence_y": "confidence"})
        merged_df.to_csv("table_final.csv", index=False)

        for _, row in merged_df.iterrows():
            self.db.add_image(folder_name=row["folder_name"],
                              image_name=row["image_name"],
                              class_predict=row["class_predict"],
                              confidence=row["confidence"],
                              registration_class=row["registration_class"],
                              registration_date=row["registration_date"],
                              count=row["count"])
        self.create_registrations()

    def create_registrations(self):
        process_data(self.db)
        self.db.close()


if __name__ == "__main__":
    processor = ImageProcessor(config_path=os.path.join("configs", "config.yml"))
    processor.process_images()
