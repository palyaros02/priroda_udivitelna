from utils.logger import Logger
from db.db import DB

from pathlib import Path
from configs.config import MainConfig
from confz import BaseConfig, FileSource
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
db = DB()

# L.info("Hello, world!")
# L.error("Hello, world!")
# L.debug("Hello, world!")


MEAN = [123.675, 116.28, 103.535]
STD = [58.395, 57.12, 57.375]

config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
detector_config = config.detector
classificator_config = config.classificator
device = config.device

pathes_to_imgs = [i for i in Path(config.src_dir).glob("*")
    if i.suffix.lower() in [".jpeg", ".jpg", ".png"]]

folder_name = Path(config.src_dir).name.split("/")[-1]

detector = YOLO(detector_config.weights)
# detector = model.to(device)

classificator = torch.load(classificator_config.weights)
# classificator = classificator.to(device)

def open_mapping(path_mapping: str) -> dict[int, str]:
        with open(path_mapping, 'r') as txt_file:
            lines = txt_file.readlines()
            lines = [i.strip() for i in lines]
            dict_map = {k: v for k, v in enumerate(lines)}
        return dict_map

mapping = open_mapping(path_mapping=config.mapping)

def extract_crops(results: list[Results], config: BaseConfig) -> dict[str, torch.Tensor]:
    dict_crops = {}
    for res_per_img in results:
        if len(res_per_img) > 0:
            crops_per_img = []
            for box in res_per_img.boxes:
                x0, y0, x1, y1 = box.xyxy.cpu().numpy().ravel().astype(np.int32)
                crop = res_per_img.orig_img[y0: y1, x0: x1]

                # Do squared crop
                # crop = letterbox(img=crop, new_shape=config.imgsz, color=(0, 0, 0))
                crop = cv2.resize(crop, config.imgsz, interpolation=cv2.INTER_LINEAR)

                # Convert Array crop to Torch tensor with [batch, channels, height, width] dimensions
                crop = torch.from_numpy(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crop = normalize(crop.float(), mean=MEAN, std=STD)
                crops_per_img.append(crop)

            dict_crops[Path(res_per_img.path).name] = torch.cat(crops_per_img) # if len(crops_per_img) else None
    return dict_crops

if len(pathes_to_imgs):

    list_predictions = []

    num_packages_det = np.ceil(len(pathes_to_imgs) / detector_config.batch_size).astype(np.int32)


    exifs = []
    for img in pathes_to_imgs:
        im = Image.open(img)
        exif = im.getexif()
        creation_time = exif.get(306)
        exifs.append(creation_time)

    print(exifs)
    print(pathes_to_imgs)

    with torch.no_grad():
        for i in tqdm(range(num_packages_det), colour="green"):
            # Inference detector
            batch_images_det = pathes_to_imgs[detector_config.batch_size * i:
                                                detector_config.batch_size * (1 + i)]
            results_det = detector(batch_images_det,
                                    iou=detector_config.iou,
                                    conf=detector_config.conf,
                                    imgsz=detector_config.imgsz,
                                    verbose=False,
                                    device=device)


            if len(results_det) > 0:
            # if True:
                # Extract crop by bboxes
                dict_crops = extract_crops(results_det, config=classificator_config)

                # Inference classificator
                for img_name, batch_images_cls in dict_crops.items():
                    # if len(batch_images_cls) > classificator_config.batch_size:
                    num_packages_cls = np.ceil(len(batch_images_cls) / classificator_config.batch_size).astype(
                        np.int32)
                    for j in range(num_packages_cls):
                        batch_images_cls = batch_images_cls[classificator_config.batch_size * j:
                                                            classificator_config.batch_size * (1 + j)]
                        logits = classificator(batch_images_cls.to(device))
                        probabilities = torch.nn.functional.softmax(logits, dim=1)
                        top_p, top_class_idx = probabilities.topk(1, dim=1)

                        # Locate torch Tensors to cpu and convert to numpy
                        top_p = top_p.cpu().numpy().ravel()
                        top_class_idx = top_class_idx.cpu().numpy().ravel()

                        class_names = [mapping[top_class_idx[idx]] for idx, _ in enumerate(batch_images_cls)]
                        list_predictions.extend([[folder_name, name, cls, prob, '--', exif] for name, cls, prob, exif in
                                                    zip(repeat(img_name, len(class_names)), class_names, top_p, exifs)])

    # Create Dataframe with predictions
    table = pd.DataFrame(list_predictions, columns=["folder_name", "image_name", "class_predict", "confidence", "registration_class", "registration_date"])

    grouped_df = table.groupby(["image_name", "class_predict"]).agg(
    count=('confidence', 'size'),
    confidence=('confidence', 'mean'))

    merged_df = table.merge(grouped_df, on=["image_name", "class_predict"], how="left").drop_duplicates(subset=["image_name", "class_predict"])

    # del confidence_x, rename confidence_y to confidence

    merged_df = merged_df.drop(columns=["confidence_x"]).rename(columns={"confidence_y": "confidence"})


    print(merged_df.head(20))

    merged_df.to_csv("table_final.csv", index=False)

    for _, row in merged_df.iterrows():
        db.add_image(folder_name=row["folder_name"], image_name=row["image_name"], class_predict=row["class_predict"], confidence=row["confidence"], registration_class=row["registration_class"], registration_date=row["registration_date"], count=row["count"])


    db.close()
