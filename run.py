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


L = Logger()
db = DB()

# L.info("Hello, world!")
# L.error("Hello, world!")
# L.debug("Hello, world!")

# def add_image(self, folder_name: str, image_name: str, class_predict: str, registration_class: str, registration_date: str, count: int, max_count: int)

config = MainConfig(config_sources=FileSource(file=os.path.join("configs", "config.yml")))
device = config.device
