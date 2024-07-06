from typing import Tuple

from confz import BaseConfig


class DetectorArgs(BaseConfig):
    weights: str
    iou: float
    conf: float
    imgsz: Tuple[int, int]
    batch_size: int


class ClassificatorArgs(BaseConfig):
    weights: str
    imgsz: Tuple[int, int]
    batch_size: int


class LoggerArgs(BaseConfig):
    log_file: str
    log_dir: str


class MainConfig(BaseConfig):
    src_dir: str
    mapping: str
    device: str
    detector: DetectorArgs
    logger: LoggerArgs
    classificator: ClassificatorArgs