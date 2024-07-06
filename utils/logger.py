import os

import loguru
from confz import FileSource

from configs.config import MainConfig

config = MainConfig(config_sources=FileSource(file=os.path.join("./configs", "config.yml")))
logger_config = config.logger

class Logger:
    def __init__(self):
        self.logger = loguru.logger
        self.logger.add(f"{logger_config.log_dir}/{logger_config.log_file}", rotation="10 MB")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)