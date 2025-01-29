import logging
import sys
from dataclasses import dataclass

from loguru import logger


@dataclass()
class XaresSettings:
    env_root: str = "./env"
    audio_ready_filename: str = ".audio_tar_ready"
    encoded_ready_filename: str = ".encoded_tar_ready"

    def __post_init__(self):
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)

        # Make the logger with this format the default for all loggers in this package
        logger.configure(
            handlers=[
                {
                    "sink": sys.stdout,
                    "format": "<fg #FF6900>(X-ARES)</fg #FF6900> [<yellow>{time:YYYY-MM-DD HH:mm:ss}</yellow>] "
                    "<level>{message}</level>",
                    "level": "DEBUG",
                    "colorize": True,
                }
            ]
        )
        logger.level("ERROR", color="<red>")
        logger.level("INFO", color="<white>")
