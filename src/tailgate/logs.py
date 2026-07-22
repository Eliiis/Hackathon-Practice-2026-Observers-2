import logging
import sys

from .config import ARTIFACTS

LOG_PATH = ARTIFACTS / "run.log"


def get(name="tailgate"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(LOG_PATH, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
