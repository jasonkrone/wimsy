import os
import logging

STR_TO_LOG_LEVEL = {
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

DEFAULT_LOG_LEVEL = "INFO"

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("JPT")

level = STR_TO_LOG_LEVEL[os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)]
logger.setLevel(level)
