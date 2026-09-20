import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config.settings import settings
import sys

_LOG_FORMAT="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

_LOG_DIR=Path(__file__).resolve().parents[2]/"logs"
_LOG_DIR.mkdir(parents=True,exist_ok=True)

_LEVEL =logging.DEBUG if settings.DEBUG else logging.INFO

def _build_logger(name:str)->logging.Logger:
    logger =logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(_LEVEL)
    logger.propagate=False

    console=logging.StreamHandler(sys.stdout)
    console.setLevel(_LEVEL)
    console.setFormatter(logging.Formatter(_LOG_FORMAT,datefmt=_DATE_FORMAT))
    logger.addHandler(console)

    file_handler=RotatingFileHandler(
        _LOG_DIR / "app.log",maxBytes=10*1024*1024,backupCount=5,encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT,datefmt=_DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger

def get_logger(name:str ="agentfm")->logging.Logger:
    return _build_logger(name)