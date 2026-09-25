import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config.settings import settings
import sys

_LOG_FORMAT="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT="%Y-%m-%d %H:%M:%S"

# Windows 控制台默认可能是 GBK(cp936)，而本项目的日志与脚本输出里含 emoji(/✓) 和中文。
# 不切换编码的话，写日志会抛 UnicodeEncodeError（日志系统会打印一大段 traceback），
# print 则可能直接把脚本打断。这里统一按 UTF-8 输出，遇到无法编码的字符用 ? 替代而非崩溃。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # 非标准流（如被重定向/被替换）时忽略
    pass

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