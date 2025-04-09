import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from logging import Logger

class CustomLogger(Logger):
    def success(self, msg, *args, **kwargs):
        self.log(25, msg, *args, **kwargs)  # 25 是介于 WARNING(30) 和 INFO(20) 之间的自定义级别

def setup_logger():
    logging.addLevelName(25, 'SUCCESS')
    logger = logging.getLogger("emoji_bot")
    logger.__class__ = CustomLogger  # 替换 logger 类
    logger.setLevel(logging.DEBUG)

    logger = logging.getLogger("emoji_bot")
    logger.setLevel(logging.DEBUG)
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 文件处理器
    file_handler = RotatingFileHandler(
        log_dir / "bot.log",
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()