import logging
import os
import json
from config import LOG_DIR

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line_no": record.lineno,
            "threadName": record.threadName
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logger(name):
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log"))
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = JsonFormatter('%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers if not already added
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
