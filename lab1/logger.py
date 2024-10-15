import logging
from pathlib import Path


def get_logger(name: str, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / f"{name}.log"
    log_file.unlink() if log_file.exists() else None

    formatter = logging.Formatter('%(name)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
