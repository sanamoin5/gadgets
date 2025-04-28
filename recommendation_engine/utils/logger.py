import logging
import os

def setup_logger(log_dir, log_filename="training.log"):
    """
    Setup a Python logger that logs to both console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Reset handlers to avoid duplicate logs

    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Stream Handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

