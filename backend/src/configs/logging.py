import logging
import queue
from logging.handlers import QueueHandler, QueueListener
from datetime import datetime
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter."""
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        if not log_record.get('asctime'):
            log_record['asctime'] = datetime.utcnow().isoformat()


def setup_logging():
    """asynchronous logging using QueueHandler and QueueListener."""
    try:
        log_queue = queue.Queue(maxsize=1000)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomJsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        console_handler.setLevel(logging.INFO)

        listener = QueueListener(log_queue, console_handler)
        queue_handler = QueueHandler(log_queue)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(queue_handler)

        listener.start()
        logging.info("Logging is configured successfully.")
        return listener
    except Exception as e:
        print(f"Logging could not be configured: {e}")
        raise
