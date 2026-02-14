from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("ai_labeller")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_path is not None:
        has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        if not has_file_handler:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
