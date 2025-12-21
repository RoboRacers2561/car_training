# car_training/utils/logging_utils.py
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parents[8].resolve()


def setup_logger(
    task_name: str,
    name: str = "car_training",
    console_level: int | None = logging.INFO,
    file_level: int | None = logging.INFO,
) -> logging.Logger:
    """
    Create a dedicated logger that won't spam duplicate lines when Isaac reloads code.
    - Logs to logs/sb3/<task_name>/YYYY-MM-DD/HH-MM-SS.log
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # important: don't double-print via root/Kit handlers

    # If handlers already exist, don't add again (Isaac can reload modules)
    if getattr(logger, "_configured", False):
        return logger
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H-%M-%S")

    log_dir = os.path.join(PROJECT_ROOT, "logs/sb3", task_name, current_date)
    filename = f"{current_time}.log"


    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_fmt = logging.Formatter(
        fmt="%(levelname)s | %(message)s"
    )

    if console_level is not None:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(console_level)
        sh.setFormatter(console_fmt)
        logger.addHandler(sh)

    if file_level is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / filename, mode="a", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)

    logger._configured = True
    logger.info("Logger initialized: %s (level=%s)", name)
    return logger
