"""
src/utils/logger.py
Centralised logging — writes to console + file.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs/", level: str = "INFO") -> logging.Logger:
    """
    Create or retrieve a named logger.

    Args:
        name:    Logger name (usually __name__ of the calling module)
        log_dir: Directory to write .log files
        level:   Logging level string (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured Python Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
