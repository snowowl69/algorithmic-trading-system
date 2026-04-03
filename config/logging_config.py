"""
Structured logging configuration.
Provides a consistent logger factory for every module.
"""

import logging
import sys
from config.settings import SYSTEM


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard format bound to *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, SYSTEM.log_level.upper(), logging.INFO))
    return logger
