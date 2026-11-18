"""
Structured logging configuration.
Provides a consistent logger factory for every module.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a standard format bound to *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Use UTF-8 stream to avoid UnicodeEncodeError on Windows cp1252 console
        import io
        utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        handler = logging.StreamHandler(utf8_stream)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    from config.settings import SYSTEM
    logger.setLevel(getattr(logging, SYSTEM.log_level.upper(), logging.INFO))
    return logger
