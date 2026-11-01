"""
VisionX - Centralized Logging Utility
Provides formatted console and file logging across all VisionX modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "VisionX",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a standardized logger instance.

    Args:
        name: The logger name, typically __name__ or module identifier.
        level: Minimum logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional Path to write log output to disk.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if the logger is retrieved multiple times
    if logger.handlers:
        return logger

    # Log Formatter: [Timestamp] [LEVEL] [Module:Line] - Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler (writes to sys.stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str = "VisionX") -> logging.Logger:
    """
    Convenience function to get a module-level logger.
    """
    return setup_logger(name=name)
