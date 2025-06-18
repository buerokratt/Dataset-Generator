"""
Logging utilities for the synthetic dataset generation service using Loguru
"""

from loguru import logger
from typing import Optional
import sys
from pathlib import Path
from datetime import datetime

# Track if logger has been initialized
LOGGER_INITIALIZED = False
LOGFILE_NAME = "synthetic_data_service.log"


def setup_logger(
    name: str = "synthetic-data-service",
    level: Optional[str] = "DEBUG",
    log_to_file: bool = True,
) -> None:
    """
    Setup the Loguru logger with the specified name and level.
    Since Loguru manages a singleton logger, this just configures the output format and level.

    Args:
        name: Logger name (used for context tagging)
        level: Logging level as a string (e.g., "INFO", "DEBUG")
        log_to_file: Whether to log to a file in addition to stdout
    """
    global LOGGER_INITIALIZED

    # Only set up the logger once
    if LOGGER_INITIALIZED:
        return

    logger.remove()  # Remove default handler to prevent duplicate logs

    # Add console handler
    logger.add(
        sink=sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{name}</cyan> | "
        "<level>{message}</level>",
        enqueue=True,  # for multi-threading support
    )

    # Add file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create log file path with date
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"{date_str}_{LOGFILE_NAME}"
        log_path = logs_dir / log_filename

        logger.add(
            sink=str(log_path),
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
            rotation="10 MB",
            retention="1 month",
            compression="zip",
            enqueue=True,
        )

        # Only log this message once
        if not LOGGER_INITIALIZED:
            logger.info(f"Log file created at: {log_path.absolute()}")

    # Mark logger as initialized
    LOGGER_INITIALIZED = True
