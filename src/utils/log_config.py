"""
Centralized logging configuration for the project.
Provides consistent logging format, log rotation, and global settings.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import colorama
from colorama import Fore, Style
from datetime import datetime

# Initialize colorama for cross-platform colored terminal output
colorama.init()

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Log colors for different levels
LOG_COLORS = {
    "DEBUG": Fore.BLUE,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Style.BRIGHT,
}

# Log format strings
CONSOLE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
FILE_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages in the console"""

    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


def configure_logger(
    name=None,
    log_dir=None,
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    log_file_size_mb=10,
    backup_count=5,
    add_timestamp=True,
):
    """
    Configure a logger with console and file handlers, with log rotation

    Args:
        name (str, optional): Logger name (uses root logger if None)
        log_dir (str, optional): Directory to save log files (uses DEFAULT_LOG_DIR if None)
        console_level (int): Logging level for console output
        file_level (int): Logging level for file output
        log_file_size_mb (int): Maximum log file size in MB before rotation
        backup_count (int): Number of backup log files to keep
        add_timestamp (bool): Whether to add timestamp to log filename

    Returns:
        logging.Logger: Configured logger
    """
    # Get or create logger
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(min(console_level, file_level))  # Set to the more verbose level

    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    console_formatter = ColoredFormatter(CONSOLE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    file_formatter = logging.Formatter(FILE_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (with rotation)
    if log_dir:
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename
        base_filename = name if name else "application"
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_filename}_{timestamp}.log"
        else:
            filename = f"{base_filename}.log"

        log_file = os.path.join(log_dir, filename)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=log_file_size_mb * 1024 * 1024,  # Convert MB to bytes
            backupCount=backup_count,
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_level_from_string(level_str):
    """
    Convert a string log level to the corresponding logging level constant

    Args:
        level_str (str): Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        int: Corresponding logging level constant
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return levels.get(level_str.upper(), logging.INFO)


def get_child_logger(parent_logger, child_name):
    """
    Get a child logger that inherits the parent's handlers

    Args:
        parent_logger (logging.Logger): Parent logger
        child_name (str): Child logger name

    Returns:
        logging.Logger: Child logger
    """
    return logging.getLogger(f"{parent_logger.name}.{child_name}")
