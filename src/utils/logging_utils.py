"""
Logging utility functions for consistent logging across the project
"""

import os
import sys
import logging
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Define color codes for different log levels
LOG_COLORS = {
    "DEBUG": Fore.BLUE,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.RED + Style.BRIGHT,
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages"""

    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(name, log_file=None, level=logging.INFO, add_console_handler=True):
    """
    Set up a logger with console and file handlers

    Args:
        name (str): Logger name
        log_file (str): Path to log file (None for no file logging)
        level (int): Logging level
        add_console_handler (bool): Whether to add a console handler

    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler if requested
    if add_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def setup_experiment_logger(experiment_name, output_dir="logs", level=logging.INFO):
    """
    Set up a logger for an experiment with timestamped log file

    Args:
        experiment_name (str): Name of the experiment
        output_dir (str): Directory to save log file
        level (int): Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.log")

    # Set up logger
    return setup_logger(experiment_name, log_file=log_file, level=level)


def get_log_level(verbosity):
    """
    Convert verbosity level to logging level

    Args:
        verbosity (int): Verbosity level (0-4)

    Returns:
        int: Logging level
    """
    levels = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG,
    }
    return levels.get(verbosity, logging.INFO)
