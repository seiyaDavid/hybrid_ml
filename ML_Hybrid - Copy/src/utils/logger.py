"""
Logging configuration and utilities for the ML Hybrid Theme Analysis system.

This module provides centralized logging configuration with proper formatting,
file rotation, and different log levels for different components.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any
from .config import config


def setup_logging() -> None:
    """
    Configure logging for the application.

    Sets up loguru logger with file and console handlers,
    proper formatting, and rotation policies.
    """
    try:
        # Remove default handler
        logger.remove()

        # Get logging configuration
        log_config = config.get_logging_config()

        # Create logs directory
        log_file = Path(log_config.get("file", "logs/app.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Console handler
        logger.add(
            sys.stdout,
            format=log_config.get(
                "format",
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            ),
            level=log_config.get("level", "INFO"),
            colorize=True,
        )

        # File handler with rotation
        logger.add(
            log_file,
            format=log_config.get(
                "format",
                "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            ),
            level=log_config.get("level", "INFO"),
            rotation=log_config.get("rotation", "10 MB"),
            retention=log_config.get("retention", "30 days"),
            compression="zip",
        )

        logger.info("Logging configured successfully")

    except Exception as e:
        print(f"Failed to configure logging: {e}")
        # Fallback to basic logging
        logger.add(sys.stdout, level="INFO")


def get_logger(name: str):
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name for the logger

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize logging
setup_logging()
