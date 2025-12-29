"""Logging configuration for TabMixNN package."""

import logging
import sys
from typing import Optional

# Package-level logger
_PACKAGE_LOGGER = None
_LOG_LEVEL = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for the given module.
    
    Parameters
    ----------
    name : str
        Name of the module (e.g., 'estimator', 'model', 'families')
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Examples
    --------
    >>> from tabmixnn.logging_config import get_logger
    >>> logger = get_logger("estimator")
    >>> logger.info("Starting training...")
    """
    logger = logging.getLogger(f"tabmixnn.{name}")
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = False  # Prevent duplicate logs
    
    return logger


def set_log_level(level: str) -> None:
    """
    Set the logging level for all TabMixNN loggers.
    
    Parameters
    ----------
    level : str
        One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    
    Examples
    --------
    >>> from tabmixnn.logging_config import set_log_level
    >>> set_log_level('DEBUG')  # Enable verbose logging
    """
    global _LOG_LEVEL
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    if level.upper() not in level_map:
        raise ValueError(f"Invalid log level: {level}. Must be one of {list(level_map.keys())}")
    
    _LOG_LEVEL = level_map[level.upper()]
    
    # Update all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('tabmixnn.'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(_LOG_LEVEL)


def disable_logging() -> None:
    """
    Disable all TabMixNN logging.
    
    Useful for production environments or when running many models.
    
    Examples
    --------
    >>> from tabmixnn.logging_config import disable_logging
    >>> disable_logging()
    """
    set_log_level('CRITICAL')
    logging.getLogger('tabmixnn').disabled = True


def enable_logging() -> None:
    """
    Re-enable TabMixNN logging after calling disable_logging().
    
    Examples
    --------
    >>> from tabmixnn.logging_config import enable_logging
    >>> enable_logging()
    """
    set_log_level('INFO')
    logging.getLogger('tabmixnn').disabled = False
