import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

__all__ = ["get_graph_logger", "LoggerWithCapture"]


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        for attr in ["user_id", "request_id", "thread_id"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class LogCapturingHandler(logging.Handler):
    """Custom handler that captures formatted log messages."""

    def __init__(self):
        super().__init__()
        self.messages: List[str] = []

    def emit(self, record):
        # Format the record and store it
        formatted_message = self.format(record)
        self.messages.append(formatted_message)

    def get_messages(self) -> List[str]:
        return self.messages.copy()

    def clear_messages(self):
        self.messages.clear()


class LoggerWithCapture:
    """Wrapper around logger that captures formatted messages."""

    def __init__(self, logger: logging.Logger, capture_handler: LogCapturingHandler):
        self._logger = logger
        self._capture_handler = capture_handler

    def _log_and_return(self, level: str, message: str) -> str:
        """Log message and return the formatted text."""
        # Get the last message count before logging
        before_count = len(self._capture_handler.messages)

        # Log the message
        getattr(self._logger, level.lower())(message)

        # Get the newly added message
        if len(self._capture_handler.messages) > before_count:
            return self._capture_handler.messages[-1]
        return message  # Fallback

    def debug(self, message: str) -> str:
        return self._log_and_return("DEBUG", message)

    def info(self, message: str) -> str:
        return self._log_and_return("INFO", message)

    def warning(self, message: str) -> str:
        return self._log_and_return("WARNING", message)

    def error(self, message: str) -> str:
        return self._log_and_return("ERROR", message)

    def critical(self, message: str) -> str:
        return self._log_and_return("CRITICAL", message)

    def get_all_messages(self) -> List[str]:
        """Get all captured messages."""
        return self._capture_handler.get_messages()

    def clear_messages(self):
        """Clear all captured messages."""
        self._capture_handler.clear_messages()

    # Delegate other logger methods
    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_environment() -> str:
    """Get the current environment (development/production)."""
    return os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()


def is_production() -> bool:
    """Check if running in production environment."""
    env = get_environment()
    return env in ["production", "prod", "live"]


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: Optional[bool] = None,
    format_string: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 10,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    Automatically configures for development vs production environments.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        console_output: Whether to output to console
        format_string: Custom format string
        max_bytes: Maximum size of log file before rotation (production)
        backup_count: Number of backup files to keep (production)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Environment-aware defaults
    prod_env = is_production()

    if level is None:
        level = "WARNING" if prod_env else "INFO"

    if console_output is None:
        console_output = not prod_env  # Disable console in production by default

    logger.setLevel(getattr(logging, level.upper()))

    # Default format
    if format_string is None:
        if prod_env:
            # Use JSON format for production
            format_string = None  # Will use JSONFormatter
        else:
            format_string = "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if prod_env and format_string is None:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColoredFormatter(format_string)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if prod_env:
            # Use rotating file handler in production
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
        else:
            # Use regular file handler in development
            file_handler = logging.FileHandler(log_file)

        file_handler.setLevel(getattr(logging, level.upper()))

        if prod_env and format_string is None:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                format_string
                or "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
            )

        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_graph_logger(module_name: str) -> LoggerWithCapture:
    """
    Get a logger for the project with consistent configuration and message capturing.
    Automatically configures for development vs production environments.

    Args:
        module_name: Usually pass __name__ from the calling module

    Returns:
        Configured logger instance with message capturing capability
    """
    # Extract the actual module name from the full path
    if module_name.startswith("src."):
        logger_name = module_name[4:]  # Remove "src." prefix
    else:
        logger_name = module_name

    # Set up project logs directory
    project_root = Path(__file__).parent.parent.parent
    log_file = project_root / "logs" / f"aria-{datetime.now().strftime('%Y-%m-%d')}.log"

    # Environment-aware configuration
    prod_env = is_production()

    # Create the base logger
    base_logger = setup_logger(
        name=logger_name,
        level="WARNING" if prod_env else "INFO",
        log_file=str(log_file),
        console_output=not prod_env,  # Disable console in production
        format_string=(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
            if not prod_env
            else None
        ),
    )

    # Create capture handler with the same format
    capture_handler = LogCapturingHandler()
    capture_handler.setLevel(getattr(logging, "WARNING" if prod_env else "INFO"))

    if prod_env:
        capture_formatter = JSONFormatter()
    else:
        capture_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"
        )

    capture_handler.setFormatter(capture_formatter)
    base_logger.addHandler(capture_handler)

    return LoggerWithCapture(base_logger, capture_handler)


# Convenience functions for quick logging
def debug(message: str, logger_name: Optional[str] = None):
    """Log a debug message."""
    logger = get_graph_logger(logger_name or "aria.debug")
    logger.debug(message)


def info(message: str, logger_name: Optional[str] = None):
    """Log an info message."""
    logger = get_graph_logger(logger_name or "aria.info")
    logger.info(message)


def warning(message: str, logger_name: Optional[str] = None):
    """Log a warning message."""
    logger = get_graph_logger(logger_name or "aria.warning")
    logger.warning(message)


def error(message: str, logger_name: Optional[str] = None):
    """Log an error message."""
    logger = get_graph_logger(logger_name or "aria.error")
    logger.error(message)


def critical(message: str, logger_name: Optional[str] = None):
    """Log a critical message."""
    logger = get_graph_logger(logger_name or "aria.critical")
    logger.critical(message)
