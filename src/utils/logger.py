"""Central loguru configuration for structured console logging."""

from __future__ import annotations

import sys

from loguru import logger

LOG_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger(level: str = "INFO") -> None:
    """Replace default loguru handlers with a timestamped console handler."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )


__all__ = ["logger", "setup_logger"]
