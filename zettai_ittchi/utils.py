"""Shared utilities: logging, IDs, timestamps."""

from __future__ import annotations

import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_level: str = "info") -> None:
    """Configure root logger for console + rotating file output."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

        # File
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "zettai.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def generate_request_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def timestamp() -> int:
    return int(time.time())
