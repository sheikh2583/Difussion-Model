"""
utils.logging — structured logging helpers.

Two complementary logging surfaces are provided:

setup_logger(name, log_dir)
    Builds and returns a standard Python Logger that writes INFO-level
    messages simultaneously to stdout and to a timestamped .log file
    inside `log_dir`.  Every Trainer instance creates one logger per
    algorithm name so log files are kept separate per run.

JsonlLogger
    Append-only structured event log that writes one JSON object per
    line to a designated .jsonl file.  Used by the Trainer to record
    machine-readable epoch events (loss, time, memory) that can be
    post-processed without parsing free-form log text.  Each record is
    automatically timestamped with a UTC ISO-8601 string.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone


def setup_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    log_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


class JsonlLogger:
    """Append-only structured event log, one JSON object per line."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")
