"""Logging: quiet rich console + rotating JSON-lines file in <data_dir>/logs."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import UTC, datetime

from rich.logging import RichHandler

from jarvis.core.config import Settings

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything passed via `extra={...}` becomes a structured field.
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(settings: Settings) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)

    console = RichHandler(show_path=False, rich_tracebacks=True)
    console.setLevel(settings.logging.console_level.upper())
    root.addHandler(console)

    log_dir = settings.app.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "jarvis.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(settings.logging.level.upper())
    file_handler.setFormatter(
        JsonFormatter()
        if settings.logging.json_file
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
