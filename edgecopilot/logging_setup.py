"""Structured JSON logging (one object per line) for log shippers (Loki, ELK, journald)."""
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": round(time.time(), 3), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in ("camera", "alert_id", "hazard", "user", "event"):
            if hasattr(record, k):
                d[k] = getattr(record, k)
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def setup(level: str = "INFO", json_logs: bool = True) -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel("WARNING")
