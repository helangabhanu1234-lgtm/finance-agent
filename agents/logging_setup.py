"""Logging: human-readable agent.log + machine-readable trace.jsonl (one line per tool call)."""
import json
import logging
from datetime import datetime

import config

_configured = False


def setup_logging() -> logging.Logger:
    global _configured
    logger = logging.getLogger("finance_agent")
    if _configured:
        return logger
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(config.LOGS_DIR / "agent.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True
    return logger


def write_trace(event: dict) -> None:
    """Append a tool-call event to logs/trace.jsonl (used as evidence for the report/demo)."""
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now().isoformat(timespec="seconds"), **event}
        with (config.LOGS_DIR / "trace.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass   # logging must never break the agent
