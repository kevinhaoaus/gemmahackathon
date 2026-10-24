"""Structured logging: every line carries node_id and the current term."""
from __future__ import annotations

import logging
import sys


class _Ctx:
    node_id = "-"
    term = 0


CTX = _Ctx()


class _CtxFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.node_id = CTX.node_id
        record.term = CTX.term
        return True


def setup_logging(node_id: str, level: str = "INFO") -> None:
    CTX.node_id = node_id
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_CtxFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d level=%(levelname)s node=%(node_id)s term=%(term)s "
        "mod=%(name)s msg=%(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "uvicorn.access", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_term(term: int) -> None:
    CTX.term = term


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"swarmmind.{name}")
