"""Logging setup.

One configuration function, called once at process start. Modules call
`get_logger(__name__)` and never touch handlers themselves, so log output stays
consistent whether the code runs in the ETL job, the training script, a
notebook, or the API container.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from src.utils.config import get_settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Install a single stdout handler on the root logger.

    Idempotent: repeated calls are ignored unless `force` is set. stdout rather
    than stderr because Docker and Render both stream stdout as the service log.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    resolved = (level or get_settings().log_level).upper()

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(resolved)

    # These libraries are chatty at INFO and drown out our own progress lines.
    for noisy in ("matplotlib", "numba", "shap", "httpx", "urllib3",
                  "google_genai", "asyncio", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def log_duration(logger: logging.Logger, label: str,
                 level: int = logging.INFO) -> Iterator[None]:
    """Time a block and log how long it took.

    Used liberally through the ETL and training pipelines: on a 55M-row ingest
    the difference between "hung" and "working" is knowing the last step's cost.

        with log_duration(log, "ingest bureau_balance"):
            ...
    """
    start = time.perf_counter()
    logger.log(level, "%s ...", label)
    try:
        yield
    except Exception:
        logger.error("%s FAILED after %.1fs", label, time.perf_counter() - start)
        raise
    logger.log(level, "%s done in %.1fs", label, time.perf_counter() - start)
