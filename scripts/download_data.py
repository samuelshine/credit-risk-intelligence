#!/usr/bin/env python
"""CLI wrapper around `src.data.acquire.ensure_dataset`.

    python scripts/download_data.py

Run once before the ETL. The Docker entrypoint calls the same function, so the
behaviour is identical whether you fetch the data by hand or let the container
do it on first boot.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.acquire import DatasetUnavailable, ensure_dataset  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402


def main() -> int:
    configure_logging()
    log = get_logger("download_data")
    try:
        ensure_dataset()
    except DatasetUnavailable as exc:
        log.error("Could not obtain the dataset.\n%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
