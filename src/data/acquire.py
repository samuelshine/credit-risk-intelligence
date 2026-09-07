"""Getting the Home Credit dataset onto disk.

The dataset is never committed (it is ~2.7 GB unzipped and Kaggle's competition
rules forbid redistribution), so every environment - a laptop, the ETL
container, the Render instance on first boot - has to fetch it. This module is
the single place that happens, and it accepts three routes in priority order:

1. Already-extracted CSVs in `data/raw/`  -> nothing to do.
2. A `home-credit-default-risk.zip` sitting in `data/` -> extract it.
   This is the manual escape hatch for anyone who would rather click "Download
   All" on Kaggle than manage an API token.
3. The Kaggle API, using `KAGGLE_API_TOKEN`.

Route 2 exists because the competition requires accepting its rules in a
browser before *any* download works, API included - so an evaluator may well
already have the zip in hand.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import get_settings
from src.utils.logger import get_logger, log_duration

log = get_logger(__name__)

COMPETITION = "home-credit-default-risk"
ZIP_NAME = f"{COMPETITION}.zip"

#: The ten files the competition ships. `expected_rows` is the published row
#: count, used as an integrity check after ingest - a truncated download is
#: otherwise silent and would quietly corrupt every downstream number.
#: `None` means "not asserted" (the description file is metadata, not data).
@dataclass(frozen=True)
class SourceFile:
    filename: str
    table: str
    expected_rows: int | None
    description: str


SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile("application_train.csv", "application_train", 307_511,
               "One row per loan application, with the TARGET default label."),
    SourceFile("application_test.csv", "application_test", 48_744,
               "Unlabelled applications held out by the competition."),
    SourceFile("bureau.csv", "bureau", 1_716_428,
               "Prior credits at other institutions, reported to the credit bureau."),
    SourceFile("bureau_balance.csv", "bureau_balance", 27_299_925,
               "Monthly status history for each bureau credit."),
    SourceFile("previous_application.csv", "previous_application", 1_670_214,
               "Earlier Home Credit applications by the same clients."),
    SourceFile("installments_payments.csv", "installments_payments", 13_605_401,
               "Scheduled vs actual repayments - the repayment-behaviour signal."),
    SourceFile("credit_card_balance.csv", "credit_card_balance", 3_840_312,
               "Monthly credit-card balance snapshots."),
    SourceFile("POS_CASH_balance.csv", "pos_cash_balance", 10_001_358,
               "Monthly point-of-sale and cash-loan balance snapshots."),
    SourceFile("sample_submission.csv", "sample_submission", 48_744,
               "Competition submission template. Not used by the platform."),
    SourceFile("HomeCredit_columns_description.csv", "column_descriptions", None,
               "Official column glossary. Feeds the chatbot's schema card."),
)

#: Files the platform genuinely needs. `sample_submission` is ignored.
REQUIRED_FILENAMES = tuple(
    f.filename for f in SOURCE_FILES if f.filename != "sample_submission.csv"
)


class DatasetUnavailable(RuntimeError):
    """Raised when the data is absent and cannot be fetched automatically."""


def missing_files(raw_dir: Path) -> list[str]:
    """Which required CSVs are not yet on disk."""
    return [name for name in REQUIRED_FILENAMES if not (raw_dir / name).exists()]


def is_available(raw_dir: Path | None = None) -> bool:
    """True when every required CSV is present."""
    raw_dir = raw_dir or get_settings().raw_data_dir
    return not missing_files(raw_dir)


def _extract_zip(zip_path: Path, raw_dir: Path, *, remove_zip: bool) -> None:
    """Unpack the competition archive into `raw_dir`.

    Home Credit ships a flat archive, but we still resolve each member against
    the destination to refuse path traversal rather than trusting the archive.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    with log_duration(log, f"extract {zip_path.name}"):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = (raw_dir / Path(member.filename).name).resolve()
                if not str(target).startswith(str(raw_dir.resolve())):
                    raise DatasetUnavailable(
                        f"Refusing unsafe archive path: {member.filename!r}"
                    )
                if target.exists():
                    log.debug("already extracted, skipping %s", target.name)
                    continue
                with archive.open(member) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                log.info("  extracted %-42s %8.1f MB",
                         target.name, target.stat().st_size / 1e6)

    if remove_zip:
        # Disk is the binding constraint on a small cloud instance: the archive
        # is ~700 MB and is useless once unpacked.
        size_mb = zip_path.stat().st_size / 1e6
        zip_path.unlink()
        log.info("removed %s, reclaiming %.0f MB", zip_path.name, size_mb)


def _download_from_kaggle(dest_dir: Path) -> Path:
    """Fetch the competition archive via the Kaggle API.

    Imported lazily: the Kaggle SDK prints an auth banner at import time when no
    credentials exist, which would pollute the logs of every process that merely
    imports this module.
    """
    settings = get_settings()
    if settings.kaggle_api_token and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = settings.kaggle_api_token

    has_token = bool(
        os.environ.get("KAGGLE_API_TOKEN")
        or (Path.home() / ".kaggle" / "access_token").exists()
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
    )
    if not has_token:
        raise DatasetUnavailable(
            "No Kaggle credentials found.\n"
            "  Either set KAGGLE_API_TOKEN in .env (generate one at\n"
            "  https://www.kaggle.com/settings/api), or download the archive by\n"
            f"  hand and place it at {dest_dir / ZIP_NAME}.\n"
            "  Either route requires accepting the competition rules first at\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )

    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    api = KaggleApi()
    api.authenticate()
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with log_duration(log, f"download {COMPETITION} from Kaggle (~700 MB)"):
            api.competition_download_files(COMPETITION, path=str(dest_dir), quiet=False)
    except Exception as exc:
        # The SDK raises requests.HTTPError, not DatasetUnavailable, on a 403 -
        # which is what Kaggle returns for a valid token that has not accepted
        # the competition's rules yet, not for a bad token. Distinguishing the
        # two here saves a confusing raw traceback for the common case.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 403:
            raise DatasetUnavailable(
                f"Kaggle rejected the download (403 Forbidden). Your token is "
                f"valid, but this competition requires accepting its rules in "
                f"the browser first - the API enforces that separately from "
                f"authentication:\n"
                f"  https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
                f"Click 'I Understand and Accept' there, then rerun this."
            ) from exc
        raise

    zip_path = dest_dir / ZIP_NAME
    if not zip_path.exists():
        raise DatasetUnavailable(
            f"Kaggle reported success but {zip_path} is missing. This usually "
            f"means the competition rules have not been accepted for your "
            f"account: https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )
    return zip_path


def ensure_dataset(*, allow_download: bool = True) -> Path:
    """Guarantee the raw CSVs exist, and return the directory holding them.

    Idempotent and safe to call on every container start: when the data is
    already present this returns immediately without touching the network.
    """
    settings = get_settings()
    raw_dir = settings.raw_data_dir

    absent = missing_files(raw_dir)
    if not absent:
        log.info("dataset present: %d files in %s", len(REQUIRED_FILENAMES), raw_dir)
        return raw_dir
    log.info("dataset incomplete, %d file(s) missing: %s",
             len(absent), ", ".join(absent))

    # Route 2: a manually-downloaded archive, in data/ or data/raw/.
    for candidate in (settings.data_dir / ZIP_NAME, raw_dir / ZIP_NAME):
        if candidate.exists():
            log.info("found archive at %s", candidate)
            _extract_zip(candidate, raw_dir, remove_zip=True)
            break
    else:
        # Route 3: the Kaggle API.
        if not allow_download:
            raise DatasetUnavailable(
                f"Dataset missing from {raw_dir} and downloading is disabled."
            )
        zip_path = _download_from_kaggle(settings.data_dir)
        _extract_zip(zip_path, raw_dir, remove_zip=True)

    still_absent = missing_files(raw_dir)
    if still_absent:
        raise DatasetUnavailable(
            f"Still missing after extraction: {', '.join(still_absent)}"
        )

    total_gb = sum(
        (raw_dir / f.filename).stat().st_size
        for f in SOURCE_FILES if (raw_dir / f.filename).exists()
    ) / 1e9
    log.info("dataset ready: %s (%.2f GB)", raw_dir, total_gb)
    return raw_dir
