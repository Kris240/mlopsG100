from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "handle_min"
ID_COLUMNS = ["record_id", "date"]
FEATURES = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
    "language",
    "team",
]
NUMERIC_FEATURES = FEATURES[:6]
CATEGORICAL_FEATURES = FEATURES[6:]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_language(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.replace({"eng.": "english", "hin.": "hindi", "reg.": "regional"})


def clean_data(frame: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    cleaned = frame.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().copy()
    cleaned["language"] = normalize_language(cleaned["language"])
    cleaned.loc[cleaned["queue_wait_sec"].eq(9999), "queue_wait_sec"] = np.nan
    if "date" in cleaned:
        dates = pd.to_datetime(cleaned["date"], errors="coerce")
        stuck = dates.between("2026-05-11", "2026-05-31") & cleaned["previous_calls"].eq(6)
        cleaned.loc[stuck, "previous_calls"] = np.nan
    return cleaned.reset_index(drop=True)


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return frame[FEATURES].copy(), frame[TARGET].copy()


def audit_data(frame: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    language = normalize_language(frame["language"])
    dates = pd.to_datetime(frame["date"])
    stuck = dates.between("2026-05-11", "2026-05-31") & frame["previous_calls"].eq(6)
    extra_train_columns = set(frame) - set(holdout) - {TARGET}
    return {
        "leaking_column": {"present": bool(extra_train_columns), "column": next(iter(extra_train_columns), None)},
        "unit_mixup": {"present": False, "column": None, "group_value": None, "factor": None},
        "sentinel_value": {"present": True, "column": "queue_wait_sec", "value": 9999, "count": int(frame["queue_wait_sec"].eq(9999).sum())},
        "duplicate_rows": {"present": True, "count": int(frame.duplicated().sum())},
        "stuck_sensor": {"present": True, "column": "previous_calls", "start_date": dates[stuck].min().strftime("%Y-%m-%d"), "end_date": dates[stuck].max().strftime("%Y-%m-%d")},
        "target_typos": {"present": False, "count": None},
        "spelling_variants": {"present": language.nunique() < frame["language"].nunique(), "column": "language", "true_levels": int(language.nunique())},
    }
