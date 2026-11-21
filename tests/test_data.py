import pandas as pd

from src.mp030.data import clean_data


def test_clean_data_removes_duplicate_sentinel_and_normalizes_language():
    row = {"record_id": "1", "date": "2026-01-01", "queue_wait_sec": 9999, "previous_calls": 2, "language": " REG.", "handle_min": 10}
    cleaned = clean_data(pd.DataFrame([row, row]))
    assert len(cleaned) == 1
    assert pd.isna(cleaned.loc[0, "queue_wait_sec"])
    assert cleaned.loc[0, "language"] == "regional"


def test_stuck_sensor_becomes_missing():
    row = {"record_id": "1", "date": "2026-05-15", "queue_wait_sec": 2, "previous_calls": 6, "language": "hindi", "handle_min": 10}
    assert pd.isna(clean_data(pd.DataFrame([row])).loc[0, "previous_calls"])
