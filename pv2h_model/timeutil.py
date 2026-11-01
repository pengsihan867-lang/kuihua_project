"""无年份 UTC 时间戳与分钟偏移互转（与官方 tools/pv2h.py 一致）。"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from pv2h_model import config

BASE = datetime(2001, 1, 1)


def minute_of_time(value: str) -> int:
    parsed = datetime.strptime("2001-" + value, "%Y-%m-%d %H:%M")
    if parsed.strftime("%m-%d %H:%M") != value:
        raise ValueError(f"Expected MM-DD HH:mm: {value!r}")
    return int((parsed - BASE).total_seconds() // 60)


def time_of_minute(value: int) -> str:
    return (BASE + timedelta(minutes=int(value))).strftime("%m-%d %H:%M")


def issue_minutes(split: str) -> np.ndarray:
    start, end = config.SPLIT_BOUNDS[split]
    return np.arange(start, end - 120, 15, dtype=np.int64)


def latest_power_minute(issue_minute: int) -> int:
    return int(issue_minute) - 15


def latest_cloud_minute(issue_minute: int) -> int:
    return ((int(issue_minute) - 20) // 10) * 10
