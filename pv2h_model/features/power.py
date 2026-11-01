"""功率 / 时间 / 晴空特征核心（数组接口，离线与在线共用）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv2h_model import config

POWER_FEAT_COLS = (
    [f"lag_{k}" for k in range(1, 9)]
    + [f"kt_lag_{k}" for k in range(1, 9)]
    + [
        "diff_1",
        "diff_2",
        "roll_mean_4",
        "roll_std_4",
        "roll_mean_8",
        "roll_std_8",
        "kt_roll_mean_4",
        "lag_day",
        "kt_lag_day",
        "tod_sin",
        "tod_cos",
        "bj_hour",
        "clearsky",
        "kt",
        "power",
        "other1_kt",
        "other1_power",
        "other2_kt",
        "other2_power",
    ]
)


def _safe_kt(y: float, cs: float, eps: float = 0.05) -> float:
    if not np.isfinite(y) or not np.isfinite(cs) or cs < eps:
        return np.nan
    return float(np.clip(y / cs, 0.0, 1.5))


def build_power_feature_tables(
    minutes: np.ndarray,
    power: np.ndarray,
    missing: np.ndarray,
    clearsky: dict[str, np.ndarray],
    station_ids: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    输入:
      minutes: [T] 15min 网格
      power/missing: [n_station, T]
      clearsky[sid]: [T]
    所有 lag/rolling 仅向后看，满足因果约束。
    """
    station_ids = station_ids or list(config.STATION_IDS)
    tables: dict[str, pd.DataFrame] = {}
    for i, sid in enumerate(station_ids):
        y = power[i].astype(float).copy()
        y[missing[i]] = np.nan
        cs = clearsky[sid].astype(float)
        s = pd.Series(y, index=minutes)
        kt = pd.Series([_safe_kt(a, b) for a, b in zip(y, cs)], index=minutes)
        df = pd.DataFrame({"minute": minutes, "power": y, "clearsky": cs, "kt": kt.to_numpy()})
        for lag in range(1, 9):
            df[f"lag_{lag}"] = s.shift(lag).to_numpy()
            df[f"kt_lag_{lag}"] = kt.shift(lag).to_numpy()
        df["diff_1"] = s.diff(1).to_numpy()
        df["diff_2"] = s.diff(2).to_numpy()
        df["roll_mean_4"] = s.shift(1).rolling(4, min_periods=1).mean().to_numpy()
        df["roll_std_4"] = s.shift(1).rolling(4, min_periods=1).std().to_numpy()
        df["roll_mean_8"] = s.shift(1).rolling(8, min_periods=1).mean().to_numpy()
        df["roll_std_8"] = s.shift(1).rolling(8, min_periods=1).std().to_numpy()
        df["kt_roll_mean_4"] = kt.shift(1).rolling(4, min_periods=1).mean().to_numpy()
        df["lag_day"] = s.shift(96).to_numpy()
        df["kt_lag_day"] = kt.shift(96).to_numpy()
        mod = minutes % 1440
        df["tod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
        df["tod_cos"] = np.cos(2 * np.pi * mod / 1440.0)
        df["bj_hour"] = ((minutes + 8 * 60) % 1440) // 60
        tables[sid] = df.set_index("minute")

    for sid in station_ids:
        others = [o for o in station_ids if o != sid]
        for j, o in enumerate(others):
            tables[sid][f"other{j + 1}_kt"] = tables[o]["kt"]
            tables[sid][f"other{j + 1}_power"] = tables[o]["power"]
    return tables


def lookup_power_feats(tables: dict[str, pd.DataFrame], station_id: str, power_minute: int) -> dict[str, float]:
    df = tables[station_id]
    m = (int(power_minute) // 15) * 15
    if m not in df.index:
        return {c: np.nan for c in POWER_FEAT_COLS}
    row = df.loc[m]
    out = {}
    for c in POWER_FEAT_COLS:
        v = row[c] if c in row.index else np.nan
        out[c] = float(v) if np.isfinite(v) else (float(v) if c in row.index and pd.notna(v) else np.nan)
    return out
