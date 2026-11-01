"""功率与场站读取。训练可依赖官方 Release；推理仅需 stats + inputs。"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.timeutil import minute_of_time


def load_stations() -> pd.DataFrame:
    return pd.read_csv(config.DATA_ROOT / "stations.csv", dtype={"station_id": str})


def load_power_raw(split: str | None = None) -> pd.DataFrame:
    splits = ("train", "val") if split is None else (split,)
    frames = []
    for name in splits:
        path = config.DATA_ROOT / "power" / f"{name}.csv.gz"
        df = pd.read_csv(path, dtype={"station_id": str}, float_precision="round_trip")
        df["split"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["minute"] = out["timestamp"].map(minute_of_time)
    return out


class PowerStore:
    """稠密 [站, 15min] 功率表，覆盖 train+val 网格。"""

    def __init__(self, power: pd.DataFrame | None = None, end_minute: int | None = None):
        raw = load_power_raw() if power is None else power.copy()
        if "minute" not in raw.columns:
            raw["minute"] = raw["timestamp"].map(minute_of_time)
        self.raw = raw
        self.station_ids = list(config.STATION_IDS)
        end = int(end_minute if end_minute is not None else config.SPLIT_BOUNDS["val"][1])
        self.minutes = np.arange(0, end, 15, dtype=np.int64)
        self.power = np.full((len(self.station_ids), len(self.minutes)), np.nan, dtype=np.float64)
        self.missing = np.ones((len(self.station_ids), len(self.minutes)), dtype=bool)
        self._minute_to_idx = {int(m): i for i, m in enumerate(self.minutes)}
        for i, sid in enumerate(self.station_ids):
            sub = raw.loc[raw.station_id == sid]
            idx = sub["minute"].map(self._minute_to_idx)
            valid = idx.notna()
            ii = idx[valid].astype(int).to_numpy()
            vals = sub.loc[valid, "power_ratio"].to_numpy(dtype=float)
            miss = sub.loc[valid, "is_missing"].fillna(1).to_numpy(dtype=bool)
            self.power[i, ii] = vals
            self.missing[i, ii] = miss | ~np.isfinite(vals)
            self.power[i, self.missing[i]] = np.nan

    def series(self, station_id: str) -> pd.Series:
        i = self.station_ids.index(station_id)
        return pd.Series(self.power[i], index=self.minutes, name=station_id)


def make_release(cache_blocks: int = 60):
    if str(config.TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(config.TOOLS_DIR))
    from pv2h import Release  # noqa: WPS433

    return Release(config.DATA_ROOT, cache_blocks=cache_blocks)
