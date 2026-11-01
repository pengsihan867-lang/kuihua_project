"""任意分钟晴空值：pvlib + 冻结的 lat/lon/ghi_norm/scale（覆盖测试期）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.stats import load_stats

_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def clearsky_values(station_id: str, minutes: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """返回与 minutes 对齐的容量归一化晴空代理。"""
    stats = stats or load_stats()
    minutes = np.asarray(minutes, dtype=np.int64)
    if minutes.size == 0:
        return np.asarray([], dtype=float)

    # 优先查网格缓存（训练期预计算）
    if station_id in _CACHE:
        grid_m, grid_v = _CACHE[station_id]
        idx = {int(m): i for i, m in enumerate(grid_m)}
        out = np.full(len(minutes), np.nan, dtype=float)
        miss = []
        for j, m in enumerate(minutes):
            m15 = (int(m) // 15) * 15
            i = idx.get(m15)
            if i is not None:
                out[j] = grid_v[i]
            else:
                miss.append(j)
        if not miss:
            return out
        # 仅补算缺失分钟
        extra = clearsky_values_raw(station_id, minutes[miss], stats)
        out[miss] = extra
        return out
    return clearsky_values_raw(station_id, minutes, stats)


def clearsky_values_raw(station_id: str, minutes: np.ndarray, stats: dict) -> np.ndarray:
    meta = stats["stations"][station_id]
    lat, lon = float(meta["lat"]), float(meta["lon"])
    ghi_norm = float(meta["ghi_norm"])
    scale = float(meta["scale"])
    minutes = np.asarray(minutes, dtype=np.int64)
    import pvlib

    base = pd.Timestamp("2001-01-01 00:00:00")
    times = (base + pd.to_timedelta(minutes, unit="m")).tz_localize("UTC")
    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC")
    ghi = loc.get_clearsky(times, model="ineichen")["ghi"].to_numpy(dtype=float)
    return np.clip(ghi / max(ghi_norm, 1.0) * scale, 0.0, 1.5)


def clearsky_table(minutes: np.ndarray, stats: dict | None = None) -> dict[str, np.ndarray]:
    stats = stats or load_stats()
    return {sid: clearsky_values(sid, minutes, stats) for sid in config.STATION_IDS}


def build_clearsky_cache(minutes: np.ndarray, stats: dict | None = None, path: Path | None = None) -> Path:
    """预计算整段网格并写入 cache/clearsky.npz，同时填充内存缓存。"""
    stats = stats or load_stats()
    minutes = np.asarray(minutes, dtype=np.int64)
    arrays = {sid: clearsky_values_raw(sid, minutes, stats) for sid in config.STATION_IDS}
    for sid, arr in arrays.items():
        _CACHE[sid] = (minutes, arr)
    path = Path(path or (config.CACHE_DIR / "clearsky.npz"))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, minutes=minutes, **arrays)
    return path


def load_clearsky_cache(path: Path | None = None) -> bool:
    path = Path(path or (config.CACHE_DIR / "clearsky.npz"))
    if not path.exists():
        return False
    data = np.load(path)
    minutes = data["minutes"]
    for sid in config.STATION_IDS:
        _CACHE[sid] = (minutes, data[sid])
    return True
