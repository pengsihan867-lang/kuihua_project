"""仅用训练集拟合的预处理统计量：经纬度、晴空归一化常数、夜间中位数。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.data import PowerStore, load_power_raw


def _beijing_hour(minutes: np.ndarray) -> np.ndarray:
    return ((np.asarray(minutes, dtype=np.int64) + 8 * 60) % 1440) // 60


def _night_mask(minutes: np.ndarray) -> np.ndarray:
    return np.isin(_beijing_hour(minutes), [0, 1, 2, 3, 4, 21, 22, 23])


def _infer_lat_lon(minutes: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    days = minutes // 1440
    rows = []
    for day in np.unique(days):
        mask = days == day
        yy = y[mask]
        mm = minutes[mask]
        if np.isfinite(yy).sum() < 20:
            continue
        dmax = np.nanmax(yy)
        if not np.isfinite(dmax) or dmax < 0.3:
            continue
        active = yy > max(0.05, 0.15 * dmax)
        if active.sum() < 8:
            continue
        am = mm[active]
        sunrise, sunset = am.min(), am.max()
        daylen_h = (sunset - sunrise) / 60.0
        noon_utc_min = ((sunrise + sunset) / 2.0) % 1440
        lon = float(np.clip(120.0 - (noon_utc_min - 12 * 60) / 4.0, 95.0, 125.0))
        lat = float(np.clip(25.0 + (daylen_h - 11.5) * 8.0, 18.0, 45.0))
        rows.append((lat, lon, dmax))
    if not rows:
        return 25.0, 110.0
    arr = np.asarray(rows)
    w = arr[:, 2]
    return float(np.average(arr[:, 0], weights=w)), float(np.average(arr[:, 1], weights=w))


def _night_median(minutes: np.ndarray, y: np.ndarray, miss: np.ndarray) -> float:
    night = _night_mask(minutes) & ~miss & np.isfinite(y)
    vals = y[night]
    if vals.size == 0:
        return 0.0
    med = float(np.nanmedian(vals))
    q1, q3 = np.nanpercentile(vals, [25, 75])
    iqr = max(q3 - q1, 1e-4)
    keep = (vals >= 0) & (vals <= med + 5 * iqr)
    return float(np.nanmedian(vals[keep])) if keep.any() else med


def _empirical_clearsky(minutes: np.ndarray, y: np.ndarray, window_days: int) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    slots = (minutes % 1440) // 15
    days = minutes // 1440
    n_days = int(days.max()) + 1 if len(days) else 0
    mat = np.full((n_days, 96), np.nan, dtype=float)
    for d, s, val in zip(days, slots, y):
        if np.isfinite(val):
            mat[int(d), int(s)] = val
    for d in range(n_days):
        hist = mat[max(0, d - window_days) : d]
        if hist.size == 0:
            continue
        with np.errstate(all="ignore"):
            q = np.nanpercentile(hist, 95, axis=0)
        for s in range(96):
            idx = np.where((days == d) & (slots == s))[0]
            if idx.size:
                out[idx] = q[s]
    return out


def _ghi_raw(minutes: np.ndarray, lat: float, lon: float) -> np.ndarray:
    import pvlib

    base = pd.Timestamp("2001-01-01 00:00:00")
    times = (base + pd.to_timedelta(minutes, unit="m")).tz_localize("UTC")
    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC")
    return loc.get_clearsky(times, model="ineichen")["ghi"].to_numpy(dtype=float)


def fit_stats(store: PowerStore | None = None, save_path: Path | None = None) -> dict:
    """只用训练集功率拟合，写入 models/stats.json。"""
    train_end = config.SPLIT_BOUNDS["train"][1]
    raw = load_power_raw("train")
    store = store or PowerStore(raw, end_minute=train_end)
    # 截断到训练集网格
    mask_m = store.minutes < train_end
    minutes = store.minutes[mask_m]

    stations = {}
    for i, sid in enumerate(store.station_ids):
        y = store.power[i, mask_m]
        miss = store.missing[i, mask_m]
        lat, lon = _infer_lat_lon(minutes, y)
        ghi = _ghi_raw(minutes, lat, lon)
        pos = ghi[ghi > 0]
        ghi_norm = float(np.nanpercentile(pos, 99)) if pos.size else 1000.0
        ghi_norm = max(ghi_norm, 1.0)
        cs = np.clip(ghi / ghi_norm, 0.0, 1.5)
        emp = _empirical_clearsky(minutes, y, config.EMPIRICAL_CLEARSKY_DAYS)
        ok = np.isfinite(cs) & np.isfinite(emp) & (cs > 0.05) & (emp > 0.05)
        scale = float(np.clip(np.nanmedian(emp[ok] / cs[ok]), 0.3, 3.0)) if ok.sum() > 100 else 1.0
        stations[sid] = {
            "lat": lat,
            "lon": lon,
            "ghi_norm": ghi_norm,
            "scale": scale,
            "night_median": _night_median(minutes, y, miss),
            "method": "pvlib",
        }

    payload = {
        "method": "pvlib+empirical",
        "fit_split": "train",
        "train_end_minute": train_end,
        "stations": stations,
    }
    path = Path(save_path or (config.MODELS_DIR / "stats.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def load_stats(path: Path | None = None) -> dict:
    path = Path(path or (config.MODELS_DIR / "stats.json"))
    if not path.exists():
        raise FileNotFoundError(f"缺少统计量文件: {path}，请先运行 train.py")
    return json.loads(path.read_text())


def night_medians(stats: dict | None = None) -> dict[str, float]:
    stats = stats or load_stats()
    return {sid: float(stats["stations"][sid]["night_median"]) for sid in config.STATION_IDS}
