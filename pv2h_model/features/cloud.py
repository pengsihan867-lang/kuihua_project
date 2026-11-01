"""场站邻域云图特征；单帧提取 + d30/d60 趋势，离线/在线共用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv2h_model import config


def _window_stats(arr: np.ndarray, row: int, col: int, radius: int) -> tuple[float, float, float]:
    h, w = arr.shape
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    patch = arr[r0:r1, c0:c1]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    center = arr[row, col] if (0 <= row < h and 0 <= col < w) else np.nan
    return float(np.mean(finite)), float(np.std(finite)), float(center) if np.isfinite(center) else np.nan


def station_pixel(stations: pd.DataFrame, station_id: str, band: str) -> tuple[int, int]:
    row = stations.loc[stations.station_id == station_id].iloc[0]
    if band == "B03":
        return int(row.row_B03), int(row.col_B03)
    if band == "B04":
        return int(row.row_B04), int(row.col_B04)
    return int(row.row_coarse), int(row.col_coarse)


def radius_for_band(band: str, radius_coarse: int) -> int:
    if band == "B03":
        return radius_coarse * 4
    if band == "B04":
        return radius_coarse * 2
    return radius_coarse


def extract_frame_features(
    cloud: dict[str, np.ndarray],
    present: np.ndarray,
    stations: pd.DataFrame,
    bands: tuple[str, ...] = config.CLOUD_BANDS,
    radii: tuple[int, ...] = config.CLOUD_RADII_COARSE,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    band_list = list(bands)
    for bi, band in enumerate(band_list):
        if band not in cloud:
            for sid in config.STATION_IDS:
                for rad_c in radii:
                    prefix = f"{sid}_{band}_r{rad_c}"
                    feats[f"{prefix}_mean"] = np.nan
                    feats[f"{prefix}_std"] = np.nan
                    if rad_c == 0:
                        feats[f"{prefix}_center"] = np.nan
                feats[f"{sid}_{band}_present"] = 0.0
            continue
        arr = cloud[band]
        if arr.ndim == 3:
            arr = arr[-1]
        is_present = float(present[bi]) if present is not None and len(present) > bi else 1.0
        for sid in config.STATION_IDS:
            r, c = station_pixel(stations, sid, band)
            for rad_c in radii:
                rad = radius_for_band(band, rad_c)
                mean, std, center = _window_stats(arr, r, c, rad)
                prefix = f"{sid}_{band}_r{rad_c}"
                feats[f"{prefix}_mean"] = mean
                feats[f"{prefix}_std"] = std
                if rad_c == 0:
                    feats[f"{prefix}_center"] = center
            feats[f"{sid}_{band}_present"] = is_present
    return feats


def feature_names(
    bands: tuple[str, ...] = config.CLOUD_BANDS,
    radii: tuple[int, ...] = config.CLOUD_RADII_COARSE,
) -> list[str]:
    names = []
    for band in bands:
        for sid in config.STATION_IDS:
            for rad_c in radii:
                prefix = f"{sid}_{band}_r{rad_c}"
                names.append(f"{prefix}_mean")
                names.append(f"{prefix}_std")
                if rad_c == 0:
                    names.append(f"{prefix}_center")
            names.append(f"{sid}_{band}_present")
    return names


def key_trend_raw_names(station_id: str) -> list[str]:
    return [
        f"{station_id}_B03_r0_center",
        f"{station_id}_B03_r3_mean",
        f"{station_id}_B13_r0_center",
        f"{station_id}_B13_r3_mean",
    ]


def attach_trends(row: dict[str, float], prev30: dict[str, float] | None, prev60: dict[str, float] | None, station_id: str) -> dict[str, float]:
    """在已有单帧特征上附加 d30/d60；缺帧时为 NaN。"""
    out = dict(row)
    for k in key_trend_raw_names(station_id):
        cur = row.get(k, np.nan)
        v30 = prev30.get(k, np.nan) if prev30 else np.nan
        v60 = prev60.get(k, np.nan) if prev60 else np.nan
        out[f"{k}_d30"] = (cur - v30) if np.isfinite(cur) and np.isfinite(v30) else np.nan
        out[f"{k}_d60"] = (cur - v60) if np.isfinite(cur) and np.isfinite(v60) else np.nan
    out["cloud_age_min"] = 0.0
    return out
