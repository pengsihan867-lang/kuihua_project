"""官方 inputs dict -> 24 行特征表；与离线 dataset 共用功率/云核心。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.features.cloud import attach_trends, extract_frame_features
from pv2h_model.features.power import POWER_FEAT_COLS, build_power_feature_tables
from pv2h_model.features.solar import clearsky_values
from pv2h_model.stats import load_stats
from pv2h_model.timeutil import latest_cloud_minute, minute_of_time


def _power_minutes_from_inputs(inputs: dict) -> np.ndarray:
    if "power_timestamps" in inputs and inputs["power_timestamps"]:
        return np.asarray([minute_of_time(t) for t in inputs["power_timestamps"]], dtype=np.int64)
    issue = minute_of_time(inputs["issue_time"])
    n = len(np.asarray(inputs["power_ratio"])[0])
    return issue - 15 * np.arange(n, 0, -1)


def _cloud_minutes_from_inputs(inputs: dict) -> np.ndarray:
    if "cloud_timestamps" in inputs and inputs["cloud_timestamps"]:
        return np.asarray([minute_of_time(t) for t in inputs["cloud_timestamps"]], dtype=np.int64)
    issue = minute_of_time(inputs["issue_time"])
    end = latest_cloud_minute(issue)
    n = len(next(iter(inputs["cloud"].values())))
    return end - 10 * np.arange(n - 1, -1, -1)


def _frame_at(cloud: dict, present: np.ndarray, cloud_bands: list[str], idx: int, stations: pd.DataFrame) -> dict[str, float]:
    bands = config.CLOUD_BANDS
    band_idx = {b: cloud_bands.index(b) for b in bands if b in cloud_bands}
    frame = {}
    present_sel = []
    for b in bands:
        if b not in cloud:
            present_sel.append(0.0)
            continue
        arr = np.asarray(cloud[b])
        frame[b] = arr[idx]
        if b in band_idx and present is not None:
            present_sel.append(float(present[idx, band_idx[b]]))
        else:
            present_sel.append(1.0 if np.isfinite(arr[idx]).any() else 0.0)
    return extract_frame_features(frame, np.asarray(present_sel, dtype=float), stations, bands=bands)


def features_from_inputs(inputs: dict, stats: dict | None = None) -> pd.DataFrame:
    """
    构造与训练样本列兼容的特征行（3 站 × 8 lead）。
    功率窗口应 ≥192 步，云帧 ≥7，以保证 lag_day 与 d60。
    """
    stats = stats or load_stats()
    issue_time = inputs["issue_time"]
    issue = minute_of_time(issue_time)
    stations = inputs["stations"]
    station_ids = list(inputs["station_ids"])
    leads = np.asarray(inputs["lead_minutes"], dtype=int)
    power = np.asarray(inputs["power_ratio"], dtype=float)
    missing = np.asarray(inputs["power_is_missing"], dtype=bool)
    p_minutes = _power_minutes_from_inputs(inputs)

    # 晴空：功率网格 + 各 lead 目标
    need_minutes = sorted(set(p_minutes.tolist()) | {issue + int(l) for l in leads} | {issue - 15})
    cs_cache = {sid: {} for sid in station_ids}
    for sid in station_ids:
        vals = clearsky_values(sid, np.asarray(need_minutes, dtype=np.int64), stats)
        for m, v in zip(need_minutes, vals):
            cs_cache[sid][int(m)] = float(v)

    clearsky_arr = {
        sid: np.asarray([cs_cache[sid].get(int(m), np.nan) for m in p_minutes], dtype=float) for sid in station_ids
    }
    # 对齐 station 顺序到 STATION_IDS 以复用 other_* 逻辑
    order = [sid for sid in config.STATION_IDS if sid in station_ids]
    idx_map = {sid: station_ids.index(sid) for sid in order}
    power_ord = np.stack([power[idx_map[s]] for s in order], axis=0)
    miss_ord = np.stack([missing[idx_map[s]] for s in order], axis=0)
    cs_ord = {s: clearsky_arr[s] for s in order}
    tables = build_power_feature_tables(p_minutes, power_ord, miss_ord, cs_ord, order)

    # 云特征：最新帧 + -30/-60
    cloud_feats_by_sid: dict[str, dict[str, float]] = {sid: {} for sid in station_ids}
    if "cloud" in inputs and inputs["cloud"]:
        c_minutes = _cloud_minutes_from_inputs(inputs)
        present = np.asarray(inputs["cloud_present"])
        cloud_bands = list(inputs["cloud_bands"])
        c_cut = latest_cloud_minute(issue)
        # 找最接近 c_cut / c_cut-30 / c_cut-60 的帧索引（不超过截止）
        def nearest_idx(target: int) -> int | None:
            ok = np.where(c_minutes <= target)[0]
            if ok.size == 0:
                return None
            return int(ok[np.argmin(np.abs(c_minutes[ok] - target))])

        i0 = nearest_idx(c_cut)
        i30 = nearest_idx(c_cut - 30)
        i60 = nearest_idx(c_cut - 60)
        feats0 = _frame_at(inputs["cloud"], present, cloud_bands, i0, stations) if i0 is not None else {}
        feats30 = _frame_at(inputs["cloud"], present, cloud_bands, i30, stations) if i30 is not None else None
        feats60 = _frame_at(inputs["cloud"], present, cloud_bands, i60, stations) if i60 is not None else None
        for sid in station_ids:
            raw = {k: v for k, v in feats0.items() if k.startswith(f"{sid}_")}
            p30 = {k: v for k, v in (feats30 or {}).items() if k.startswith(f"{sid}_")} or None
            p60 = {k: v for k, v in (feats60 or {}).items() if k.startswith(f"{sid}_")} or None
            cloud_feats_by_sid[sid] = attach_trends(raw, p30, p60, sid)

    p_cut = issue - 15
    rows = []
    for sid in station_ids:
        tab = tables[sid]
        m15 = (p_cut // 15) * 15
        if m15 in tab.index:
            prow = tab.loc[m15]
            pfeats = {f"p_{c}": float(prow[c]) if c in prow.index and np.isfinite(prow[c]) else np.nan for c in POWER_FEAT_COLS}
        else:
            pfeats = {f"p_{c}": np.nan for c in POWER_FEAT_COLS}
        cfeats = {f"c_{k}": v for k, v in cloud_feats_by_sid.get(sid, {}).items()}
        for lead in leads:
            target = issue + int(lead)
            cs1 = cs_cache[sid].get(int(target), np.nan)
            # 目标时刻可能不在功率网格上，用最近 15min 向下取整
            if not np.isfinite(cs1):
                cs1 = cs_cache[sid].get(int((target // 15) * 15), np.nan)
            cs0 = cs_cache[sid].get(int(m15), np.nan)
            row = {
                "station_id": sid,
                "issue_minute": issue,
                "issue_time": issue_time,
                "lead_minutes": int(lead),
                "clearsky_target": cs1,
                "clearsky_issue": cs0,
                "p_clearsky_target": cs1,
                "lead_minutes_feat": float(lead),
                "lead_sin": float(np.sin(2 * np.pi * lead / 120.0)),
                "lead_cos": float(np.cos(2 * np.pi * lead / 120.0)),
            }
            row.update(pfeats)
            row.update(cfeats)
            rows.append(row)
    return pd.DataFrame(rows)
