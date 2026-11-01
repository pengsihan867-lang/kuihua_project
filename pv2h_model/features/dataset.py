"""离线样本表：(station, issue, lead) -> 特征 + 标签。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.data import PowerStore, load_stations
from pv2h_model.features.cloud import feature_names as cloud_feature_names, key_trend_raw_names
from pv2h_model.features.power import POWER_FEAT_COLS, build_power_feature_tables
from pv2h_model.features.solar import clearsky_table
from pv2h_model.stats import load_stats
from pv2h_model.timeutil import issue_minutes, latest_cloud_minute, time_of_minute


def _prepare_cloud_index(cloud_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if cloud_df is None:
        return None
    if "minute" in cloud_df.columns:
        return cloud_df.drop_duplicates("minute").set_index("minute").sort_index()
    return cloud_df.sort_index()


def _lookup_rows(df: pd.DataFrame | None, minutes: np.ndarray, cols: list[str]) -> pd.DataFrame:
    minutes = np.asarray(minutes, dtype=np.int64)
    if df is None or df.empty:
        return pd.DataFrame({c: np.nan for c in cols}, index=minutes)
    use_cols = [c for c in cols if c in df.columns]
    src = df[use_cols]
    exact = src.reindex(minutes)
    need = exact.isna().all(axis=1) if use_cols else pd.Series(False, index=exact.index)
    if need.any():
        left = pd.DataFrame({"minute": minutes})
        right = src.reset_index().rename(columns={src.index.name or "index": "minute"})
        if "minute" not in right.columns:
            right = src.reset_index()
            right = right.rename(columns={right.columns[0]: "minute"})
        merged = pd.merge_asof(
            left.sort_values("minute"),
            right.sort_values("minute"),
            on="minute",
            direction="backward",
        )
        exact = merged.set_index("minute")[use_cols].reindex(minutes)
    for c in cols:
        if c not in exact.columns:
            exact[c] = np.nan
    return exact[cols]


def build_samples(
    store: PowerStore,
    cloud_df: pd.DataFrame | None,
    split: str = "train",
    stats: dict | None = None,
    issue_list: np.ndarray | None = None,
) -> pd.DataFrame:
    stats = stats or load_stats()
    if issue_list is None:
        issue_list = issue_minutes(split)
    issue_list = np.asarray(issue_list, dtype=np.int64)
    clearsky = clearsky_table(store.minutes, stats)
    power_tables = build_power_feature_tables(store.minutes, store.power, store.missing, clearsky, store.station_ids)
    cloud_indexed = _prepare_cloud_index(cloud_df)
    cloud_cols_all = cloud_feature_names()
    cs_maps = {sid: pd.Series(clearsky[sid], index=store.minutes) for sid in store.station_ids}
    power_maps = {sid: store.series(sid) for sid in store.station_ids}

    frames = []
    for sid in store.station_ids:
        ptab = power_tables[sid]
        issues_rep = np.repeat(issue_list, len(config.LEADS))
        leads_rep = np.tile(config.LEADS, len(issue_list))
        targets = issues_rep + leads_rep
        p_cut = issues_rep - 15
        c_cut = np.array([latest_cloud_minute(int(x)) for x in issues_rep], dtype=np.int64)

        p_feats = _lookup_rows(ptab, p_cut, list(POWER_FEAT_COLS)).add_prefix("p_")
        p_feats.index = np.arange(len(p_feats))

        y = power_maps[sid].reindex(targets).to_numpy(dtype=float)
        is_missing = (~np.isfinite(y)).astype(int)
        cs_target = cs_maps[sid].reindex((targets // 15) * 15).to_numpy(dtype=float)
        cs_issue = cs_maps[sid].reindex((p_cut // 15) * 15).to_numpy(dtype=float)

        out = pd.DataFrame(
            {
                "station_id": sid,
                "issue_minute": issues_rep,
                "issue_time": [time_of_minute(int(x)) for x in issues_rep],
                "lead_minutes": leads_rep.astype(int),
                "target_minute": targets,
                "power_cutoff": p_cut,
                "cloud_cutoff": c_cut,
                "y": y,
                "is_missing": is_missing,
                "clearsky_target": cs_target,
                "clearsky_issue": cs_issue,
                "weight": 1.0 / np.maximum(np.nan_to_num(y, nan=0.0), config.SCORE_FLOOR) ** 2,
                "p_clearsky_target": cs_target,
                "lead_minutes_feat": leads_rep.astype(float),
                "lead_sin": np.sin(2 * np.pi * leads_rep / 120.0),
                "lead_cos": np.cos(2 * np.pi * leads_rep / 120.0),
            }
        )
        out = pd.concat([out, p_feats.reset_index(drop=True)], axis=1)

        if cloud_indexed is not None:
            sid_cols = [c for c in cloud_cols_all if c.startswith(f"{sid}_")]
            crow = _lookup_rows(cloud_indexed, c_cut, sid_cols).add_prefix("c_")
            crow.index = np.arange(len(crow))
            for k in key_trend_raw_names(sid):
                if k not in cloud_indexed.columns:
                    crow[f"c_{k}_d30"] = np.nan
                    crow[f"c_{k}_d60"] = np.nan
                    continue
                cur = _lookup_rows(cloud_indexed, c_cut, [k])[k].to_numpy()
                v30 = _lookup_rows(cloud_indexed, c_cut - 30, [k])[k].to_numpy()
                v60 = _lookup_rows(cloud_indexed, c_cut - 60, [k])[k].to_numpy()
                crow[f"c_{k}_d30"] = np.where(np.isfinite(cur) & np.isfinite(v30), cur - v30, np.nan)
                crow[f"c_{k}_d60"] = np.where(np.isfinite(cur) & np.isfinite(v60), cur - v60, np.nan)
            crow["c_cloud_age_min"] = 0.0
            out = pd.concat([out, crow.reset_index(drop=True)], axis=1)

        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def feature_columns(df: pd.DataFrame, use_cloud: bool = True) -> list[str]:
    cols = [c for c in df.columns if c.startswith("p_")]
    cols += ["lead_minutes_feat", "lead_sin", "lead_cos", "clearsky_target", "clearsky_issue"]
    if use_cloud:
        cols += [c for c in df.columns if c.startswith("c_")]
    seen, out = set(), []
    for c in cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            out.append(c)
    return out


def split_xy(df: pd.DataFrame, feature_cols: list[str]):
    valid = df["is_missing"].eq(0) & np.isfinite(df["y"].to_numpy(dtype=float))
    use = df.loc[valid].copy()
    return use[feature_cols], use["y"].to_numpy(dtype=float), use["weight"].to_numpy(dtype=float), use
