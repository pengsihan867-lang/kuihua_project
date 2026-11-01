"""本地复刻南网日均准确率（与 tools/evaluate.py 同口径，用于 hold 选轮）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pv2h_model import config


def score_rows(df: pd.DataFrame, pred_col: str = "pred", y_col: str = "y") -> dict:
    use = df.loc[df["is_missing"].eq(0) & np.isfinite(df[y_col].to_numpy(dtype=float))].copy()
    pred = np.where(np.isfinite(use[pred_col].to_numpy(dtype=float)), use[pred_col].to_numpy(dtype=float), 0.0)
    y = use[y_col].to_numpy(dtype=float)
    use = use.assign(e2=((pred - y) / np.maximum(y, config.SCORE_FLOOR)) ** 2)
    t = use.groupby(["station_id", "target_minute"], as_index=False)["e2"].mean()
    t["err"] = np.sqrt(t["e2"])
    t["local_day"] = (t["target_minute"] + 8 * 60) // 1440
    daily = t.groupby(["station_id", "local_day"])["err"].mean()
    acc = (1 - daily).groupby(level="station_id").mean().reindex(config.STATION_IDS)
    return {
        "score_percent": float(acc.mean() * 100),
        "stations": {sid: float(acc.loc[sid] * 100) for sid in config.STATION_IDS},
    }


def postprocess(pred: np.ndarray, df: pd.DataFrame, night_medians: dict[str, float]) -> np.ndarray:
    out = np.asarray(pred, dtype=float).copy()
    out[~np.isfinite(out)] = 0.0
    out = np.maximum(out, 0.0)
    night = df["clearsky_target"].fillna(0).to_numpy(dtype=float) < config.NIGHT_CLEARSKY_EPS
    sid = df["station_id"].to_numpy()
    for s, med in night_medians.items():
        out[night & (sid == s)] = med
    return out
