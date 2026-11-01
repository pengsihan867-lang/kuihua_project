"""回测报表：官方南网指标 + 曲线绘图数据准备。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.eval.scoring import score_rows
from pv2h_model.timeutil import minute_of_time, time_of_minute


def ensure_tools_path(data_root: Path) -> None:
    tools = Path(data_root) / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))


def official_score(predictions: pd.DataFrame, data_root: Path, split: str = "val") -> dict:
    """与 tools/evaluate.py 完全一致的南网日均准确率。"""
    ensure_tools_path(data_root)
    from evaluate import score_predictions  # noqa: WPS433

    station_ids = pd.read_csv(Path(data_root) / "stations.csv", dtype={"station_id": str}).station_id.tolist()
    power = pd.read_csv(
        Path(data_root) / "power" / f"{split}.csv.gz",
        dtype={"station_id": str},
        float_precision="round_trip",
    )
    cols = ["station_id", "issue_time", "lead_minutes", "pred_power_ratio"]
    return score_predictions(predictions[cols].copy(), power, station_ids, split=split)


def score_rows_metric(samples: pd.DataFrame, pred: np.ndarray | pd.Series) -> dict:
    """行级样本上的南网口径（用于与官方分对照；需覆盖完整 issue 网格时应用 official_score）。"""
    use = samples.copy()
    use["pred"] = np.asarray(pred, dtype=float)
    return score_rows(use, pred_col="pred")


def nanwang_report_table(official: dict) -> pd.DataFrame:
    """
    展示用表：3 个场站 + 三站等权，共 4 行。
    两列南网指标（TASK §6.3）：
      - 场站南网日均准确率 A_s
      - 三站等权南网日均准确率（仅汇总行填值，场站行留空或重复说明）
    """
    per = official["station_daily_accuracy_percent"]
    total = official["score_percent"]
    rows = []
    for sid in config.STATION_IDS:
        rows.append(
            {
                "统计对象": sid,
                "场站南网日均准确率_%": round(per[sid], 4),
                "三站等权南网日均准确率_%": np.nan,
            }
        )
    rows.append(
        {
            "统计对象": "三站等权",
            "场站南网日均准确率_%": np.nan,
            "三站等权南网日均准确率_%": round(total, 4),
        }
    )
    return pd.DataFrame(rows)


def predictions_from_samples(samples: pd.DataFrame, pred: np.ndarray | pd.Series) -> pd.DataFrame:
    out = samples[["station_id", "issue_time", "lead_minutes"]].copy()
    if "issue_time" not in out.columns or out["issue_time"].isna().any():
        out["issue_time"] = samples["issue_minute"].map(time_of_minute)
    out["pred_power_ratio"] = np.asarray(pred, dtype=float)
    out["pred_power_ratio"] = out["pred_power_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def curve_frame(
    samples: pd.DataFrame,
    pred: np.ndarray | pd.Series,
    station_id: str,
    lead_minutes: int,
    target_start: str | None = None,
    target_end: str | None = None,
) -> pd.DataFrame:
    """按目标时刻筛选，返回 pred / 实测 / clearsky 曲线数据。"""
    pred_s = pd.Series(np.asarray(pred, dtype=float), index=samples.index)
    use = samples.loc[samples["station_id"].eq(station_id) & samples["lead_minutes"].eq(int(lead_minutes))].copy()
    use["pred"] = pred_s.loc[use.index].to_numpy()
    use["y_true"] = use["y"]
    if target_start:
        t0 = minute_of_time(target_start)
        use = use.loc[use["target_minute"] >= t0]
    if target_end:
        t1 = minute_of_time(target_end)
        use = use.loc[use["target_minute"] <= t1]
    use = use.loc[use["is_missing"].eq(0) & np.isfinite(use["y_true"].to_numpy(dtype=float))]
    use["target_time"] = use["target_minute"].map(time_of_minute)
    return use.sort_values("target_minute")
