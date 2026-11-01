#!/usr/bin/env python3
"""逐起报推理 + 官方 evaluate.py 打分，并按月汇总。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv2h_model import config  # noqa: E402
from pv2h_model.eval.scoring import score_rows  # noqa: E402
from predict import predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=config.DATA_ROOT)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--output", type=Path, default=ROOT / "val_predictions.csv")
    ap.add_argument("--score", type=Path, default=ROOT / "score.json")
    ap.add_argument("--monthly", type=Path, default=ROOT / "score_by_month.csv")
    ap.add_argument("--limit", type=int, default=0, help="仅跑前 N 个起报（调试）")
    args = ap.parse_args()

    sys.path.insert(0, str(args.data / "tools"))
    from evaluate import score_predictions  # noqa: WPS433
    from pv2h import Release, issue_minutes, time_of_minute  # noqa: WPS433

    rows = []
    with Release(args.data, cache_blocks=80) as dataset:
        issues = issue_minutes(args.split)
        if args.limit:
            issues = issues[: args.limit]
        for issue in tqdm(issues, desc=f"eval_{args.split}"):
            ts = time_of_minute(int(issue))
            inputs = dataset.inputs(
                ts,
                power_steps=config.POWER_STEPS,
                cloud_frames=config.CLOUD_FRAMES,
                bands=config.CLOUD_BANDS,
            )
            pred = predict(inputs)
            pred["issue_minute"] = int(issue)
            rows.append(pred)
    preds = pd.concat(rows, ignore_index=True)
    preds[["station_id", "issue_time", "lead_minutes", "pred_power_ratio"]].to_csv(args.output, index=False)

    if args.limit:
        print("limit mode: skip official full-grid score", flush=True)
        return

    power = pd.read_csv(
        args.data / "power" / f"{args.split}.csv.gz",
        dtype={"station_id": str},
        float_precision="round_trip",
    )
    score = score_predictions(
        preds[["station_id", "issue_time", "lead_minutes", "pred_power_ratio"]],
        power,
        list(config.STATION_IDS),
        split=args.split,
    )
    args.score.write_text(json.dumps(score, indent=2, ensure_ascii=False))
    print(json.dumps(score, indent=2, ensure_ascii=False), flush=True)

    # 按月：对齐样本标签后用 score_rows
    samples = pd.read_parquet(config.CACHE_DIR / f"samples_{args.split}.parquet")
    merged = samples.merge(
        preds.rename(columns={"pred_power_ratio": "pred"}),
        on=["station_id", "issue_minute", "lead_minutes"],
        how="inner",
    )
    monthly_rows = []
    merged["month"] = (merged["issue_minute"] // 1440 // 31).astype(int)  # rough; better use issue_time
    merged["month"] = merged["issue_time"].str.slice(0, 2)
    for month, g in merged.groupby("month"):
        s = score_rows(g, "pred")
        monthly_rows.append({"month": month, "score_percent": s["score_percent"], "n_rows": len(g), **s["stations"]})
    pd.DataFrame(monthly_rows).sort_values("month").to_csv(args.monthly, index=False)
    print("wrote", args.output, args.score, args.monthly, flush=True)


if __name__ == "__main__":
    main()
