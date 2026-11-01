#!/usr/bin/env python3
"""分块逐起报推理，可断点续跑；全部完成后调用官方评分。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    ap.add_argument("--split", default="val")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--outdir", type=Path, default=ROOT / "cache" / "e2e_chunks")
    args = ap.parse_args()

    sys.path.insert(0, str(args.data / "tools"))
    from evaluate import score_predictions  # noqa: WPS433
    from pv2h import Release, issue_minutes, time_of_minute  # noqa: WPS433

    args.outdir.mkdir(parents=True, exist_ok=True)
    issues = issue_minutes(args.split)
    done = {int(p.stem.split("_")[1]) for p in args.outdir.glob("chunk_*.parquet")}

    with Release(args.data, cache_blocks=80) as dataset:
        for start in range(0, len(issues), args.chunk):
            if start in done:
                continue
            chunk_issues = issues[start : start + args.chunk]
            rows = []
            for issue in tqdm(chunk_issues, desc=f"chunk_{start}"):
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
            pd.concat(rows, ignore_index=True).to_parquet(args.outdir / f"chunk_{start:05d}.parquet", index=False)
            print("saved", start, flush=True)

    parts = sorted(args.outdir.glob("chunk_*.parquet"))
    preds = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out_csv = ROOT / "val_predictions.csv"
    preds[["station_id", "issue_time", "lead_minutes", "pred_power_ratio"]].to_csv(out_csv, index=False)

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
    (ROOT / "score.json").write_text(json.dumps(score, indent=2, ensure_ascii=False))
    print(json.dumps(score, indent=2, ensure_ascii=False), flush=True)

    samples = pd.read_parquet(config.CACHE_DIR / f"samples_{args.split}.parquet")
    merged = samples.merge(
        preds.rename(columns={"pred_power_ratio": "pred"}),
        on=["station_id", "issue_minute", "lead_minutes"],
        how="inner",
    )
    merged["month"] = merged["issue_time"].str.slice(0, 2)
    monthly = []
    for month, g in merged.groupby("month"):
        s = score_rows(g, "pred")
        monthly.append({"month": month, "score_percent": s["score_percent"], "n_rows": len(g), **s["stations"]})
    pd.DataFrame(monthly).sort_values("month").to_csv(ROOT / "score_by_month.csv", index=False)
    print("done rows=", len(preds), flush=True)


if __name__ == "__main__":
    main()
