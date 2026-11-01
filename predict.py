#!/usr/bin/env python3
"""推理入口：predict(inputs) -> DataFrame，字段符合 TASK.md 第 4 节。

也可 CLI：
  python predict.py --data .. --issue "09-01 10:00" --output preds.csv
  python predict.py --data .. --split val --output val_predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv2h_model import config  # noqa: E402
from pv2h_model.models.blend import BlendPredictor  # noqa: E402

_PREDICTOR: BlendPredictor | None = None


def get_predictor() -> BlendPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = BlendPredictor(config.MODELS_DIR)
    return _PREDICTOR


def predict(inputs: dict) -> pd.DataFrame:
    """主办方调用的标准入口。"""
    return get_predictor().predict_from_inputs(inputs)


def _run_issue(data_root: Path, issue: str, output: Path) -> None:
    sys.path.insert(0, str(data_root / "tools"))
    from pv2h import Release  # noqa: WPS433

    with Release(data_root) as dataset:
        inputs = dataset.inputs(
            issue,
            power_steps=config.POWER_STEPS,
            cloud_frames=config.CLOUD_FRAMES,
            bands=config.CLOUD_BANDS,
        )
        pred = predict(inputs)
    pred.to_csv(output, index=False)
    print(output, flush=True)


def _run_split(data_root: Path, split: str, output: Path) -> None:
    sys.path.insert(0, str(data_root / "tools"))
    from pv2h import Release, issue_minutes, time_of_minute  # noqa: WPS433
    from tqdm import tqdm

    rows = []
    with Release(data_root, cache_blocks=80) as dataset:
        issues = issue_minutes(split)
        for issue in tqdm(issues, desc=f"predict_{split}"):
            ts = time_of_minute(int(issue))
            inputs = dataset.inputs(
                ts,
                power_steps=config.POWER_STEPS,
                cloud_frames=config.CLOUD_FRAMES,
                bands=config.CLOUD_BANDS,
            )
            rows.append(predict(inputs))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output, index=False)
    print(output, "rows=", len(out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=config.DATA_ROOT)
    ap.add_argument("--issue", default=None)
    ap.add_argument("--split", choices=["train", "val"], default=None)
    ap.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = ap.parse_args()
    if args.split:
        _run_split(args.data, args.split, args.output)
    else:
        _run_issue(args.data, args.issue or "09-01 10:00", args.output)


if __name__ == "__main__":
    main()
