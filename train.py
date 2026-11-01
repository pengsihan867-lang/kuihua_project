#!/usr/bin/env python3
"""一键训练：统计量(仅训练集) -> 云特征缓存 -> 样本 -> 10 成员 -> blend.json。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pv2h_model import config  # noqa: E402
from pv2h_model.cloud_cache import build_cloud_cache  # noqa: E402
from pv2h_model.data import PowerStore  # noqa: E402
from pv2h_model.features.dataset import build_samples  # noqa: E402
from pv2h_model.models.members import train_all_members  # noqa: E402
from pv2h_model.stats import fit_stats, load_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["stats", "cloud", "samples", "members", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--force-cloud", action="store_true")
    args = ap.parse_args()

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ("stats", "all"):
        print("=== fit stats (train only) ===", flush=True)
        fit_stats()
        print("wrote", config.MODELS_DIR / "stats.json", flush=True)

    if args.stage in ("cloud", "all"):
        cloud_path = config.CACHE_DIR / "cloud_feats.parquet"
        if cloud_path.exists() and not args.force_cloud:
            print("reuse", cloud_path, flush=True)
        else:
            print("=== build cloud cache ===", flush=True)
            build_cloud_cache(n_jobs=args.jobs)

    if args.stage in ("samples", "all"):
        print("=== build samples ===", flush=True)
        import pandas as pd

        from pv2h_model.features.solar import build_clearsky_cache, load_clearsky_cache

        stats = load_stats()
        store = PowerStore()
        if not load_clearsky_cache():
            print("building clearsky cache...", flush=True)
            build_clearsky_cache(store.minutes, stats)
        cloud = pd.read_parquet(config.CACHE_DIR / "cloud_feats.parquet")
        for split in ("train", "val"):
            df = build_samples(store, cloud, split=split, stats=stats)
            path = config.CACHE_DIR / f"samples_{split}.parquet"
            df.to_parquet(path, index=False)
            print(split, df.shape, "->", path, flush=True)

    if args.stage in ("members", "all"):
        print("=== train members ===", flush=True)
        import pandas as pd

        tr = pd.read_parquet(config.CACHE_DIR / "samples_train.parquet")
        va = pd.read_parquet(config.CACHE_DIR / "samples_val.parquet")
        cfg = train_all_members(tr, va)
        print("val score_rows", cfg["val_score_rows"]["score_percent"], flush=True)


if __name__ == "__main__":
    main()
