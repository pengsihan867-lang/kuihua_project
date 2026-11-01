"""从日 ZIP 构建云特征缓存（训练用；推理走 online 单帧提取）。"""
from __future__ import annotations

import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pv2h_model import config
from pv2h_model.data import load_stations
from pv2h_model.features.cloud import extract_frame_features, feature_names
from pv2h_model.timeutil import time_of_minute

if str(config.TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(config.TOOLS_DIR))
from pv2h import BANDS, read_block  # noqa: E402


def _day_zip_path(day: int) -> Path:
    split = "train" if day < 243 else "val"
    name = time_of_minute(day * 1440)[:5] + ".zip"
    return config.DATA_ROOT / "cloud" / split / name


def _process_day(day: int, bands: tuple[str, ...]) -> list[dict]:
    path = _day_zip_path(day)
    stations = load_stations()
    rows: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read("metadata.json"))
        timestamps = meta["timestamps"]
        present = meta["present"]
        chunk = int(meta["chunk_frames"])
        block_cache: dict[tuple[str, int], np.ndarray] = {}
        for slot, ts in enumerate(timestamps):
            minute = day * 1440 + slot * 10
            block, within = divmod(slot, chunk)
            cloud, present_flags = {}, []
            for band in bands:
                key = (band, block)
                if key not in block_cache:
                    block_cache[key] = read_block(archive, band, block)
                cloud[band] = block_cache[key][within]
                present_flags.append(present[slot][BANDS.index(band)])
            feats = extract_frame_features(
                cloud, np.asarray(present_flags, dtype=np.float32), stations, bands=bands
            )
            row = {"minute": minute, "timestamp": ts, "day": day, "slot": slot}
            row.update(feats)
            rows.append(row)
    return rows


def build_cloud_cache(
    out_path: Path | None = None,
    n_jobs: int = config.N_JOBS,
    days: list[int] | None = None,
) -> Path:
    out_path = Path(out_path or (config.CACHE_DIR / "cloud_feats.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bands = config.CLOUD_BANDS
    if days is None:
        days = list(range(304))
    all_rows: list[dict] = []
    if n_jobs <= 1:
        for day in tqdm(days, desc="cloud_cache"):
            all_rows.extend(_process_day(day, bands))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {pool.submit(_process_day, d, bands): d for d in days}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="cloud_cache"):
                all_rows.extend(fut.result())
    df = pd.DataFrame(all_rows).sort_values("minute").reset_index(drop=True)
    for name in feature_names(bands=bands):
        if name not in df.columns:
            df[name] = np.nan
    df.to_parquet(out_path, index=False)
    return out_path
