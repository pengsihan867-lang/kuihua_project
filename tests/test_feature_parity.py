"""在线特征与离线样本一致性抽检。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv2h_model import config
from pv2h_model.data import make_release
from pv2h_model.features.dataset import feature_columns
from pv2h_model.features.online import features_from_inputs
from pv2h_model.stats import load_stats
from pv2h_model.timeutil import time_of_minute


@unittest.skipUnless(
    (config.CACHE_DIR / "samples_val.parquet").exists() and (config.MODELS_DIR / "stats.json").exists(),
    "need samples and stats",
)
class TestFeatureParity(unittest.TestCase):
    def test_online_matches_offline(self):
        stats = load_stats()
        samples = pd.read_parquet(config.CACHE_DIR / "samples_val.parquet")
        feat = feature_columns(samples, use_cloud=True)
        issues = (
            samples.loc[samples["p_power"] > 0.2, "issue_minute"]
            .drop_duplicates()
            .sample(n=3, random_state=0)
            .tolist()
        )
        with make_release(cache_blocks=40) as data:
            for issue in issues:
                ts = time_of_minute(int(issue))
                inputs = data.inputs(
                    ts,
                    power_steps=config.POWER_STEPS,
                    cloud_frames=config.CLOUD_FRAMES,
                    bands=config.CLOUD_BANDS,
                )
                online = features_from_inputs(inputs, stats).sort_values(["station_id", "lead_minutes"]).reset_index(drop=True)
                off = samples.loc[samples["issue_minute"] == int(issue)].sort_values(["station_id", "lead_minutes"]).reset_index(drop=True)
                self.assertEqual(len(online), 24)
                self.assertEqual(len(off), 24)
                bad = []
                for c in feat:
                    if c not in offline_cols(off) or c not in online.columns:
                        continue
                    a = off[c].to_numpy(dtype=float)
                    b = online[c].to_numpy(dtype=float)
                    both_nan = ~np.isfinite(a) & ~np.isfinite(b)
                    ok = both_nan | (np.abs(a - b) < 1e-5)
                    if ok.mean() < 0.95:
                        bad.append((c, float(ok.mean())))
                self.assertLessEqual(len(bad), 5, msg=f"issue={ts} mismatches={bad[:10]}")


def offline_cols(df: pd.DataFrame) -> set[str]:
    return set(df.columns)


if __name__ == "__main__":
    unittest.main()
