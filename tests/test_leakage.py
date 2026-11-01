"""防泄漏：统计量仅用训练集；特征截止时刻合规。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv2h_model import config
from pv2h_model.stats import fit_stats, load_stats
from pv2h_model.timeutil import latest_cloud_minute, latest_power_minute


class TestLeakage(unittest.TestCase):
    def test_cutoff_rules(self):
        for issue in (0, 15, 243 * 1440, 243 * 1440 + 75):
            self.assertEqual(latest_power_minute(issue), issue - 15)
            self.assertEqual(latest_cloud_minute(issue), ((issue - 20) // 10) * 10)
            self.assertLessEqual(latest_cloud_minute(issue), issue - 20)

    def test_stats_train_only(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "stats.json"
            payload = fit_stats(save_path=out)
            self.assertEqual(payload["fit_split"], "train")
            self.assertEqual(payload["train_end_minute"], config.SPLIT_BOUNDS["train"][1])
            for sid in config.STATION_IDS:
                st = payload["stations"][sid]
                for k in ("lat", "lon", "ghi_norm", "scale", "night_median"):
                    self.assertIn(k, st)

    def test_loaded_stats_path(self):
        path = config.MODELS_DIR / "stats.json"
        if not path.exists():
            self.skipTest("stats not fitted")
        stats = load_stats()
        self.assertEqual(stats["fit_split"], "train")


if __name__ == "__main__":
    unittest.main()
