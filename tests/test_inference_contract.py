"""推理契约：24 行、字段、唯一键、有限值；缺测输入仍可输出。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pv2h_model import config
from pv2h_model.data import make_release
from predict import predict

REQUIRED = {"station_id", "issue_time", "lead_minutes", "pred_power_ratio"}


@unittest.skipUnless((config.MODELS_DIR / "blend.json").exists(), "models not trained")
class TestInferenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with make_release() as data:
            cls.inputs = data.inputs(
                "09-01 02:00",
                power_steps=config.POWER_STEPS,
                cloud_frames=config.CLOUD_FRAMES,
                bands=config.CLOUD_BANDS,
            )

    def _check(self, pred, issue: str):
        self.assertTrue(REQUIRED.issubset(pred.columns))
        self.assertEqual(len(pred), 24)
        self.assertTrue(pred["issue_time"].eq(issue).all())
        self.assertEqual(set(pred["station_id"]), set(config.STATION_IDS))
        self.assertEqual(sorted(pred["lead_minutes"].unique().tolist()), list(map(int, config.LEADS)))
        self.assertEqual(int(pred.duplicated(["station_id", "issue_time", "lead_minutes"]).sum()), 0)
        self.assertTrue(np.isfinite(pred["pred_power_ratio"].to_numpy(dtype=float)).all())

    def test_normal(self):
        self._check(predict(self.inputs), self.inputs["issue_time"])

    def test_missing_cloud(self):
        inputs = dict(self.inputs)
        inputs["cloud"] = {b: np.full_like(v, np.nan) for b, v in self.inputs["cloud"].items()}
        inputs["cloud_present"] = np.zeros_like(self.inputs["cloud_present"])
        self._check(predict(inputs), inputs["issue_time"])

    def test_missing_power(self):
        inputs = dict(self.inputs)
        inputs["power_ratio"] = np.full_like(self.inputs["power_ratio"], np.nan)
        inputs["power_is_missing"] = np.ones_like(self.inputs["power_is_missing"], dtype=bool)
        self._check(predict(inputs), inputs["issue_time"])


if __name__ == "__main__":
    unittest.main()
