"""加载 models/ 下 10 成员，等权融合 + 夜间后处理 + 有限值兜底。"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.eval.scoring import postprocess
from pv2h_model.features.online import features_from_inputs
from pv2h_model.stats import load_stats, night_medians


class BlendPredictor:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self.stats = load_stats(self.models_dir / "stats.json")
        self.nights = night_medians(self.stats)
        blend_path = self.models_dir / "blend.json"
        self.cfg = json.loads(blend_path.read_text())
        self.members = list(self.cfg["members"])
        w = np.asarray(self.cfg["weights"], dtype=float)
        self.weights = w / w.sum()
        self.feat_cols = json.loads((self.models_dir / "feature_columns.json").read_text())
        self._models: dict[str, object] = {}
        self._load_members()

    def _load_members(self) -> None:
        for name in self.members:
            if name == "mlp":
                obj = joblib.load(self.models_dir / "mlp.joblib")
                if isinstance(obj, dict) and obj.get("kind") == "lgbm":
                    self._models[name] = lgb.Booster(model_file=str(self.models_dir / obj["path"]))
                else:
                    self._models[name] = obj["pipes"] if isinstance(obj, dict) else [obj]
            elif name == "lgbm_perlead":
                models = {}
                for lead in config.LEADS:
                    p = self.models_dir / "perlead" / f"lgbm_lead{int(lead)}.txt"
                    models[int(lead)] = lgb.Booster(model_file=str(p))
                self._models[name] = models
            elif name == "lgbm_kt":
                self._models[name] = lgb.Booster(model_file=str(self.models_dir / "lgbm_kt.txt"))
            else:
                self._models[name] = lgb.Booster(model_file=str(self.models_dir / f"{name}.txt"))

    def _predict_member(self, name: str, df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        if name == "mlp":
            model = self._models[name]
            if isinstance(model, list):
                arr = np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                arr = np.clip(arr, -50.0, 50.0)
                return np.mean([p.predict(arr) for p in model], axis=0)
            return model.predict(X)
        if name == "lgbm_perlead":
            out = np.full(len(df), np.nan)
            leads = df["lead_minutes"].to_numpy(dtype=int)
            for lead, booster in self._models[name].items():
                m = leads == lead
                if m.any():
                    out[m] = booster.predict(X.loc[m])
            return out
        if name == "lgbm_kt":
            kt = self._models[name].predict(X)
            cs = df["clearsky_target"].to_numpy(dtype=float)
            return np.clip(kt, 0, 3) * cs
        return self._models[name].predict(X)

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feat_cols)
        cols = []
        for name in self.members:
            cols.append(self._predict_member(name, df, X))
        P = np.column_stack(cols)
        raw = np.nan_to_num(P, nan=0.0) @ self.weights
        out = postprocess(raw, df, self.nights)
        # 兜底：仍非有限则用智能持久性 / 夜间中位数
        bad = ~np.isfinite(out)
        if bad.any():
            last = df.get("p_power", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
            cs = df["clearsky_target"].to_numpy(dtype=float)
            cs0 = df.get("clearsky_issue", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
            kt = np.where(np.isfinite(last) & np.isfinite(cs0) & (cs0 > 0.05), np.clip(last / cs0, 0, 1.5), np.nan)
            smart = np.where(np.isfinite(kt) & np.isfinite(cs), kt * cs, last)
            for i in np.where(bad)[0]:
                sid = df.iloc[i]["station_id"]
                if np.isfinite(cs[i]) and cs[i] < config.NIGHT_CLEARSKY_EPS:
                    out[i] = self.nights.get(sid, 0.0)
                elif np.isfinite(smart[i]):
                    out[i] = float(smart[i])
                else:
                    out[i] = self.nights.get(sid, 0.0)
        return out

    def predict_from_inputs(self, inputs: dict) -> pd.DataFrame:
        df = features_from_inputs(inputs, self.stats)
        pred = self.predict_frame(df)
        return pd.DataFrame(
            {
                "station_id": df["station_id"],
                "issue_time": df["issue_time"],
                "lead_minutes": df["lead_minutes"].astype(int),
                "pred_power_ratio": pred.astype(float),
            }
        )
