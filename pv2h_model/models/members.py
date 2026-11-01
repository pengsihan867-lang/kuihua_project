"""10 成员训练：1–7 月拟合、8 月官方指标选轮、1–8 月重训。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pv2h_model import config
from pv2h_model.eval.scoring import postprocess, score_rows
from pv2h_model.features.dataset import feature_columns, split_xy
from pv2h_model.models.train_lgbm import train_lgbm
from pv2h_model.stats import night_medians

META_COLS = [
    "station_id",
    "issue_minute",
    "lead_minutes",
    "target_minute",
    "y",
    "is_missing",
    "weight",
    "clearsky_target",
    "clearsky_issue",
    "p_kt",
    "p_power",
    "lead_minutes_feat",
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _best_rounds(predict_at, hold_df: pd.DataFrame, nights: dict, checkpoints=None) -> tuple[int, float, np.ndarray]:
    checkpoints = checkpoints or config.CHECKPOINTS
    best = (None, -np.inf, None)
    for n in checkpoints:
        p = predict_at(n)
        s = score_rows(hold_df.assign(pred=postprocess(p, hold_df, nights)))["score_percent"]
        if s > best[1]:
            best = (n, s, p)
    return best  # type: ignore[return-value]


def train_all_members(train_df: pd.DataFrame, val_df: pd.DataFrame, out_dir: Path | None = None) -> dict:
    out = Path(out_dir or config.MODELS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "perlead").mkdir(exist_ok=True)
    nights = night_medians()
    feat = feature_columns(train_df, use_cloud=True)
    (out / "feature_columns.json").write_text(json.dumps(feat, indent=2, ensure_ascii=False))

    hold_start = config.HOLD_START
    fit_mask = (train_df["issue_minute"] < hold_start).to_numpy()
    Xf, yf, wf, fit_use = split_xy(train_df.loc[fit_mask], feat)
    Xh, yh, wh, hold_use = split_xy(train_df.loc[~fit_mask], feat)
    Xa, ya, wa, all_use = split_xy(train_df, feat)
    Xv = val_df[feat]
    _log(f"fit={Xf.shape} hold={Xh.shape} all={Xa.shape} val={Xv.shape} n_feat={len(feat)}")

    hold_pred = hold_use[META_COLS].copy()
    val_pred = val_df[[c for c in META_COLS if c in val_df.columns]].copy()
    chosen: dict = {}
    cs_h = hold_use["clearsky_target"].to_numpy(float)
    cs_a = all_use["clearsky_target"].to_numpy(float)
    cs_v = val_df["clearsky_target"].to_numpy(float)
    lead_f = fit_use["lead_minutes"].to_numpy()
    lead_h = hold_use["lead_minutes"].to_numpy()
    lead_a = all_use["lead_minutes"].to_numpy()
    lead_v = val_df["lead_minutes"].to_numpy()

    # ---- lgbm_pc2
    name = "lgbm_pc2"
    t0 = time.time()
    params = config.MEMBER_PARAMS[name]
    b = train_lgbm(Xf, yf, wf, params=params, num_boost_round=1000)
    n, s, ph = _best_rounds(lambda k: b.predict(Xh, num_iteration=k), hold_use, nights)
    nr = int(round(n * config.REFIT_MULT))
    bf = train_lgbm(Xa, ya, wa, params=params, num_boost_round=nr)
    hold_pred[name], val_pred[name] = ph, bf.predict(Xv)
    bf.save_model(str(out / f"{name}.txt"))
    chosen[name] = {"n_hold": n, "hold_score": s, "n_refit": nr, "params": params}
    _log(f"{name} n*={n} hold={s:.3f} refit={nr} ({time.time() - t0:.0f}s)")

    # ---- lgbm_kt
    name = "lgbm_kt"
    t0 = time.time()
    cs_f = fit_use["clearsky_target"].to_numpy(float)
    df_ = cs_f > 0.05
    da_ = cs_a > 0.05
    ktf = np.clip(yf[df_] / cs_f[df_], 0, 3)
    kta = np.clip(ya[da_] / cs_a[da_], 0, 3)
    wkf = cs_f[df_] ** 2 / np.maximum(yf[df_], config.SCORE_FLOOR) ** 2
    wka = cs_a[da_] ** 2 / np.maximum(ya[da_], config.SCORE_FLOOR) ** 2
    b = train_lgbm(Xf.loc[df_], ktf, wkf, num_boost_round=1000)
    n, s, ph = _best_rounds(lambda k: b.predict(Xh, num_iteration=k) * cs_h, hold_use, nights)
    nr = int(round(n * config.REFIT_MULT))
    bf = train_lgbm(Xa.loc[da_], kta, wka, num_boost_round=nr)
    hold_pred[name], val_pred[name] = ph, bf.predict(Xv) * cs_v
    bf.save_model(str(out / f"{name}.txt"))
    chosen[name] = {"n_hold": n, "hold_score": s, "n_refit": nr}
    _log(f"{name} n*={n} hold={s:.3f} refit={nr} ({time.time() - t0:.0f}s)")

    # ---- lgbm_l1
    name = "lgbm_l1"
    t0 = time.time()
    pl1 = config.MEMBER_PARAMS[name]
    w1f = 1 / np.maximum(yf, config.SCORE_FLOOR)
    w1a = 1 / np.maximum(ya, config.SCORE_FLOOR)
    b = train_lgbm(Xf, yf, w1f, params=pl1, num_boost_round=1000)
    n, s, ph = _best_rounds(lambda k: b.predict(Xh, num_iteration=k), hold_use, nights)
    nr = int(round(n * config.REFIT_MULT))
    bf = train_lgbm(Xa, ya, w1a, params=pl1, num_boost_round=nr)
    hold_pred[name], val_pred[name] = ph, bf.predict(Xv)
    bf.save_model(str(out / f"{name}.txt"))
    chosen[name] = {"n_hold": n, "hold_score": s, "n_refit": nr, "params": pl1}
    _log(f"{name} n*={n} hold={s:.3f} refit={nr} ({time.time() - t0:.0f}s)")

    # ---- per-lead
    name = "lgbm_perlead"
    t0 = time.time()
    ph_all = np.full(len(Xh), np.nan)
    pv_all = np.full(len(Xv), np.nan)
    per_lead_n = {}
    for lead in config.LEADS:
        mf, mh, ma, mv = lead_f == lead, lead_h == lead, lead_a == lead, lead_v == lead
        b = train_lgbm(Xf.loc[mf], yf[mf], wf[mf], num_boost_round=1000)
        hsub = hold_use.loc[mh]
        n, s, ph = _best_rounds(lambda k, mh=mh, b=b: b.predict(Xh.loc[mh], num_iteration=k), hsub, nights)
        nr = int(round(n * config.REFIT_MULT))
        bf = train_lgbm(Xa.loc[ma], ya[ma], wa[ma], num_boost_round=nr)
        ph_all[mh] = ph
        pv_all[mv] = bf.predict(Xv.loc[mv])
        bf.save_model(str(out / "perlead" / f"lgbm_lead{int(lead)}.txt"))
        per_lead_n[int(lead)] = {"n_hold": n, "hold_score": s, "n_refit": nr}
    hold_pred[name], val_pred[name] = ph_all, pv_all
    chosen[name] = per_lead_n
    _log(f"{name} done ({time.time() - t0:.0f}s)")

    # ---- remaining LGBM variants
    for name in ("lgbm_pc3", "lgbm_pc_xt", "lgbm_pc_dart", "lgbm_pc_lr03", "lgbm_pc_deep"):
        t0 = time.time()
        params = config.MEMBER_PARAMS[name]
        max_r = 1000 if params.get("learning_rate", 0.05) >= 0.05 else 1600
        if params.get("boosting_type") == "dart":
            max_r = 1000
        cps = list(config.CHECKPOINTS) + ([1300, 1600] if max_r > 1000 else [])
        b = train_lgbm(Xf, yf, wf, params=params, num_boost_round=max_r)
        n, s, ph = _best_rounds(lambda k, b=b: b.predict(Xh, num_iteration=k), hold_use, nights, checkpoints=cps)
        nr = int(round(n * config.REFIT_MULT))
        bf = train_lgbm(Xa, ya, wa, params=params, num_boost_round=nr)
        hold_pred[name], val_pred[name] = ph, bf.predict(Xv)
        bf.save_model(str(out / f"{name}.txt"))
        chosen[name] = {"n_hold": n, "hold_score": s, "n_refit": nr, "params": params}
        _log(f"{name} n*={n} hold={s:.3f} refit={nr} ({time.time() - t0:.0f}s)")

    hold_pred.to_parquet(out / "members_hold.parquet", index=False)
    val_pred.to_parquet(out / "members_val.parquet", index=False)

    # equal blend score on val
    members = list(config.BLEND_MEMBERS)
    Pv = np.nan_to_num(val_pred[members].to_numpy(float), nan=0.0)
    blend = postprocess(Pv.mean(axis=1), val_df.loc[val_pred.index] if False else val_df, nights)
    # val_pred rows align with val_df via META subset — use val_df valid rows order from split
    # Actually val_pred was built from full val_df META cols, same index order as val_df
    val_full = val_df.copy()
    for m in members:
        val_full[m] = val_pred[m].to_numpy()
    raw = np.nan_to_num(val_full[members].to_numpy(float), nan=0.0).mean(axis=1)
    scored = score_rows(val_full.assign(pred=postprocess(raw, val_full, nights)))

    blend_cfg = {
        "members": members,
        "weights": [1.0 / len(members)] * len(members),
        "postprocess": "clip>=0; night(clearsky<0.02)->station median",
        "protocol": "Aug official-metric rounds, Jan-Aug refit *1.15, equal blend",
        "val_score_rows": scored,
        "chosen": chosen,
        "night_medians": nights,
        "power_steps": config.POWER_STEPS,
        "cloud_frames": config.CLOUD_FRAMES,
        "cloud_bands": list(config.CLOUD_BANDS),
    }
    (out / "blend.json").write_text(json.dumps(blend_cfg, indent=2, ensure_ascii=False))
    _log(f"blend val score_rows={scored['score_percent']:.4f}")
    return blend_cfg
