"""LightGBM 训练封装。"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from pv2h_model import config


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    params: dict | None = None,
    num_boost_round: int = 800,
) -> lgb.Booster:
    p = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": config.SEED,
        "num_threads": config.N_JOBS,
    }
    if params:
        p.update(params)
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train, free_raw_data=False)
    return lgb.train(
        p,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=0)],
    )
