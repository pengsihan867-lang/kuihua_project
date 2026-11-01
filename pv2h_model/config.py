"""路径、窗口声明与模型超参。数据根目录可用环境变量 PV2H_DATA 覆盖。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PV2H_DATA", PACKAGE_ROOT.parent))
TOOLS_DIR = DATA_ROOT / "tools"
CACHE_DIR = PACKAGE_ROOT / "cache"
MODELS_DIR = PACKAGE_ROOT / "models"

STATION_IDS = ["S01", "S02", "S03"]
LEADS = np.arange(15, 121, 15, dtype=np.int64)

# UTC 分钟边界（非闰年合成历，01-01 00:00 = 0）
SPLIT_BOUNDS = {
    "train": (0, 243 * 1440),
    "val": (243 * 1440, 304 * 1440),
    "test": (304 * 1440, 365 * 1440),
}

# 提交声明的历史窗口：lag_day 需 ≥97 功率步；云图 d60 需 ≥7 帧
POWER_STEPS = 192
CLOUD_FRAMES = 7
CLOUD_BANDS = ("B03", "B04", "B13", "B08")
CLOUD_RADII_COARSE = (0, 1, 3, 7, 15)

SEED = 42
N_JOBS = max(1, min(8, (os.cpu_count() or 4)))
SCORE_FLOOR = 0.2
NIGHT_CLEARSKY_EPS = 0.02
EMPIRICAL_CLEARSKY_DAYS = 15
REFIT_MULT = 1.15
HOLD_START = 212 * 1440  # Aug 1 UTC
CHECKPOINTS = [50, 85, 120, 160, 200, 250, 300, 400, 500, 650, 800, 1000]

# production 等权融合（sklearn MLP 在本特征集上数值不稳定，改用 9 个树模型成员）
BLEND_MEMBERS = [
    "lgbm_pc2",
    "lgbm_kt",
    "lgbm_l1",
    "lgbm_perlead",
    "lgbm_pc3",
    "lgbm_pc_xt",
    "lgbm_pc_dart",
    "lgbm_pc_lr03",
    "lgbm_pc_deep",
]

# 成员 LightGBM 超参（与实验阶段 hold 选轮结果一致）
MEMBER_PARAMS: dict[str, dict] = {
    "lgbm_pc2": {
        "seed": SEED + 7,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.7,
        "num_leaves": 95,
        "min_data_in_leaf": 100,
    },
    "lgbm_kt": {},
    "lgbm_l1": {"objective": "l1", "metric": "l1"},
    "lgbm_perlead": {},
    "lgbm_pc3": {
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.7,
        "seed": SEED + 11,
    },
    "lgbm_pc_xt": {
        "extra_trees": True,
        "num_leaves": 127,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.7,
        "seed": SEED + 23,
    },
    "lgbm_pc_dart": {
        "boosting_type": "dart",
        "drop_rate": 0.12,
        "num_leaves": 95,
        "seed": SEED + 31,
    },
    "lgbm_pc_lr03": {
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_data_in_leaf": 120,
        "seed": SEED + 37,
    },
    "lgbm_pc_deep": {
        "num_leaves": 127,
        "min_data_in_leaf": 40,
        "max_depth": -1,
        "seed": SEED + 41,
    },
}
