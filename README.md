# PV2H 光伏超短期功率预测 — 最终模型包

面向主办方 `TASK.md` §8.1 的一次性提交包：三站联合 0–2h 容量归一化功率预测，评价指标为南网日均准确率。

## 1. 模型结构、配置与权重

- **结构**：9 个异构 LightGBM 成员等权融合（联合模型，三站共享一套特征与融合器）
  - 功率+云图：`lgbm_pc2 / pc3 / pc_xt / pc_dart / pc_lr03 / pc_deep`
  - 晴空指数目标：`lgbm_kt`
  - L1 目标：`lgbm_l1`
  - 分时效：`lgbm_perlead`（8 个 lead 各一棵树）
  - （实验阶段曾尝试 MLP；在本特征集上 sklearn MLP 数值不稳定，提交版不采用）
- **权重与配置目录**：[`models/`](models/)
  - `blend.json`：成员列表、等权权重、训练协议摘要
  - `feature_columns.json`：推理特征列顺序
  - `stats.json`：训练集拟合的预处理统计量
  - `lgbm_*.txt`、`perlead/lgbm_lead*.txt`、`mlp.joblib`
- **未使用**外部预训练模型或其他第三方模型权重。

## 2. 推理入口

标准函数（与官方 `tools/example_inference.py` 同形）：

```python
from predict import predict
# inputs = Release(...).inputs(issue, power_steps=192, cloud_frames=7, bands=(...))
pred = predict(inputs)  # DataFrame: station_id, issue_time, lead_minutes, pred_power_ratio
```

CLI：

```bash
python predict.py --data /path/to/pv2h_v1 --issue "09-01 10:00" --output preds.csv
python predict.py --data /path/to/pv2h_v1 --split val --output val_predictions.csv
```

环境变量 `PV2H_DATA` 可覆盖数据包根目录（默认假定本仓库与 `cloud/` `power/` 同级）。

## 3. 数据处理流程

1. **功率**：容量归一化值直接使用；缺测按 `is_missing` 置 NaN，不改写评分标签。
2. **云图**：B03/B04/B08/B13；场站邻域窗口统计 + 相对 −30/−60 min 差分；缺帧 / 缺波段 → NaN + present=0。
3. **晴空**：pvlib Ineichen + 训练集冻结的 lat/lon/`ghi_norm`/`scale`，可计算任意分钟（含测试期）。
4. **特征**：功率滞后/滚动/跨站、lead 编码、晴空与云图特征；因果截止：功率 `t−15`，云图 `floor10(t−20)`。
5. **后处理**：预测裁剪 `≥0`；目标晴空 `<0.02` 时用该站夜间中位数；非有限值回退智能持久性 / 夜间中位数。

## 4. 预处理统计量（`models/stats.json`）

| 字段 | 含义 |
|---|---|
| `stations[Sxx].lat/lon` | 仅用训练功率估计的场站位置 |
| `ghi_norm` | 训练集 GHI 99 分位，冻结为归一化分母 |
| `scale` | 与经验晴空对齐的标定系数 |
| `night_median` | 训练期夜间功率中位数（后处理） |
| `fit_split` | 恒为 `train` |

## 5. 环境、依赖与安装

- Python **3.10+**（开发环境 3.10.6）
- 安装：

```bash
python -m pip install -r requirements.txt
```

依赖见 [`requirements.txt`](requirements.txt)（已锁定主要包版本）。官方读数器位于数据包 `tools/`，需与本包同级提供。

## 6. 历史窗口、场站对应与资源

| 项目 | 声明 |
|---|---|
| 功率历史 | **192** 步 × 15 min（48h），满足 `lag_day` |
| 云图历史 | **7** 帧 × 10 min，满足 d60 |
| 云波段 | B03, B04, B13, B08（主办方可传入全部 12 波段，本包只取上述子集） |
| 场站—模型 | **联合模型**：S01/S02/S03 同一套 10 成员融合 |
| 推理资源 | CPU 即可；建议 ≥8GB 内存；单次起报约 0.1s 量级（模型），含云图解码约 0.1–0.2s；全验证集 5848 起报约 10 分钟 |
| 验证成绩 | 在线官方南网日均准确率 **84.94%**（S01 86.17 / S02 85.55 / S03 83.09） |

## 7. 训练方法与资源说明

```bash
python train.py --stage all          # 统计量 → 云缓存 → 样本 → 成员
python train.py --stage members      # 仅重训成员（需已有 samples_*.parquet）
```

协议：1–7 月拟合 → 8 月按官方指标选 boosting 轮数 → 1–8 月重训（轮数 ×1.15）→ 等权融合。验证集仅用于报告，不参与统计量拟合与权重学习。

实验过程见 [`EXPERIMENTS.txt`](EXPERIMENTS.txt)。**未使用**外部预训练模型。

### 验证评分

```bash
python scripts/evaluate_split.py --data .. --split val
python ../tools/evaluate.py --data .. --predictions val_predictions.csv --output score.json
```

### 打包

```bash
python scripts/build_submission.py
# 产物: dist/pv2h_submission.zip
```
