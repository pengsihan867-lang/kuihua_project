#!/usr/bin/env python3
"""打包提交 zip（代码 + models + 说明；不含数据与 cache）。"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
INCLUDE = [
    "README.md",
    "EXPERIMENTS.txt",
    "requirements.txt",
    "predict.py",
    "train.py",
    "pv2h_model",
    "models",
    "scripts",
    "tests",
]


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    staging = DIST / "pv2h_submission"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name in INCLUDE:
        src = ROOT / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = staging / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    "members_*.parquet",
                    "*.log",
                    "resume_members.py",
                ),
            )
            # 二次清理缓存目录（部分环境 copytree ignore 仍会留下空/残留）
            for p in dst.rglob("__pycache__"):
                shutil.rmtree(p, ignore_errors=True)
        else:
            shutil.copy2(src, dst)

    zip_path = DIST / "pv2h_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())
    print("wrote", zip_path, "size_mb=", round(zip_path.stat().st_size / 1e6, 2))

    # 自检：导入 predict
    sys.path.insert(0, str(staging))
    from pv2h_model.models.blend import BlendPredictor  # noqa: WPS433

    bp = BlendPredictor(staging / "models")
    assert bp.members
    print("self-check ok, members=", bp.members)


if __name__ == "__main__":
    main()
