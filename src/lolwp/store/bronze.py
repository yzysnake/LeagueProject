"""bronze 层：把 13 个年度 CSV 转成 Parquet，只保留 BP 时点可见的列 + label。

为什么要这一层：
  - 全量 CSV 约 700 MB，每次全表扫描要 1-2 分钟；转 Parquet 后降到秒级。
  - **在最早的环节就把赛后统计列物理丢掉**，后面想泄漏都没得泄漏
    （PLAN.md §4.8.1 的列级防线，落在存储层而不只是断言层）。
  - 保留 `data_snapshot_sha`，任何一行都能追溯到 data/manifest.jsonl 里的哪个快照。
"""
from __future__ import annotations

import json
import os

import pandas as pd

from lolwp.features.whitelist import ALLOWED_RAW, LABEL_ONLY, split_raw_columns

RAW_DIR = "data/raw"
BRONZE_DIR = "data/bronze"
MANIFEST = "data/manifest.jsonl"

KEEP = sorted(ALLOWED_RAW | LABEL_ONLY)


def _snapshot_shas() -> dict[int, str]:
    """manifest 是 append-only 日志，同一 year 取最后一条。"""
    out: dict[int, str] = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["year"]] = r["sha256"]
    return out


def build(raw_dir: str = RAW_DIR, out_dir: str = BRONZE_DIR) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    shas = _snapshot_shas()
    frames, report = [], []

    files = sorted(f for f in os.listdir(raw_dir) if f.endswith("OraclesElixir.csv"))
    for fn in files:
        year = int(fn[:4])
        path = os.path.join(raw_dir, fn)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        buckets = split_raw_columns(header)
        usecols = [c for c in header if c in KEEP]

        df = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
        # 只有 result 转数值；其余一律保持字符串，避免 pandas 把 gameid 之类推断坏
        df["result"] = pd.to_numeric(df["result"], errors="coerce").astype("Int8")
        df["data_snapshot_sha"] = shas.get(year, "")
        frames.append(df)
        report.append(
            {
                "year": year,
                "raw_cols": len(header),
                "kept": len(usecols),
                "dropped_forbidden": len(buckets["forbidden"]),
                "rows": len(df),
            }
        )
        out = os.path.join(out_dir, f"{year}.parquet")
        tmp = out + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)

    all_df = pd.concat(frames, ignore_index=True)
    tmp = os.path.join(out_dir, "all.parquet.tmp")
    all_df.to_parquet(tmp, index=False)
    os.replace(tmp, os.path.join(out_dir, "all.parquet"))

    print(f"{'年':<6}{'原始列':>7}{'保留':>6}{'丢弃':>6}{'行数':>9}")
    for r in report:
        print(f"{r['year']:<6}{r['raw_cols']:>7}{r['kept']:>6}{r['dropped_forbidden']:>6}{r['rows']:>9}")
    print(f"\nbronze 合计 {len(all_df)} 行 / {all_df.gameid.nunique()} 场 / {len(all_df.columns)} 列")
    return all_df


if __name__ == "__main__":
    build()
