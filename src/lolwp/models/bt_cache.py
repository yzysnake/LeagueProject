"""分层 BT 特征的内容寻址缓存。

为什么需要：`rolling_ratings()` 在全量 dev 段上实测 288 s ~ 4,538 s
（收敛难度随数据窗口与 warm start 质量波动，标签打乱时最慢）。
而它的输出**只依赖 (数据快照, 配置)**，与下游的 k₁、特征选择、模型口径全都无关。

之前我因为改了一个与 BT 无关的常数（champ_meta 的 k₁）就整段重算了 BT，
白等了近一小时。缓存键取 (数据指纹, 配置指纹)，改别的东西一律命中缓存。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

import pandas as pd

CACHE_DIR = "data/cache/bt"


def _fingerprint(df: pd.DataFrame, cfg) -> str:
    """数据指纹用 (行数, 时间范围, game_id 的 sha) —— 不读全表内容，够用且快。"""
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    h.update(str(df.kickoff_utc.min()).encode())
    h.update(str(df.kickoff_utc.max()).encode())
    h.update(hashlib.sha256("|".join(df.game_id.astype(str)).encode()).digest())
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def rolling_ratings_cached(df: pd.DataFrame, cfg=None, *, cache_dir: str = CACHE_DIR,
                           force: bool = False) -> pd.DataFrame:
    from lolwp.models.hier_bt import HierBTConfig, rolling_ratings

    cfg = cfg or HierBTConfig()
    key = _fingerprint(df, cfg)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.parquet")
    if os.path.exists(path) and not force:
        out = pd.read_parquet(path)
        if len(out) == len(df):
            out.index = df.index
            return out
    out = rolling_ratings(df, cfg)
    tmp = path + ".tmp"
    out.reset_index(drop=True).to_parquet(tmp, index=False)
    os.replace(tmp, path)
    out.index = df.index
    return out
