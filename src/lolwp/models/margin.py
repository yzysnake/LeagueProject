"""分层 BT 的训练目标：赛后经济差的软标签（margin of victory，预注册修订 A6）。

胜负是很粗的观测：险胜和碾压记的是同一个 1。终局经济差把它们分开：

    y_margin = σ(κ · Δ)，Δ = 终局（蓝 − 红）总经济 ÷ 时长，单位「千金 / 10 分钟」

κ = 0.5：胜方的 Δ 中位数 3.34 → 0.84，险胜（5% 分位 0.66）→ 0.58，碾压（95% 分位 6.34）→ 0.96；
翻盘局（胜方经济落后，1.8%）会被记成经济领先方占优。调参段（2016–2018）在 5 组 (λ, κ) × 2 种接法里
选出「只用软标签、κ=0.5、替换胜负标签」，验收段（2019 起）BT4_LR +0.00390 log loss（3.04× MDE），
四个半段全部为正（reports/exp_model_improvements.md §2，复现：scripts/exp_margin_bt.py）。

**只当历史局的标签用**：快照只吃开赛早于快照时刻的比赛，任何一局的经济差都进不了它自己的特征；
本局的赛后统计照旧禁用（features/whitelist.py）。缺经济或时长的局退回胜负标签。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "y_margin"
WIN = "y_blue_win"
MARGIN_KAPPA = 0.5


def margin_target(y, blue_gold, red_gold, seconds, kappa: float = MARGIN_KAPPA) -> np.ndarray:
    """逐场软标签；经济或时长缺失（或时长不为正）的位置退回 y。"""
    y = np.asarray(y, dtype=float)
    bg, rg, s = (np.asarray(v, dtype=float) for v in (blue_gold, red_gold, seconds))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (bg - rg) / 1000.0 / (s / 600.0)
    soft = 1.0 / (1.0 + np.exp(-kappa * d))
    return np.where(np.isfinite(soft) & (s > 0), soft, y)


def with_margin_target(games: pd.DataFrame, margins: pd.DataFrame | None = None) -> pd.DataFrame:
    """silver 口径的比赛表加一列 y_margin。margins 缺省读 data/silver/margins.parquet（store.margins）。"""
    if margins is None:
        from lolwp.store.margins import load

        margins = load()
    m = margins.reindex(games.game_id.astype(str))
    out = games.copy()
    out[TARGET] = margin_target(games[WIN].to_numpy(), m.blue_gold.to_numpy(), m.red_gold.to_numpy(),
                                m.gamelength_s.to_numpy())
    out.attrs = dict(games.attrs)
    return out
