"""rolling-origin walk-forward 切分（PLAN.md §6.1）。

三条红线：
  1. 分组键是 series_id 不是 game_id。同一 Bo3/Bo5 多局共享双方实力、当日状态、
     BP 博弈状态，甚至"打到第 5 局"这件事本身就携带系列赛结果信息。
  2. embargo 7 天。Elo/rating 是递归状态，会跨 fold 边界渗透。
  3. 任何依赖训练段全局分布的统计量必须在每个 fold 的训练段内重估
     （本模块只负责切分，重估由调用方保证；见 §4.1 分类表）。

汇总方式：所有 fold 的 per-game 预测拼成一条长向量算总 log loss，
**不是各 fold 指标平均**——后者会让小 fold 获得与大 fold 相同的权重。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

WindowKind = Literal["expanding", "36mo", "24mo", "18mo", "12mo", "9mo", "6mo"]
_MONTHS: dict[str, int] = {"36mo": 36, "24mo": 24, "18mo": 18, "12mo": 12, "9mo": 9, "6mo": 6}


@dataclass(frozen=True)
class WFConfig:
    step_weeks: int = 4
    valid_weeks: int = 4
    min_train_games: int = 500
    embargo_days: int = 7
    window: WindowKind = "expanding"
    time_col: str = "kickoff_utc"
    series_col: str = "series_id"


@dataclass(frozen=True)
class Fold:
    idx: int
    train_idx: pd.Index
    valid_idx: pd.Index
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    train_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_valid(self) -> int:
        return len(self.valid_idx)


def make_folds(df: pd.DataFrame, cfg: WFConfig | None = None) -> list[Fold]:
    """按时间推进产出 fold。df 必须已按时间排序且一场一行。"""
    cfg = cfg or WFConfig()
    t = df[cfg.time_col]
    if not t.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序排列")

    step = pd.Timedelta(weeks=cfg.step_weeks)
    span = pd.Timedelta(weeks=cfg.valid_weeks)
    embargo = pd.Timedelta(days=cfg.embargo_days)

    # 第一个 origin：累计满 min_train 场之后，再往后推一个 embargo
    if len(df) <= cfg.min_train_games:
        return []
    first_valid = t.iloc[cfg.min_train_games] + embargo
    # 对齐到 step 网格，避免 fold 边界随数据量抖动
    last_valid_end = t.iloc[-1]

    folds: list[Fold] = []
    v_start = first_valid
    i = 0
    while v_start + span <= last_valid_end + span:
        v_end = v_start + span
        train_cut = v_start - embargo

        if cfg.window == "expanding":
            train_mask = t < train_cut
        else:
            lo = train_cut - pd.DateOffset(months=_MONTHS[cfg.window])
            train_mask = (t < train_cut) & (t >= lo)

        valid_mask = (t >= v_start) & (t < v_end)
        n_tr, n_va = int(train_mask.sum()), int(valid_mask.sum())
        if n_tr >= cfg.min_train_games and n_va > 0:
            folds.append(
                Fold(
                    idx=i,
                    train_idx=df.index[train_mask],
                    valid_idx=df.index[valid_mask],
                    valid_start=v_start,
                    valid_end=v_end,
                    train_end=train_cut,
                )
            )
            i += 1
        v_start = v_start + step
        if v_start > last_valid_end:
            break
    return folds


def assert_no_series_straddle(df: pd.DataFrame, folds: list[Fold], cfg: WFConfig | None = None) -> None:
    """红线 1 的硬断言：同一 series 不得同时出现在 train 与 valid。"""
    cfg = cfg or WFConfig()
    for f in folds:
        tr = set(df.loc[f.train_idx, cfg.series_col])
        va = set(df.loc[f.valid_idx, cfg.series_col])
        overlap = tr & va
        if overlap:
            raise AssertionError(
                f"fold {f.idx}: {len(overlap)} 个 series 同时在 train 与 valid，例 {list(overlap)[:3]}"
            )


def assert_embargo_respected(df: pd.DataFrame, folds: list[Fold], cfg: WFConfig | None = None) -> None:
    """红线 2 的硬断言：train 的最后一场与 valid 的第一场之间至少隔 embargo_days。"""
    cfg = cfg or WFConfig()
    gap = pd.Timedelta(days=cfg.embargo_days)
    for f in folds:
        if not len(f.train_idx):
            continue
        last_train = df.loc[f.train_idx, cfg.time_col].max()
        first_valid = df.loc[f.valid_idx, cfg.time_col].min()
        if first_valid - last_train < gap:
            raise AssertionError(
                f"fold {f.idx}: embargo 不足，train 末 {last_train} → valid 首 {first_valid}"
            )


def oos_frame(df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """把所有 fold 的 valid 段拼成一张 OOS 表，带 fold 编号。

    同一场可能落进多个 fold 的 valid 吗？不会——valid 窗口宽度等于推进步长，
    互不重叠。这里仍然断言一次，因为一旦 step≠valid_weeks 就会重复计数。
    """
    parts = [df.loc[f.valid_idx].assign(fold=f.idx) for f in folds]
    oos = pd.concat(parts) if parts else df.iloc[:0].assign(fold=pd.Series(dtype=int))
    dup = oos.game_id.duplicated().sum()
    if dup:
        raise AssertionError(f"{dup} 场比赛落进了多个 fold 的 valid，会被重复计数")
    return oos.sort_values("kickoff_utc", kind="mergesort")


def summary(df: pd.DataFrame, folds: list[Fold], cfg: WFConfig | None = None) -> dict[str, object]:
    cfg = cfg or WFConfig()
    oos = oos_frame(df, folds)
    return {
        "window": cfg.window,
        "n_folds": len(folds),
        "oos_games": len(oos),
        "oos_series": int(oos[cfg.series_col].nunique()) if len(oos) else 0,
        "first_valid": str(folds[0].valid_start) if folds else None,
        "last_valid": str(folds[-1].valid_end) if folds else None,
        "train_sizes": [f.n_train for f in folds[:3]] + ["..."] + [f.n_train for f in folds[-2:]]
        if len(folds) > 5
        else [f.n_train for f in folds],
    }
