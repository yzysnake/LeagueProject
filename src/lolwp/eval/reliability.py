"""split-half 可靠性闸门（PLAN.md §4.5.2）。

**不要凭断言砍特征，也不要凭直觉加特征。让数据裁决。**

做法：按 series 随机对半分（不是按时间——按时间分会把 meta 漂移混进来，
测到的就不是"这个量稳不稳"而是"meta 变没变"），两边各算一次统计量，
求 Pearson r，再用 Spearman-Brown 修正到全样本：r_full = 2r / (1 + r)。

门槛 r_full ≥ 0.20 才准进入建模阶段。低于门槛记录"不可测量"并砍掉。

为什么可靠性是前置条件：一个统计量如果在两个随机半样本上都对不上自己，
那它就没有可被模型学到的稳定结构，后面无论用多复杂的低秩分解都是在拟合噪声。
先测可靠性比先建模便宜两个数量级。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

ROLES = ("top", "jng", "mid", "bot", "sup")
GATE = 0.20


@dataclass
class ReliabilityResult:
    name: str
    r_half: float
    r_full: float
    n_units: int
    min_obs: int
    passed: bool
    note: str = ""

    def line(self) -> str:
        v = "进入建模 ✓" if self.passed else "【不可测量】砍掉 ✗"
        return (f"{self.name:<28} r_half={self.r_half:+.4f}  r_full={self.r_full:+.4f}  "
                f"单元={self.n_units:>6,}  最小样本={self.min_obs:>3}  {v}")


def _spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if r > -1 else float("nan")


def split_by_series(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 series 随机对半。同一系列必须整块进同一半，否则两半之间相关，
    r 会被人为抬高。"""
    rng = np.random.default_rng(seed)
    sids = df.series_id.unique()
    mask = rng.random(len(sids)) < 0.5
    left = set(sids[mask])
    is_left = df.series_id.isin(left)
    return df[is_left], df[~is_left]


def _champ_stats(df: pd.DataFrame) -> dict[tuple, tuple[float, int]]:
    """单英雄胜率：(champ_key,) -> (win_rate, n)。"""
    acc: dict[tuple, list[float]] = {}
    for r in df.itertuples(index=False):
        y = float(r.y_blue_win)
        for keys, yy in ((r.blue_champ_keys, y), (r.red_champ_keys, 1 - y)):
            for c in keys:
                if c is None or pd.isna(c):
                    continue
                k = (int(c),)
                a = acc.setdefault(k, [0.0, 0.0])
                a[0] += yy
                a[1] += 1
    return {k: (v[0] / v[1], int(v[1])) for k, v in acc.items() if v[1] > 0}


def _pair_stats(df: pd.DataFrame, *, same_team: bool) -> dict[tuple, tuple[float, int]]:
    """same_team=True 是协同（同队两英雄），False 是克制（跨队两英雄）。

    克制用**有序对**（a 在蓝、b 在红），因为克制是反对称的。
    """
    acc: dict[tuple, list[float]] = {}
    for r in df.itertuples(index=False):
        y = float(r.y_blue_win)
        b = [int(c) for c in r.blue_champ_keys if c is not None and not pd.isna(c)]
        rd = [int(c) for c in r.red_champ_keys if c is not None and not pd.isna(c)]
        if same_team:
            for side, yy in ((b, y), (rd, 1 - y)):
                for x, z in combinations(sorted(side), 2):
                    a = acc.setdefault((x, z), [0.0, 0.0])
                    a[0] += yy
                    a[1] += 1
        else:
            for x in b:
                for z in rd:
                    a = acc.setdefault((x, z), [0.0, 0.0])
                    a[0] += y
                    a[1] += 1
    return {k: (v[0] / v[1], int(v[1])) for k, v in acc.items() if v[1] > 0}


def _player_champ_stats(df: pd.DataFrame) -> dict[tuple, tuple[float, int]]:
    acc: dict[tuple, list[float]] = {}
    for r in df.itertuples(index=False):
        y = float(r.y_blue_win)
        for pids, keys, yy in ((r.blue_player_ids, r.blue_champ_keys, y),
                               (r.red_player_ids, r.red_champ_keys, 1 - y)):
            for p, c in zip(pids, keys):
                if p is None or c is None or pd.isna(c) or (isinstance(p, float) and pd.isna(p)):
                    continue
                k = (str(p), int(c))
                a = acc.setdefault(k, [0.0, 0.0])
                a[0] += yy
                a[1] += 1
    return {k: (v[0] / v[1], int(v[1])) for k, v in acc.items() if v[1] > 0}


def reliability(
    df: pd.DataFrame, statfn, name: str, *, min_obs: int = 10, seed: int = 0
) -> ReliabilityResult:
    """对半分 → 两边各算统计量 → 只保留两边都达到 min_obs 的单元 → Pearson r。"""
    a, b = split_by_series(df, seed)
    sa, sb = statfn(a), statfn(b)
    keys = [k for k in set(sa) & set(sb) if sa[k][1] >= min_obs and sb[k][1] >= min_obs]
    if len(keys) < 20:
        return ReliabilityResult(name, float("nan"), float("nan"), len(keys), min_obs,
                                 False, f"共同单元只有 {len(keys)} 个，样本不足以估计可靠性")
    va = np.array([sa[k][0] for k in keys])
    vb = np.array([sb[k][0] for k in keys])
    if va.std() == 0 or vb.std() == 0:
        return ReliabilityResult(name, 0.0, 0.0, len(keys), min_obs, False, "统计量无变异")
    r = float(np.corrcoef(va, vb)[0, 1])
    rf = _spearman_brown(r)
    return ReliabilityResult(name, r, rf, len(keys), min_obs, bool(rf >= GATE))


def run_gate(df: pd.DataFrame, *, seed: int = 0) -> list[ReliabilityResult]:
    return [
        reliability(df, _champ_stats, "单英雄胜率", min_obs=30, seed=seed),
        reliability(df, _player_champ_stats, "选手×英雄胜率", min_obs=10, seed=seed),
        reliability(df, lambda d: _pair_stats(d, same_team=True), "英雄协同（同队 pair）",
                    min_obs=10, seed=seed),
        reliability(df, lambda d: _pair_stats(d, same_team=False), "英雄克制（跨队有序对）",
                    min_obs=10, seed=seed),
    ]
