"""三组位置搭配的英雄协同：ADC+辅助、打野+上单、打野+中单——教练 BP 时真正会掂量的那几对。

和阶段 7 的「同队全部 10 对」不同（那里大半是上单+辅助这种不相干的对，信号被稀释，见
reports/stage7_falsification.md）：这里只看三对，每对按位置区分（打野 A + 中单 B ≠ 打野 B + 中单 A）。

    syn(位置对, 英雄 a, 英雄 b) = 这对英雄之前各局「赛果 − 模型预期」的时间衰减和 ÷（衰减后局数 + k）

模型预期已经含队伍实力与单英雄版本强度，剩下的才是「搭配」本身；k 是收缩（出场少的对往 0 收）。
一局的特征 = 蓝方三对之和 − 红方三对之和。只用开赛更早的局，同一时刻开赛的局互相看不见。

**未进生产**：2026-10-25 实测验收段 +0.00044（0.82× MDE），登记在 ops/backlog.py 等数据积累后重测。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ROLES = (top, jng, mid, bot, sup)
PAIRS: dict[str, tuple[int, int]] = {"bot+sup": (3, 4), "jng+top": (1, 0), "jng+mid": (1, 2)}


def _key(c) -> int | None:
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return None
    return int(c)


def role_pair_synergy(games: pd.DataFrame, p: np.ndarray, k: float = 30.0, hl_days: float = 730.0,
                      pairs: dict[str, tuple[int, int]] = PAIRS) -> np.ndarray:
    """逐场 蓝 − 红 的位置搭配协同（as-of）。games 按开赛升序，含 blue/red_champ_keys（按位置排）、
    y_blue_win、kickoff_utc；p 是同序的模型蓝方胜率（残差的基准）。"""
    kick = games.kickoff_utc.to_numpy()
    # 秒：不能用 asi8——silver 的时间是微秒精度，asi8 给的是微秒整数，按纳秒算的衰减会慢一千倍
    t = (games.kickoff_utc - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    y = games.y_blue_win.to_numpy(dtype=float)
    p = np.asarray(p, dtype=float)
    lam = np.log(2) / (hl_days * 86400.0)
    st: dict[tuple, tuple[float, float, float]] = {}

    def keys(champs) -> list[tuple]:
        cs = [_key(c) for c in list(champs)] if champs is not None else []
        if len(cs) < 5:
            return []
        return [(nm, cs[i], cs[j]) for nm, (i, j) in pairs.items()
                if cs[i] is not None and cs[j] is not None]

    def read(key: tuple, now: float) -> float:
        if key not in st:
            return 0.0
        num, den, t0 = st[key]
        f = np.exp(-lam * (now - t0))
        return num * f / (den * f + k)

    bc, rc = games.blue_champ_keys.to_numpy(), games.red_champ_keys.to_numpy()
    out = np.zeros(len(games))
    i = 0
    while i < len(games):
        j = i
        while j < len(games) and kick[j] == kick[i]:
            j += 1
        for q in range(i, j):
            out[q] = sum(read(kk, t[q]) for kk in keys(bc[q])) - sum(read(kk, t[q]) for kk in keys(rc[q]))
        for q in range(i, j):
            for champs, r in ((bc[q], y[q] - p[q]), (rc[q], p[q] - y[q])):
                for kk in keys(champs):
                    num, den, t0 = st.get(kk, (0.0, 0.0, t[q]))
                    f = np.exp(-lam * (t[q] - t0))
                    st[kk] = (num * f + r, den * f + 1.0, t[q])
        i = j
    return out
