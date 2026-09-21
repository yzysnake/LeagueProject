"""BO 内的序列上下文特征（PLAN.md §4.2 G0/G5）。

**时间正确性的边界要划清楚**：
  · `game_in_series`（本局是系列第几局）**安全**——它只取决于此前已打完的局。
  · `series_score_diff`（本系列到目前为止的比分差）**安全**——同上。
  · `series_length`（这个系列总共打几局）**不安全**——打到第 5 局意味着
    知道了前 4 局是 2-2，而那在第 1 局的 BP 时点是未知的。**本模块不产出它。**

为什么可能有信号（实测 tier≥1、2019 起、23,702 场）：
    局号 1  蓝方胜率 0.5403
    局号 2            0.5070
    局号 3            0.5298
    局号 4            0.5034
    局号 5            0.5312
奇偶交替。多数赛制里败方选边，所以偶数局往往是**弱队在蓝方**，
抵消了一部分蓝方优势。这意味着局号携带的信息可能与 side 先验和实力项冗余——
所以必须走门禁，不能因为"看起来有规律"就加进去。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    """产出序列上下文特征。df 需含 series_id / game_in_series / blue_team_id /
    y_blue_win，且按 kickoff_utc 升序。"""
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")

    g = df.game_in_series.fillna(1).astype(int).to_numpy()
    out = {
        "game_in_series": g.astype(float),
        # 奇偶本身：赛制里的选边规则是离散的，线性项抓不住交替
        "game_is_even": (g % 2 == 0).astype(float),
        "is_first_game": (g == 1).astype(float),
    }

    # ---- series_score_diff：本系列到目前为止，蓝方这支队赢了几局减输了几局 ----
    # 单遍扫描，**只用此前已结算的局**
    score: dict[str, dict[str, int]] = {}
    diff = np.zeros(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        sid = str(r.series_id)
        bt = str(r.blue_team_id)
        s = score.setdefault(sid, {})
        diff[i] = float(s.get(bt, 0) - sum(v for k, v in s.items() if k != bt))
        # 结算后更新
        winner = bt if r.y_blue_win else str(r.red_team_id)
        s[winner] = s.get(winner, 0) + 1
    out["series_score_diff"] = diff
    # 领先方在决胜局的压力：|比分差| 与"是否赛点"
    out["is_match_point"] = (np.abs(diff) >= 2).astype(float)
    return pd.DataFrame(out, index=df.index)
