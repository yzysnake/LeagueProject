"""esportsTeamId → OE teamid 的确定性解析。

没有官方对照表，也**不允许用 difflib/fuzzy 按队名匹配**（手册 §3.11 明令禁止：
有实测先例把 Beijing JDG Esports 静默映射成了 LNG Esports——另一支队伍，无报错）。

正确做法是**用已解析的选手反推**：
  feed 给 esportsPlayerId → primary crosswalk → oe:player → silver 里该选手最近
  效力的 oe teamid → 多数表决。
选手对照表本身是「同场 + 同位置 + 同英雄」硬 join 出来的，零 fuzzy，
所以这条链路整体也是确定性的。

降级：多数票不过半（转会期、青训队上场）→ 返回 None，由调用方退化为赛区平均
并告警，绝不猜。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TeamResolution:
    oe_team_id: str | None
    oe_team_name: str | None
    votes: int
    total: int
    confidence: float
    detail: str


def build_player_team_index(silver: pd.DataFrame) -> dict[str, tuple[str, str, pd.Timestamp]]:
    """oe:player → (最近的 oe teamid, teamname, 该场时间)。"""
    idx: dict[str, tuple[str, str, pd.Timestamp]] = {}
    for r in silver.sort_values("kickoff_utc").itertuples(index=False):
        for pfx in ("blue", "red"):
            tid = getattr(r, f"{pfx}_team_id")
            tnm = getattr(r, f"{pfx}_team_name")
            for p in getattr(r, f"{pfx}_player_ids"):
                if p is not None and not pd.isna(p):
                    idx[str(p)] = (str(tid), str(tnm), r.kickoff_utc)
    return idx


def resolve_team(
    oe_player_ids: list[str | None],
    index: dict[str, tuple[str, str, pd.Timestamp]],
    *,
    min_votes: int = 3,
) -> TeamResolution:
    """多数表决。5 人里至少 min_votes 人指向同一支队才认。"""
    votes = Counter()
    names: dict[str, str] = {}
    n = 0
    for p in oe_player_ids:
        if not p:
            continue
        hit = index.get(str(p))
        if hit is None:
            continue
        votes[hit[0]] += 1
        names[hit[0]] = hit[1]
        n += 1
    if not votes:
        return TeamResolution(None, None, 0, n, 0.0, "无任何选手可反查到队伍")
    tid, v = votes.most_common(1)[0]
    if v < min_votes:
        return TeamResolution(
            None, None, v, n, v / max(n, 1),
            f"多数票 {v}/{n} 未达 {min_votes}，可能是转会期或青训队，拒绝猜测",
        )
    return TeamResolution(tid, names[tid], v, n, v / max(n, 1),
                          f"{v}/{n} 名选手指向 {names[tid]}")
