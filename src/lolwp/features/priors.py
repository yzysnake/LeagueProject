"""单遍时序状态机：先读状态写特征，再用赛果更新状态。

PLAN.md §4.1 / §4.2。顺序绝对不能反——反了就是把赛果喂进了本场的特征，
而且不会报错，只会让指标好得不像话。

在线递归类特征（side 先验、team/player Elo）**不需要**在每个 fold 内重估
（§6.1.2 红线 3 的豁免项）：它们逐场只用 t 时刻之前的信息，全局跑一遍即可，
所有 fold 共用。需要重估的是依赖训练段全局分布的统计量（EB 收缩强度、
标准化器、校准器等），那些不在本模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROLES = ("top", "jng", "mid", "bot", "sup")


@dataclass
class EloConfig:
    k: float = 24.0
    init: float = 1500.0
    scale: float = 400.0
    # 休赛期回归：r ← init + decay*(r-init)。跨赛季不重置但要收缩。
    season_decay: float = 0.65
    season_gap_days: int = 45


@dataclass
class SidePriorConfig:
    """按赛区收缩的蓝方先验。k 的含义是"需要几百场才相信这个赛区和全局不一样"。"""
    k: float = 300.0
    global_init: float = 0.5


@dataclass
class _EloState:
    cfg: EloConfig
    r: dict[str, float] = field(default_factory=dict)
    last_seen: dict[str, pd.Timestamp] = field(default_factory=dict)

    def get(self, key: str, now: pd.Timestamp) -> float:
        v = self.r.get(key, self.cfg.init)
        last = self.last_seen.get(key)
        if last is not None:
            gap = (now - last).days
            if gap >= self.cfg.season_gap_days:
                v = self.cfg.init + self.cfg.season_decay * (v - self.cfg.init)
        return v

    def commit(self, key: str, value: float, now: pd.Timestamp) -> None:
        self.r[key] = value
        self.last_seen[key] = now


def _expect(ra: float, rb: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))


def run_state_machine(
    df: pd.DataFrame,
    *,
    elo: EloConfig | None = None,
    side: SidePriorConfig | None = None,
) -> pd.DataFrame:
    """单遍产出 as-of 特征。df 必须按 kickoff_utc 升序、一场一行。

    返回与 df 同索引的特征表，**每一列都只用了本场开赛前的信息**。
    """
    elo = elo or EloConfig()
    side = side or SidePriorConfig()
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")

    team = _EloState(elo)
    player = _EloState(elo)

    # side 先验的 as-of 累加器：全局 + 按赛区
    g_wins = g_n = 0.0
    lg_wins: dict[str, float] = {}
    lg_n: dict[str, float] = {}

    out = {
        "side_prior": np.empty(len(df)),
        "team_elo_diff": np.empty(len(df)),
        "player_elo_diff": np.empty(len(df)),
        "blue_team_elo": np.empty(len(df)),
        "red_team_elo": np.empty(len(df)),
        "team_games_min": np.empty(len(df)),
        "player_games_min": np.empty(len(df)),
    }
    team_seen: dict[str, int] = {}
    player_seen: dict[str, int] = {}

    for i, row in enumerate(df.itertuples(index=False)):
        now = row.kickoff_utc
        lg = row.league_canon
        y = int(row.y_blue_win)

        # ---------- 先读状态，写特征 ----------
        gl = (g_wins + side.k * side.global_init) / (g_n + side.k)
        lw, ln = lg_wins.get(lg, 0.0), lg_n.get(lg, 0.0)
        out["side_prior"][i] = (lw + side.k * gl) / (ln + side.k)

        bt, rt = str(row.blue_team_id), str(row.red_team_id)
        rb, rr = team.get(bt, now), team.get(rt, now)
        out["blue_team_elo"][i] = rb
        out["red_team_elo"][i] = rr
        out["team_elo_diff"][i] = rb - rr
        out["team_games_min"][i] = min(team_seen.get(bt, 0), team_seen.get(rt, 0))

        bp = [str(p) for p in row.blue_player_ids if p is not None and not pd.isna(p)]
        rp = [str(p) for p in row.red_player_ids if p is not None and not pd.isna(p)]
        pb = float(np.mean([player.get(p, now) for p in bp])) if bp else elo.init
        pr = float(np.mean([player.get(p, now) for p in rp])) if rp else elo.init
        out["player_elo_diff"][i] = pb - pr
        out["player_games_min"][i] = min(
            [player_seen.get(p, 0) for p in bp + rp] or [0]
        )

        # ---------- 再用赛果更新状态 ----------
        g_wins += y
        g_n += 1
        lg_wins[lg] = lw + y
        lg_n[lg] = ln + 1

        e_team = _expect(rb, rr, elo.scale)
        team.commit(bt, rb + elo.k * (y - e_team), now)
        team.commit(rt, rr + elo.k * ((1 - y) - (1 - e_team)), now)
        team_seen[bt] = team_seen.get(bt, 0) + 1
        team_seen[rt] = team_seen.get(rt, 0) + 1

        e_pl = _expect(pb, pr, elo.scale)
        for p in bp:
            player.commit(p, player.get(p, now) + elo.k * (y - e_pl), now)
            player_seen[p] = player_seen.get(p, 0) + 1
        for p in rp:
            player.commit(p, player.get(p, now) + elo.k * ((1 - y) - (1 - e_pl)), now)
            player_seen[p] = player_seen.get(p, 0) + 1

    return pd.DataFrame(out, index=df.index)
