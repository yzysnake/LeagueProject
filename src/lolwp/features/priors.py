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
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
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

    feats = pd.DataFrame(out, index=df.index)
    if not return_state:
        return feats
    # 最终状态快照：推理时用它 + 下面 features_from_state() 的同一套公式，
    # 保证 training/serving 不会 skew（PLAN.md §7.4）。
    state = {
        "team_elo": dict(team.r),
        "team_last": {k: str(v) for k, v in team.last_seen.items()},
        "player_elo": dict(player.r),
        "player_last": {k: str(v) for k, v in player.last_seen.items()},
        "team_games": dict(team_seen),
        "player_games": dict(player_seen),
        "side_global": {"wins": g_wins, "n": g_n},
        "side_league": {k: [lg_wins[k], lg_n[k]] for k in lg_n},
        "elo_cfg": {"k": elo.k, "init": elo.init, "scale": elo.scale,
                     "season_decay": elo.season_decay, "season_gap_days": elo.season_gap_days},
        "side_cfg": {"k": side.k, "global_init": side.global_init},
        "as_of": str(df.kickoff_utc.max()),
        "n_games": len(df),
    }
    return feats, state


def features_from_state(
    state: dict,
    *,
    league_canon: str,
    blue_team_id: str,
    red_team_id: str,
    blue_player_ids: list[str],
    red_player_ids: list[str],
) -> dict[str, float]:
    """从状态快照算单场特征。**必须与 run_state_machine 的公式逐字一致。**

    training/serving skew 是这类系统最常见的静默故障：两条路径各写一遍公式，
    差一点点，线上指标比离线差却查不出原因。tests/test_serving_skew.py 用
    golden test 锁死两者一致。
    """
    ecfg = state["elo_cfg"]
    scfg = state["side_cfg"]
    init = ecfg["init"]

    gw, gn = state["side_global"]["wins"], state["side_global"]["n"]
    gl = (gw + scfg["k"] * scfg["global_init"]) / (gn + scfg["k"])
    lw, ln = state["side_league"].get(league_canon, [0.0, 0.0])
    side_prior = (lw + scfg["k"] * gl) / (ln + scfg["k"])

    te = state["team_elo"]
    rb = te.get(str(blue_team_id), init)
    rr = te.get(str(red_team_id), init)

    pe = state["player_elo"]
    bp = [pe.get(str(p), init) for p in blue_player_ids if p]
    rp = [pe.get(str(p), init) for p in red_player_ids if p]
    pb = float(np.mean(bp)) if bp else init
    pr = float(np.mean(rp)) if rp else init

    tg, pg = state["team_games"], state["player_games"]
    return {
        "side_prior": side_prior,
        "team_elo_diff": rb - rr,
        "player_elo_diff": pb - pr,
        "blue_team_elo": rb,
        "red_team_elo": rr,
        "team_games_min": float(min(tg.get(str(blue_team_id), 0), tg.get(str(red_team_id), 0))),
        "player_games_min": float(min(
            [pg.get(str(p), 0) for p in list(blue_player_ids) + list(red_player_ids) if p] or [0]
        )),
    }
