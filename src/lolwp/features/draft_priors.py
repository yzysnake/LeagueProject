"""draft 层的 as-of 特征：英雄版本强度（G6）与选手熟练度计数（G7）。

PLAN.md §4.2。与 priors.py 同样是单遍时序状态机——先读状态写特征，再用赛果更新。

两个设计要点，偏离了会静默出错：

1. **英雄强度只借 exposure，不借 win rate**（§4.2 G6 的 v1.1 修正）。
   胜负只吃 tier-1（同质池子），全赛区池子只用来标定两样东西：
   该 position 的基准 logit θ̄_p、以及收缩强度 k₁。
   v1.0 的写法把所有赛区的胜负拍进同一个累加器，恰恰就是"借了 win rate"。

2. **熟练度用计数不用胜率**（§4.3）。
   2025 年 675 个选手 × 159 个英雄 = 107,325 个格子只有 30,800 条观测，
   平均每格 0.29 条。胜率的分母太小，是纯噪声；而场次本身是个干净的计数。
   0→14 场舞台局有效果、之后饱和，所以取 min(g,14)/14。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROLES = ("top", "jng", "mid", "bot", "sup")
HL_CHAMP_DAYS = 75.0     # 英雄 meta 半衰期，§4.4 实测 pick 分布相似度中点约 75 天
HL_MASTERY_DAYS = 180.0  # 熟练度衰减最慢
MASTERY_CAP = 14.0       # 舞台局饱和点
_EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


@dataclass
class _Decayed:
    """指数衰减的 (加权胜场, 加权场次)。读取时才衰减到 now，避免每场全表扫。"""
    half_life_days: float
    w: dict[tuple, float] = field(default_factory=dict)
    n: dict[tuple, float] = field(default_factory=dict)
    t: dict[tuple, pd.Timestamp] = field(default_factory=dict)

    def _decay(self, key: tuple, now: pd.Timestamp) -> tuple[float, float]:
        last = self.t.get(key)
        if last is None:
            return 0.0, 0.0
        dt = (now - last).total_seconds() / 86400.0
        f = 0.5 ** (dt / self.half_life_days)
        return self.w.get(key, 0.0) * f, self.n.get(key, 0.0) * f

    def read(self, key: tuple, now: pd.Timestamp) -> tuple[float, float]:
        return self._decay(key, now)

    def update(self, key: tuple, y: float, now: pd.Timestamp) -> None:
        w, n = self._decay(key, now)
        self.w[key] = w + y
        self.n[key] = n + 1.0
        self.t[key] = now


def run_draft_state_machine(
    df: pd.DataFrame,
    *,
    tier_col: str = "league_tier",
    min_tier_for_winrate: int = 1,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """单遍产出 champ_meta_diff / mastery_diff 等 as-of 特征。

    df 必须按 kickoff_utc 升序、一场一行，且包含 blue/red_champ_keys 与 player_ids。
    **传全赛区的 df**：tier-1 用于胜负，全赛区用于标定。
    """
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")

    cs_t1 = _Decayed(HL_CHAMP_DAYS)    # 只吃 tier-1 的胜负
    cs_all = _Decayed(HL_CHAMP_DAYS)   # 全赛区，只用于标定 θ̄_p 与 k₁
    mastery = _Decayed(HL_MASTERY_DAYS)
    seen_pc: set[tuple] = set()        # (player, champ) 是否上过舞台

    n = len(df)
    out = {k: np.zeros(n) for k in (
        "champ_meta_diff", "mastery_diff", "n_first_time_champ_diff",
        "min_champ_games_diff", "champ_cover_min",
    )}

    for i, row in enumerate(df.itertuples(index=False)):
        now = row.kickoff_utc
        y = float(row.y_blue_win)
        is_t1 = int(getattr(row, tier_col)) >= min_tier_for_winrate

        # θ̄_p 与 k₁ 由全赛区池子的矩估计给出。两者都只用 <t 的信息，
        # 因此是 as-of 的，属于 §4.1 分类表里"在线递归、无需 fold 内重估"的一类。
        theta_bar: dict[str, float] = {}
        for pos in ROLES:
            tw = tn = 0.0
            for (c, p), _ in list(cs_all.n.items())[:0]:  # 占位，避免全表扫
                pass
            tw, tn = cs_all.read(("__POS__", pos), now)
            theta_bar[pos] = _logit(tw / tn) if tn > 0 else 0.0

        # k₁ 由阶段 6 的方差成分定标给出（configs/shrinkage_constants.yaml）。
        # 手册的 476 无来源；本数据的矩估计与 SVI 两条独立路线给出：
        #   champion 层         k = 774.1 / 773.3（差 0.05%）
        #   champion:position 层 k = 673.4   ← 本模块用的是 (英雄,位置) 组合键，取这个
        # 收缩比手册强 41%，what-if 单次换英雄的 Δp 会相应缩小。
        k1 = 673.0

        sides = {}
        for side in ("blue", "red"):
            keys = getattr(row, f"{side}_champ_keys")
            pids = getattr(row, f"{side}_player_ids")
            tot_logit = 0.0
            mast, mins, firsts, covered = [], [], 0, 0
            for pos, c, pid in zip(ROLES, keys, pids):
                if c is None or pd.isna(c):
                    tot_logit += theta_bar[pos]
                    continue
                ck = int(c)
                w, ne = cs_t1.read((ck, pos), now)
                if ne > 0:
                    lo = _logit(w / ne)
                    tot_logit += (lo * ne + k1 * theta_bar[pos]) / (ne + k1)
                    covered += 1
                else:
                    tot_logit += theta_bar[pos]
                if pid is not None and not pd.isna(pid):
                    _, g = mastery.read((str(pid), ck), now)
                    mast.append(min(g, MASTERY_CAP) / MASTERY_CAP)
                    mins.append(g)
                    if (str(pid), ck) not in seen_pc:
                        firsts += 1
            sides[side] = (
                tot_logit,
                float(np.mean(mast)) if mast else 0.0,
                float(min(mins)) if mins else 0.0,
                firsts,
                covered,
            )

        b, r = sides["blue"], sides["red"]
        out["champ_meta_diff"][i] = b[0] - r[0]
        out["mastery_diff"][i] = b[1] - r[1]
        out["min_champ_games_diff"][i] = b[2] - r[2]
        out["n_first_time_champ_diff"][i] = b[3] - r[3]
        out["champ_cover_min"][i] = min(b[4], r[4])

        # ---------- 结算后更新 ----------
        for side, yy in (("blue", y), ("red", 1.0 - y)):
            keys = getattr(row, f"{side}_champ_keys")
            pids = getattr(row, f"{side}_player_ids")
            for pos, c, pid in zip(ROLES, keys, pids):
                if c is None or pd.isna(c):
                    continue
                ck = int(c)
                cs_all.update((ck, pos), yy, now)
                cs_all.update(("__POS__", pos), yy, now)   # position 基准
                if is_t1:
                    cs_t1.update((ck, pos), yy, now)
                if pid is not None and not pd.isna(pid):
                    mastery.update((str(pid), ck), yy, now)
                    seen_pc.add((str(pid), ck))

    feats = pd.DataFrame(out, index=df.index)
    if not return_state:
        return feats
    now = df.kickoff_utc.iloc[-1]
    # 只导出衰减到 as_of 时刻的值，推理时再按实际时间继续衰减
    champ = {}
    for key in cs_t1.n:
        if key[0] == "__POS__":
            continue
        w, n_ = cs_t1.read(key, now)
        if n_ > 1e-4:
            champ[f"{key[0]}|{key[1]}"] = [w, n_]
    pos_base = {}
    for pos in ROLES:
        w, n_ = cs_all.read(("__POS__", pos), now)
        pos_base[pos] = _logit(w / n_) if n_ > 0 else 0.0
    mast = {}
    for key in mastery.n:
        _, g = mastery.read(key, now)
        if g > 1e-3:
            mast[f"{key[0]}|{key[1]}"] = g
    state = {
        "champ_t1": champ,
        "pos_base_logit": pos_base,
        "mastery": mast,
        "k1": 673.0,
        "hl_champ_days": HL_CHAMP_DAYS,
        "hl_mastery_days": HL_MASTERY_DAYS,
        "as_of": str(now),
        "n_seen_pc": len(seen_pc),
    }
    return feats, state


def draft_features_from_state(
    state: dict,
    *,
    blue_champ_keys: list[int],
    red_champ_keys: list[int],
    blue_player_ids: list[str] | None = None,
    red_player_ids: list[str] | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, float]:
    """从状态快照算单场 draft 特征。**必须与 run_draft_state_machine 的公式一致。**

    now 给定时按实际时间继续衰减；不给则视作快照时刻（最常见的推理场景：
    预测的就是"现在"这场比赛，距快照通常只有几天，衰减可忽略但仍然算上）。
    """
    k1 = state["k1"]
    base = state["pos_base_logit"]
    champ = state["champ_t1"]
    mast = state.get("mastery", {})
    as_of = pd.Timestamp(state["as_of"])
    f = 1.0
    if now is not None:
        dt = (pd.Timestamp(now) - as_of).total_seconds() / 86400.0
        f = 0.5 ** (max(dt, 0.0) / state["hl_champ_days"])
    fm = 1.0
    if now is not None:
        dt = (pd.Timestamp(now) - as_of).total_seconds() / 86400.0
        fm = 0.5 ** (max(dt, 0.0) / state["hl_mastery_days"])

    def side_logit(keys: list[int]) -> tuple[float, int]:
        tot, cov = 0.0, 0
        for pos, c in zip(ROLES, keys):
            if c is None or (isinstance(c, float) and math.isnan(c)):
                tot += base[pos]
                continue
            w, n_ = champ.get(f"{int(c)}|{pos}", [0.0, 0.0])
            w, n_ = w * f, n_ * f
            if n_ > 0:
                tot += (_logit(w / n_) * n_ + k1 * base[pos]) / (n_ + k1)
                cov += 1
            else:
                tot += base[pos]
        return tot, cov

    def side_mastery(keys: list[int], pids: list[str] | None) -> tuple[float, float, int]:
        if not pids:
            return 0.0, 0.0, 0
        vals, mins, firsts = [], [], 0
        for c, pid in zip(keys, pids):
            if c is None or not pid:
                continue
            g = mast.get(f"{pid}|{int(c)}", 0.0) * fm
            vals.append(min(g, MASTERY_CAP) / MASTERY_CAP)
            mins.append(g)
            if g <= 0:
                firsts += 1
        return (
            float(np.mean(vals)) if vals else 0.0,
            float(min(mins)) if mins else 0.0,
            firsts,
        )

    bl, bc = side_logit(blue_champ_keys)
    rl, rc = side_logit(red_champ_keys)
    bm, bmin, bf = side_mastery(blue_champ_keys, blue_player_ids)
    rm, rmin, rf = side_mastery(red_champ_keys, red_player_ids)
    return {
        "champ_meta_diff": bl - rl,
        "mastery_diff": bm - rm,
        "min_champ_games_diff": bmin - rmin,
        "n_first_time_champ_diff": float(bf - rf),
        "champ_cover_min": float(min(bc, rc)),
    }
