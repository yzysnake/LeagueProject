"""Glicko-2 双轨（队伍层 + 选手层）——Elo 的对照实现，主产出是 RD 而不是 log loss。

为什么要单独写一套而不是接着调 Elo（PLAN.md §4.2 G2 / §4.6.3 / §4.6.4）：
Elo 只给点估计，一支只打过 2 场的新队和一支打过 300 场的老队，rating 都是一个数，
模型无从分辨"1520 分"哪个可信。Glicko-2 多带一个 RD（rating deviation），它有两个
Elo 给不了的用处——
  (a) **冷启动不再虚假自信**：新队 RD=350，g(RD) 会自动把它的 rating 差往 0 压，
      预测不敢偏离 0.5 太远；Elo 做不到这件事，只能靠外面加 `has_new_team` 补丁。
  (b) **UI 的预测区间有了尺子**：§4.6.4 要求"国际赛的预测区间必须明显放宽"。
      RD 就是那个宽度的来源，`win_prob_band()` 把它直接算成概率区间。

--------------------------------------------------------------------------
§6.1.2 红线 3 归类：**在线递归类，豁免逐 fold 重估。**

理由（必须逐条成立，否则这个豁免就是错的）：
  1. 本模块的所有超参（τ=0.5、RD₀=350、σ₀=0.06、SCALE=173.7178）都是 Glickman
     论文里的常数，**没有一个是在本数据上标定的**，因此不存在"用未来数据定超参"。
  2. 特征的因果性由 rating period 的语义保证：一个 period 内的所有比赛，一律读
     period **起点**的快照；period 内的赛果攒在 pending 里，period 结束才批量写回。
     所以第 t 场的特征只依赖严格早于本 period 起点的比赛，与"训练段切到哪一天"无关。
  3. 唯一一个看起来像全局量的东西是 side 先验——但它不在本模块（在 priors.py）。
全局跑一遍状态机、所有 fold 共用是正确的；逐 fold 重估只会得到逐位相同的结果。
（这与 §4.1 分类表把 "Glicko RD" 明确放在"fold 内可复用"那一行一致。）

--------------------------------------------------------------------------
rating period 怎么切：代价与理由

Glicko-2 **不是逐场算法**，Glickman 建议一个 period 里每个选手有 10–15 场。
本数据实测（data/silver/games.parquet，98,592 场 dev 段）：
  * 每队每周 3.26 场（中位数 3，p90 6）→ 按周切只有 3 场/period，离 10–15 差得远；
  * 要凑到 10–15 场得按 3–4 周切，那 rating 就会陈旧到一个赛段只更新 3 次。
三种切法的代价：
  | 切法        | 优点                                  | 代价 |
  |---|---|---|
  | `"game"`   | 信息最新：Bo3 第 2 局能看到第 1 局的结果 | 退化成 Glicko，σ 的批量估计几乎没有样本支撑；且同一 series 内的强相关被当独立 |
  | `"week"`   | 真正的批量，σ 有 3 场左右的样本；与赛程天然对齐 | 同周内的比赛互相看不见，特征最陈旧可达 7 天 |
  | `"month"`  | σ 样本够（~13 场），最接近论文建议     | rating 陈旧到一个赛段只动 3 次，赛季初几乎不更新 |
**默认取 `"week"`**：它是唯一同时满足"是个真批量"和"陈旧度不超过一个赛程周"的切法。
三种都实现了，`benchmark()` 会把三种的 log loss 并排跑出来——差别是实测的，不是猜的。

休赛期 RD 膨胀用连续时间版本（论文的 step 6 是"每空一个 period 加一个 σ²"，
这里推广成"每空 `inflation_period_days` 天加一个 σ²"）。这样三种切法的 RD 才可比：
按周切与按场切，同样过一周注入的方差量相同。实测最大空档：全局 34 天，
单队 3,286 天，单选手 4,102 天——所以 `max_rd` 的截断是必需的，不是装饰。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "ANCHOR",
    "SCALE",
    "Glicko2Config",
    "Rating",
    "drift",
    "expected_score",
    "features_from_glicko_state",
    "g_factor",
    "rd_report",
    "run_glicko_state_machine",
    "solve_volatility",
    "update_rating",
    "win_prob_band",
]

# Glickman 定义的 Elo↔内部尺度换算。改这两个常数就不是 Glicko-2 了。
SCALE = 173.7178
ANCHOR = 1500.0

PeriodKind = Literal["game", "day", "week", "biweek", "month"]
PlayerMode = Literal["pairwise", "team_mean", "off"]

# (mu_j, phi_j, score, weight)：一条对局记录，内部尺度。
Opponent = tuple[float, float, float, float]


# ==========================================================================
# 1. 纯函数层：Glickman 论文的算法，逐步对应，可单测
# ==========================================================================


@dataclass(frozen=True)
class Rating:
    """一个实体的 Glicko-2 状态。mu/phi 是内部尺度，sigma 是波动率。

    存内部尺度而不是 Elo 尺度，是因为除了展示以外所有公式都在内部尺度上；
    每次读写都换算一遍既慢又容易漏。展示用 `.rating` / `.rd`。
    """

    mu: float = 0.0
    phi: float = 350.0 / SCALE
    sigma: float = 0.06

    @property
    def rating(self) -> float:
        """Elo 尺度的 rating，可以直接和 priors.py 的 team_elo 摆在一起看。"""
        return ANCHOR + SCALE * self.mu

    @property
    def rd(self) -> float:
        """Elo 尺度的 RD。350 = 一无所知，30–60 = 打了很多场。"""
        return SCALE * self.phi


@dataclass(frozen=True)
class Glicko2Config:
    """全部取 Glickman 论文的默认值。**任何一个都不许在 OOS 上调**（见模块 docstring）。"""

    init_rating: float = ANCHOR
    init_rd: float = 350.0
    init_vol: float = 0.06
    # τ 约束波动率的变化速度。论文建议 0.3–1.2，越小越不允许 rating 突变。
    tau: float = 0.5
    max_rd: float = 350.0
    # RD 下限。0.0 = 纯论文版（不设限）。设了下限相当于承认"再多场次也留一点疑问"。
    min_rd: float = 0.0
    period: PeriodKind = "week"
    # RD 膨胀的时间单位：每过这么多天注入一个 σ²。与 period 解耦，见模块 docstring。
    inflation_period_days: float = 7.0
    player_mode: PlayerMode = "pairwise"
    # 5v5 拆成 25 对 1v1 时每对的权重。None = 自动取 1/n_对手，
    # 这样"一场比赛"仍然只提供一场比赛的信息量（5 个结果完全相关，不是 5 次独立观测）。
    pairwise_weight: float | None = None
    # 休赛期 μ 回归。1.0 = 关闭（纯 Glicko-2：不确定性涨回来就够了，不动点估计）。
    season_decay: float = 1.0
    season_gap_days: float = 45.0
    eps: float = 1e-6
    max_iter: int = 100

    def initial(self) -> Rating:
        return Rating(
            (self.init_rating - ANCHOR) / SCALE, self.init_rd / SCALE, self.init_vol
        )


def g_factor(phi: float) -> float:
    """g(φ)：对手越不确定，这一局对我的 rating 影响越小。论文 step 3。"""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def expected_score(mu: float, mu_j: float, phi_j: float) -> float:
    """E(μ, μ_j, φ_j)：注意它被对手的 φ_j **压向 0.5**，这正是 Elo 没有的性质。"""
    return 1.0 / (1.0 + math.exp(-g_factor(phi_j) * (mu - mu_j)))


def solve_volatility(
    sigma: float,
    delta: float,
    phi: float,
    v: float,
    *,
    tau: float = 0.5,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """论文 step 5：用 Illinois 算法解 f(x)=0，返回新的 σ'。

    f(x) = e^x(Δ²−φ²−v−e^x) / (2(φ²+v+e^x)²) − (x−a)/τ²，a = ln σ²。
    这里刻意不用 scipy.optimize：Illinois 是论文指定的求根法，逐字实现才能和
    参考实现对数；换个求根器在 Δ² 远大于 φ²+v 时会落到不同的根上。
    """
    a = math.log(sigma * sigma)
    d2 = delta * delta
    p2 = phi * phi
    tau2 = tau * tau

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (d2 - p2 - v - ex)
        den = 2.0 * (p2 + v + ex) ** 2
        return num / den - (x - a) / tau2

    big = d2 - p2 - v
    if big > 0.0:
        b = math.log(big)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
            if k > max_iter:  # 兜底：理论上不会发生，发生了也不能死循环
                break
        b = a - k * tau

    fa, fb = f(a), f(b)
    lo = a
    it = 0
    while abs(b - lo) > eps and it < max_iter:
        c = lo + (lo - b) * fa / (fb - fa)
        fc = f(c)
        if fc * fb <= 0.0:
            lo, fa = b, fb
        else:
            fa = fa / 2.0
        b, fb = c, fc
        it += 1
    return math.exp(lo / 2.0)


def drift(r: Rating, elapsed: float, max_phi: float) -> Rating:
    """休赛期膨胀：φ ← min(√(φ² + elapsed·σ²), φ_max)。elapsed 的单位是 period。

    论文 step 6 的连续时间推广。elapsed=1 时逐字等于论文的"空过一个 period"。
    截断是必需的：本数据里单队最长空档 3,286 天，不截断 φ 会飞到无意义的量级。
    """
    if elapsed <= 0.0:
        return r
    phi = math.sqrt(r.phi * r.phi + elapsed * r.sigma * r.sigma)
    return Rating(r.mu, min(phi, max_phi), r.sigma)


def update_rating(
    r: Rating,
    opponents: Sequence[Opponent],
    *,
    elapsed: float = 1.0,
    cfg: Glicko2Config | None = None,
) -> Rating:
    """一个 rating period 的批量更新。opponents 为空 = 没打，只膨胀 RD（step 6）。

    elapsed=1.0 且所有 weight=1.0 时，本函数逐字等于 Glickman 论文的 step 3–8，
    tests/test_glicko2.py 用论文第 3 页的算例（1500/200/0.06 打 3 场 → 1464.06/151.52/0.05999）
    锁死这一点。weight 是本项目加的：5v5 拆成 25 对 1v1 时，那 25 个结果完全相关，
    按 1 计权会让 RD 收缩快 5 倍（假信息量），所以默认每对计 1/5。
    """
    cfg = cfg or Glicko2Config()
    max_phi = cfg.max_rd / SCALE
    min_phi = cfg.min_rd / SCALE
    if not opponents:
        return drift(r, elapsed, max_phi)

    inv_v = 0.0
    s = 0.0
    for mu_j, phi_j, score, w in opponents:
        gj = g_factor(phi_j)
        e = 1.0 / (1.0 + math.exp(-gj * (r.mu - mu_j)))
        inv_v += w * gj * gj * e * (1.0 - e)
        s += w * gj * (score - e)
    if inv_v <= 0.0:  # 数值退化（E 完全饱和），当作没打
        return drift(r, elapsed, max_phi)

    v = 1.0 / inv_v
    delta = v * s
    sigma_new = solve_volatility(
        r.sigma, delta, r.phi, v, tau=cfg.tau, eps=cfg.eps, max_iter=cfg.max_iter
    )
    phi_star = min(
        math.sqrt(r.phi * r.phi + elapsed * sigma_new * sigma_new), max_phi
    )
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + inv_v)
    phi_new = min(max(phi_new, min_phi), max_phi)
    return Rating(r.mu + phi_new * phi_new * s, phi_new, sigma_new)


def win_prob_band(
    blue: Rating, red: Rating, *, z: float = 1.96
) -> tuple[float, float, float]:
    """Glicko 自己的胜率 + 由 RD 直接给出的区间。§4.6.4 的 UI 宽度就用这个。

    点估计用 g(√(φ_b²+φ_r²)) 压过的 logit——两边越不确定，预测越靠近 0.5。
    区间把 μ 差的后验 sd = √(φ_b²+φ_r²) 传播过去。国际赛/新队伍的区间会自动变宽，
    不需要在 UI 层写任何"如果是国际赛就放宽"的硬编码分支。
    """
    sd = math.sqrt(blue.phi * blue.phi + red.phi * red.phi)
    g = g_factor(sd)
    d = blue.mu - red.mu
    lo, hi = d - z * sd, d + z * sd
    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-g * x))

    return sig(d), sig(lo), sig(hi)


# ==========================================================================
# 2. rating period 的切分
# ==========================================================================


def _period_start_days(ts: pd.Series, kind: PeriodKind) -> np.ndarray:
    """返回每一场所属 rating period 的**起点**（自 epoch 的天数，float）。

    用 period 起点而不是比赛时刻，是为了保证同一 period 内每一场读到的状态完全相同
    ——这正是"period 内互相看不见"这条语义的实现。
    """
    days = ts.to_numpy(dtype="datetime64[ns]").astype("int64") / 86_400e9
    if kind == "game":
        return days
    if kind == "month":
        ym = (ts.dt.year * 12 + ts.dt.month - 1).to_numpy()
        starts = pd.to_datetime(
            {"year": ym // 12, "month": ym % 12 + 1, "day": 1}, utc=True
        )
        return starts.to_numpy(dtype="datetime64[ns]").astype("int64") / 86_400e9
    width = {"day": 1.0, "week": 7.0, "biweek": 14.0}[kind]
    return np.floor(days / width) * width


# ==========================================================================
# 3. 状态机（priors.py 的 return_state 模式）
# ==========================================================================


class _Track:
    """一条 rating 轨道（队伍层或选手层）。批量更新 + 惰性膨胀。

    惰性膨胀（读到才算）而不是每个 period 给所有实体算一遍，是因为本数据有 2,046 支
    队伍和 10,863 名选手，但一个 period 只有几十个实体在打；全量膨胀是 O(实体×period)，
    惰性是 O(出场次数)。两者数值上完全等价（膨胀只依赖时间差）。
    """

    __slots__ = ("_init", "_pending", "_prior", "cfg", "last_t", "max_phi", "n", "r")

    def __init__(self, cfg: Glicko2Config) -> None:
        self.cfg = cfg
        self.r: dict[str, Rating] = {}
        self.last_t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.max_phi = cfg.max_rd / SCALE
        self._init = cfg.initial()
        # 本 period 的缓存：key -> (用于更新的先验 Rating, elapsed, 用于出特征的 Rating)
        self._prior: dict[str, tuple[Rating, float, Rating]] = {}
        self._pending: dict[str, list[Opponent]] = {}

    def view(self, key: str, t: float) -> Rating:
        """读 period 起点 t 的 as-of 状态（含休赛期膨胀与可选的赛季间回归）。"""
        hit = self._prior.get(key)
        if hit is not None:
            return hit[2]
        base = self.r.get(key)
        if base is None:
            base = self._init
            self.r[key] = base
            self.last_t[key] = t
        # df 已断言升序，max 只防 period 定义被改错时出负数
        elapsed = max(t - self.last_t[key], 0.0)
        if (
            self.cfg.season_decay != 1.0
            and elapsed * self.cfg.inflation_period_days >= self.cfg.season_gap_days
        ):
            mu0 = (self.cfg.init_rating - ANCHOR) / SCALE
            base = Rating(mu0 + self.cfg.season_decay * (base.mu - mu0), base.phi, base.sigma)
        shown = drift(base, elapsed, self.max_phi)
        self._prior[key] = (base, elapsed, shown)
        return shown

    def record(self, key: str, opps: Iterable[Opponent]) -> None:
        self._pending.setdefault(key, []).extend(opps)

    def flush(self, t: float) -> None:
        """period 结束：把攒下的对局一次性写回。**这里才允许改 self.r。**"""
        for key, opps in self._pending.items():
            base, elapsed, _ = self._prior[key]
            self.r[key] = update_rating(base, opps, elapsed=elapsed, cfg=self.cfg)
            self.last_t[key] = t
            self.n[key] = self.n.get(key, 0) + len(opps)
        self._pending.clear()
        self._prior.clear()

    def snapshot(self) -> dict[str, list[float]]:
        return {
            k: [v.mu, v.phi, v.sigma, self.last_t.get(k, 0.0), float(self.n.get(k, 0))]
            for k, v in self.r.items()
        }


def _clean_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    return [str(p) for p in raw if p is not None and not pd.isna(p)]


_TEAM_COLS = (
    "team_glicko_diff",
    "blue_team_glicko",
    "red_team_glicko",
    "team_rd_blue",
    "team_rd_red",
    "team_rd_joint",
    "team_glicko_g_diff",
    "team_glicko_prob",
    "team_vol_blue",
    "team_vol_red",
    "team_n_games_min",
)
_PLAYER_COLS = (
    "player_glicko_diff",
    "player_rd_blue",
    "player_rd_red",
    "player_rd_max_blue",
    "player_rd_max_red",
    "player_glicko_g_diff",
    "player_n_games_min",
)
FEATURE_COLS = _TEAM_COLS + _PLAYER_COLS


def run_glicko_state_machine(
    df: pd.DataFrame,
    *,
    cfg: Glicko2Config | None = None,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """单遍产出 Glicko-2 的 as-of 特征。df 必须按 kickoff_utc 升序、一场一行。

    与 priors.py 的区别只有一处，但它是本模块的全部要点：**更新是按 rating period
    批量的**。同一 period 内的比赛读同一份快照，period 结束才写回。
    因此 `period="week"` 下，同一周的第 3 场看不到第 1 场的结果——这是 Glicko-2 的
    定义，不是 bug；代价（特征陈旧最多 7 天）在模块 docstring 里量化过。
    """
    cfg = cfg or Glicko2Config()
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")

    n = len(df)
    pstart = _period_start_days(df.kickoff_utc, cfg.period) / cfg.inflation_period_days
    team = _Track(cfg)
    player = _Track(cfg)
    out = {c: np.empty(n) for c in FEATURE_COLS}

    cur = math.nan
    for i, row in enumerate(df.itertuples(index=False)):
        t = float(pstart[i])
        if t != cur:
            if not math.isnan(cur):
                team.flush(cur)
                player.flush(cur)
            cur = t

        y = float(row.y_blue_win)
        bt, rt = str(row.blue_team_id), str(row.red_team_id)
        rb, rr = team.view(bt, t), team.view(rt, t)

        # ---------- 先读状态，写特征 ----------
        sd = math.sqrt(rb.phi * rb.phi + rr.phi * rr.phi)
        gj = g_factor(sd)
        out["team_glicko_diff"][i] = rb.rating - rr.rating
        out["blue_team_glicko"][i] = rb.rating
        out["red_team_glicko"][i] = rr.rating
        out["team_rd_blue"][i] = rb.rd
        out["team_rd_red"][i] = rr.rd
        out["team_rd_joint"][i] = SCALE * sd
        out["team_glicko_g_diff"][i] = gj * (rb.rating - rr.rating)
        out["team_glicko_prob"][i] = 1.0 / (1.0 + math.exp(-gj * (rb.mu - rr.mu)))
        out["team_vol_blue"][i] = rb.sigma
        out["team_vol_red"][i] = rr.sigma
        out["team_n_games_min"][i] = min(team.n.get(bt, 0), team.n.get(rt, 0))

        bp, rp = _clean_ids(row.blue_player_ids), _clean_ids(row.red_player_ids)
        bv = [player.view(p, t) for p in bp]
        rv = [player.view(p, t) for p in rp]
        if bv and rv:
            mb = sum(x.mu for x in bv) / len(bv)
            mr = sum(x.mu for x in rv) / len(rv)
            pb = math.sqrt(sum(x.phi * x.phi for x in bv) / len(bv))
            pr = math.sqrt(sum(x.phi * x.phi for x in rv) / len(rv))
            out["player_glicko_diff"][i] = SCALE * (mb - mr)
            out["player_rd_blue"][i] = SCALE * pb
            out["player_rd_red"][i] = SCALE * pr
            out["player_rd_max_blue"][i] = SCALE * max(x.phi for x in bv)
            out["player_rd_max_red"][i] = SCALE * max(x.phi for x in rv)
            out["player_glicko_g_diff"][i] = (
                g_factor(math.sqrt(pb * pb + pr * pr)) * SCALE * (mb - mr)
            )
            out["player_n_games_min"][i] = min(player.n.get(p, 0) for p in bp + rp)
        else:
            # 选手名单缺失：给中性值，并让 RD 顶到上限（"我什么都不知道"）
            out["player_glicko_diff"][i] = 0.0
            out["player_rd_blue"][i] = cfg.max_rd
            out["player_rd_red"][i] = cfg.max_rd
            out["player_rd_max_blue"][i] = cfg.max_rd
            out["player_rd_max_red"][i] = cfg.max_rd
            out["player_glicko_g_diff"][i] = 0.0
            out["player_n_games_min"][i] = 0

        # ---------- 再把赛果攒进 pending（period 结束才写回） ----------
        team.record(bt, [(rr.mu, rr.phi, y, 1.0)])
        team.record(rt, [(rb.mu, rb.phi, 1.0 - y, 1.0)])

        if cfg.player_mode == "pairwise" and bv and rv:
            wb = cfg.pairwise_weight if cfg.pairwise_weight is not None else 1.0 / len(rv)
            wr = cfg.pairwise_weight if cfg.pairwise_weight is not None else 1.0 / len(bv)
            red_opp = [(x.mu, x.phi, y, wb) for x in rv]
            blue_opp = [(x.mu, x.phi, 1.0 - y, wr) for x in bv]
            for p in bp:
                player.record(p, red_opp)
            for p in rp:
                player.record(p, blue_opp)
        elif cfg.player_mode == "team_mean" and bv and rv:
            # Elo 现行做法的 Glicko 版：5 人共用一个"聚合对手"。
            # 聚合对手的 φ 取 RMS 而不是 /√5——5 个人的表现不独立，别假装信息量有 5 份。
            opp_r = (sum(x.mu for x in rv) / len(rv),
                     math.sqrt(sum(x.phi**2 for x in rv) / len(rv)), y, 1.0)
            opp_b = (sum(x.mu for x in bv) / len(bv),
                     math.sqrt(sum(x.phi**2 for x in bv) / len(bv)), 1.0 - y, 1.0)
            for p in bp:
                player.record(p, [opp_r])
            for p in rp:
                player.record(p, [opp_b])

    if not math.isnan(cur):
        team.flush(cur)
        player.flush(cur)

    feats = pd.DataFrame(out, index=df.index)
    if not return_state:
        return feats
    state = {
        "team": team.snapshot(),
        "player": player.snapshot(),
        "cfg": {
            "init_rating": cfg.init_rating, "init_rd": cfg.init_rd,
            "init_vol": cfg.init_vol, "tau": cfg.tau, "max_rd": cfg.max_rd,
            "min_rd": cfg.min_rd, "period": cfg.period,
            "inflation_period_days": cfg.inflation_period_days,
            "player_mode": cfg.player_mode, "pairwise_weight": cfg.pairwise_weight,
            "season_decay": cfg.season_decay, "season_gap_days": cfg.season_gap_days,
        },
        "as_of_days": float(pstart[-1]) * cfg.inflation_period_days if n else 0.0,
        "as_of": str(df.kickoff_utc.max()) if n else None,
        "n_games": n,
    }
    return feats, state


def _state_view(state: dict[str, Any], track: str, key: str, t: float) -> Rating:
    c = state["cfg"]
    row = state[track].get(str(key))
    if row is None:
        return Rating((c["init_rating"] - ANCHOR) / SCALE, c["init_rd"] / SCALE, c["init_vol"])
    mu, phi, sigma, last_t, _ = row
    base = Rating(mu, phi, sigma)
    elapsed = max(0.0, t - last_t)
    if c["season_decay"] != 1.0 and elapsed * c["inflation_period_days"] >= c["season_gap_days"]:
        mu0 = (c["init_rating"] - ANCHOR) / SCALE
        base = Rating(mu0 + c["season_decay"] * (base.mu - mu0), base.phi, base.sigma)
    return drift(base, elapsed, c["max_rd"] / SCALE)


def features_from_glicko_state(
    state: dict[str, Any],
    *,
    blue_team_id: str,
    red_team_id: str,
    blue_player_ids: Sequence[str],
    red_player_ids: Sequence[str],
    now: pd.Timestamp | None = None,
) -> dict[str, float]:
    """从快照算单场特征。**必须与 run_glicko_state_machine 的公式逐字一致。**

    training/serving skew 是这类系统最常见的静默故障（PLAN.md §7.4）：两条路径
    各写一遍公式，差一点点，线上比离线差却查不出原因。tests/test_glicko2.py 用
    golden test 把两条路径锁死。`now` 给了就会按休赛期膨胀 RD——线上推理时一定要给，
    否则一支半年没打的队会顶着半年前那个窄 RD 出现在界面上。
    """
    c = state["cfg"]
    if now is None:
        t = float(state.get("as_of_days", 0.0)) / c["inflation_period_days"]
    else:
        t = pd.Timestamp(now).timestamp() / 86_400.0 / c["inflation_period_days"]

    rb = _state_view(state, "team", str(blue_team_id), t)
    rr = _state_view(state, "team", str(red_team_id), t)
    bp = _clean_ids(list(blue_player_ids))
    rp = _clean_ids(list(red_player_ids))
    bv = [_state_view(state, "player", p, t) for p in bp]
    rv = [_state_view(state, "player", p, t) for p in rp]

    sd = math.sqrt(rb.phi**2 + rr.phi**2)
    gj = g_factor(sd)
    tn = state["team"]
    pn = state["player"]
    feats = {
        "team_glicko_diff": rb.rating - rr.rating,
        "blue_team_glicko": rb.rating,
        "red_team_glicko": rr.rating,
        "team_rd_blue": rb.rd,
        "team_rd_red": rr.rd,
        "team_rd_joint": SCALE * sd,
        "team_glicko_g_diff": gj * (rb.rating - rr.rating),
        "team_glicko_prob": 1.0 / (1.0 + math.exp(-gj * (rb.mu - rr.mu))),
        "team_vol_blue": rb.sigma,
        "team_vol_red": rr.sigma,
        "team_n_games_min": float(min(
            tn.get(str(blue_team_id), [0, 0, 0, 0, 0])[4],
            tn.get(str(red_team_id), [0, 0, 0, 0, 0])[4],
        )),
    }
    if bv and rv:
        mb = sum(x.mu for x in bv) / len(bv)
        mr = sum(x.mu for x in rv) / len(rv)
        pb = math.sqrt(sum(x.phi**2 for x in bv) / len(bv))
        pr = math.sqrt(sum(x.phi**2 for x in rv) / len(rv))
        feats |= {
            "player_glicko_diff": SCALE * (mb - mr),
            "player_rd_blue": SCALE * pb,
            "player_rd_red": SCALE * pr,
            "player_rd_max_blue": SCALE * max(x.phi for x in bv),
            "player_rd_max_red": SCALE * max(x.phi for x in rv),
            "player_glicko_g_diff": g_factor(math.sqrt(pb * pb + pr * pr)) * SCALE * (mb - mr),
            "player_n_games_min": float(min(pn.get(p, [0, 0, 0, 0, 0])[4] for p in bp + rp)),
        }
    else:
        feats |= {
            "player_glicko_diff": 0.0,
            "player_rd_blue": c["max_rd"], "player_rd_red": c["max_rd"],
            "player_rd_max_blue": c["max_rd"], "player_rd_max_red": c["max_rd"],
            "player_glicko_g_diff": 0.0, "player_n_games_min": 0.0,
        }
    return feats


# ==========================================================================
# 4. RD 的画像与 Elo 对照实验（可复现：python -m lolwp.models.glicko2）
# ==========================================================================


def rd_report(df: pd.DataFrame, feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """RD 的分布画像：冷启动 vs 老手、赛季初 vs 赛季末。

    这是本模块真正的产出——log loss 的差别大概率很小，RD 是 Elo 完全给不出来的东西。
    """
    d = pd.concat([df[["kickoff_utc", "league_tier", "league_canon"]], feats], axis=1)
    d = d.assign(
        month=d.kickoff_utc.dt.month,
        year=d.kickoff_utc.dt.year,
        player_rd=(d.player_rd_blue + d.player_rd_red) / 2,
        team_rd=(d.team_rd_blue + d.team_rd_red) / 2,
    )
    exp_bins = [-1, 0, 5, 20, 100, 500, 10**9]
    exp_lab = ["0 场（首秀）", "1-5", "6-20", "21-100", "101-500", ">500"]
    by_exp_t = d.groupby(
        pd.cut(d.team_n_games_min, exp_bins, labels=exp_lab), observed=True
    ).agg(n=("team_rd", "size"), rd_mean=("team_rd", "mean"),
          rd_p10=("team_rd", lambda s: s.quantile(.10)),
          rd_p90=("team_rd", lambda s: s.quantile(.90)))
    by_exp_p = d.groupby(
        pd.cut(d.player_n_games_min, exp_bins, labels=exp_lab), observed=True
    ).agg(n=("player_rd_max_blue", "size"),
          rd_mean=("player_rd", "mean"),
          rd_max_mean=("player_rd_max_blue", "mean"))
    # 赛季初 vs 赛季末：LoL 赛历 1-2 月开赛、10-11 月世界赛收尾
    seg = pd.cut(d.month, [0, 2, 5, 8, 11, 12],
                 labels=["1-2 赛季初", "3-5", "6-8", "9-11 赛季末", "12 休赛"])
    by_seg = d.groupby(seg, observed=True).agg(
        n=("team_rd", "size"), team_rd=("team_rd", "mean"),
        player_rd=("player_rd", "mean"), team_rd_joint=("team_rd_joint", "mean"))
    intl = d.assign(intl=d.league_tier >= 2).groupby("intl", observed=True).agg(
        n=("team_rd", "size"), team_rd=("team_rd", "mean"),
        player_rd=("player_rd", "mean"), team_rd_joint=("team_rd_joint", "mean"))
    return {
        "by_team_experience": by_exp_t,
        "by_player_experience": by_exp_p,
        "by_season_segment": by_seg,
        "by_international": intl,
    }


# G-阶梯：把 L2/L3 的 Elo 特征原地换成 Glicko，其余口径（标准化、C、fold）完全不动。
G_LADDER: dict[str, list[str]] = {
    "G2": ["side_prior", "team_glicko_diff"],
    "G3": ["side_prior", "team_glicko_diff", "player_glicko_diff"],
    "G3g": ["side_prior", "team_glicko_g_diff", "player_glicko_g_diff"],
    "G3rd": ["side_prior", "team_glicko_diff", "player_glicko_diff",
             "team_rd_joint", "player_rd_max_blue", "player_rd_max_red"],
    "L3rd": ["side_prior", "team_elo_diff", "player_elo_diff",
             "team_rd_joint", "player_rd_max_blue", "player_rd_max_red"],
    "L3G3": ["side_prior", "team_elo_diff", "player_elo_diff",
             "team_glicko_diff", "player_glicko_diff"],
}


def benchmark(
    *,
    periods: Sequence[PeriodKind] = ("week",),
    since: str = "2019-01-01",
    min_tier: int = 1,
) -> dict[str, Any]:
    """与现行 Elo 的同口径对照（PLAN.md §6.1 walk-forward）。

    口径必须与阶段 4 的基线逐字相同，否则 log loss 没法比。做法是**复用**
    `eval.baselines.run_ladder`（同一个标准化器、同一个 C、同一组 fold），
    只是临时往它的 LADDER 里塞几行 Glicko 版本的特征清单，跑完就撤掉。
    这样"差异来自特征"这个结论才站得住——代码路径是同一条。
    """
    from lolwp.eval import baselines, metrics
    from lolwp.eval.walkforward import make_folds, summary
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine

    full = load_dev()
    elo_feats = run_state_machine(full)
    frames: dict[str, pd.DataFrame] = {}
    for per in periods:
        frames[per] = run_glicko_state_machine(full, cfg=Glicko2Config(period=per))

    mask = full.league_tier >= min_tier
    if since:
        mask &= full.kickoff_utc >= pd.Timestamp(since, tz="UTC")
    ev = full[mask].reset_index(drop=True)
    folds = make_folds(ev)

    results: dict[str, Any] = {"folds": summary(ev, folds), "rows": []}
    preds: dict[str, pd.DataFrame] = {}
    saved = dict(baselines.LADDER)
    try:
        for per, gf in frames.items():
            feats = pd.concat([elo_feats, gf], axis=1)[mask].reset_index(drop=True)
            baselines.LADDER.update(G_LADDER)
            names = ["L2", "L3", *G_LADDER]
            p = baselines.run_ladder(ev, feats, folds, names=names)
            preds[per] = p
            for nm, g in p.groupby("model"):
                row = metrics.evaluate(g.y.to_numpy(), g.p.to_numpy())
                row |= {"period": per, "model": nm}
                results["rows"].append(row)
    finally:
        baselines.LADDER.clear()
        baselines.LADDER.update(saved)
    results["preds"] = preds
    results["frames"] = frames
    results["eval_df"] = ev
    return results


def main() -> None:  # pragma: no cover - 手动跑的实验入口
    res = benchmark(periods=("week", "game", "month"))
    tab = pd.DataFrame(res["rows"])[
        ["period", "model", "n", "log_loss", "brier", "auc", "ece", "calib_slope"]
    ]
    print(res["folds"])
    print(tab.round(5).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
