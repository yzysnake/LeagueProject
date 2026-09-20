"""离线定标工具：分层贝叶斯读方差成分（PLAN.md §5.4「离线定标工具」/ §4.3）。

**这个模块不上线。** 它是一把尺子，回答一个纯定量的问题：

    选手层 / 队伍层 / 英雄层 / 选手×英雄交互层，各自的真实效应离散度是多少？

项目里所有 EB 收缩强度 `k = σ²_noise / σ²_effect` 都由这四个数决定，而当前
`draft_priors.py` 里的 `k₁ = 476` 是照抄手册的参考值，**从未在本仓库的数据上估过**。
本模块把它估出来，写进 `configs/shrinkage_constants.yaml`。

=========================== 红线 3（§6.1.2）归属：哪一类 ===========================
手册 §4.1 把 `k` 明确归到「**每个 fold 必须重估**（依赖训练段全局分布）」那一栏，
把在线递归量（Elo/Glicko）归到豁免栏。本模块产出的是 `k`，所以**属于前者**，
理由是：σ 的估计用到了整段数据的**聚合分布**（所有实体的胜率离散度），
拿全 dev 段估出来的 σ 去标定某个 fold 的收缩强度，就是用该 fold 之后的数据调超参。

因此本模块的纪律是：

1. **SVI 分层模型（`fit_svi`）只跑一次，只作为测量与报告**，它的输出**不是**任何
   fold 内被消费的统计量。yaml 里的 `k` 是「默认值 / 先验 / 报告口径」，
   不是"某个 fold 的最优 k"。
2. 真正要在生产里逐 fold 重估的，是**矩估计** `moment_variance_components()`——
   它接受一个 DataFrame，调用方传训练段切片即可，秒级，满足 §4.1 的成本预算。
   本模块还用 `moment_stability_over_origins()` 实测了逐 fold 重估到底会让 k 动多少。
3. SVI 的用途是**交叉验证矩估计**（阶段 6 验收④要求两者「差异 <3 倍」），
   以及给出矩估计给不了的东西：**队内选手偏差** σ_player 与队伍层 σ_team 的分离。
   矩估计只能测到「选手胜率的边际离散度」，而那个量 98% 是他队友的功劳。
4. 这把尺子准不准由 `simulate_design()` / `synthetic_recovery()` 自证：
   在**真实的**出场结构上按已知 σ 造 y，再用同一套 SVI 读回来。
   报告 §5 的偏倚方向（σ_champ 近无偏、σ_player 系统性低估、σ_team 会被吃掉）
   全部来自这里，没有它，主结果里每个数字都无法判断偏在哪边。

=========================== 单位：logit 还是胜率 ===========================
分层模型的参数天然在 **logit** 上；手册的参考值 0.1294 / 0.1330 / 0.0229 在 **胜率**
上（`k = p(1-p)/σ² = 0.25/σ²` 只有在胜率单位下才等于 476）。两者换算用 p=0.5
处的斜率：`σ_winrate ≈ 0.25 · σ_logit`。所有对外报出的 k 一律在胜率单位下算，
report 里两种单位都列，避免下一个人再踩一次。

还有一个更容易错的地方：**各层进 logit 的重数不同**。
本模型按任务书的写法，队伍效应是通过 5 名选手的 nesting 进来的，
所以一侧的 logit 里队伍项的系数是 **5**，而选手偏差/英雄各 5 个独立项相加、
SD 是 **√5·σ**。所以「实体自身的胜率偏移 SD」与「该层对一侧 logit 的总贡献 SD」
是两个数，混用会差 5 倍以上。见 `LevelSummary`。
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")

# 国际赛事：队伍的「赛区」不能取这些 league，否则 region 层会把 Worlds 参赛队
# 全并成一个"赛区"，region_effect 就退化成"是否打过国际赛"。
INTERNATIONAL: frozenset[str] = frozenset(
    {"WLDs", "MSI", "EWC", "ASI", "MSC", "FST", "IEM", "RIFT", "UNKNOWN"}
)

BERNOULLI_VAR_AT_HALF = 0.25  # σ²_noise，单场二元结果在 p=0.5 处的方差
LOGIT_TO_WINRATE = 0.25       # dσ(x)/dx |_{x=0}，把 logit 单位的效应折成胜率单位


# --------------------------------------------------------------------------- #
# 设计矩阵
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VCDesign:
    """分层模型的整数索引设计。全部是 (G,) 或 (G,5) 的 int32。"""

    y: np.ndarray                 # (G,) 蓝方是否获胜
    league: np.ndarray            # (G,) 本场所属赛事（side 先验按它分层）
    blue_team: np.ndarray         # (G,)
    red_team: np.ndarray          # (G,)
    blue_player: np.ndarray       # (G,5)
    red_player: np.ndarray        # (G,5)
    blue_champ: np.ndarray        # (G,5)
    red_champ: np.ndarray         # (G,5)
    blue_pc: np.ndarray           # (G,5) (player, champion) 交互单元
    red_pc: np.ndarray            # (G,5)
    team_region: np.ndarray       # (T,) 每支队伍的归属赛区（国际赛不计）
    n_league: int
    n_region: int
    n_team: int
    n_player: int
    n_champ: int
    n_pc: int
    labels: dict[str, list[str]] = field(default_factory=dict, repr=False)

    @property
    def n_games(self) -> int:
        return int(self.y.shape[0])


def _stack5(col: pd.Series) -> np.ndarray:
    """把 list 列摊成 (G,5) 的 object 数组，保留 None 以便后续判缺。"""
    return np.array([list(v) for v in col], dtype=object)


def _keep_mask(df: pd.DataFrame) -> np.ndarray:
    """缺 playerid / champion / teamid 的整场丢弃（§5.5 的离线口径）。"""
    def _bad(a: np.ndarray) -> np.ndarray:
        return np.array([[x is None or (isinstance(x, float) and math.isnan(x))
                          for x in row] for row in a]).any(axis=1)

    keep = ~(_bad(_stack5(df.blue_player_ids)) | _bad(_stack5(df.red_player_ids))
             | _bad(_stack5(df.blue_champ_keys)) | _bad(_stack5(df.red_champ_keys)))
    keep = keep & df.blue_team_id.notna().to_numpy() & df.red_team_id.notna().to_numpy()
    return np.asarray(keep, dtype=bool)


def build_design(df: pd.DataFrame) -> VCDesign:
    """从 silver 表建索引设计。缺 playerid / champion 的整场丢弃（§5.5 离线口径）。"""
    keep = _keep_mask(df)
    df = df.loc[keep].reset_index(drop=True)
    bp, rp = _stack5(df.blue_player_ids), _stack5(df.red_player_ids)
    bc, rc = _stack5(df.blue_champ_keys), _stack5(df.red_champ_keys)
    def _index(values: Sequence[Any]) -> tuple[dict[Any, int], list[str]]:
        uniq = sorted({str(v) for v in values})
        return {v: i for i, v in enumerate(uniq)}, uniq

    league_map, league_lab = _index(df.league_canon.tolist())
    team_ids = df.blue_team_id.tolist() + df.red_team_id.tolist()
    team_map, team_lab = _index(team_ids)
    player_map, player_lab = _index(
        [str(x) for row in np.concatenate([bp, rp]) for x in row]
    )
    champ_map, champ_lab = _index(
        [str(int(x)) for row in np.concatenate([bc, rc]) for x in row]
    )

    def _pid(a: np.ndarray) -> np.ndarray:
        return np.array([[player_map[str(x)] for x in row] for row in a], np.int32)

    def _cid(a: np.ndarray) -> np.ndarray:
        return np.array([[champ_map[str(int(x))] for x in row] for row in a], np.int32)

    bpi, rpi, bci, rci = _pid(bp), _pid(rp), _cid(bc), _cid(rc)

    # player×champion 单元：只给真实出现过的组合建索引（35k 量级，稀疏）
    pc_keys = np.concatenate(
        [bpi.astype(np.int64) * len(champ_lab) + bci,
         rpi.astype(np.int64) * len(champ_lab) + rci]
    ).ravel()
    uniq_pc = np.unique(pc_keys)
    pc_pos = {int(k): i for i, k in enumerate(uniq_pc)}

    def _pcid(pi: np.ndarray, ci: np.ndarray) -> np.ndarray:
        flat = pi.astype(np.int64) * len(champ_lab) + ci
        return np.vectorize(pc_pos.__getitem__)(flat).astype(np.int32)

    # 队伍 → 赛区：取该队在【非国际赛】里出场最多的 league；只打国际赛的归 INTL
    reg_counter: dict[str, dict[str, int]] = {}
    for tcol in ("blue_team_id", "red_team_id"):
        for t, lg in zip(df[tcol], df.league_canon, strict=True):
            if lg in INTERNATIONAL:
                continue
            reg_counter.setdefault(str(t), {}).setdefault(lg, 0)
            reg_counter[str(t)][lg] += 1
    region_lab = sorted({lg for d in reg_counter.values() for lg in d} | {"INTL"})
    region_map = {r: i for i, r in enumerate(region_lab)}
    team_region = np.full(len(team_lab), region_map["INTL"], np.int32)
    for t, counts in reg_counter.items():
        team_region[team_map[t]] = region_map[max(counts, key=counts.__getitem__)]

    return VCDesign(
        y=df.y_blue_win.to_numpy().astype(np.int32),
        league=np.array([league_map[x] for x in df.league_canon], np.int32),
        blue_team=np.array([team_map[str(x)] for x in df.blue_team_id], np.int32),
        red_team=np.array([team_map[str(x)] for x in df.red_team_id], np.int32),
        blue_player=bpi, red_player=rpi, blue_champ=bci, red_champ=rci,
        blue_pc=_pcid(bpi, bci), red_pc=_pcid(rpi, rci),
        team_region=team_region,
        n_league=len(league_lab), n_region=len(region_lab), n_team=len(team_lab),
        n_player=len(player_lab), n_champ=len(champ_lab), n_pc=len(uniq_pc),
        labels={"league": league_lab, "region": region_lab, "team": team_lab,
                "player": player_lab, "champion": champ_lab},
    )


# --------------------------------------------------------------------------- #
# NumPyro 分层 logistic
# --------------------------------------------------------------------------- #
def hierarchical_model(
    d: VCDesign,
    *,
    use_player: bool = True,
    use_champ: bool = True,
    use_pc: bool = True,
    prior_scale: float = 0.5,
) -> None:
    """y ~ Bernoulli(logit = side + Σ_blue(·) − Σ_red(·))，全部非中心参数化。

    嵌套按任务书：player_effect ~ N(team_effect, σ_player²)，
    team_effect ~ N(region_effect, σ_team²)，champ ~ N(0, σ_champ²)，
    player:champ ~ N(0, σ_pc²)。**因为 5 名选手共享同一个 team_effect，
    队伍项在一侧 logit 里的系数是 5**（见模块 docstring 的单位说明）。

    非中心化不是风格问题：中心化写法下 σ→0 时后验呈漏斗形，
    mean-field SVI 会直接卡死在 σ≈0。
    """
    import numpyro
    import numpyro.distributions as dist

    side_mu = numpyro.sample("side_mu", dist.Normal(0.0, 1.0))
    sigma_side = numpyro.sample("sigma_side", dist.HalfNormal(prior_scale))
    with numpyro.plate("league", d.n_league):
        side_raw = numpyro.sample("side_raw", dist.Normal(0.0, 1.0))
    sigma_region = numpyro.sample("sigma_region", dist.HalfNormal(prior_scale))
    with numpyro.plate("region", d.n_region):
        region_raw = numpyro.sample("region_raw", dist.Normal(0.0, 1.0))
    sigma_team = numpyro.sample("sigma_team", dist.HalfNormal(prior_scale))
    with numpyro.plate("team", d.n_team):
        team_raw = numpyro.sample("team_raw", dist.Normal(0.0, 1.0))
    region = sigma_region * region_raw
    team = region[d.team_region] + sigma_team * team_raw

    logits = side_mu + sigma_side * side_raw[d.league]
    logits = logits + 5.0 * (team[d.blue_team] - team[d.red_team])

    if use_player:
        sigma_player = numpyro.sample("sigma_player", dist.HalfNormal(prior_scale))
        with numpyro.plate("player", d.n_player):
            player_raw = numpyro.sample("player_raw", dist.Normal(0.0, 1.0))
        dev = sigma_player * player_raw
        logits = logits + dev[d.blue_player].sum(-1) - dev[d.red_player].sum(-1)
    if use_champ:
        sigma_champ = numpyro.sample("sigma_champ", dist.HalfNormal(prior_scale))
        with numpyro.plate("champion", d.n_champ):
            champ_raw = numpyro.sample("champ_raw", dist.Normal(0.0, 1.0))
        champ = sigma_champ * champ_raw
        logits = logits + champ[d.blue_champ].sum(-1) - champ[d.red_champ].sum(-1)

    if use_pc:
        sigma_pc = numpyro.sample("sigma_pc", dist.HalfNormal(prior_scale))
        with numpyro.plate("pc", d.n_pc):
            pc_raw = numpyro.sample("pc_raw", dist.Normal(0.0, 1.0))
        pc = sigma_pc * pc_raw
        logits = logits + pc[d.blue_pc].sum(-1) - pc[d.red_pc].sum(-1)

    with numpyro.plate("games", d.n_games):
        numpyro.sample("y", dist.Bernoulli(logits=logits), obs=d.y)


SIGMA_SITES: tuple[str, ...] = (
    "sigma_side", "sigma_region", "sigma_team", "sigma_player",
    "sigma_champ", "sigma_pc",
)


@dataclass
class SVIFit:
    """一次 SVI 的全部产物。`sigmas` 是后验样本，键是 SIGMA_SITES 里出现的那些。"""

    seed: int
    losses: np.ndarray                  # 每步 ELBO 的负值
    sigmas: dict[str, np.ndarray]       # site -> (n_draw,) 后验样本（logit 单位）
    effects: dict[str, np.ndarray]      # site -> 后验均值（keep_effects=True 时才填）
    n_games: int
    use_pc: bool
    prior_scale: float
    steps: int
    lr: float
    wall_seconds: float

    def summary(self) -> pd.DataFrame:
        rows = []
        for k, v in self.sigmas.items():
            lo, med, hi = np.percentile(v, [2.5, 50.0, 97.5])
            rows.append({"site": k, "mean": float(v.mean()), "median": float(med),
                         "lo95": float(lo), "hi95": float(hi)})
        return pd.DataFrame(rows)

    def elbo_diagnostics(self, tail_frac: float = 0.1) -> dict[str, float]:
        """收敛诊断：尾段 ELBO 的相对漂移。|drift| 远小于尾段抖动才算收敛。"""
        n = len(self.losses)
        tail = self.losses[int(n * (1 - tail_frac)):]
        half = len(tail) // 2
        drift = float(tail[:half].mean() - tail[half:].mean())
        return {
            "final_loss": float(self.losses[-1]),
            "tail_mean": float(tail.mean()),
            "tail_std": float(tail.std()),
            "tail_drift": drift,
            "tail_drift_over_std": float(drift / (tail.std() + 1e-12)),
        }


def fit_svi(
    d: VCDesign,
    *,
    seed: int = 0,
    steps: int = 20_000,
    lr: float = 0.02,
    use_player: bool = True,
    use_champ: bool = True,
    use_pc: bool = True,
    prior_scale: float = 0.5,
    n_draw: int = 4000,
    keep_effects: bool = False,
    progress: bool = False,
) -> SVIFit:
    """AutoNormal + Adam。手册 §5.4 明确要 SVI 不要 NUTS（3–5k 随机效应 NUTS 跑不动）。

    mean-field 会低估后验相关，进而**低估**层级尺度的后验宽度——所以本函数的 95% CI
    只能当"下界宽度"读，真正的稳健性由「多种子 + 矩估计交叉验证」提供。
    """
    import time

    import jax
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.infer.initialization import init_to_median

    guide = AutoNormal(hierarchical_model, init_loc_fn=init_to_median)
    svi = SVI(hierarchical_model, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    t0 = time.time()
    res = svi.run(
        jax.random.PRNGKey(seed), steps, d,
        use_player=use_player, use_champ=use_champ, use_pc=use_pc,
        prior_scale=prior_scale, progress_bar=progress,
    )
    post = guide.sample_posterior(
        jax.random.PRNGKey(seed + 10_000), res.params, sample_shape=(n_draw,)
    )
    sigmas = {k: np.asarray(post[k]).reshape(-1) for k in SIGMA_SITES if k in post}
    effects = (
        {k: np.asarray(v).mean(axis=0) for k, v in post.items()} if keep_effects else {}
    )
    return SVIFit(
        seed=seed, losses=np.asarray(res.losses), sigmas=sigmas, effects=effects,
        n_games=d.n_games, use_pc=use_pc, prior_scale=prior_scale,
        steps=steps, lr=lr, wall_seconds=time.time() - t0,
    )


# --------------------------------------------------------------------------- #
# 独立矩估计（验收④要求的交叉验证；也是生产里唯一该逐 fold 重估的那个）
# --------------------------------------------------------------------------- #
def dl_variance(n: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """beta-binomial 的 DerSimonian–Laird / Kleinman 矩估计。返回 (σ², p̄)。

    E[Σ n_e(p̂_e−p̄)²] = (E−1)·p̄(1−p̄) + σ²·(N − Σn_e²/N)，解出 σ²。
    **必须扣掉二项噪声那一项**：不扣的话 n_e 中位数只有 2–3 的英雄/选手层
    会给出大得离谱的"效应 SD"，那全是抽样噪声。σ² 可能为负，如实返回。
    """
    n = np.asarray(n, float)
    w = np.asarray(w, float)
    total = n.sum()
    if total <= 0 or len(n) < 2:
        return float("nan"), float("nan")
    p_bar = w.sum() / total
    p_hat = w / n
    s = float((n * (p_hat - p_bar) ** 2).sum())
    noise = (len(n) - 1) * p_bar * (1.0 - p_bar)
    denom = total - (n**2).sum() / total
    return (s - noise) / denom, float(p_bar)


@dataclass
class MomentEstimate:
    level: str
    n_units: int
    n_obs: int
    median_obs_per_unit: float
    var: float                  # 可能为负，不截断
    sigma: float                # sqrt(max(var,0))，胜率单位
    ci: tuple[float, float]     # 实体级 cluster bootstrap 的 σ 95% CI
    k: float                    # 0.25 / σ²
    debug: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {"level": self.level, "n_units": self.n_units, "n_obs": self.n_obs,
                "median_obs": self.median_obs_per_unit, "sigma_wr": self.sigma,
                "sigma_lo95": self.ci[0], "sigma_hi95": self.ci[1], "k": self.k}


def _bootstrap_sigma(
    n: np.ndarray, w: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    if n_boot <= 0:                      # n_boot=0 用于"只要点估计"的批量扫描
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    e = len(n)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, e, e)
        v, _ = dl_variance(n[idx], w[idx])
        out[b] = math.sqrt(max(v, 0.0))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _unit_counts(keys: np.ndarray, wins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    w = np.bincount(inv, weights=wins, minlength=len(uniq))
    return n, w


def _mk(level: str, n: np.ndarray, w: np.ndarray, *, n_boot: int,
        seed: int) -> MomentEstimate:
    var, _ = dl_variance(n, w)
    sigma = math.sqrt(max(var, 0.0))
    return MomentEstimate(
        level=level, n_units=len(n), n_obs=int(n.sum()),
        median_obs_per_unit=float(np.median(n)), var=float(var), sigma=sigma,
        ci=_bootstrap_sigma(n, w, n_boot=n_boot, seed=seed),
        k=(BERNOULLI_VAR_AT_HALF / var) if var > 0 else float("inf"),
    )


def moment_variance_components(
    df: pd.DataFrame, *, n_boot: int = 400, seed: int = 0
) -> dict[str, MomentEstimate]:
    """胜率单位的矩估计，与手册 0.1294 / 0.1330 / 0.0229 同口径。

    **这是生产里该逐 fold 重估的那个函数**（§4.1：`k` 属于"每个 fold 必须重估"）。
    调用方传训练段切片即可，23k 场上耗时亚秒级（bootstrap 除外）。

    口径说明：
    - team / player：一场算一条观测，y 是该队/该选手所在方是否获胜。
    - champion：一次 pick 算一条观测，y 是选它那一方是否获胜。
      所以同一场会给两个不同英雄各记一条（镜像 pick 时是同一个英雄记两条）。
    - champion×position：位置由 `blue_champ_keys` 的下标给出（ROLES 顺序）。
    """
    rows_team: list[tuple[str, int]] = []
    rows_player: list[tuple[str, int]] = []
    rows_champ: list[tuple[str, int]] = []
    rows_cpos: list[tuple[str, int]] = []
    for side, opp in (("blue", "red"), ("red", "blue")):
        _ = opp
        y = df.y_blue_win.to_numpy() if side == "blue" else 1 - df.y_blue_win.to_numpy()
        rows_team += list(zip(df[f"{side}_team_id"].astype(str), y, strict=True))
        for pids, cks, yy in zip(
            df[f"{side}_player_ids"], df[f"{side}_champ_keys"], y, strict=True
        ):
            for pos, (p, c) in enumerate(zip(pids, cks, strict=True)):
                if p is not None:
                    rows_player.append((str(p), int(yy)))
                if c is not None:
                    rows_champ.append((str(int(c)), int(yy)))
                    rows_cpos.append((f"{int(c)}|{ROLES[pos]}", int(yy)))

    out: dict[str, MomentEstimate] = {}
    for name, rows in (("team", rows_team), ("player", rows_player),
                       ("champion", rows_champ), ("champion:position", rows_cpos)):
        keys = np.array([r[0] for r in rows])
        wins = np.array([r[1] for r in rows], float)
        n, w = _unit_counts(keys, wins)
        out[name] = _mk(name, n, w, n_boot=n_boot, seed=seed)
    out["player:champion"] = player_champ_residual_moment(
        df, n_boot=n_boot, seed=seed
    )
    return out


def player_champ_residual_moment(
    df: pd.DataFrame, *, n_boot: int = 400, seed: int = 0, n_perm: int = 20
) -> MomentEstimate:
    """选手×英雄**残差**的矩估计，并用置换零分布扣掉手册 §4.3 点名的那个正偏。

    步骤：
    1. 基准取 **player-year** 胜率（去掉跨年状态漂移，§4.3 的做法）。
    2. 残差 r = y − p̂_{player-year}，按 (player, champion) 单元聚合。
    3. 把 champion 标签**在 player-year 内置换**重跑，得到零分布均值 var₀，
       从观测 var 里扣掉。不扣的话会高估 —— 手册实测 var₀ ≈ 0.0002–0.0009。
    """
    recs: list[tuple[str, str, str, int]] = []
    for side in ("blue", "red"):
        y = df.y_blue_win.to_numpy() if side == "blue" else 1 - df.y_blue_win.to_numpy()
        for pids, cks, yr, yy in zip(
            df[f"{side}_player_ids"], df[f"{side}_champ_keys"],
            df.season_year.astype(str), y, strict=True
        ):
            for p, c in zip(pids, cks, strict=True):
                if p is not None and c is not None:
                    recs.append((f"{p}|{yr}", str(int(c)), str(p), int(yy)))
    py = np.array([r[0] for r in recs])
    ch = np.array([r[1] for r in recs])
    pl = np.array([r[2] for r in recs])
    y = np.array([r[3] for r in recs], float)

    uniq_py, inv_py = np.unique(py, return_inverse=True)
    base = np.bincount(inv_py, weights=y, minlength=len(uniq_py)) / np.bincount(
        inv_py, minlength=len(uniq_py)
    )
    resid = y - base[inv_py]

    def _cell_var(champ_labels: np.ndarray) -> float:
        cell = np.char.add(np.char.add(pl, "|"), champ_labels)
        uniq, inv = np.unique(cell, return_inverse=True)
        n = np.bincount(inv, minlength=len(uniq)).astype(float)
        s = np.bincount(inv, weights=resid, minlength=len(uniq))
        m = s / n
        within = float(((resid - m[inv]) ** 2).sum()) / max(len(resid) - len(uniq), 1)
        total = float((n * (m - resid.mean()) ** 2).sum())
        noise = (len(uniq) - 1) * within
        denom = n.sum() - (n**2).sum() / n.sum()
        return (total - noise) / denom

    obs_var = _cell_var(ch)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    order = np.argsort(inv_py, kind="stable")
    bounds = np.searchsorted(inv_py[order], np.arange(len(uniq_py) + 1))
    for b in range(n_perm):
        perm = ch.copy()
        for i in range(len(uniq_py)):
            sl = order[bounds[i]:bounds[i + 1]]
            perm[sl] = ch[rng.permutation(sl)]
        null[b] = _cell_var(perm)
    var = obs_var - float(null.mean())
    sigma = math.sqrt(max(var, 0.0))

    # bootstrap：对 player 重抽（cluster = 选手）。
    # ⚠️ 被抽中两次的选手**必须当成两个不同的选手**——否则他的两份拷贝会并进同一个
    #    (player,champ) 格子，格内方差不变而 n 翻倍，between 项被凭空放大，
    #    点估计 0 也能 bootstrap 出 [0.11,0.12] 的 CI（第一版就是这么错的）。
    uniq_pl = np.unique(pl)
    idx_by_pl = {p: np.flatnonzero(pl == p) for p in uniq_pl}
    boots = np.empty(max(min(n_boot, 200), 0))
    for b in range(len(boots)):
        pick = rng.choice(uniq_pl, len(uniq_pl), replace=True)
        sel = np.concatenate([idx_by_pl[p] for p in pick])
        rep = np.concatenate([
            np.full(len(idx_by_pl[p]), j) for j, p in enumerate(pick)
        ]).astype(str)
        cell = np.char.add(np.char.add(rep, "|"), ch[sel])
        uq, iv = np.unique(cell, return_inverse=True)
        n = np.bincount(iv, minlength=len(uq)).astype(float)
        ssum = np.bincount(iv, weights=resid[sel], minlength=len(uq))
        m = ssum / n
        within = float(((resid[sel] - m[iv]) ** 2).sum()) / max(len(sel) - len(uq), 1)
        total = float((n * (m - resid[sel].mean()) ** 2).sum())
        v = (total - (len(uq) - 1) * within) / (n.sum() - (n**2).sum() / n.sum())
        boots[b] = math.sqrt(max(v - float(null.mean()), 0.0))

    cells, cell_inv = np.unique(
        np.char.add(np.char.add(pl, "|"), ch), return_inverse=True
    )
    est = MomentEstimate(
        level="player:champion", n_units=len(cells), n_obs=len(resid),
        median_obs_per_unit=float(np.median(np.bincount(cell_inv))),
        var=float(var), sigma=sigma,
        ci=((float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            if len(boots) else (float("nan"), float("nan"))),
        k=(BERNOULLI_VAR_AT_HALF / var) if var > 0 else float("inf"),
    )
    est.debug = {"obs_var": float(obs_var), "null_var_mean": float(null.mean()),
                 "null_var_sd": float(null.std()),
                 "sigma_before_null": math.sqrt(max(obs_var, 0.0))}
    return est


# --------------------------------------------------------------------------- #
# logit ↔ 胜率、k 换算、与 SVI 的对照
# --------------------------------------------------------------------------- #
@dataclass
class LevelSummary:
    """一层的完整读数。三个 σ 是三个不同的量，别混。"""

    level: str
    sigma_logit: tuple[float, float, float]      # (lo95, med, hi95) 单个实体，logit
    entity_winrate: tuple[float, float, float]   # 单个实体自身的胜率偏移 SD
    side_logit_sd: float                         # 该层对一侧 logit 的总贡献 SD
    k: tuple[float, float, float]                # 由 entity_winrate 换算，(hi..lo 反序)


def shrinkage_k(sigma_winrate: float) -> float:
    """k = σ²_noise/σ²_effect，σ_noise² = p(1−p) = 0.25。σ→0 时 k→∞。"""
    if sigma_winrate <= 0:
        return float("inf")
    return BERNOULLI_VAR_AT_HALF / sigma_winrate**2


# 每层在一侧 logit 里的重数：队伍是 5 名选手共享同一个效应 → 系数 5；
# 选手偏差 / 英雄 / 选手×英雄是 5 个独立同分布项相加 → SD 乘 √5。
_SIDE_MULTIPLIER: dict[str, float] = {
    "sigma_team": 5.0,
    "sigma_region": 5.0,
    "sigma_player": math.sqrt(5.0),
    "sigma_champ": math.sqrt(5.0),
    "sigma_pc": math.sqrt(5.0),
    "sigma_side": 1.0,
}
_ENTITY_LOGIT_MULTIPLIER: dict[str, float] = {
    # 「这个实体自身对本场 logit 的贡献」的尺度：队伍是 5×（它决定 5 个槽位）
    "sigma_team": 5.0,
    "sigma_region": 5.0,
    "sigma_player": 1.0,
    "sigma_champ": 1.0,
    "sigma_pc": 1.0,
    "sigma_side": 1.0,
}


def pool_sigmas(fits: Sequence[SVIFit]) -> dict[str, np.ndarray]:
    """把多个种子的后验样本拼起来。

    mean-field 的单次 CI 只反映"给定这个局部最优"的不确定度，**不含 VI 自身的
    随机性**。把种子拼起来报出的 CI 才同时含这两部分，是更诚实的区间。
    """
    keys = [k for k in SIGMA_SITES if all(k in f.sigmas for f in fits)]
    return {k: np.concatenate([f.sigmas[k] for f in fits]) for k in keys}


def summarize_levels(sigmas: dict[str, np.ndarray]) -> list[LevelSummary]:
    out = []
    for site, s in sigmas.items():
        lo, med, hi = (float(x) for x in np.percentile(s, [2.5, 50.0, 97.5]))
        em = _ENTITY_LOGIT_MULTIPLIER[site]
        wr = tuple(LOGIT_TO_WINRATE * em * x for x in (lo, med, hi))
        out.append(LevelSummary(
            level=site, sigma_logit=(lo, med, hi),
            entity_winrate=wr,  # type: ignore[arg-type]
            side_logit_sd=_SIDE_MULTIPLIER[site] * med,
            k=(shrinkage_k(wr[2]), shrinkage_k(wr[1]), shrinkage_k(wr[0])),
        ))
    return out


def team_marginal_winrate_sd(sigmas: dict[str, np.ndarray]) -> np.ndarray:
    """模型隐含的「队伍边际胜率 SD」——用来和矩估计的 team/player 层直接对表。

    一支队面对平均对手时的 logit = 5·t + Σ_{5 人} dev，
    SD = √(25σ_team² + 5σ_player²)，再乘 0.25 折成胜率。
    矩估计的 team 层和 player 层测的都是这个量（选手的胜率就是他队伍的胜率），
    这解释了手册里 player 0.1294 与 team 0.1330 为什么几乎一样。
    """
    st = sigmas["sigma_team"]
    sp = sigmas["sigma_player"]
    return LOGIT_TO_WINRATE * np.sqrt(25.0 * st**2 + 5.0 * sp**2)


def moment_stability_over_origins(
    df: pd.DataFrame, origins: Sequence[str], *, n_boot: int = 0
) -> pd.DataFrame:
    """把矩估计沿扩张训练窗重算，实测「逐 fold 重估」到底会让 k 动多少。

    这是 §6.1.2 红线 3 的经验依据：如果 k 在各 origin 间基本不动，
    那么"全段估一次"与"逐 fold 重估"的差别可以量化，而不是靠嘴说。
    """
    rows = []
    for o in origins:
        sub = df[df.kickoff_utc < pd.Timestamp(o, tz="UTC")]
        if len(sub) < 500:
            continue
        est = moment_variance_components(sub, n_boot=n_boot)
        row: dict[str, Any] = {"origin": o, "n_games": len(sub)}
        for lvl, e in est.items():
            row[f"sigma_{lvl}"] = e.sigma
            row[f"k_{lvl}"] = e.k
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 时间外推检验：测出来的 σ_player 到底是信号还是过拟合
# --------------------------------------------------------------------------- #
def subset_design(d: VCDesign, mask: np.ndarray) -> VCDesign:
    """按行掩码切子设计。**实体索引空间保持不变**（不重新编号），

    这样训练段拟合出的 effects 可以直接下标到测试段——测试段里训练段没见过的实体
    自然落到后验 ≈ 先验 ≈ 0，就是 §5.5 要求的 fallback 行为。
    """
    m = np.asarray(mask, bool)
    return VCDesign(
        y=d.y[m], league=d.league[m],
        blue_team=d.blue_team[m], red_team=d.red_team[m],
        blue_player=d.blue_player[m], red_player=d.red_player[m],
        blue_champ=d.blue_champ[m], red_champ=d.red_champ[m],
        blue_pc=d.blue_pc[m], red_pc=d.red_pc[m],
        team_region=d.team_region,
        n_league=d.n_league, n_region=d.n_region, n_team=d.n_team,
        n_player=d.n_player, n_champ=d.n_champ, n_pc=d.n_pc, labels=d.labels,
    )


def predict_logits(effects: dict[str, np.ndarray], d: VCDesign) -> np.ndarray:
    """用后验均值算 logit。mean-field guide 下 E[σ·raw] = E[σ]·E[raw]，所以可以逐项相乘。"""
    def g(k: str) -> np.ndarray:
        return np.asarray(effects[k])

    region = g("sigma_region") * g("region_raw")
    team = region[d.team_region] + g("sigma_team") * g("team_raw")
    out = g("side_mu") + g("sigma_side") * g("side_raw")[d.league]
    out = out + 5.0 * (team[d.blue_team] - team[d.red_team])
    if "player_raw" in effects:
        dev = g("sigma_player") * g("player_raw")
        out = out + dev[d.blue_player].sum(-1) - dev[d.red_player].sum(-1)
    if "champ_raw" in effects:
        ch = g("sigma_champ") * g("champ_raw")
        out = out + ch[d.blue_champ].sum(-1) - ch[d.red_champ].sum(-1)
    if "pc_raw" in effects:
        pc = g("sigma_pc") * g("pc_raw")
        out = out + pc[d.blue_pc].sum(-1) - pc[d.red_pc].sum(-1)
    return np.asarray(out, float)


def temporal_predictive_check(
    df: pd.DataFrame,
    *,
    split: str,
    steps: int = 30_000,
    lr: float = 0.02,
    seed: int = 0,
    variants: Sequence[tuple[str, bool, bool, bool]] = (
        ("team only", False, False, False),
        ("+player", True, False, False),
        ("+player+champ", True, True, False),
        ("+player+champ+pc", True, True, True),
    ),
) -> pd.DataFrame:
    """在 `split` 之前拟合、之后评估。**这是判断 σ_player 是不是过拟合的唯一硬证据。**

    分层模型里一个层的 σ 大，只说明"这一层的随机效应吃掉了很多训练集方差"，
    不说明它能外推——2,029 个选手 vs 240 支队伍，选手层的自由度是队伍层的 8.5 倍，
    完全可能只是在记住训练集。所以必须做一次严格向前的时间外推。

    注意这是**静态**效应（无时间衰减），外推到一年后必然劣于在线 Elo；
    这里只看**层与层之间的相对差**，不与 L2/L3 的绝对数字对比。
    """
    d = build_design(df)
    # build_design 会丢掉缺 id 的场次，行掩码必须用同一套过滤后再切
    before = (df.kickoff_utc < pd.Timestamp(split, tz="UTC")).to_numpy()
    tr = before[_keep_mask(df)]
    d_tr, d_te = subset_design(d, tr), subset_design(d, ~tr)
    rows = []
    for name, up, uc, upc in variants:
        fit = fit_svi(d_tr, seed=seed, steps=steps, lr=lr, use_player=up,
                      use_champ=uc, use_pc=upc, keep_effects=True, n_draw=600)
        lg = predict_logits(fit.effects, d_te)
        p = 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))
        y = d_te.y
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        rows.append({"variant": name, "n_train": d_tr.n_games, "n_test": d_te.n_games,
                     "oos_logloss": ll, "oos_brier": float(((p - y) ** 2).mean()),
                     "sigma_player": float(np.median(fit.sigmas["sigma_player"]))
                     if "sigma_player" in fit.sigmas else float("nan"),
                     "wall_s": fit.wall_seconds})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 跨种子稳定性：手册验收⑧那条 max|Δlogit| < 1e-3 到底过不过
# --------------------------------------------------------------------------- #
def cross_seed_logit_stability(
    d: VCDesign,
    *,
    seeds: Sequence[int] = (0, 1),
    steps: int = 30_000,
    lr: float = 0.02,
    n_draw: int = 600,
) -> dict[str, float]:
    """两个种子各拟合一次，比对**逐场 logit**，不是比 σ。

    为什么必须单独测这个：σ 的跨种子 CV 只有 1–3%，看上去很稳；但 σ 稳不等于
    per-entity 的 rating 稳。本设计有 3.8 万个随机效应，其中大量个体弱可识别
    （一个只打过 3 场的选手，他的效应和队友的效应换来换去似然几乎不变），
    mean-field SVI 会停在不同的局部最优。**只报 σ 会把这件事完全盖住。**

    手册验收⑧给 BT/SVI 这一支定的是 T2 档 B「固定 seed bit-identical 且跨 seed
    `max|Δlogit| < 1e-3`」。本函数产出的就是那条门禁的实测值，**报告 §3.3
    判它不过，靠的是这里的数字**。对本模块无所谓（它只对外报 σ），
    但谁要把 SVI/BT 的 per-entity rating 送进生产，得先看这个数。
    """
    fits = [
        fit_svi(d, seed=s, steps=steps, lr=lr, keep_effects=True, n_draw=n_draw)
        for s in seeds
    ]
    lg = [predict_logits(f.effects, d) for f in fits]
    dl = np.abs(lg[0] - lg[1])
    prob = [1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))) for x in lg]
    dp = np.abs(prob[0] - prob[1])
    return {
        "n_games": float(d.n_games),
        "max_abs_dlogit": float(dl.max()),
        "p99_abs_dlogit": float(np.percentile(dl, 99)),
        "median_abs_dlogit": float(np.median(dl)),
        "corr_logit": float(np.corrcoef(lg[0], lg[1])[0, 1]),
        "max_abs_dp": float(dp.max()),
        "mean_abs_dp": float(dp.mean()),
        "gate_1e_3_passed": float(dl.max() < 1e-3),
    }


# --------------------------------------------------------------------------- #
# 合成恢复检验：这把尺子本身准不准
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticTruth:
    """合成数据的真参数（logit 单位）。默认值就是报告 §5 那张表的固定背景。"""

    sigma_player: float
    sigma_team: float = 0.06
    sigma_pc: float = 0.0
    sigma_champ: float = 0.07
    sigma_region: float = 0.12
    sigma_side: float = 0.05
    side_mu: float = 0.10


def simulate_design(d: VCDesign, truth: SyntheticTruth, *, seed: int) -> VCDesign:
    """保留真实索引结构（同一批场次、同一张出场表），只按已知 σ 重新生成 y。

    **必须用真实的索引结构，不能随机造图**：σ_player 与 σ_team 能不能分开，
    完全取决于「谁和谁同队、每个实体打了多少场」这张二部图的共线程度，
    而那是这份数据自己的性质。换成随机图就测不出真实的塌陷风险（见 §5 / §7.1：
    真 σ_player=0.30 时 σ_team 被吃到 0.0016，全 dev 口径上这个塌陷真的发生了）。

    抽样顺序（region→team→player→champ→pc→side→y）与 y 的伯努利抽取共用同一个
    `default_rng(1000+seed)`，改动顺序会改变全部数字，报告里的表就对不上了。
    """
    rng = np.random.default_rng(1000 + seed)
    region = rng.normal(0.0, truth.sigma_region, d.n_region)
    team = region[d.team_region] + rng.normal(0.0, truth.sigma_team, d.n_team)
    dev = rng.normal(0.0, truth.sigma_player, d.n_player)
    champ = rng.normal(0.0, truth.sigma_champ, d.n_champ)
    pc = (rng.normal(0.0, truth.sigma_pc, d.n_pc) if truth.sigma_pc > 0
          else np.zeros(d.n_pc))
    side = truth.side_mu + rng.normal(0.0, truth.sigma_side, d.n_league)
    lg = side[d.league] + 5.0 * (team[d.blue_team] - team[d.red_team])
    lg = lg + dev[d.blue_player].sum(-1) - dev[d.red_player].sum(-1)
    lg = lg + champ[d.blue_champ].sum(-1) - champ[d.red_champ].sum(-1)
    lg = lg + pc[d.blue_pc].sum(-1) - pc[d.red_pc].sum(-1)
    p = 1.0 / (1.0 + np.exp(-lg))
    return replace(d, y=(rng.random(d.n_games) < p).astype(np.int32))


def synthetic_recovery(
    d: VCDesign,
    truths: Sequence[tuple[SyntheticTruth, int]],
    *,
    steps: int = 30_000,
    lr: float = 0.02,
) -> pd.DataFrame:
    """对每个 (真参数, 种子) 生成一份合成 y 再跑同一套 SVI，看能不能读回来。

    **报告里最重要的一段就靠这个函数**：没有它，主结果里每个 σ 的偏倚方向都无从判断。
    `(truth, seed)` 里的 seed 同时用作生成种子（`1000+seed`）与 SVI 种子，
    这样一行表 = 一个可复现的整数。
    """
    rows: list[dict[str, Any]] = []
    for truth, seed in truths:
        ds = simulate_design(d, truth, seed=seed)
        f = fit_svi(ds, seed=seed, steps=steps, lr=lr, use_pc=True)
        s = f.summary().set_index("site")
        row: dict[str, Any] = {
            "true_player": truth.sigma_player, "true_team": truth.sigma_team,
            "true_champ": truth.sigma_champ, "true_pc": truth.sigma_pc,
            "base_rate": float(ds.y.mean()), "seed": seed,
            "wall_s": round(f.wall_seconds, 1),
        }
        for site, tag in (("sigma_player", "player"), ("sigma_team", "team"),
                          ("sigma_champ", "champ"), ("sigma_pc", "pc")):
            row[f"est_{tag}"] = float(s.loc[site, "median"])
            row[f"lo95_{tag}"] = float(s.loc[site, "lo95"])
            row[f"hi95_{tag}"] = float(s.loc[site, "hi95"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# yaml 落盘
# --------------------------------------------------------------------------- #
def _plain(obj: Any) -> Any:
    """递归把 numpy 标量换成 python 标量。

    `yaml.safe_dump` 见到 `np.float64` 直接抛 RepresenterError，
    而 `round(np.float64)` 返回的还是 `np.float64`——这个坑必须在出口统一堵死。
    """
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_shrinkage_yaml(
    path: str,
    *,
    sigmas: dict[str, np.ndarray],
    moments: dict[str, MomentEstimate],
    meta: dict[str, Any],
) -> str:
    """写 configs/shrinkage_constants.yaml。

    **落盘的 k 取矩估计**，不取 SVI：SVI 是 mean-field 近似、且只跑一次全段，
    而矩估计能在每个 fold 内秒级重估（§4.1 的要求）。SVI 的数字同时写进文件，
    作为"两条独立路线是否同量级"的存档证据（验收④）。
    """
    import yaml

    levels = {s.level: s for s in summarize_levels(sigmas)}
    site_of = {"player": "sigma_player", "team": "sigma_team",
               "champion": "sigma_champ", "player:champion": "sigma_pc"}
    doc: dict[str, Any] = {
        "_generated_by": "lolwp.models.variance_components",
        "_discipline": (
            "k 属于 PLAN §4.1「每个 fold 必须重估」那一类。本文件给的是默认值/报告口径；"
            "生产里逐 fold 调用 moment_variance_components(train_slice) 重估。"
        ),
        "_units": "sigma_* 为胜率单位；k = 0.25 / sigma^2",
        "meta": meta,
        "moment": {}, "svi_logit": {}, "k": {},
    }
    for lvl, e in moments.items():
        doc["moment"][lvl] = {
            "sigma_winrate": round(e.sigma, 6),
            "ci95": [round(e.ci[0], 6), round(e.ci[1], 6)],
            "k": (None if math.isinf(e.k) else round(e.k, 1)),
            "n_units": e.n_units, "n_obs": e.n_obs,
            "median_obs_per_unit": e.median_obs_per_unit,
        }
        doc["k"][lvl] = None if math.isinf(e.k) else round(float(e.k))
    for lvl, site in site_of.items():
        if site not in levels:
            continue
        s = levels[site]
        doc["svi_logit"][lvl] = {
            "sigma_logit_median": round(s.sigma_logit[1], 6),
            "ci95_logit": [round(s.sigma_logit[0], 6), round(s.sigma_logit[2], 6)],
            "entity_sigma_winrate": round(s.entity_winrate[1], 6),
            "entity_ci95_winrate": [round(s.entity_winrate[0], 6),
                                    round(s.entity_winrate[2], 6)],
            "side_logit_sd": round(s.side_logit_sd, 6),
            "k_from_svi": (None if math.isinf(s.k[1]) else round(s.k[1], 1)),
        }
    # 只有分层模型能给的那一层：**队内选手偏差**。矩估计给不了——
    # 一个选手的胜率就是他队伍的胜率，矩估计测到的 player 层 98% 是队友的功劳。
    if "sigma_player" in levels:
        s = levels["sigma_player"]
        doc["k"]["player_within_team"] = (
            None if math.isinf(s.k[1]) else round(float(s.k[1]))
        )
        doc["svi_only"] = {
            "player_within_team": {
                "sigma_logit_median": round(s.sigma_logit[1], 6),
                "ci95_logit": [round(s.sigma_logit[0], 6), round(s.sigma_logit[2], 6)],
                "sigma_winrate": round(s.entity_winrate[1], 6),
                "k": None if math.isinf(s.k[1]) else round(s.k[1], 1),
                "_note": "矩估计测不到这一层；只有分层模型能把队伍与选手分开",
            }
        }
    # 对表：模型隐含的队伍边际胜率 SD vs 矩估计的 team 层（验收④的"同量级"判据）
    if {"sigma_team", "sigma_player"} <= set(sigmas):
        agg = team_marginal_winrate_sd(sigmas)
        lo, med, hi = (float(x) for x in np.percentile(agg, [2.5, 50.0, 97.5]))
        mom = moments.get("team")
        doc["crosscheck_team_marginal"] = {
            "svi_implied_sigma_winrate": round(med, 6),
            "svi_ci95": [round(lo, 6), round(hi, 6)],
            "moment_sigma_winrate": None if mom is None else round(mom.sigma, 6),
            "ratio_svi_over_moment": (
                None if mom is None or mom.sigma <= 0 else round(med / mom.sigma, 3)
            ),
        }
    # 先序列化到字符串再落盘：`yaml.safe_dump(fh)` 直接写文件的话，一旦中途抛
    # RepresenterError（np.float64 就会），文件已经被 "w" 截成 0 字节 —— 本仓库
    # 的 configs/shrinkage_constants.yaml 就这么空了一次，从 traceback 看不出来。
    text = yaml.safe_dump(_plain(doc), allow_unicode=True, sort_keys=False, width=100)
    with open(path, "w") as fh:
        fh.write(text)
    return json.dumps({"written": path, "bytes": len(text)}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# CLI：一次跑完，产出 yaml + ELBO 曲线 csv
# --------------------------------------------------------------------------- #
DEFAULT_MIN_TIER = 1
DEFAULT_SINCE = "2019-01-01"
YAML_PATH = "configs/shrinkage_constants.yaml"
ELBO_CSV = "reports/stage6_variance_components_elbo.csv"


def main(
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 30_000,
    lr: float = 0.02,
    min_tier: int = DEFAULT_MIN_TIER,
    since: str = DEFAULT_SINCE,
    n_boot: int = 300,
    yaml_path: str = YAML_PATH,
    elbo_csv: str = ELBO_CSV,
) -> dict[str, Any]:
    """跑一次完整定标。**只用 load_dev()，holdout 物理不可达。**"""
    from lolwp.features.loader import load_dev

    df = load_dev(min_tier=min_tier, since=since)
    d = build_design(df)
    fits = [fit_svi(d, seed=s, steps=steps, lr=lr) for s in seeds]
    sig = pool_sigmas(fits)
    moments = moment_variance_components(df, n_boot=n_boot, seed=7)

    step_idx = np.arange(0, steps, max(steps // 600, 1))
    elbo = pd.DataFrame({"step": step_idx})
    for f in fits:
        elbo[f"neg_elbo_seed{f.seed}"] = f.losses[step_idx]
    elbo.to_csv(elbo_csv, index=False)

    meta = {
        "dataset": "data/silver/games.parquet",
        "filter": f"load_dev(min_tier={min_tier}, since={since!r})  # holdout 物理排除",
        "holdout_start": str(df.attrs.get("holdout_start")),
        "n_games_loaded": len(df),
        "n_games_modelled": d.n_games,
        "n_player": d.n_player, "n_team": d.n_team, "n_region": d.n_region,
        "n_champion": d.n_champ, "n_player_champion_cells": d.n_pc,
        "svi": {"guide": "AutoNormal", "optimizer": f"Adam(lr={lr})",
                "steps": steps, "seeds": list(seeds),
                "wall_seconds_total": round(sum(f.wall_seconds for f in fits), 1)},
    }
    write_shrinkage_yaml(yaml_path, sigmas=sig, moments=moments, meta=meta)
    return {"sigmas": sig, "moments": moments, "fits": fits, "meta": meta,
            "design": d}


if __name__ == "__main__":  # pragma: no cover
    out = main()
    print(pd.DataFrame(
        [s.__dict__ for s in summarize_levels(out["sigmas"])]).to_string(index=False))
    print(pd.DataFrame(
        [e.as_row() for e in out["moments"].values()]).to_string(index=False))
