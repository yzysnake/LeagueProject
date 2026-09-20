"""阶段 8 对照模型：CatBoost、zero-init residual DeepSet、等权 logit 平均集成。

PLAN.md §5.4 的 E1 / E2。这个模块的目的**不是**找到更好的模型，而是给
「为什么主力是分组 L2 logistic」提供可被别人复现的反驳材料：

- **E1 CatBoost**：同样的已工程化标量特征，换一个函数族（树的加性组合 + 交互）。
  如果 GBDT 赢不了线性，说明 BT4 那几个特征与 logit 的关系确实近似线性，
  剩下的不是"模型不够强"而是"信息不够多"。手册自己的诚实预期就是测不出差别。
- **E2 zero-init residual DeepSet**：对 10 个 (选手, 英雄, 位置) token 做置换不变池化，
  输出乘一个 **初始化为 0** 的标量 α 加到 BT4 的 logit 上。α=0 时它逐位等于 BT4，
  所以"最差退化为 baseline"是结构保证而不是祈祷。
- **E3 等权 logit 平均**：手册唯一允许的集成形式（不学权重，§5.4 表格）。

三条必须写下来的工程纪律（偏离了会静默出错，不会报错）：

1. **绝不把 playerid / champion 当高基数类别喂给 CatBoost**。2,560 个 playerid、
   每人中位 26 场，ordered target statistic 的方差巨大；那本质上和阶段 7 里
   1,412 个自由参数的 embedding 是同一个失败换个包装
   （实测 log loss 0.63076 → 0.99658，校准斜率塌到 0.0911）。
2. **`has_time=True` 必须开**。CatBoost 的 ordered boosting 默认会打乱样本顺序来算
   target statistic；本项目的每一列特征都是 as-of 的，打乱顺序等于让早期样本
   看到晚期样本的标签。这个泄漏不会让任何断言失败，只会让 OOS 指标变好看。
3. **超参在训练段内层选**。每折取训练段最后 10%（按 series 边界对齐）做内层验证，
   绝不碰该折的 valid 段。

DeepSet 为什么手写 numpy 而不用 torch：本仓库依赖里没有 torch，而这个网络只有
几百个参数、两层 tanh MLP，解析梯度十几行就能写完并用有限差分校验
（`tests` 与 `gradient_check()` 都验过）。为了 530 个参数拉进一个 2 GB 的依赖不划算。
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lolwp.features.draft_priors import (
    HL_CHAMP_DAYS,
    HL_MASTERY_DAYS,
    MASTERY_CAP,
    ROLES,
    _Decayed,
    _logit,
)

# --------------------------------------------------------------------------
# 特征口径
# --------------------------------------------------------------------------

#: 当前生产候选 BT4 的列。对照模型必须吃**完全相同**的信息，
#: 否则比较的是特征工程而不是函数族。
BT4_COLS: tuple[str, ...] = (
    "side_prior",
    "bt_player_diff",
    "bt_chem_diff",
    "bt_league_diff",
    "champ_meta_diff",
)

#: BT4 的分组 L2 结构：每组一个 λ。side 基本不需要罚，强度项与 draft 项分开。
BT4_GROUPS: dict[str, list[str]] = {
    "side": ["side_prior"],
    "strength": ["bt_player_diff", "bt_chem_diff", "bt_league_diff"],
    "draft": ["champ_meta_diff"],
}

#: 扩展标量集（手册 §5.1 点名可以喂给 GBDT 的那些）。
#: 全部是已工程化的低基数标量，**没有一个是高基数类别**。
EXTRA_SCALARS: tuple[str, ...] = (
    "team_elo_diff",
    "player_elo_diff",
    "mastery_diff",
    "n_first_time_champ_diff",
    "min_champ_games_diff",
    "champ_cover_min",
    "bt_known_players",
    "bt_team_level_diff",
    "team_games_min",
    "player_games_min",
)

#: 从原始 df 直接取的上下文标量（不是 as-of 状态机产物，取值在开赛前即已确定）。
CONTEXT_SCALARS: tuple[str, ...] = (
    "patch_age_days",
    "league_tier",
    "game_in_series",
    "playoffs",
)


def context_frame(df: pd.DataFrame) -> pd.DataFrame:
    """把 `CONTEXT_SCALARS` 从原始表里抠出来并转成 float（`playoffs` 可能是 bool）。"""
    out = {}
    for c in CONTEXT_SCALARS:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(0.0).to_numpy()
    return pd.DataFrame(out, index=df.index)


def inner_cut(n: int, frac: float, series: Sequence[Any] | np.ndarray | None) -> int:
    """训练段内层验证的切点：最后 `frac` 比例，必要时前移到 series 边界。

    与 `models/grouped_ridge._inner_cut` 同一套语义 —— 内层验证段绝不能从一个
    Bo5 中间切开，否则同一系列的前几局在训练段、后几局在验证段，选出来的超参偏乐观。
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"inner_frac 必须在 (0,1) 内，给的是 {frac}")
    cut = n - max(1, round(n * frac))
    if series is None:
        return max(cut, 1)
    s = np.asarray(series)
    while 1 < cut < n and s[cut - 1] == s[cut]:
        cut -= 1
    return max(cut, 1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 sigmoid。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def to_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """概率 → logit，带夹逼。集成在 logit 空间做平均（§5.4：等权 logit 平均）。"""
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def logit_average(*ps: np.ndarray) -> np.ndarray:
    """固定等权 logit 平均。**不学权重** —— 学权重要再切一层 OOF，
    会缩短有效训练窗口，且摧毁 what-if 页面赖以生存的可解释性（§5.4 明确不做）。
    """
    if not ps:
        raise ValueError("至少要给一个概率向量")
    z = np.mean([to_logit(p) for p in ps], axis=0)
    return sigmoid(z)


# --------------------------------------------------------------------------
# E1 · CatBoost
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CatBoostConfig:
    """手册 §5.4 E1 给的起点。`has_time=True` 是红线，不提供关掉它的入口。"""

    depth: int = 5
    l2_leaf_reg: float = 20.0
    iterations: int = 500
    learning_rate: float = 0.03
    one_hot_max_size: int = 8
    max_ctr_complexity: int = 1
    thread_count: int = 4
    random_seed: int = 0

    def kwargs(self) -> dict[str, Any]:
        return {
            "has_time": True,  # ★ 强制 ordered target statistics 按时间序，不得关闭
            "one_hot_max_size": self.one_hot_max_size,
            "max_ctr_complexity": self.max_ctr_complexity,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "thread_count": self.thread_count,
            "random_seed": self.random_seed,
            "verbose": 0,
            "allow_writing_files": False,
        }


#: 内层网格。刻意只有 4 个点：每折都要重选，格点多了成本线性上涨而收益在噪声里。
CATBOOST_GRID: tuple[tuple[int, float], ...] = ((4, 20.0), (4, 100.0), (6, 20.0), (6, 100.0))


def catboost_fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    *,
    cfg: CatBoostConfig | None = None,
    grid: Sequence[tuple[int, float]] | None = CATBOOST_GRID,
    inner_frac: float = 0.10,
    series_tr: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """单折：训练段内层选 (depth, l2_leaf_reg, 迭代数)，再整段重拟合并预测 valid。

    **内层验证段永远取训练段的最后 10%**（按时间，不是随机）——
    随机切会让内层验证段包含训练段末尾之后的信息分布之外的东西，
    而且同一 series 会横跨两侧。选出的迭代数 `best_iteration_` 直接带到整段重拟合。
    """
    from catboost import CatBoostClassifier

    cfg = cfg or CatBoostConfig()
    n = len(X_tr)
    best = (cfg.depth, cfg.l2_leaf_reg, cfg.iterations)
    trace: list[dict[str, float]] = []

    if grid and n >= 400:
        cut = inner_cut(n, inner_frac, series_tr)
        Xi, yi = X_tr[:cut], y_tr[:cut]
        Xv, yv = X_tr[cut:], y_tr[cut:]
        if len(np.unique(yi)) == 2 and len(np.unique(yv)) == 2:
            best_ll = math.inf
            for depth, l2 in grid:
                kw = cfg.kwargs() | {"depth": depth, "l2_leaf_reg": l2}
                m = CatBoostClassifier(**kw)
                m.fit(Xi, yi, eval_set=(Xv, yv), use_best_model=True,
                      early_stopping_rounds=60)
                ll = float(m.best_score_["validation"]["Logloss"])
                it = int(m.best_iteration_ or 0) + 1
                trace.append({"depth": depth, "l2_leaf_reg": l2, "inner_ll": ll, "iters": it})
                if ll < best_ll:
                    best_ll, best = ll, (depth, l2, max(it, 50))

    depth, l2, iters = best
    model = CatBoostClassifier(**(cfg.kwargs() | {"depth": depth, "l2_leaf_reg": l2,
                                                  "iterations": iters}))
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_va)[:, 1]
    return p, {"depth": depth, "l2_leaf_reg": l2, "iterations": iters,
               "n_trees": int(model.tree_count_), "trace": trace, "model": model}


# --------------------------------------------------------------------------
# E2 · token 化：10 个 (选手, 英雄, 位置)
# --------------------------------------------------------------------------

#: token 的特征名。**没有一个是 identity embedding** —— 那正是被证伪的路线。
#: 每个 token 只带「这个位置上的英雄现在多强 / 这个人对这个英雄多熟 / 这个人多强」，
#: DeepSet 要学的是这些量之间的**交互**（例如优势集中在一路 vs 均摊到五路），
#: 而 BT4 的 logit 里它们只以「两队求和后相减」的加性形式出现。
TOKEN_FEATURES: tuple[str, ...] = (
    "side",
    "role_top", "role_jng", "role_mid", "role_bot", "role_sup",
    "champ_meta_logit",
    "champ_meta_logit_signed",
    "mastery_frac",
    "mastery_frac_signed",
    "first_time",
    "player_form_logit",
    "player_form_logit_signed",
    "player_games_log",
)

#: 选手层收缩强度。阶段 6 的方差成分定标给出 player 层 k ≈ 14。
K_PLAYER: float = 14.0
#: 英雄×位置层收缩强度。阶段 6 的矩估计与 SVI 两条路线都给 673（手册的 476 弱 41%）。
K_CHAMP: float = 673.0


def build_tokens(df: pd.DataFrame, *, min_tier_for_winrate: int = 1) -> np.ndarray:
    """单遍时序状态机，给每一场产出 (10, len(TOKEN_FEATURES)) 的 token 矩阵。

    返回形状 `(n, 10, d)`，前 5 个 token 是蓝方五路、后 5 个是红方五路。
    **先读状态写特征、再用赛果更新状态** —— 与 `features/priors.py`、
    `features/draft_priors.py` 同一套写法，保证每个数只用了本场开赛前的信息。

    与 `draft_priors` 的关系：那里输出的是两队求和后的 **diff**，这里保留逐 token 的原值。
    DeepSet 的全部立论基础就是"求和之前还有东西"，所以不能复用聚合后的列。
    """
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")

    d = len(TOKEN_FEATURES)
    n = len(df)
    out = np.zeros((n, 10, d), dtype=np.float32)

    cs_t1 = _Decayed(HL_CHAMP_DAYS)     # 英雄×位置胜负，只吃 tier-1
    cs_all = _Decayed(HL_CHAMP_DAYS)    # 全赛区，只用来标定位置基准 θ̄_p
    mastery = _Decayed(HL_MASTERY_DAYS)
    pform = _Decayed(HL_MASTERY_DAYS)   # 选手近期战绩（衰减胜率），与 BT 潜变量不同源
    seen_pc: set[tuple[str, int]] = set()

    idx = {name: i for i, name in enumerate(TOKEN_FEATURES)}
    role_cols = [idx[f"role_{r}"] for r in ROLES]

    for i, row in enumerate(df.itertuples(index=False)):
        now = row.kickoff_utc
        y = float(row.y_blue_win)
        is_t1 = int(row.league_tier) >= min_tier_for_winrate

        theta_bar = {}
        for pos in ROLES:
            tw, tn = cs_all.read(("__POS__", pos), now)
            theta_bar[pos] = _logit(tw / tn) if tn > 0 else 0.0

        for s, (side, sign) in enumerate((("blue", 1.0), ("red", -1.0))):
            keys = getattr(row, f"{side}_champ_keys")
            pids = getattr(row, f"{side}_player_ids")
            for j, (pos, c, pid) in enumerate(zip(ROLES, keys, pids)):
                t = s * 5 + j
                out[i, t, idx["side"]] = sign
                out[i, t, role_cols[j]] = 1.0

                if c is not None and not pd.isna(c):
                    ck = int(c)
                    w, ne = cs_t1.read((ck, pos), now)
                    cm = ((_logit(w / ne) * ne + K_CHAMP * theta_bar[pos]) / (ne + K_CHAMP)
                          if ne > 0 else theta_bar[pos])
                else:
                    ck = -1
                    cm = theta_bar[pos]
                out[i, t, idx["champ_meta_logit"]] = cm
                out[i, t, idx["champ_meta_logit_signed"]] = cm * sign

                if pid is not None and not pd.isna(pid) and ck >= 0:
                    key = (str(pid), ck)
                    _, g = mastery.read(key, now)
                    frac = min(g, MASTERY_CAP) / MASTERY_CAP
                    out[i, t, idx["mastery_frac"]] = frac
                    out[i, t, idx["mastery_frac_signed"]] = frac * sign
                    out[i, t, idx["first_time"]] = 0.0 if key in seen_pc else 1.0

                if pid is not None and not pd.isna(pid):
                    pw, pn = pform.read((str(pid),), now)
                    form = (_logit(pw / pn) * pn) / (pn + K_PLAYER) if pn > 0 else 0.0
                    out[i, t, idx["player_form_logit"]] = form
                    out[i, t, idx["player_form_logit_signed"]] = form * sign
                    out[i, t, idx["player_games_log"]] = math.log1p(pn)

        # ---------- 结算后更新 ----------
        for side, yy in (("blue", y), ("red", 1.0 - y)):
            keys = getattr(row, f"{side}_champ_keys")
            pids = getattr(row, f"{side}_player_ids")
            for pos, c, pid in zip(ROLES, keys, pids):
                if c is not None and not pd.isna(c):
                    ck = int(c)
                    cs_all.update((ck, pos), yy, now)
                    cs_all.update(("__POS__", pos), yy, now)
                    if is_t1:
                        cs_t1.update((ck, pos), yy, now)
                    if pid is not None and not pd.isna(pid):
                        mastery.update((str(pid), ck), yy, now)
                        seen_pc.add((str(pid), ck))
                if pid is not None and not pd.isna(pid):
                    pform.update((str(pid),), yy, now)

    return out


# --------------------------------------------------------------------------
# E2 · zero-init residual DeepSet
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeepSetConfig:
    """参数量必须克制。

    阶段 7 的教训写死在这里：1,412 个自由参数的 champion embedding 让
    log loss 从 0.631 崩到 0.997、校准斜率塌到 0.091。所以这里的 token
    **不含任何 identity embedding**，隐藏层默认 16 维，全部参数约 530 个
    （`ZeroInitDeepSet.n_params` 会精确报出来）。
    """

    hidden: int = 16
    l2: float = 1e-3
    lr: float = 0.01
    batch_size: int = 512
    max_epochs: int = 60
    patience: int = 8
    inner_frac: float = 0.10
    seed: int = 0


@dataclass
class ZeroInitDeepSet:
    """φ(2 层 tanh MLP) → sum pooling → ρ(线性) → ×α(init=0) → 加到 base logit 上。

    **为什么 α 初始化为 0 不会让网络学不动**（ReZero 那一套）：
    ∂L/∂α = mean(g·z) 一般不为 0，α 会先离开 0；α≠0 之后内层参数才开始收到梯度。
    代价是前几个 epoch 几乎在原地，收益是**任何时刻停下来都不会比 baseline 更差**，
    这正是我们要的"最差退化为 BT4"的结构性保证。

    优化器是手写 Adam。梯度是解析的，`gradient_check()` 用有限差分校验过。
    """

    cfg: DeepSetConfig = field(default_factory=DeepSetConfig)
    W1: np.ndarray = field(default_factory=lambda: np.empty(0))
    b1: np.ndarray = field(default_factory=lambda: np.empty(0))
    W2: np.ndarray = field(default_factory=lambda: np.empty(0))
    b2: np.ndarray = field(default_factory=lambda: np.empty(0))
    w3: np.ndarray = field(default_factory=lambda: np.empty(0))
    b3: float = 0.0
    alpha: float = 0.0
    mu_: np.ndarray = field(default_factory=lambda: np.empty(0))
    sd_: np.ndarray = field(default_factory=lambda: np.empty(0))
    history_: list[dict[str, float]] = field(default_factory=list)
    best_epoch_: int = 0

    # ------------------------------------------------------------- 参数管理
    def init_params(self, d_in: int) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        h = self.cfg.hidden
        # Xavier/tanh 尺度。α 单独置 0。
        self.W1 = rng.normal(0.0, math.sqrt(1.0 / d_in), (d_in, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0.0, math.sqrt(1.0 / h), (h, h))
        self.b2 = np.zeros(h)
        self.w3 = rng.normal(0.0, math.sqrt(1.0 / h), h)
        self.b3 = 0.0
        self.alpha = 0.0

    @property
    def n_params(self) -> int:
        """自由参数总数。报告里必须写出来，和阶段 7 的 1,412 对照。"""
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size
                   + self.w3.size + 2)

    def _pack(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.w3,
                np.array([self.b3]), np.array([self.alpha])]

    def _unpack(self, arrs: Sequence[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2, self.w3 = (a.copy() for a in arrs[:5])
        self.b3 = float(arrs[5][0])
        self.alpha = float(arrs[6][0])

    # --------------------------------------------------------------- 前向
    def _forward(self, T: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """T: (N, 10, d) 已标准化。返回 (残差 score, 缓存)。"""
        A1 = T @ self.W1 + self.b1              # (N,10,h)
        H1 = np.tanh(A1)
        A2 = H1 @ self.W2 + self.b2             # (N,10,h)
        H2 = np.tanh(A2)
        S = H2.sum(axis=1)                      # (N,h)  ← 置换不变的 sum pooling
        z = S @ self.w3 + self.b3               # (N,)
        return self.alpha * z, {"T": T, "H1": H1, "H2": H2, "S": S, "z": z}

    def _standardize(self, tokens: np.ndarray) -> np.ndarray:
        return (tokens.astype(np.float64) - self.mu_) / self.sd_

    def residual(self, tokens: np.ndarray) -> np.ndarray:
        """给定 token 张量返回加到 base logit 上的残差。"""
        score, _ = self._forward(self._standardize(tokens))
        return score

    def predict_proba(self, tokens: np.ndarray, base_logit: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(base_logit, dtype=float) + self.residual(tokens))

    # --------------------------------------------------------------- 训练
    def _grads(self, T: np.ndarray, base: np.ndarray, y: np.ndarray
               ) -> tuple[float, list[np.ndarray]]:
        n = len(y)
        score, c = self._forward(T)
        p = sigmoid(base + score)
        loss = float(-np.mean(y * np.log(np.clip(p, 1e-15, 1)) +
                              (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))))
        g = (p - y) / n                          # dL/dscore

        d_alpha = float(g @ c["z"])
        dz = g * self.alpha                      # (N,)
        d_w3 = c["S"].T @ dz
        d_b3 = float(dz.sum())
        dS = np.outer(dz, self.w3)               # (N,h)
        dH2 = np.repeat(dS[:, None, :], T.shape[1], axis=1)
        dA2 = dH2 * (1.0 - c["H2"] ** 2)
        d_W2 = np.einsum("nth,ntk->hk", c["H1"], dA2)
        d_b2 = dA2.sum(axis=(0, 1))
        dH1 = dA2 @ self.W2.T
        dA1 = dH1 * (1.0 - c["H1"] ** 2)
        d_W1 = np.einsum("ntd,nth->dh", T, dA1)
        d_b1 = dA1.sum(axis=(0, 1))

        lam = self.cfg.l2
        d_W1 += lam * self.W1
        d_W2 += lam * self.W2
        d_w3 += lam * self.w3
        loss += 0.5 * lam * float((self.W1 ** 2).sum() + (self.W2 ** 2).sum()
                                  + (self.w3 ** 2).sum())
        return loss, [d_W1, d_b1, d_W2, d_b2, d_w3, np.array([d_b3]), np.array([d_alpha])]

    def fit(
        self,
        tokens: np.ndarray,
        base_logit: np.ndarray,
        y: np.ndarray,
        *,
        series: np.ndarray | None = None,
    ) -> ZeroInitDeepSet:
        """训练段内再切最后 10% 做早停（按 series 边界对齐），返回内层最优参数。

        不在早停后用全段重训：重训会丢掉"停在第几轮"这个唯一的容量控制手段，
        而这个模型的风险恰恰是容量。宁可少用 10% 的行。
        """
        y = np.asarray(y, dtype=float).ravel()
        base = np.asarray(base_logit, dtype=float).ravel()
        Tall = tokens.astype(np.float64)
        n, _, d_in = Tall.shape

        self.mu_ = Tall.reshape(-1, d_in).mean(axis=0)
        sd = Tall.reshape(-1, d_in).std(axis=0)
        sd[sd < 1e-8] = 1.0
        self.sd_ = sd
        Z = (Tall - self.mu_) / self.sd_

        self.init_params(d_in)
        cut = inner_cut(n, self.cfg.inner_frac, series)
        Ztr, btr, ytr = Z[:cut], base[:cut], y[:cut]
        Zva, bva, yva = Z[cut:], base[cut:], y[cut:]

        params = self._pack()
        m = [np.zeros_like(a) for a in params]
        v = [np.zeros_like(a) for a in params]
        best = [a.copy() for a in params]
        best_ll, bad, step = math.inf, 0, 0
        rng = np.random.default_rng(self.cfg.seed + 1)
        b1_, b2_, eps = 0.9, 0.999, 1e-8

        for ep in range(self.cfg.max_epochs):
            order = rng.permutation(cut)
            for s in range(0, cut, self.cfg.batch_size):
                sel = order[s: s + self.cfg.batch_size]
                self._unpack(params)
                _, gs = self._grads(Ztr[sel], btr[sel], ytr[sel])
                step += 1
                for k in range(len(params)):
                    m[k] = b1_ * m[k] + (1 - b1_) * gs[k]
                    v[k] = b2_ * v[k] + (1 - b2_) * gs[k] ** 2
                    mh = m[k] / (1 - b1_ ** step)
                    vh = v[k] / (1 - b2_ ** step)
                    params[k] = params[k] - self.cfg.lr * mh / (np.sqrt(vh) + eps)
            self._unpack(params)
            if len(yva) and len(np.unique(yva)) == 2:
                pv = sigmoid(bva + self._forward(Zva)[0])
                ll = float(-np.mean(yva * np.log(np.clip(pv, 1e-15, 1)) +
                                    (1 - yva) * np.log(np.clip(1 - pv, 1e-15, 1))))
            else:
                ll = math.inf
            self.history_.append({"epoch": ep, "inner_ll": ll, "alpha": float(self.alpha)})
            if ll < best_ll - 1e-7:
                best_ll, bad = ll, 0
                best = [a.copy() for a in params]
                self.best_epoch_ = ep
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break
        self._unpack(best)
        return self

    # ----------------------------------------------------------- 梯度校验
    def gradient_check(self, T: np.ndarray, base: np.ndarray, y: np.ndarray,
                       eps: float = 1e-6) -> float:
        """解析梯度 vs 中心差分，返回最大绝对偏差。手写反传必须有这一步。"""
        _, gs = self._grads(T, base, y)
        params = self._pack()
        worst = 0.0
        rng = np.random.default_rng(0)
        for k, a in enumerate(params):
            flat = a.ravel()
            picks = rng.choice(flat.size, size=min(6, flat.size), replace=False)
            for i in picks:
                orig = flat[i]
                flat[i] = orig + eps
                self._unpack(params)
                lp = self._grads(T, base, y)[0]
                flat[i] = orig - eps
                self._unpack(params)
                lm = self._grads(T, base, y)[0]
                flat[i] = orig
                self._unpack(params)
                worst = max(worst, abs((lp - lm) / (2 * eps) - gs[k].ravel()[i]))
        return worst


# --------------------------------------------------------------------------
# 推理延迟
# --------------------------------------------------------------------------
def measure_latency(fn: Any, *args: Any, repeat: int = 200) -> dict[str, float]:
    """单次推理延迟（毫秒）：先热身 20 次，再取 `repeat` 次的中位数与 p95。

    报均值没意义 —— 第一次调用带 import / JIT / 内存分配的一次性成本，
    会把均值拉高一个量级。生产关心的是稳态。
    """
    for _ in range(20):
        fn(*args)
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts)
    return {"median_ms": float(np.median(arr)), "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()), "n": repeat}


# --------------------------------------------------------------------------
# 阶段 8 实验驱动（`python -m lolwp.models.challengers`）
# --------------------------------------------------------------------------
FEATURE_CACHE = "data/cache/stage8_features.parquet"
TOKEN_CACHE = "data/cache/stage8_tokens.npy"
PREDS_OUT = "data/cache/stage8_preds.parquet"

#: 对比表里的模型名。BT4_GL2 是主力（分组 L2），其余都是挑战者。
MODEL_NAMES: tuple[str, ...] = (
    "BT4_GL2", "BT4_LR", "CB_BT4", "CB_EXT", "DEEPSET", "ENS_CB4", "ENS_CBX",
)


def load_stage8(*, min_tier: int = 1, since: str = "2019-01-01"
                ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list]:
    """把 dev 口径的 df / 特征表 / token 张量 / folds 一次性备齐（带磁盘缓存）。

    **特征与 token 都在全赛区全时段上跑状态机，再按评测口径取子集**：
    "训宽评窄"。先切子集再跑状态机会让 2019 年之前的历史凭空消失，
    Elo / 衰减胜率全部从零起步，那是另一个模型。
    """
    import os

    from lolwp.eval.walkforward import WFConfig, make_folds
    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine

    full = load_dev()
    if os.path.exists(FEATURE_CACHE):
        F_all = pd.read_parquet(FEATURE_CACHE)
        F_all.index = full.index
    else:
        from lolwp.models.bt_cache import rolling_ratings_cached
        from lolwp.models.hier_bt import HierBTConfig
        F_all = pd.concat([run_state_machine(full), run_draft_state_machine(full),
                           rolling_ratings_cached(full, HierBTConfig(refit_every_days=14))],
                          axis=1)
        F_all.to_parquet(FEATURE_CACHE, index=False)
    F_all = pd.concat([F_all, context_frame(full)], axis=1)

    if os.path.exists(TOKEN_CACHE):
        tokens_all = np.load(TOKEN_CACHE)
        if len(tokens_all) != len(full):
            tokens_all = build_tokens(full)
            np.save(TOKEN_CACHE, tokens_all)
    else:
        tokens_all = build_tokens(full)
        np.save(TOKEN_CACHE, tokens_all)

    m = (full.league_tier >= min_tier) & (full.kickoff_utc >= since)
    pos = np.nonzero(m.to_numpy())[0]
    dev = full[m].reset_index(drop=True)
    F = F_all.loc[full[m].index].reset_index(drop=True)
    tokens = tokens_all[pos]
    folds = make_folds(dev, WFConfig())
    return dev, F, tokens, folds


def _bt4_grouped(F: pd.DataFrame, y: pd.Series, tr: pd.Index, va: pd.Index,
                 series: pd.Series, t: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """BT4 的分组 L2 拟合。返回 (valid 概率, train 段的 in-sample 概率)。

    train 段的 in-sample 概率用作 DeepSet 的 base logit。5 个参数拟 12k 行，
    in-sample 与 OOS 的差别在 1e-4 量级，不构成阶段 7 那种构造性泄漏
    （那里泄漏的是**特征**本身在训练标签上拟合出来的，自由度 1,412）。
    """
    from lolwp.models import grouped_ridge as gr

    try:
        p_va, model = gr.predict_fold(F, y, tr, va, groups=BT4_GROUPS,
                                      grid=gr.DEFAULT_GRID, series=series, time=t)
    except (ValueError, KeyError):
        p_va, model = gr.predict_fold(F, y, tr, va, groups=BT4_GROUPS, default_lambda=1.0)
    return p_va, model.predict_proba(F.loc[tr])[:, 1]


def run_stage8(*, folds_limit: int | None = None, ds_cfg: DeepSetConfig | None = None,
               cb_cfg: CatBoostConfig | None = None, verbose: bool = True) -> pd.DataFrame:
    """跑满 walk-forward，返回长表（一行 = 一场 × 一个模型）并落盘。"""
    from sklearn.linear_model import LogisticRegression

    ds_cfg = ds_cfg or DeepSetConfig()
    cb_cfg = cb_cfg or CatBoostConfig()
    dev, F, tokens, folds = load_stage8()
    if folds_limit:
        folds = folds[:folds_limit]
    y = dev.y_blue_win
    ext = list(BT4_COLS) + list(EXTRA_SCALARS) + list(CONTEXT_SCALARS)
    timing: dict[str, float] = dict.fromkeys(MODEL_NAMES, 0.0)
    recs: list[pd.DataFrame] = []
    diag: list[dict[str, Any]] = []

    for f in folds:
        tr, va = f.train_idx, f.valid_idx
        ytr = y.loc[tr].to_numpy()
        ser_tr = dev.loc[tr, "series_id"].to_numpy()
        p: dict[str, np.ndarray] = {}

        t0 = time.perf_counter()
        p["BT4_GL2"], base_tr = _bt4_grouped(F, y, tr, va, dev.series_id, dev.kickoff_utc)
        timing["BT4_GL2"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        Xtr = F.loc[tr, list(BT4_COLS)].to_numpy(float)
        Xva = F.loc[va, list(BT4_COLS)].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, ytr)
        p["BT4_LR"] = lr.predict_proba((Xva - mu) / sd)[:, 1]
        timing["BT4_LR"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        p["CB_BT4"], info4 = catboost_fit_predict(Xtr, ytr, Xva, cfg=cb_cfg, series_tr=ser_tr)
        timing["CB_BT4"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        p["CB_EXT"], infox = catboost_fit_predict(
            F.loc[tr, ext].to_numpy(float), ytr, F.loc[va, ext].to_numpy(float),
            cfg=cb_cfg, series_tr=ser_tr)
        timing["CB_EXT"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ds = ZeroInitDeepSet(ds_cfg)
        ds.fit(tokens[tr.to_numpy()], to_logit(base_tr), ytr, series=ser_tr)
        p["DEEPSET"] = ds.predict_proba(tokens[va.to_numpy()], to_logit(p["BT4_GL2"]))
        timing["DEEPSET"] += time.perf_counter() - t0

        p["ENS_CB4"] = logit_average(p["BT4_GL2"], p["CB_BT4"])
        p["ENS_CBX"] = logit_average(p["BT4_GL2"], p["CB_EXT"])

        diag.append({"fold": f.idx, "n_train": len(tr), "n_valid": len(va),
                     "cb4_depth": info4["depth"], "cb4_l2": info4["l2_leaf_reg"],
                     "cb4_iters": info4["iterations"],
                     "cbx_depth": infox["depth"], "cbx_l2": infox["l2_leaf_reg"],
                     "cbx_iters": infox["iterations"],
                     "ds_alpha": float(ds.alpha), "ds_epoch": ds.best_epoch_,
                     "ds_resid_sd": float(np.std(ds.residual(tokens[va.to_numpy()])))})
        for nm in MODEL_NAMES:
            recs.append(pd.DataFrame({
                "game_id": dev.loc[va, "game_id"].to_numpy(),
                "series_id": dev.loc[va, "series_id"].to_numpy(),
                "kickoff_utc": dev.loc[va, "kickoff_utc"].to_numpy(),
                "league_tier": dev.loc[va, "league_tier"].to_numpy(),
                "fold": f.idx, "model": nm,
                "y": y.loc[va].to_numpy(), "p": p[nm]}))
        if verbose:
            print(f"fold {f.idx:>3} n_tr={len(tr):>6} n_va={len(va):>4} "
                  f"α={ds.alpha:+.4f}@{ds.best_epoch_} "
                  f"cb4=({info4['depth']},{info4['l2_leaf_reg']:g},{info4['iterations']})",
                  flush=True)

    out = pd.concat(recs, ignore_index=True)
    out.to_parquet(PREDS_OUT, index=False)
    pd.DataFrame(diag).to_csv(PREDS_OUT.replace(".parquet", "_diag.csv"), index=False)
    pd.Series(timing).to_csv(PREDS_OUT.replace(".parquet", "_timing.csv"))
    return out




# --------------------------------------------------------------------------
# 阶段 8 裁决（`python -m lolwp.models.challengers --analyze`）
# --------------------------------------------------------------------------
def analyze(preds: pd.DataFrame | None = None, *, ref: str = "BT4_GL2",
            n_boot: int = 4000, seed: int = 0) -> dict[str, Any]:
    """对每个挑战者给出：指标全家桶 + 相对参照模型的配对裁决。

    裁决口径（三条都必须同时报，缺一条都能被挑出毛病）：
    - **ΔLL 的 series 级 block bootstrap CI**，以及由它反解的 MDE = 2.8·sd(boot)。
      §6.3.1 做法 A —— 直接用经验分布，不经参数化的 deff，
      因为 deff 依赖 ρ 的估计口径，而本项目实测三种 ρ 口径并不一致。
    - **DM 检验**（聚类稳健 + Harvey 修正），给 p 值。
    - **DeLong** 只作辅助：它假设样本独立，而同一 Bo5 的 5 局强相关，p 值偏乐观。

    符号约定：ΔLL > 0 表示**挑战者比参照更好**（log loss 更低）。
    注意 `compare.paired_bootstrap_delta` 的原生符号相反，这里统一翻过来。
    """
    from lolwp.eval import compare, metrics, power
    from lolwp.eval.delong import delong_roc_test

    if preds is None:
        preds = pd.read_parquet(PREDS_OUT)
    models = [m for m in MODEL_NAMES if m in set(preds.model)]
    spine, P = compare.align_predictions(preds, models)
    y = spine.y.to_numpy()
    groups = spine.series_id.to_numpy()
    p_ref = P[ref].to_numpy(dtype=float)

    rows: dict[str, dict[str, Any]] = {}
    pvals: list[float] = []
    keys: list[str] = []
    for m in models:
        p = P[m].to_numpy(dtype=float)
        r: dict[str, Any] = dict(metrics.evaluate(y, p))
        ll = compare.block_bootstrap_ci(y, p, metrics.log_loss, groups, n_boot, seed)
        r |= {"ll_lo": ll.lo, "ll_hi": ll.hi, "ll_se": ll.se}
        if m != ref:
            b = compare.paired_bootstrap_delta(y, p, p_ref, metrics.log_loss,
                                               groups, n_boot, seed)
            d = compare.loss_diff(y, p_ref, p)          # >0 ⟺ 挑战者更好
            dm = compare.diebold_mariano(d, groups)
            pw = power.paired_power(d, groups)
            dl = delong_roc_test(y, p, p_ref)
            r |= {
                "delta_ll": -b.point, "delta_lo": -b.hi, "delta_hi": -b.lo,
                "delta_se": b.se, "mde_boot": b.mde,
                "exceeds_mde_boot": bool(abs(b.point) > b.mde),
                "dm": dm.dm_harvey, "dm_p": dm.p_value,
                "t_clustered": pw["t_clustered"], "mde_deff": pw["mde"],
                "exceeds_mde_deff": bool(pw["exceeds_mde"]),
                "rho": pw["icc"], "deff": pw["deff"], "n_eff": pw["n_eff"],
                "delong_z": dl.z, "delong_p": dl.p_value,
                "footer": power.footer(pw),
            }
            pvals.append(dm.p_value)
            keys.append(m)
        rows[m] = r

    bh = compare.benjamini_hochberg(pvals) if pvals else None
    if bh is not None:
        for k, rej, q in zip(keys, bh.reject, bh.qvalues):
            rows[k] |= {"bh_reject": bool(rej), "bh_q": float(q)}
    return {"rows": rows, "spine": spine, "P": P, "bh": bh, "ref": ref}


#: §6.7 红线：任一条触发就几乎必有泄漏，必须先排查再谈指标。
RED_LINES: dict[str, float] = {"accuracy": 0.67, "log_loss": 0.60, "fold_auc": 0.78}


def red_line_check(preds: pd.DataFrame) -> pd.DataFrame:
    """对每个模型逐条核对红线。**只要有一条触发就不许往下解读任何指标。**

    fold AUC 只在该折 valid 段 ≥ 100 场且两类都有时才计入 —— 60 场的 fold
    AUC 方差极大，0.78 完全可能是噪声，把它算成泄漏会浪费一整轮排查。
    """
    from lolwp.eval import metrics

    out = []
    for m, g in preds.groupby("model", sort=False):
        y, p = g.y.to_numpy(), g.p.to_numpy(dtype=float)
        aucs = []
        for _, gf in g.groupby("fold"):
            if len(gf) >= 100 and 0 < gf.y.mean() < 1:
                aucs.append(metrics.auc(gf.y.to_numpy(), gf.p.to_numpy(dtype=float)))
        acc = float(((p > 0.5) == (y == 1)).mean())
        ll = metrics.log_loss(y, p)
        mx = max(aucs) if aucs else float("nan")
        out.append({"model": m, "accuracy": acc, "log_loss": ll, "max_fold_auc": mx,
                    "n_folds_scored": len(aucs),
                    "trip_accuracy": acc > RED_LINES["accuracy"],
                    "trip_log_loss": ll < RED_LINES["log_loss"],
                    "trip_fold_auc": bool(np.nanmax([mx, -1]) > RED_LINES["fold_auc"])})
    t = pd.DataFrame(out)
    t["any_trip"] = t[["trip_accuracy", "trip_log_loss", "trip_fold_auc"]].any(axis=1)
    return t


def latency_bench(*, repeat: int = 300) -> pd.DataFrame:
    """单场推理延迟：在最后一折的训练段上各训一次，然后对**一行**反复计时。

    为什么是一行而不是一批：线上 what-if 页面每改一个英雄就要重算一次，
    批量吞吐无关紧要，关心的是单次往返。批量摊薄后的数字会好看 10-100 倍，
    报出来是误导。
    """
    from sklearn.linear_model import LogisticRegression

    dev, F, tokens, folds = load_stage8()
    f = folds[-1]
    tr, va = f.train_idx, f.valid_idx
    y = dev.y_blue_win
    ytr = y.loc[tr].to_numpy()
    ser_tr = dev.loc[tr, "series_id"].to_numpy()
    ext = list(BT4_COLS) + list(EXTRA_SCALARS) + list(CONTEXT_SCALARS)

    from lolwp.models import grouped_ridge as gr
    gl2 = gr.GroupedRidgeLogistic(BT4_GROUPS)
    gl2.fit_lambda_grid(F.loc[tr], y.loc[tr], series=ser_tr)
    Xtr = F.loc[tr, list(BT4_COLS)].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, ytr)
    _, i4 = catboost_fit_predict(Xtr, ytr, Xtr[:2], series_tr=ser_tr)
    _, ix = catboost_fit_predict(F.loc[tr, ext].to_numpy(float), ytr,
                                 F.loc[tr, ext].to_numpy(float)[:2], series_tr=ser_tr)
    ds = ZeroInitDeepSet(DeepSetConfig())
    ds.fit(tokens[tr.to_numpy()], to_logit(gl2.predict_proba(F.loc[tr])[:, 1]), ytr,
           series=ser_tr)

    one = F.loc[va[:1]]
    one_bt4 = one[list(BT4_COLS)].to_numpy(float)
    one_ext = one[ext].to_numpy(float)
    one_tok = tokens[va.to_numpy()[:1]]
    cb4, cbx = i4["model"], ix["model"]

    def f_gl2() -> np.ndarray:
        return gl2.predict_proba(one)[:, 1]

    def f_lr() -> np.ndarray:
        return lr.predict_proba((one_bt4 - mu) / sd)[:, 1]

    def f_cb4() -> np.ndarray:
        return cb4.predict_proba(one_bt4)[:, 1]

    def f_cbx() -> np.ndarray:
        return cbx.predict_proba(one_ext)[:, 1]

    def f_ds() -> np.ndarray:
        return ds.predict_proba(one_tok, to_logit(gl2.predict_proba(one)[:, 1]))

    def f_ens() -> np.ndarray:
        return logit_average(gl2.predict_proba(one)[:, 1], cb4.predict_proba(one_bt4)[:, 1])

    fns = {"BT4_GL2": f_gl2, "BT4_LR": f_lr, "CB_BT4": f_cb4, "CB_EXT": f_cbx,
           "DEEPSET": f_ds, "ENS_CB4": f_ens}
    rows = []
    for nm, fn in fns.items():
        r = measure_latency(fn, repeat=repeat)
        r["model"] = nm
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="阶段 8 对照模型（CatBoost / DeepSet / 集成）")
    ap.add_argument("--folds", type=int, default=0, help="只跑前 N 折（调试用），0=全部")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--analyze", action="store_true", help="只读已有预测做裁决")
    ap.add_argument("--latency", action="store_true", help="只测单场推理延迟")
    a = ap.parse_args()
    t0 = time.time()

    if a.latency:
        tbl = latency_bench()
        print(tbl.round(4).to_string(), flush=True)
        tbl.to_csv("data/cache/stage8_latency.csv")
        return

    if not a.analyze:
        run_stage8(folds_limit=a.folds or None,
                   ds_cfg=DeepSetConfig(hidden=a.hidden),
                   cb_cfg=CatBoostConfig(thread_count=a.threads))
        print(f"训练全部完成 {time.time() - t0:.0f}s → {PREDS_OUT}", flush=True)

    preds = pd.read_parquet(PREDS_OUT)
    rl = red_line_check(preds)
    print("\n=== 红线核对（§6.7）===")
    print(rl.round(4).to_string(index=False), flush=True)
    if bool(rl.any_trip.any()):
        print("⚠⚠ 有红线触发，先排查泄漏再解读任何指标", flush=True)

    res = analyze(preds)
    dump = {m: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in r.items() if not isinstance(v, (np.ndarray, pd.DataFrame))}
            for m, r in res["rows"].items()}
    with open("reports/stage8_challengers.json", "w") as fh:
        json.dump(dump, fh, indent=2, ensure_ascii=False, default=str)
    rl.to_csv("data/cache/stage8_redlines.csv", index=False)

    print("\n=== 指标 ===")
    hdr = f"{'model':<10}{'logloss':>10}{'Brier':>9}{'AUC':>8}{'ECE':>8}{'slope':>8}{'acc':>8}"
    print(hdr)
    for m, r in res["rows"].items():
        print(f"{m:<10}{r['log_loss']:>10.5f}{r['brier']:>9.5f}{r['auc']:>8.4f}"
              f"{r['ece']:>8.4f}{r['calib_slope']:>8.4f}{r['accuracy']:>8.4f}")
    print(f"\n=== 相对 {res['ref']} 的配对裁决（ΔLL>0 = 挑战者更好）===")
    for m, r in res["rows"].items():
        if "delta_ll" not in r:
            continue
        print(f"{m:<10} ΔLL={r['delta_ll']:+.5f} "
              f"95%CI[{r['delta_lo']:+.5f},{r['delta_hi']:+.5f}] "
              f"MDE_boot={r['mde_boot']:.5f} {'超过✓' if r['exceeds_mde_boot'] else '低于✗'} "
              f"DM*={r['dm']:+.3f}(p={r['dm_p']:.4g}) BHq={r.get('bh_q', float('nan')):.4g} "
              f"DeLong z={r['delong_z']:+.3f}(p={r['delong_p']:.4g})")
        print(f"           {r['footer']}")
    print(f"\n完成 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
