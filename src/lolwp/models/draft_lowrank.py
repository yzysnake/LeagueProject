"""协同 / 克制的低秩建模 + zero-init α（PLAN.md §4.5.3 / §4.5.4 / §4.5.5）。

为什么不查表：173 个英雄有 C(173,2)=14,878 个 pair，实测 2019 年起 tier-1 里
只观测到 4,171 个组合达到 10 场，中位数远低于可估计的量。显式 pairwise 是
9,990 到 29,420 个参数对 2 万场样本，手册实测 log loss 0.7756，比抛硬币还差。

低秩替代：每个英雄一个 d 维 embedding u_c。
  synergy = Σ_{i<j∈blue}⟨u_i,u_j⟩ − Σ_{i<j∈red}⟨u_i,u_j⟩      对称
  counter = Σ_{i∈blue,j∈red} u_iᵀ M u_j，  M = −Mᵀ（反对称）    d=8 时仅 28 个自由参数
反对称保证 counter(a,b) = −counter(b,a)，这是克制的定义性质，
用一般矩阵会让模型浪费一半参数去学这个约束。

**zero-init α（§4.5.4，最重要的一个工程技巧）**：
draft 项乘一个初始化为 0 的可学标量再加进 logit。三重收益——
  1. 最差退化为纯实力基线，不可能比 base 更糟；
  2. 回滚就是 α := 0，一行代码；
  3. **α̂ 的 bootstrap 区间本身就是"BP 到底值多少胜率"的带区间答案**，
     这是整个项目最有科研价值的单个数字。

**泄漏陷阱（§4.5.5）**：embedding 的预训练语料**必须在每个 fold 的训练段内**
重新拟合。用全量数据预训练再去打 OOS，等于把未来的 meta 信息喂进了特征——
这类泄漏不报错，只会让指标好看。本模块的 fit() 只吃传进来的训练段。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROLES = ("top", "jng", "mid", "bot", "sup")


@dataclass
class LowRankConfig:
    dim: int = 8
    l2: float = 1.0
    max_iter: int = 200
    seed: int = 0
    use_counter: bool = True


class DraftLowRank:
    """在给定训练段上拟合英雄 embedding，产出 synergy / counter 标量特征。"""

    def __init__(self, cfg: LowRankConfig | None = None) -> None:
        self.cfg = cfg or LowRankConfig()
        self.U: np.ndarray | None = None
        self.M: np.ndarray | None = None
        self.index: dict[int, int] = {}

    # ---------- 内部：把一场比赛编码成英雄下标 ----------
    def _encode(self, blue: list, red: list) -> tuple[list[int], list[int]]:
        b = [self.index[int(c)] for c in blue if c is not None and int(c) in self.index]
        r = [self.index[int(c)] for c in red if c is not None and int(c) in self.index]
        return b, r

    def _terms(self, b: list[int], r: list[int], U: np.ndarray, M: np.ndarray | None):
        syn = 0.0
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                syn += float(U[b[i]] @ U[b[j]])
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                syn -= float(U[r[i]] @ U[r[j]])
        cnt = 0.0
        if M is not None and b and r:
            cnt = float(U[b].sum(0) @ M @ U[r].sum(0))
        return syn, cnt

    def fit(self, games, y: np.ndarray) -> DraftLowRank:
        """games 是 (blue_keys, red_keys) 的序列。**只吃训练段。**"""
        from scipy.optimize import minimize

        keys = sorted({int(c) for b, r in games for c in list(b) + list(r)
                       if c is not None and not np.isnan(float(c))})
        self.index = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        d = self.cfg.dim
        enc = [self._encode(b, r) for b, r in games]
        rng = np.random.default_rng(self.cfg.seed)
        # 反对称 M 只存上三角，d=8 → 28 个自由参数
        tri = np.triu_indices(d, k=1)
        n_m = len(tri[0]) if self.cfg.use_counter else 0
        x0 = np.concatenate([rng.normal(0, 0.01, n * d), np.zeros(n_m)])
        y = np.asarray(y, dtype=float)

        def unpack(x):
            U = x[: n * d].reshape(n, d)
            if not self.cfg.use_counter:
                return U, None
            M = np.zeros((d, d))
            M[tri] = x[n * d:]
            return U, M - M.T

        def obj(x):
            U, M = unpack(x)
            z = np.empty(len(enc))
            for i, (b, r) in enumerate(enc):
                s, c = self._terms(b, r, U, M)
                z[i] = s + c
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            eps = 1e-12
            nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            return nll + self.cfg.l2 * float((x**2).sum()) / len(enc)

        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": self.cfg.max_iter, "maxfun": self.cfg.max_iter * 3})
        self.U, self.M = unpack(res.x)
        self.n_iter_ = int(res.nit)
        return self

    def transform(self, games) -> np.ndarray:
        """返回 (n, 2)：[synergy_diff, counter_diff]。"""
        out = np.zeros((len(games), 2))
        for i, (b, r) in enumerate(games):
            bi, ri = self._encode(b, r)
            s, c = self._terms(bi, ri, self.U, self.M)
            out[i] = (s, c)
        return out
