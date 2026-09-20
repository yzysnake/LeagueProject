"""协同 / 克制的低秩建模 + zero-init α（PLAN.md §4.5.3 / §4.5.4 / §4.5.5）。

为什么不查表：173 个英雄有 C(173,2)=14,878 个 pair，实测 tier-1 里只有 4,171 个
组合达到 10 场。显式 pairwise 是 9,990~29,420 个参数对 2 万场样本，
手册实测 log loss 0.7756，比抛硬币的 0.6912 还差。

低秩替代：每个英雄一个 d 维 embedding u_c。
  synergy = Σ_{i<j∈blue}⟨u_i,u_j⟩ − Σ_{i<j∈red}⟨u_i,u_j⟩            对称
  counter = (Σ_{i∈blue} u_i)ᵀ M (Σ_{j∈red} u_j)，M = −Mᵀ            反对称
反对称保证 counter(a,b) = −counter(b,a)，这是克制的定义性质；
d=8 时 M 只有 28 个自由参数，用一般矩阵会浪费一半参数去学这个约束。

**向量化**：朴素写法对每场遍历 10 个 pair，90 折 × 200 次迭代是 Python 里跑不完的。
用恒等式 Σ_{i<j}⟨u_i,u_j⟩ = ½(‖Σu_i‖² − Σ‖u_i‖²) 把它变成一次 scatter-add
加两次矩阵乘，并给出解析梯度。实测全量拟合从"跑不完"降到秒级。

**zero-init α（§4.5.4）**：draft 项乘一个初始化为 0 的可学标量再加进 logit。
最差退化为纯实力基线；回滚就是 α := 0；而 **α̂ 的 bootstrap 区间本身就是
"BP 到底值多少胜率"的带区间答案**。

**泄漏陷阱（§4.5.5）**：embedding 必须在**每个 fold 的训练段内**重新拟合。
用全量预训练再打 OOS 等于把未来 meta 喂进特征，不报错、只让指标好看。
fit() 只吃传进来的训练段。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROLES = ("top", "jng", "mid", "bot", "sup")


@dataclass
class LowRankConfig:
    dim: int = 8
    l2: float = 3.0
    max_iter: int = 300
    seed: int = 0
    use_counter: bool = True


def _index_matrix(games, index: dict[int, int], n_slots: int = 5) -> np.ndarray:
    """(n_games, 5) 的英雄下标矩阵，缺失用 -1。"""
    out = np.full((len(games), n_slots), -1, dtype=np.int64)
    for i, keys in enumerate(games):
        j = 0
        for c in keys:
            if c is None or (isinstance(c, float) and np.isnan(c)):
                continue
            k = index.get(int(c))
            if k is not None and j < n_slots:
                out[i, j] = k
                j += 1
    return out


class DraftLowRank:
    def __init__(self, cfg: LowRankConfig | None = None) -> None:
        self.cfg = cfg or LowRankConfig()
        self.U: np.ndarray | None = None
        self.M: np.ndarray | None = None
        self.index: dict[int, int] = {}

    # ---------- 向量化的前向与梯度 ----------
    @staticmethod
    def _gather(U: np.ndarray, idx: np.ndarray):
        """返回 (S, Q, mask)：S=Σu，Q=Σ‖u‖²。缺失槽位贡献 0。"""
        mask = idx >= 0
        safe = np.where(mask, idx, 0)
        emb = U[safe] * mask[:, :, None]          # (n,5,d)
        S = emb.sum(1)                             # (n,d)
        Q = (emb**2).sum((1, 2))                   # (n,)
        return S, Q, emb, mask, safe

    def _forward(self, U, M, ib, ir):
        Sb, Qb, Eb, Mb, Ab = self._gather(U, ib)
        Sr, Qr, Er, Mr, Ar = self._gather(U, ir)
        syn = 0.5 * ((Sb**2).sum(1) - Qb) - 0.5 * ((Sr**2).sum(1) - Qr)
        cnt = ((Sb @ M) * Sr).sum(1) if M is not None else np.zeros(len(syn))
        return syn, cnt, (Sb, Sr, Eb, Er, Mb, Mr, Ab, Ar)

    def fit(self, blue_games, red_games, y: np.ndarray) -> DraftLowRank:
        from scipy.optimize import minimize

        keys = sorted({int(c) for g in (blue_games, red_games) for ks in g for c in ks
                       if c is not None and not (isinstance(c, float) and np.isnan(c))})
        self.index = {k: i for i, k in enumerate(keys)}
        n, d = len(keys), self.cfg.dim
        ib = _index_matrix(blue_games, self.index)
        ir = _index_matrix(red_games, self.index)
        y = np.asarray(y, dtype=float)
        ng = len(y)
        tri = np.triu_indices(d, k=1)
        n_m = len(tri[0]) if self.cfg.use_counter else 0
        rng = np.random.default_rng(self.cfg.seed)
        x0 = np.concatenate([rng.normal(0, 0.05, n * d), np.zeros(n_m)])
        lam = self.cfg.l2 / ng

        def unpack(x):
            U = x[: n * d].reshape(n, d)
            if not self.cfg.use_counter:
                return U, None
            Mu = np.zeros((d, d))
            Mu[tri] = x[n * d:]
            return U, Mu - Mu.T

        def fg(x):
            U, M = unpack(x)
            syn, cnt, (Sb, Sr, Eb, Er, Mb, Mr, Ab, Ar) = self._forward(U, M, ib, ir)
            z = np.clip(syn + cnt, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            nll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            g_z = (p - y) / ng                                     # (n,)

            gU = np.zeros_like(U)
            # synergy: d/du_i = S_side − u_i（蓝为 +，红为 −）
            for S, E, Mk, A, sgn in ((Sb, Eb, Mb, Ab, +1.0), (Sr, Er, Mr, Ar, -1.0)):
                contrib = (S[:, None, :] - E) * Mk[:, :, None]     # (n,5,d)
                w = (sgn * g_z)[:, None, None] * contrib
                np.add.at(gU, A[Mk], w[Mk])
            # counter: d/du_i(blue) = M S_r ；d/du_j(red) = Mᵀ S_b = −M S_b
            if M is not None:
                cb = (Sr @ M.T)[:, None, :] * Mb[:, :, None]
                cr = (Sb @ M)[:, None, :] * Mr[:, :, None]
                np.add.at(gU, Ab[Mb], (g_z[:, None, None] * cb)[Mb])
                np.add.at(gU, Ar[Mr], (g_z[:, None, None] * cr)[Mr])
                gM_full = (Sb * g_z[:, None]).T @ Sr               # (d,d)
                gM = (gM_full - gM_full.T)[tri]
            else:
                gM = np.zeros(0)

            grad = np.concatenate([gU.ravel(), gM]) + 2 * lam * x
            return nll + lam * float((x**2).sum()), grad

        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.cfg.max_iter, "maxfun": self.cfg.max_iter * 2})
        self.U, self.M = unpack(res.x)
        self.n_iter_ = int(res.nit)
        self.nll_ = float(res.fun)
        return self

    def transform(self, blue_games, red_games) -> np.ndarray:
        """返回 (n, 2)：[synergy_diff, counter_diff]。"""
        ib = _index_matrix(blue_games, self.index)
        ir = _index_matrix(red_games, self.index)
        syn, cnt, _ = self._forward(self.U, self.M, ib, ir)
        return np.column_stack([syn, cnt])
