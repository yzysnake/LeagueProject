"""DeLong 检验：两个相关 ROC 曲线下面积之差的显著性（PLAN.md §6.6.3）。

来源与许可
----------
* 检验本体：DeLong, DeLong & Clarke-Pearson (1988), *Biometrics* 44(3):837-845。
* 快速算法：Xu Sun & Weichao Xu (2014), "Fast Implementation of DeLong's Algorithm
  for Comparing the Areas Under Correlated Receiver Operating Characteristic
  Curves", *IEEE Signal Processing Letters* 21(11):1389-1393。把原始 O(n²) 的
  成对比较降到 O(n log n)：结构分量 V10 / V01 可以直接由 midrank 写出来。
* 公开参考实现：``github.com/yandexdataschool/roc_comparison`` 的
  ``compare_auc_delong_xu.py``（该仓库未在 PyPI 发布，手册要求 vendor 一份）。

本文件**不是**逐行复制。midrank、结构分量、协方差三段都按论文公式重新推导后
自己写的（沿用论文的 V10 / V01 / S10 / S01 记号便于对照），并用 bootstrap 做了
数值验收（见 ``tests/test_delong.py``：AUC 点估计与 ``metrics.auc`` 的实测
max|diff| = 1.1e-16，解析方差与 2000 次 bootstrap 经验方差的相对误差实测
2.5%，配对差方差 3.5%，且 6 个种子上无系统性偏向）。因此本文件不携带
上游仓库的版权义务；引用关系写在这里是为了可追溯，不是为了规避署名。

算法（记 m 个正样本 X、n 个负样本 Y，Z = X ⊕ Y）
------------------------------------------------
    TX = midrank(X within X)      TY = midrank(Y within Y)
    TZ = midrank(Z within Z)
    AUC   = (Σ TZ[正] − m(m+1)/2) / (m·n)          ← 含并列的 Mann-Whitney U
    V10_i = (TZ_i − TX_i) / n            i = 1..m   ← 每个正样本的结构分量
    V01_j = 1 − (TZ_{m+j} − TY_j) / m    j = 1..n   ← 每个负样本的结构分量
    S     = cov(V10)/m + cov(V01)/n                 ← AUC 向量的协方差矩阵
两模型比较取 l = (1, −1)：var(ΔAUC) = l S lᵀ，z = ΔAUC / sqrt(var)。

⚠️ 本项目特有的限制（必读）
---------------------------
**标准 DeLong 假设样本相互独立。本项目的样本按 series 聚类**——同一个 BO3/BO5
里的多局共享双方阵容、状态与当天的临场因素。阶段 3 实测（reports/
stage3_power_and_icc.md）配对 log loss 差的 ICC 在 **0.086–0.312** 之间。
聚类让 DeLong 低估标准误、**p 值偏乐观**；方向是固定的，绝不会偏保守。

**偏差幅度是本模块自己跑出来的，不是引用**。L2 vs L3，23,060 场 / 12,274
series（m̄ = 1.879），series 级 block bootstrap B = 10,000：

======================  ==========  ==================  ======
量                      DeLong 解析  series block boot   比值
======================  ==========  ==================  ======
se(AUC_L3)，单模型       0.003428    0.003585            1.046
se(ΔAUC = L3 − L2)      0.000975    0.000991            1.017
p(ΔAUC)                 4.83e-4     6.0e-4              ×1.24
======================  ==========  ==================  ======

换 seed 复核，配对的 se_ratio 在 **1.00–1.02** 之间抖动（bootstrap 自身的
Monte Carlo 误差就有 ±1.5%），所以只能说"配对上的聚类惩罚 ≤2%"，不能报得更细。

**为什么这么小——这是本模块实测出来的一条非平凡结论。**
series 层的共享效应会**同时**抬高两个模型的 AUC，相减时基本抵消。合成的强聚类
数据（每 series 5 局、组内相关极高）上同样如此：单模型 AUC 的 se_ratio ≈ 1.25，
而配对 ΔAUC 的 se_ratio ≈ 1.00，见
``tests/test_delong.py::test_cluster_penalty_largely_cancels_in_the_paired_difference``。

对照：**同一个 L2 vs L3 配对在 log loss 上 deff = 1.274（标准误 ×1.13）**，
比 AUC 侧大一个数量级。**不能把 log loss 侧的 deff 挪用到 AUC 上**，反之亦然——
这是阶段 3 那条"ICC 是每个比较各自的属性"的直接推论，这里又多了一个维度：
它还是**每个指标**各自的属性。

另一条校准证据：把 bootstrap 换成 game 级（= 假装独立），se = 0.000966，
与 DeLong 解析的 0.000975 只差 **0.9%**（B=10,000）。说明解析公式在
n = 23,060 上本身是准的；series 级 / game 级 = **1.027**，那 2–3 个百分点
才是聚类的份额，不是实现误差。

所以：**AUC 检验仅作辅助（审稿人的形式要求），主判据是 log loss 的聚类稳健检验**
（§6.6.1 的 series 级 block bootstrap + §6.3 的 ICC / deff 修正）。凡 DeLong 的 p
落在 0.01–0.05 这一段，一律以 ``paired_auc_block_bootstrap`` 的 p 为准。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "DelongResult",
    "delong_roc_test",
    "delong_roc_variance",
    "paired_auc_block_bootstrap",
    "pairwise_delong_table",
]


# --------------------------------------------------------------------------
# 内核
# --------------------------------------------------------------------------

def _validate_scores(*arrays: np.ndarray) -> None:
    """NaN 会让 _midrank 的内层 while 永久挂起（NaN != NaN 使 j 停在 i，外层不推进）。

    [实测 2026-09-19] 含单个 NaN 的 6 元素输入 10 秒未返回。
    这是**挂起**不是报错，在轮询/CI 里会表现为任务永远不结束，比崩溃更难查。
    """
    for a in arrays:
        arr = np.asarray(a, dtype=float)
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"DeLong 的输入含 {n_bad} 个 NaN/Inf。请先剔除或填补——"
                "带 NaN 进来会让 midrank 死循环而不是报错。"
            )


def _midrank(x: np.ndarray) -> np.ndarray:
    """并列取平均秩（midrank），1-based。

    并列必须取平均秩而不是任意顺序秩，否则 AUC 会随输入顺序抖动——
    本项目 L0 是常数 0.5（全部并列），L1 每折一个常数（大段并列），
    没有 midrank 这两级的 AUC 根本没有定义。
    """
    order = np.argsort(x, kind="mergesort")
    z = np.asarray(x, dtype=float)[order]
    n = len(z)
    t = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        # 秩 i+1 .. j 的平均值
        t[i:j] = 0.5 * (i + 1 + j)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _cov_rows(a: np.ndarray) -> np.ndarray:
    """行 = 变量、列 = 观测的样本协方差（ddof=1）。

    不用 ``np.cov``：k=1 时它返回 0 维数组，样本数 <2 时它返回 nan 并告警，
    而这两种退化情形在本项目里都真实存在（L0 常数预测），必须给出 0 而不是 nan。
    """
    k, n = a.shape
    if n < 2:
        return np.zeros((k, k), dtype=float)
    c = a - a.mean(axis=1, keepdims=True)
    return (c @ c.T) / (n - 1)


def _fast_delong(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) 的 O(n log n) 内核。

    参数
    ----
    y : (N,) 0/1 标签
    scores : (k, N) k 个模型的预测分数

    返回
    ----
    (aucs, cov)：aucs 形状 (k,)，cov 形状 (k, k)。
    """
    y = np.asarray(y)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    if scores.shape[1] != len(y):
        raise ValueError(f"scores 第二维 {scores.shape[1]} 与 y 长度 {len(y)} 不符")
    pos = y == 1
    m = int(pos.sum())
    n = int((~pos).sum())
    if m == 0 or n == 0:
        raise ValueError("DeLong 需要正负样本各至少一个")

    k = scores.shape[0]
    xs = scores[:, pos]          # (k, m)
    ys = scores[:, ~pos]         # (k, n)
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(xs[r])
        ty[r] = _midrank(ys[r])
        tz[r] = _midrank(np.concatenate([xs[r], ys[r]]))

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    cov = _cov_rows(v10) / m + _cov_rows(v01) / n
    return aucs, cov


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------

def delong_roc_variance(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """单模型的 AUC 与其 DeLong 解析方差。

    返回 ``(auc, var)``。``auc`` 与 ``lolwp.eval.metrics.auc`` 逐位一致
    （两者都是含 midrank 的 Mann-Whitney U，实测 max|diff| < 1e-15）。

    常数预测（全部并列）的方差恒为 0，这不是 bug：结构分量此时是常数，
    没有抽样波动可言。
    """
    _validate_scores(p)
    aucs, cov = _fast_delong(y, np.asarray(p, dtype=float)[None, :])
    return float(aucs[0]), float(max(cov[0, 0], 0.0))


@dataclass(frozen=True)
class DelongResult:
    """DeLong 两样本（配对、同一批样本）AUC 比较的结果。"""

    auc_a: float
    auc_b: float
    delta: float          # auc_a − auc_b
    se: float             # ΔAUC 的 DeLong 解析标准误（**未做聚类修正**）
    z: float
    p_value: float
    cov: np.ndarray = field(repr=False)   # 2×2 AUC 协方差矩阵
    n_pos: int = 0
    n_neg: int = 0

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """ΔAUC 的正态近似置信区间（同样未做聚类修正）。"""
        z = float(stats.norm.ppf(0.5 + level / 2.0))
        return self.delta - z * self.se, self.delta + z * self.se

    def to_row(self) -> dict[str, float]:
        lo, hi = self.ci()
        return {
            "auc_a": self.auc_a,
            "auc_b": self.auc_b,
            "delta_auc": self.delta,
            "se_delong": self.se,
            "z": self.z,
            "p_delong": self.p_value,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_pos": float(self.n_pos),
            "n_neg": float(self.n_neg),
        }


def delong_roc_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> DelongResult:
    """检验 AUC(p_a) − AUC(p_b) = 0（同一批样本上的配对比较）。

    ⚠️ p 值假设样本独立。本项目样本按 series 聚类，该 p 值**偏乐观**
    （L2 vs L3 实测：标准误被低估 ~2%，p 从 6.0e-4 变成 4.8e-4）。
    AUC 只是辅助证据，主判据是 log loss 的聚类稳健检验。
    需要聚类稳健的 AUC 结论时用 :func:`paired_auc_block_bootstrap`。

    退化情形：``p_a is p_b`` 或两者完全相同时，Δ = 0 且 var = 0，
    返回 z = 0 / p = 1，不会除零。
    """
    _validate_scores(p_a, p_b)
    scores = np.vstack([np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)])
    aucs, cov = _fast_delong(y, scores)
    delta = float(aucs[0] - aucs[1])
    # l = (1, −1)：var(ΔAUC) = S11 − 2·S12 + S22。浮点下可能出现 −1e-20，截到 0。
    var = float(max(cov[0, 0] - 2.0 * cov[0, 1] + cov[1, 1], 0.0))
    se = float(np.sqrt(var))
    if se <= 0.0:
        z = 0.0 if delta == 0.0 else float(np.sign(delta) * np.inf)
    else:
        z = delta / se
    y_arr = np.asarray(y)
    return DelongResult(
        auc_a=float(aucs[0]),
        auc_b=float(aucs[1]),
        delta=delta,
        se=se,
        z=float(z),
        p_value=float(2.0 * stats.norm.sf(abs(z))),
        cov=cov,
        n_pos=int((y_arr == 1).sum()),
        n_neg=int((y_arr != 1).sum()),
    )


# --------------------------------------------------------------------------
# 聚类稳健对照：series 级 block bootstrap
# --------------------------------------------------------------------------

def _group_offsets(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把任意分组键压成 (排序后的行序, 每组起点, 每组大小)。"""
    codes = pd.factorize(pd.Series(groups), sort=False)[0]
    order = np.argsort(codes, kind="mergesort")
    sizes = np.bincount(codes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return order, starts, sizes


def paired_auc_block_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """以 **series** 为重抽单位的配对 ΔAUC bootstrap（§6.6.1 的 AUC 版）。

    为什么需要它：DeLong 的方差公式建立在样本独立之上；本项目同一 series 内的
    多局高度相关，独立假设被违反的方向是固定的——**低估标准误、p 值偏乐观**。
    block bootstrap 直接把 series 当重抽单位，不需要 ICC / deff 这类近似
    （这也是手册 §6.3.1 要求"两套尺子合一"的做法）。

    以 game 为单位重抽会让区间系统性偏窄，等于把 DeLong 的错误换一种方式再犯一次。

    返回的 ``se_ratio = se_block / se_delong`` 就是 DeLong 乐观程度的量化。
    实测：真实数据上配对 ΔAUC 的 se_ratio ≈ 1.00–1.02，单模型 AUC ≈ 1.05；
    合成强聚类数据上分别是 ≈1.00 与 ≈1.25。想量单模型 AUC 的惩罚，
    可以把 ``p_b`` 传成常数（常数预测方差恒为 0，var(ΔAUC) 就等于 var(AUC_a)）。

    B 的取舍：se 的 Monte Carlo 相对误差约 1/sqrt(2B)，B=2,000 时 ±1.6%，
    B=10,000 时 ±0.7%。既然要量的偏差本身只有 1–5%，**B < 5,000 报不出 se_ratio
    的小数点后第二位**。
    """
    y = np.asarray(y)
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    order, starts, sizes = _group_offsets(np.asarray(groups))
    y_s, a_s, b_s = y[order], a[order], b[order]
    n_groups = len(sizes)

    ref = delong_roc_test(y, a, b)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n_bad = 0
    for i in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        lens = sizes[pick]
        out_starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        idx = (
            np.arange(int(lens.sum()))
            - np.repeat(out_starts, lens)
            + np.repeat(starts[pick], lens)
        )
        yy = y_s[idx]
        if yy.min() == yy.max():          # 极端重抽下可能单类，跳过
            boot[i] = np.nan
            n_bad += 1
            continue
        aucs, _ = _fast_delong(yy, np.vstack([a_s[idx], b_s[idx]]))
        boot[i] = aucs[0] - aucs[1]

    vals = boot[~np.isnan(boot)]
    se_block = float(np.std(vals, ddof=1))
    z_block = ref.delta / se_block if se_block > 0 else 0.0
    # 百分位法的双侧 p：0 落在 bootstrap 分布的哪一侧尾巴里
    frac_le = float((vals <= 0).mean())
    p_pct = float(min(1.0, 2.0 * min(frac_le, 1.0 - frac_le)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "delta_auc": ref.delta,
        "se_delong": ref.se,
        "p_delong": ref.p_value,
        "se_block": se_block,
        "z_block": float(z_block),
        "p_block": float(2.0 * stats.norm.sf(abs(z_block))),
        "p_block_pct": p_pct,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "se_ratio": float(se_block / ref.se) if ref.se > 0 else float("nan"),
        "n_groups": float(n_groups),
        "n_boot": float(len(vals)),
        "n_degenerate": float(n_bad),
    }


def pairwise_delong_table(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """对一组模型配对批量跑 DeLong，返回可直接进报告的表。

    ``pairs`` 写成 ``[("L1", "L0"), ...]``，delta = AUC(第一个) − AUC(第二个)。
    """
    rows = []
    for a, b in pairs:
        r = delong_roc_test(y, preds[a], preds[b]).to_row()
        rows.append({"model_a": a, "model_b": b, **r})
    return pd.DataFrame(rows)
