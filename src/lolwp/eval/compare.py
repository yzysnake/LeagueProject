"""模型对比的显著性检验套件（PLAN.md §6.6 + §6.4.1 的对比表）。

**全套检验的聚类单位是 series，不是 game。** 理由不是"保守一点比较好"，
而是实测：同一 Bo3/Bo5 的多局共享双方实力、当日状态、BP 博弈，
配对 loss 差 d_i 的组内相关（ICC）在本项目实测 0.086–0.312（reports/stage3_power_and_icc.md）。
按 game 独立算标准误会把 SE 系统性压窄 √deff 倍 —— 本模块的
`block_bootstrap_ci(..., unit="game")` 与 `DMResult.se_iid` 就是拿来把这个差值算给人看的。

四件套 + 一张表：
  1. block_bootstrap_ci      —— series 整块重采样的单模型指标 CI
  2. paired_bootstrap_delta  —— 共用同一组重采样索引的配对 Δ 及其 CI
  3. diebold_mariano         —— 聚类稳健方差 + Harvey 小样本修正
  4. benjamini_hochberg      —— exploratory 比较的 FDR 控制
  5. compare_table           —— §6.4.1 的 13 列对比表，表脚自动打印 power.footer()

预注册（experiments/preregistration.md §1）里 primary comparison 只有一个
（M_main vs L3），其余一律 exploratory，**必须走 BH**。

--------------------------------------------------------------------------
实测摘要 **[2026-09-19，data/preds_ladder.parquet，23,060 场 / 12,274 series]**

L2 vs L3 的 ΔlogLoss = +0.001602，同一个量用五种口径算标准误：

  | 口径                              | SE        | 相对 game 级 |
  |---|---|---|
  | game 级 block bootstrap           | 0.000597  | 1.000 |
  | 朴素 iid（σ_d/√n）                | 0.000595  | 0.997 |
  | CR1 聚类稳健三明治（本模块 DM 用）  | 0.000607  | 1.016 |
  | **series 级 block bootstrap**     | 0.000614  | 1.029 |
  | ICC(1)+deff（σ_d/√n_eff）          | 0.000672  | 1.126 |

两个结论：
  a) **series 级确实比 game 级宽**，但在本 OOS 上只宽 1.6%–3%，
     不是手册暗示的几十 % —— 因为 m̄ 只有 1.879 且配对差的组内相关很低。
     （合成数据上组内相关更强时实测宽 11.7%，game 级覆盖率从 95.5% 掉到 92.7%。）
  b) ICC(1) 路线（阶段 3 用的那个）在这个配对上 **高估 SE 约 9%**，
     原因是单局 series 的异方差被 ICC(1) 误认成组间方差分量，详见 `cluster_rho`。
     对比表的 `>MDE?` 因此用 bootstrap 经验分布反解的 MDE（§6.3.1 做法 A），
     ICC 路线的数字仍照 §6.3.6 打在表脚里并排给读者看。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from lolwp.eval import metrics, power

__all__ = [
    "BHResult",
    "BootResult",
    "DMResult",
    "align_predictions",
    "benjamini_hochberg",
    "block_bootstrap_ci",
    "cluster_rho",
    "compare_table",
    "diebold_mariano",
    "fixed_bin_ece",
    "format_compare_table",
    "international_leagues",
    "loss_diff",
    "paired_bootstrap_delta",
    "register_mean_metric",
    "se_comparison",
]

MetricFn = Callable[[np.ndarray, np.ndarray], float]
Unit = Literal["series", "game"]

# t* = 2.8 与 power.mde 保持同一把尺子（§6.3.1 做法 A：MDE = 2.8·sd(boot)）
T_STAR = 2.8


# --------------------------------------------------------------------------
# 0. 逐场可分解指标的快路径登记表
# --------------------------------------------------------------------------
def _per_game_brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


# log loss / Brier 都是「逐场贡献的算术平均」，重采样统计量可以写成
# sum(组和)/sum(组大小) 的比值形式 —— 每次迭代只要 O(n_series) 而不是 O(n_games)，
# 2000 次 bootstrap × 7 个模型才跑得动。AUC / ECE 不可分解，走通用慢路径。
_MEAN_METRICS: dict[Any, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    metrics.log_loss: metrics.per_game_log_loss,
    metrics.brier: _per_game_brier,
}


def register_mean_metric(
    metric_fn: MetricFn, per_game_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    """登记一个「等于逐场贡献均值」的指标，让它走 O(n_series) 的快路径。

    只登记真的满足 metric_fn(y,p) == mean(per_game_fn(y,p)) 的指标；
    登记错了不会报错，只会悄悄算错，所以这里故意不做成自动推断。
    """
    _MEAN_METRICS[metric_fn] = per_game_fn


def fixed_bin_ece(
    p_reference: np.ndarray | pd.Series, n_bins: int = 15
) -> Callable[[np.ndarray, np.ndarray], float]:
    """等频分箱边界**冻结在全样本上**的 ECE，专供 bootstrap 使用。

    为什么需要它：`metrics.ece` 每次调用都用传进来的 p 重新算分位数边界。
    bootstrap 重采样会造出大量**完全重复的 p 值**（同一场被抽中多次，
    同一 series 内多局的 p 本来就相近），分位数边界上就挤满了并列，
    等频分箱退化成大小悬殊的箱，小箱内 |acc−conf| 被噪声抬得很高。
    ECE 又是绝对值之和（不会正负抵消），于是系统性上偏。

    **[实测 2026-09-19，合成数据 n=1,561 / 600 series]**

    | 实现 | point | 95% percentile CI | bias/se | CI 含点估计？ |
    |---|---|---|---|---|
    | `metrics.ece`（每次重算边界） | 0.04078 | [0.04103, 0.08523] | **+1.88** | **否** |
    | `fixed_bin_ece`（冻结边界） | 0.04078 | [0.03916, 0.07849] | +1.74 | 是 |

    诚实地说：冻结边界只拿掉了一小部分偏（1.88 → 1.74）。
    **大头不是分箱不稳，而是分箱 ECE 本身就是正偏估计量**：
    E|acc_b − conf_b| ≥ |E acc_b − E conf_b|，而 bootstrap 重采样只有 ≈63% 的唯一观测值，
    箱内噪声更大 → ECE 更大。所以本项目的结论是：
    **不要报 ECE 的 bootstrap 置信区间**，只报点估计（compare_table 就是这么做的）；
    非要给区间不可时，用本函数并把 bias 一并写出来。

    另注：冻结边界估的是「固定这套分箱下的校准误差」，与每次重算边界的
    ECE 不是同一个估计对象（全样本上两者点估计相同，重采样下不同）。
    """
    ref = np.asarray(p_reference, dtype=float)
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))[1:-1]

    def _ece(y: np.ndarray, p: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        idx = np.digitize(p, edges)
        tot = 0.0
        for b in range(n_bins):
            m = idx == b
            k = int(m.sum())
            if k:
                tot += k * abs(y[m].mean() - p[m].mean())
        return float(tot / len(y))

    return _ece


# --------------------------------------------------------------------------
# 1. 聚类索引：一次构建，所有重采样共用
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _Clusters:
    """把 groups 压成 (codes, order, starts, counts)，避免每次迭代重新分组。

    order 是「按组排序后的行号」，starts/counts 给出每组在 order 里的区间。
    这样一次抽样只需要一次 np.repeat + 一次花式索引。
    """

    codes: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    counts: np.ndarray

    @property
    def n_groups(self) -> int:
        return len(self.counts)

    @property
    def n(self) -> int:
        return len(self.codes)

    def rows_for(self, idx: np.ndarray) -> np.ndarray:
        """给一组（可重复的）组下标，返回被选中的全部行号。"""
        cnt = self.counts[idx]
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        dest = np.empty(len(idx), dtype=np.int64)
        dest[0] = 0
        np.cumsum(cnt[:-1], out=dest[1:])
        return self.order[np.repeat(self.starts[idx] - dest, cnt) + np.arange(total)]


def _clusters(groups: Sequence[Any] | np.ndarray | pd.Series, unit: Unit = "series") -> _Clusters:
    n = len(groups)
    if unit == "game":
        # game 级 bootstrap：每场自成一组。只用于「证明 SE 确实被压窄了」的对照。
        codes = np.arange(n, dtype=np.int64)
    elif unit == "series":
        codes = pd.factorize(np.asarray(groups), sort=False)[0].astype(np.int64)
        if (codes < 0).any():
            raise ValueError("groups 含缺失值；series_id 必须每场都有（§6.1.2 红线 1）")
    else:
        raise ValueError(f"unit 只能是 'series' 或 'game'，收到 {unit!r}")
    counts = np.bincount(codes, minlength=codes.max() + 1 if n else 0).astype(np.int64)
    starts = np.empty(len(counts), dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts[:-1], out=starts[1:])
    order = np.argsort(codes, kind="stable").astype(np.int64)
    return _Clusters(codes=codes, order=order, starts=starts, counts=counts)


# --------------------------------------------------------------------------
# 2. block bootstrap
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BootResult:
    """bootstrap 结果。point 是**全样本点估计**，不是 bootstrap 均值。

    用全样本点估计而不是 boot.mean() 是故意的：后者带 bootstrap 偏差，
    报告里必须报可复现的那个数（别人用 sklearn 算一遍要能对上）。
    """

    point: float
    lo: float
    hi: float
    se: float
    bias: float
    alpha: float
    n_boot: int
    n_games: int
    n_series: int
    unit: str
    ci_method: str
    boot: np.ndarray = field(repr=False)

    @property
    def bias_ratio(self) -> float:
        """bias/se。**>0.5 就说明这个指标的 bootstrap 分布明显偏了**，
        percentile CI 不再以点估计为中心，报告时必须换 ci_method="basic" 或明说。
        实测 ECE 的 bias_ratio ≈ 1.9（ECE 是正偏统计量：重采样造成的重复样本
        会让每个等频 bin 内的 |acc−conf| 系统性变大），log loss / Brier / AUC 都 <0.05。
        """
        return self.bias / self.se if self.se else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return (self.lo, self.hi)

    @property
    def mde(self) -> float:
        """§6.3.1 做法 A：直接用经验分布反解 MDE = 2.8·sd(boot)，不经 n_eff。"""
        return T_STAR * self.se

    @property
    def p_two_sided(self) -> float:
        """bootstrap 双侧 p（Δ 是否跨 0）。仅对差值有意义。"""
        if self.n_boot == 0:
            return float("nan")
        frac = float(np.mean(self.boot > 0.0))
        return float(min(1.0, 2.0 * min(frac, 1.0 - frac)))

    def fmt(self, digits: int = 4) -> str:
        return f"{self.point:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def _boot_indices(cl: _Clusters, n_boot: int, seed: int) -> np.ndarray:
    """一次生成全部 (n_boot, n_groups) 的组下标。

    配对检验必须共用这批索引 —— 两个模型各抽各的会把
    Var(Δ) 从 Var(a−b) 放大成 Var(a)+Var(b)，在 ρ≈0.99 的同源模型上
    等于凭空造出 10 倍以上的方差（tests/test_compare.py 里实测了这一点）。
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, cl.n_groups, size=(n_boot, cl.n_groups), dtype=np.int64)


def _boot_stats(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn: MetricFn,
    cl: _Clusters,
    idx_mat: np.ndarray,
) -> np.ndarray:
    """在给定的重采样索引上算 metric。可分解指标走 O(n_series) 比值路径。"""
    per_game = _MEAN_METRICS.get(metric_fn)
    if per_game is not None:
        v = np.asarray(per_game(y, p), dtype=float)
        gsum = np.bincount(cl.codes, weights=v, minlength=cl.n_groups)
        cnt = cl.counts.astype(float)
        return gsum[idx_mat].sum(axis=1) / cnt[idx_mat].sum(axis=1)
    out = np.empty(len(idx_mat), dtype=float)
    for b, idx in enumerate(idx_mat):
        rows = cl.rows_for(idx)
        out[b] = float(metric_fn(y[rows], p[rows]))
    return out


def _make_result(
    point: float,
    boot: np.ndarray,
    cl: _Clusters,
    alpha: float,
    n_boot: int,
    unit: str,
    ci_method: str,
) -> BootResult:
    """percentile 是 §6.6.1 给的默认口径；basic 供有偏统计量（ECE）使用。

    basic CI = [2·point − q_{1−α/2}, 2·point − q_{α/2}]，
    它把一阶偏差消掉，代价是可能越出指标的定义域（ECE 可能出现负下界）。
    默认仍用 percentile，因为手册写的就是它，换口径必须是显式动作。
    """
    lo_q, hi_q = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if ci_method == "percentile":
        lo, hi = float(lo_q), float(hi_q)
    elif ci_method == "basic":
        lo, hi = float(2 * point - hi_q), float(2 * point - lo_q)
    else:
        raise ValueError(f"ci_method 只能是 'percentile' 或 'basic'，收到 {ci_method!r}")
    return BootResult(
        point=point,
        lo=lo,
        hi=hi,
        se=float(np.std(boot, ddof=1)) if n_boot > 1 else float("nan"),
        bias=float(np.mean(boot) - point) if n_boot else float("nan"),
        alpha=alpha,
        n_boot=n_boot,
        n_games=cl.n,
        n_series=cl.n_groups,
        unit=unit,
        ci_method=ci_method,
        boot=boot,
    )


def block_bootstrap_ci(
    y: np.ndarray | pd.Series,
    p: np.ndarray | pd.Series,
    metric_fn: MetricFn,
    groups: np.ndarray | pd.Series | Sequence[Any],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    unit: Unit = "series",
    ci_method: str = "percentile",
) -> BootResult:
    """series 级 block bootstrap（§6.6.1）：**整块重采样 series，不是重采样 game**。

    metric_fn 任意 (y, p) -> float，log_loss / brier / auc / ece 都能用。
    被抽中的 series 连同它内部的全部 game 一起进入重采样样本，
    因此系列内相关被自然保留；按 game 抽会把这份相关抹掉，CI 系统性偏窄。

    unit="game" 只为了做对照（算出偏窄的那个 SE 给人看），**不要用它出报告**。

    ⚙ 有偏指标的坑：**ECE 的 percentile CI 可能整段落在点估计之上**
    （实测 bias/se ≈ 1.9）。重采样会制造重复样本，让每个等频 bin 内的
    |acc−conf| 系统性变大，而 ECE 是绝对值之和、不会互相抵消。
    这不是 bug，是 ECE 这个统计量本身正偏（详见 fixed_bin_ece 的实测表）。
    **本项目的做法：ECE 只报点估计，不报区间。**
    log loss / Brier / AUC 的 bias/se 实测都在 ±0.1 以内，不受影响。
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) != len(groups):
        raise ValueError(f"长度不一致: y={len(y)} p={len(p)} groups={len(groups)}")
    cl = _clusters(groups, unit)
    point = float(metric_fn(y, p))
    boot = _boot_stats(y, p, metric_fn, cl, _boot_indices(cl, n_boot, seed))
    return _make_result(point, boot, cl, alpha, n_boot, unit, ci_method)


def paired_bootstrap_delta(
    y: np.ndarray | pd.Series,
    p_a: np.ndarray | pd.Series,
    p_b: np.ndarray | pd.Series,
    metric_fn: MetricFn,
    groups: np.ndarray | pd.Series | Sequence[Any],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    unit: Unit = "series",
    ci_method: str = "percentile",
) -> BootResult:
    """配对 block bootstrap：Δ = metric(a) − metric(b)，**两边共用同一组重采样索引**。

    符号约定与对比表一致：b 是 baseline，Δ>0 表示 a 的损失更大 = a 更差
    （对 log loss / Brier 这类越小越好的指标而言）。
    L2 vs L3 实测 Δ = +0.00160，即 L3 比 L2 好 0.00160。

    共用索引是这个函数存在的全部理由：a 与 b 的逐场损失相关系数常常 >0.99，
    Var(Δ) ≪ Var(a)+Var(b)。各抽各的等于故意把功效扔掉。
    """
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if not (len(y) == len(p_a) == len(p_b) == len(groups)):
        raise ValueError("y / p_a / p_b / groups 长度必须一致（必须逐场对齐后再传进来）")
    cl = _clusters(groups, unit)
    idx_mat = _boot_indices(cl, n_boot, seed)
    boot = _boot_stats(y, p_a, metric_fn, cl, idx_mat) - _boot_stats(
        y, p_b, metric_fn, cl, idx_mat
    )
    point = float(metric_fn(y, p_a)) - float(metric_fn(y, p_b))
    return _make_result(point, boot, cl, alpha, n_boot, unit, ci_method)


def se_comparison(
    y: np.ndarray | pd.Series,
    p_a: np.ndarray | pd.Series,
    p_b: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series | Sequence[Any],
    metric_fn: MetricFn | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """把 series 级与 game 级的 SE 都算出来摆在一起（验收要求之三）。

    ratio = se_series / se_game 应当 ≈ √deff。若 ratio ≈ 1，
    说明这个配对的 ICC ≈ 0，此时聚类不聚类无所谓 —— 但那是实测结论，不是默认假设。
    """
    metric_fn = metric_fn or metrics.log_loss
    s = paired_bootstrap_delta(y, p_a, p_b, metric_fn, groups, n_boot, seed, unit="series")
    g = paired_bootstrap_delta(y, p_a, p_b, metric_fn, groups, n_boot, seed, unit="game")
    d = loss_diff(y, p_a, p_b)
    pw = power.paired_power(d, np.asarray(groups))
    return {
        "delta": s.point,
        "se_series": s.se,
        "se_game": g.se,
        "se_ratio": s.se / g.se if g.se else float("nan"),
        "sqrt_deff": float(np.sqrt(pw["deff"])),
        "icc": pw["icc"],
        "ci_width_series": s.hi - s.lo,
        "ci_width_game": g.hi - g.lo,
    }


# --------------------------------------------------------------------------
# 3. Diebold-Mariano（聚类稳健 + Harvey 小样本修正）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DMResult:
    dm: float
    dm_harvey: float
    p_value: float
    mean_d: float
    se_cluster: float
    se_iid: float
    se_ratio: float
    harvey_factor: float
    df: int
    h: int
    n_games: int
    n_series: int

    def fmt(self) -> str:
        return f"DM*={self.dm_harvey:+.3f} (p={self.p_value:.4g}, df={self.df})"


def loss_diff(
    y: np.ndarray | pd.Series,
    p_a: np.ndarray | pd.Series,
    p_b: np.ndarray | pd.Series,
    per_game_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """逐场 loss 差 d_i = L(a) − L(b)。默认 log loss（§6.6.2：DM 必须传 log loss，
    不能用 dieboldmariano 包默认的平方误差 —— 主指标是 log loss）。"""
    f = per_game_fn or metrics.per_game_log_loss
    return np.asarray(f(np.asarray(y), np.asarray(p_a, dtype=float)), dtype=float) - np.asarray(
        f(np.asarray(y), np.asarray(p_b, dtype=float)), dtype=float
    )


def diebold_mariano(
    d: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series | Sequence[Any],
    h: int = 1,
) -> DMResult:
    """Diebold-Mariano 检验，聚类稳健方差 + Harvey et al. (1997) 小样本修正。

    **方差估计的选择（必须说明的那一条）**：
    教科书 DM 用 Newey-West 估 d_t 的长程方差，前提是依赖结构沿时间轴呈 MA(h−1)。
    本项目的依赖结构不是时间序列式的 —— 它来自 series 聚类：
    同一 Bo5 的 5 局几乎同时发生、彼此强相关，而相邻两个 series 之间基本独立
    （实测 L2→L3 的 series 级 ICC = 0.3120，见 reports/stage3_power_and_icc.md）。
    所以这里用 **CR1 聚类稳健（= block 形式的三明治估计）**：

        V̂(d̄) = c · Σ_g ( Σ_{i∈g}(d_i − d̄) )² / n²,    c = G/(G−1)

    c 就是 CR1 的有限样本修正 [G/(G−1)]·[(N−1)/(N−K)]，本处回归只有截距 K=1，
    第二个因子退化为 1。这个估计等价于「以 series 为块、块长不等的 block 方差」，
    不需要选 bandwidth，也不会像 Newey-West 那样在块内重复计一遍相关。
    **不再额外叠加 Newey-West 滞后项**，否则同一份系列内相关会被算两次。

    Harvey 修正：DM* = DM · √[(m + 1 − 2h + h(h−1)/m) / m]，参照 t_(m−1)。
    原文的 m 是「预测条数」，这里取 **m = series 数**而不是 game 数 ——
    独立单位是 series，用 game 数会把小样本修正做得过轻（df 虚高）。
    h=1 时该因子 = √((m−1)/m)，m 上万时几乎为 1，这个修正只在
    分 fold / 分赛区的小子集上才真的起作用（正是最容易过度解读的地方）。

    边界情况（§6.6.4）：d 全同号且组间无变异时 V̂=0，返回 ±inf 与 p=0，
    按符号判定，不得当成噪声。
    """
    d = np.asarray(d, dtype=float)
    if len(d) != len(groups):
        raise ValueError(f"d 与 groups 长度不一致: {len(d)} vs {len(groups)}")
    cl = _clusters(groups, "series")
    n, m = cl.n, cl.n_groups
    if n < 2 or m < 2:
        raise ValueError(f"样本太小，无法做 DM 检验: n_games={n}, n_series={m}")
    mean_d = float(d.mean())
    resid = d - mean_d
    gsum = np.bincount(cl.codes, weights=resid, minlength=m)
    c_cr1 = m / (m - 1.0)
    var_cluster = c_cr1 * float((gsum**2).sum()) / n**2
    se_cluster = float(np.sqrt(var_cluster))
    se_iid = float(np.std(d, ddof=1) / np.sqrt(n))

    harvey = float(np.sqrt(max((m + 1 - 2 * h + h * (h - 1) / m) / m, 0.0)))
    if se_cluster == 0.0:
        dm = float(np.inf) * np.sign(mean_d) if mean_d != 0 else 0.0
        dm_h = dm
        pval = 0.0 if mean_d != 0 else 1.0
    else:
        dm = mean_d / se_cluster
        dm_h = dm * harvey
        pval = float(2.0 * stats.t.sf(abs(dm_h), df=m - 1))
    return DMResult(
        dm=float(dm),
        dm_harvey=float(dm_h),
        p_value=pval,
        mean_d=mean_d,
        se_cluster=se_cluster,
        se_iid=se_iid,
        se_ratio=se_cluster / se_iid if se_iid else float("nan"),
        harvey_factor=harvey,
        df=int(m - 1),
        h=int(h),
        n_games=int(n),
        n_series=int(m),
    )


def cluster_rho(
    d: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series | Sequence[Any],
) -> dict[str, float]:
    """三种口径的「组内相关」摆在一起看。**它们在本项目的数据上并不一致。**

    1. `icc_anova`      —— power.icc_oneway 的单向随机效应 ICC(1)（阶段 3 用的那个）
    2. `rho_pairwise`   —— 同一 series 内两两配对的矩估计，分母用全样本方差，
                           不假设方差齐性；它正好是进入 deff = 1 + (m̃−1)ρ 的那个 ρ，
                           m̃ = Σm_g²/n（不等组大小下正确的 deff 用 m̃，不是 m̄）
    3. `rho_cr1`        —— 由 CR1 三明治方差反解：(V_cr1/V_iid − 1)·n/Σm_g(m_g−1)

    2 与 3 在代数上几乎等价（实测只差 2e-5），同时给出是当交叉检验；
    有信息量的对比是 1 对 2/3。

    **[实测 2026-09-19，data/preds_ladder.parquet，23,060 场 / 12,274 series]**
    L2 vs L3 的 d 上三者是 **0.3120 / 0.0266 / 0.0266**，
    对应 deff **1.274 / 1.039 / 1.039** —— ICC(1) 把 deff 高估了 23%。
    （另一个参照：只在 m≥2 的 series 内部算 Pearson r = 0.0344，
    比 0.0266 略大，因为分母换成了多局样本自己的方差。）
    对照组：L1 vs L2 上三者是 0.1154 / 0.0665 / 0.0665，deff 1.101 / 1.098，
    几乎一致 —— **说明问题不是“ICC(1) 总是偏大”，而是它在某些配对上崩掉。**
    原因已定位：

      - OOS 里 5,995/12,274 个 series 是单局（Bo1），它们对 SSB 贡献了完整的
        (d_i − d̄)²，却对 SSW 贡献 0，ICC(1) 分不清「组间差异」和「单局噪声」；
      - 而单局比赛的 var(d) = 0.01312，多局 series 内的 var(d) = 0.00642，
        **单局方差是 2.04 倍**（Bo1 多见于强弱悬殊的常规赛，d 的尾巴更肥）；
      - 于是 MSB 被单局的大方差抬高，MSW 只能看到多局 series 的小方差，
        (MSB−MSW) 被当成了组间方差分量。

    反证：把真实的 series 结构配上**完全独立**、但单局方差 ×1.5 的噪声喂给
    `icc_oneway`，它报 ICC = 0.2493（真值 0）。同样的结构配同方差 iid 噪声报 0.0000。
    所以 0.3120 主要是**异方差伪装成的组内相关**，不是真的系列内相关。

    后果：`deff = 1 + (m̄−1)·ICC` 这条路线在 L2 vs L3 上把 SE 高估了约 9%
    （√deff_anova = 1.129，而 block bootstrap 实测 se_series/se_game = 1.035、
    CR1/iid = 1.019），对应 MDE 从 0.00173 抬到 0.00188。
    好消息是裁决不翻：|Δ| = 0.00160 两边都低于 MDE，
    但余量从 15% 缩到 7%，阶段 3 报告里那句“低于可检出下限”比它看上去的要勉强。
    **对比表的 `>MDE?` 一律用 bootstrap 经验分布反解的 MDE（§6.3.1 做法 A），
    不用 ICC 路线的 MDE。** ICC 路线仍照 §6.3.6 打进表脚，两个数并排给读者看。
    """
    d = np.asarray(d, dtype=float)
    cl = _clusters(groups, "series")
    n, m = cl.n, cl.n_groups
    mu = float(d.mean())
    resid = d - mu
    gsum = np.bincount(cl.codes, weights=resid, minlength=m)
    gsq = np.bincount(cl.codes, weights=resid**2, minlength=m)
    n_pairs = float((cl.counts.astype(float) ** 2).sum() - n)  # = Σ m_g(m_g−1)
    var_pop = float(np.mean(resid**2))
    rho_pair = (
        float((gsum**2 - gsq).sum() / (n_pairs * var_pop)) if n_pairs and var_pop else 0.0
    )
    var_iid = float(np.var(d, ddof=1) / n)
    var_cr1 = (m / (m - 1.0)) * float((gsum**2).sum()) / n**2 if m > 1 else float("nan")
    rho_cr1 = (var_cr1 / var_iid - 1.0) * n / n_pairs if (n_pairs and var_iid) else float("nan")
    st = power.icc_oneway(d, np.asarray(groups))
    m_tilde = float((cl.counts.astype(float) ** 2).sum() / n)
    return {
        "icc_anova": float(st["icc"]),
        "rho_pairwise": rho_pair,
        "rho_cr1": float(rho_cr1),
        "m_bar": float(n / m),
        "m_tilde": m_tilde,
        "n_pairs": n_pairs,
        "deff_anova": power.design_effect(float(st["m_bar"]), float(st["icc"])),
        "deff_pairwise": 1.0 + (m_tilde - 1.0) * rho_pair,
        "var_ratio_cr1": var_cr1 / var_iid if var_iid else float("nan"),
        "n_games": float(n),
        "n_series": float(m),
    }


# --------------------------------------------------------------------------
# 4. Benjamini-Hochberg
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BHResult:
    reject: np.ndarray
    qvalues: np.ndarray
    alpha: float
    n_tests: int
    n_reject: int
    threshold: float


def benjamini_hochberg(pvals: Iterable[float], alpha: float = 0.05) -> BHResult:
    """BH (1995) 线性升阶法，控制 FDR ≤ alpha。

    为什么不是 Bonferroni：§6.6.5 —— 8 模型 × 5 赛区 × 3 指标 = 120 次检验，
    Bonferroni 把 α 压到 4e-4，而本项目真实效应量只有 0.002 量级、
    MDE 本来就贴着效应走，Bonferroni 下功效基本归零。探索性分析要的是 FDR。

    返回的 qvalues 是**强制单调**后的 BH 调整 p 值（q_(i) = min_{k≥i} m/k·p_(k)），
    否则会出现「p 更小的检验 q 反而更大」这种没法向人解释的东西。
    reject 与 qvalues ≤ alpha 恒等 —— 这是 BH 阶跃规则的等价形式。

    注意：primary comparison（预注册 §1，唯一一个）**不进 BH**，
    它是事先声明的单一检验；把它混进 FDR 池等于自己惩罚自己。
    """
    p = np.asarray(list(pvals), dtype=float)
    m = len(p)
    if m == 0:
        empty = np.empty(0)
        return BHResult(empty.astype(bool), empty, alpha, 0, 0, float("nan"))
    if np.isnan(p).any():
        raise ValueError("p 值含 NaN；检验失败的比较请先剔除，不要塞进 FDR 池")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p 值必须落在 [0, 1]")
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = np.minimum.accumulate((m / ranks * ps)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    below = ps <= ranks / m * alpha
    k = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0
    reject = np.zeros(m, dtype=bool)
    if k:
        reject[order[:k]] = True
    return BHResult(
        reject=reject,
        qvalues=q,
        alpha=alpha,
        n_tests=m,
        n_reject=k,
        threshold=float(ps[k - 1]) if k else 0.0,
    )


# --------------------------------------------------------------------------
# 5. §6.4.1 的 13 列对比表
# --------------------------------------------------------------------------
_INTERNATIONAL_FALLBACK: tuple[str, ...] = ("WLDs", "MSI", "FST", "EWC", "IEM", "IWCI", "MSC", "ASI")
TIER1_YAML = "eval/tier1_leagues.yaml"


def international_leagues(path: str = TIER1_YAML) -> tuple[str, ...]:
    """国际赛名单以 eval/tier1_leagues.yaml 为唯一真源，不照抄手册。

    手册 §6.4.1 写的是「MSI/WLDs/FST/IC/EWC」，但预注册修订 A1 实测
    IC = Iberian Cup（西葡二级/青训队，tier 0）已被剔除，且 WLDS 的正确写法是 WLDs。
    硬编码手册那一版会让「国际赛子集」这一列悄悄混进 244 场二级赛并漏掉世界赛。
    yaml 读不到时才退回常量（离线跑测试用），退回值与当前 yaml 一致。
    """
    try:
        import yaml

        with open(path) as fh:
            cfg = yaml.safe_load(fh) or {}
        vals = cfg.get("international") or []
        if vals:
            return tuple(str(v) for v in vals)
    except (OSError, ImportError, ValueError):
        pass
    return _INTERNATIONAL_FALLBACK


_RATING_CANDIDATES: tuple[str, ...] = ("rating_diff", "team_elo_diff", "elo_diff")
PLACEHOLDER = "—（调用方填）"

COLUMNS: tuple[str, ...] = (
    "n games/series",
    "LogLoss (95% CI)",
    "ΔLL vs {base} (DM p)",
    ">MDE?",
    "Brier (Rel/Res/Unc)",
    "AUC",
    "ECE",
    "calib slope",
    "|Δrating|<50 LL",
    "国际赛 LL",
    "训练耗时 T_state/T_refit/T_fit",
    "推理延迟 p50/p99",
    "可解释性",
)


def align_predictions(preds_long: pd.DataFrame, models: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """长表 → (spine, P)。spine 一行一场，P 是 game × model 的概率矩阵。

    配对检验的前提是**逐场对齐**。这里用 game_id 硬对齐并断言三件事：
    每个模型的 game_id 唯一、模型间 game_id 集合相同、同一场的 y 在各模型下一致。
    对不齐而不报错的话，Δ 会变成两个不同样本的均值差，DM 与配对 bootstrap 全部失效。
    """
    need = {"game_id", "series_id", "model", "y", "p"}
    missing = need - set(preds_long.columns)
    if missing:
        raise ValueError(f"preds_long 缺列: {sorted(missing)}")
    parts: dict[str, pd.Series] = {}
    spine: pd.DataFrame | None = None
    for m in models:
        sub = preds_long[preds_long.model == m]
        if sub.game_id.duplicated().any():
            raise ValueError(f"模型 {m} 有重复 game_id（同一场落进了多个 fold？）")
        sub = sub.set_index("game_id")
        if spine is None:
            keep = [c for c in sub.columns if c not in ("model", "p")]
            spine = sub[keep]
        else:
            if len(sub) != len(spine) or not sub.index.equals(spine.index):
                sub = sub.reindex(spine.index)
                if sub.p.isna().any():
                    raise ValueError(f"模型 {m} 与 {models[0]} 的 game_id 集合不同，无法配对")
            # y 的一致性必须无条件检：索引相同不等于标签相同。
            # 两份预测来自不同脚本时，y 反了号而 game_id 对得上是真实发生过的事故，
            # 而且它不会报错，只会把一个模型的 log loss 变成 1.4。
            if not np.array_equal(sub.y.to_numpy(), spine.y.to_numpy()):
                raise ValueError(f"模型 {m} 的 y 与 {models[0]} 不一致，数据对齐已经坏了")
        parts[m] = sub.p
    assert spine is not None
    return spine, pd.DataFrame(parts, index=spine.index)


def _subset_ll(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    if mask.sum() == 0:
        return float("nan"), 0
    return float(metrics.log_loss(y[mask], p[mask])), int(mask.sum())


def compare_table(
    preds_long: pd.DataFrame,
    baseline: str = "L3",
    *,
    models: Sequence[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    rating_col: str | None = None,
    rating_threshold: float = 50.0,
    international: Iterable[str] | None = None,
    extras: dict[str, dict[str, str]] | None = None,
    echo: bool = True,
) -> pd.DataFrame:
    """§6.4.1 的 13 列模型对比表。输入 data/preds_ladder.parquet 那种长表。

    列：n / LogLoss(95%CI) / ΔLL vs baseline(DM p) / >MDE? / Brier(Rel/Res/Unc) /
    AUC / ECE / calib slope / |Δrating|<50 子集 LL / 国际赛子集 LL /
    训练耗时 / 推理延迟 / 可解释性。后三列拿不到，留占位由调用方经 extras 填。

    `|Δrating|<50` 需要长表里带一列评分差（rating_diff / team_elo_diff）。
    preds_ladder.parquet 目前不带，所以这一列会显示「无 rating 列」——
    这是诚实的缺失，不是 0，不能当成"这一档没问题"。

    返回值是**给人看的字符串表**；机器要用的数值在 `tbl.attrs["raw"]`（DataFrame），
    功效脚注在 `tbl.attrs["footer"]`（str）与 `tbl.attrs["power"]`（dict）。
    echo=True 时直接把脚注打到 stdout —— §6.3.6 要求每张表都必须带着 MDE 出现。
    """
    if models is None:
        seen = preds_long.model.drop_duplicates().tolist()
        models = seen
    models = list(models)
    if baseline not in models:
        raise ValueError(f"baseline={baseline!r} 不在 models 里: {models}")
    spine, P = align_predictions(preds_long, models)
    y = spine.y.to_numpy()
    groups = spine.series_id.to_numpy()
    p_base = P[baseline].to_numpy(dtype=float)

    intl = set(international) if international is not None else set(international_leagues())
    intl_mask = (
        spine.league_canon.isin(intl).to_numpy()
        if "league_canon" in spine.columns
        else np.zeros(len(spine), dtype=bool)
    )
    if rating_col is None:
        rating_col = next((c for c in _RATING_CANDIDATES if c in spine.columns), None)
    close_mask = (
        (spine[rating_col].abs() < rating_threshold).to_numpy()
        if rating_col is not None
        else None
    )

    extras = extras or {}
    rows: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, Any]] = {}
    pw_all: dict[str, dict[str, float]] = {}
    foot: list[str] = []

    for m in models:
        p = P[m].to_numpy(dtype=float)
        ll = block_bootstrap_ci(y, p, metrics.log_loss, groups, n_boot, seed, alpha)
        dec = metrics.brier_decomposition(y, p)
        bs = metrics.brier(y, p)
        a = metrics.auc(y, p)
        e = metrics.ece(y, p)
        slope = (
            metrics.calibration_slope(y, p)[0]
            if float(np.ptp(p)) > 0
            else float("nan")
        )
        ll_intl, n_intl = _subset_ll(y, p, intl_mask)
        if close_mask is None:
            ll_close, n_close = float("nan"), 0
        else:
            ll_close, n_close = _subset_ll(y, p, close_mask)

        r: dict[str, Any] = {
            "n_games": ll.n_games,
            "n_series": ll.n_series,
            "log_loss": ll.point,
            "ll_lo": ll.lo,
            "ll_hi": ll.hi,
            "ll_se": ll.se,
            "brier": bs,
            "reliability": dec["reliability"],
            "resolution": dec["resolution"],
            "uncertainty": dec["uncertainty"],
            "auc": a,
            "ece": e,
            "calib_slope": slope,
            "ll_close": ll_close,
            "n_close": n_close,
            "ll_intl": ll_intl,
            "n_intl": n_intl,
        }
        if m == baseline:
            r |= {
                "delta_ll": 0.0, "delta_lo": np.nan, "delta_hi": np.nan,
                "dm_stat": np.nan, "dm_p": np.nan, "mde_boot": np.nan,
                "mde_icc": np.nan, "icc": np.nan, "deff": np.nan,
                "n_eff": np.nan, "exceeds_mde": False,
            }
            dcell, mcell = "基准", "—"
        else:
            dlt = paired_bootstrap_delta(
                y, p, p_base, metrics.log_loss, groups, n_boot, seed, alpha
            )
            d = loss_diff(y, p, p_base)
            dm = diebold_mariano(d, groups)
            pw = power.paired_power(d, groups)
            rho = cluster_rho(d, groups)
            pw_all[m] = pw | {f"rho_{k}": v for k, v in rho.items()}
            # §6.3.1 做法 A：判定用 bootstrap 经验分布反解的 MDE。
            # 不用 ICC 路线的 MDE —— cluster_rho() 的实测说明 ICC(1) 在本数据上
            # 被单局 series 的异方差污染，会系统性高估 deff。
            exceeds = abs(dlt.point) > dlt.mde
            r |= {
                "delta_ll": dlt.point, "delta_lo": dlt.lo, "delta_hi": dlt.hi,
                "delta_se": dlt.se, "dm_stat": dm.dm_harvey, "dm_p": dm.p_value,
                "mde_boot": dlt.mde, "mde_icc": pw["mde"], "icc": pw["icc"],
                "deff": pw["deff"], "n_eff": pw["n_eff"], "exceeds_mde": exceeds,
            }
            dcell = (
                f"{dlt.point:+.5f} [{dlt.lo:+.5f}, {dlt.hi:+.5f}] "
                f"(DM p={dm.p_value:.2g})"
            )
            mcell = "✓" if exceeds else "✗【低于可检出下限】"
            warn = ""
            if rho["icc_anova"] - rho["rho_pairwise"] > 0.05:
                warn = (
                    f"\n      ⚠ ICC(1) 不可信：ICC_anova={rho['icc_anova']:.4f}，"
                    f"但实测系列内两两相关 "
                    f"ρ_pairwise={rho['rho_pairwise']:.4f}、CR1 反解 ρ={rho['rho_cr1']:.4f}，"
                    f"MDE 相应差 {dlt.mde / pw['mde'] - 1:+.0%}。"
                    f"见 compare.cluster_rho 的实测说明（单局 series 的异方差伪装成了组内相关）。"
                )
            foot.append(
                f"  {m} vs {baseline}: {power.footer(pw)}  "
                f"MDE_boot(2.8·sd)={dlt.mde:.5f}  "
                f"{'超过 ✓' if exceeds else '未超过 ✗'}{warn}"
            )

        ex = extras.get(m, {})
        rows[m] = {
            COLUMNS[0]: f"{r['n_games']:,} / {r['n_series']:,}",
            COLUMNS[1]: f"{ll.point:.4f} [{ll.lo:.4f}, {ll.hi:.4f}]",
            COLUMNS[2].format(base=baseline): dcell,
            COLUMNS[3]: mcell,
            COLUMNS[4]: (
                f"{bs:.4f} ({dec['reliability']:.4f}/{dec['resolution']:.4f}/"
                f"{dec['uncertainty']:.4f})"
            ),
            COLUMNS[5]: f"{a:.4f}",
            COLUMNS[6]: f"{e:.4f}",
            COLUMNS[7]: "—" if np.isnan(slope) else f"{slope:.3f}",
            COLUMNS[8]: (
                "无 rating 列（传 rating_col）"
                if close_mask is None
                else f"{ll_close:.4f} (n={n_close:,})"
            ),
            COLUMNS[9]: "—" if n_intl == 0 else f"{ll_intl:.4f} (n={n_intl:,})",
            COLUMNS[10]: ex.get("train_time", PLACEHOLDER),
            COLUMNS[11]: ex.get("latency", PLACEHOLDER),
            COLUMNS[12]: ex.get("interpretability", PLACEHOLDER),
        }
        raw[m] = r

    tbl = pd.DataFrame(rows).T
    tbl.index.name = "model"
    raw_df = pd.DataFrame(raw).T
    # §6.6.5：除 primary comparison 外，本表每一行都是 exploratory，必须过 BH。
    # 这里不替调用方判断哪一行是 primary —— 全部进池是保守方向，
    # 真正的 primary 由 experiments/preregistration.md 指定，调用方自行标注。
    expl = [m for m in models if m != baseline]
    bh_line = ""
    if expl:
        bh = benjamini_hochberg([raw[m]["dm_p"] for m in expl], alpha)
        raw_df["bh_q"] = pd.Series(dict(zip(expl, bh.qvalues, strict=True)))
        raw_df["bh_reject"] = pd.Series(dict(zip(expl, bh.reject, strict=True)))
        bh_line = (
            f"BH-FDR(alpha={alpha}) 在 {bh.n_tests} 个 exploratory 比较上拒绝 "
            f"{bh.n_reject} 个（q 值在 attrs['raw'].bh_q）。"
        )
    header = (
        f"【功效脚注】baseline={baseline}，n_boot={n_boot}，seed={seed}，"
        f"CI={100 * (1 - alpha):.0f}% series 级 block bootstrap\n"
        "每个配对各自实测 ICC —— ICC 不是数据集常数（reports/stage3_power_and_icc.md 第 1 节）"
    )
    tail = (
        f"{bh_line}\n"
        "凡 |Δ| < MDE 的行已标注【低于可检出下限】，不得解读为改进（PLAN.md §6.3.6）。\n"
        "primary comparison 只有一个（experiments/preregistration.md §1），"
        "其余全部 exploratory，用 benjamini_hochberg() 控 FDR 后才准解读。"
    )
    tbl.attrs["footer"] = "\n".join([header, *foot, tail]) if foot else header + "\n" + tail
    tbl.attrs["raw"] = raw_df
    tbl.attrs["power"] = pw_all
    tbl.attrs["baseline"] = baseline
    if echo:
        print(tbl.attrs["footer"])
    return tbl


def format_compare_table(tbl: pd.DataFrame, max_colwidth: int = 56) -> str:
    """把表和脚注拼成一整块可直接贴进报告的文本。脚注缺失时直接报错 —— 验收⑤。"""
    if "footer" not in tbl.attrs:
        raise ValueError("这张表没有功效脚注，不得进报告（PLAN.md §6.3.6）")
    body = tbl.to_string(max_colwidth=max_colwidth)
    return f"{body}\n\n{tbl.attrs['footer']}"
