"""阶段 4：显著性检验套件的验收测试（PLAN.md §6.6）。

这一套测试的立场：**检验工具本身也要被检验**。
一个偷偷把标准误算窄 10% 的 bootstrap 不会报错、不会崩，
它只会让你在 reports/ 里写下一个不存在的"改进"。所以这里的重点不是
"函数能跑通"，而是三件可证伪的事：

  1. 95% CI 真的覆盖 95%（合成真值 + 400 次重复实测覆盖率）；
  2. 配对 bootstrap 真的共用了同一组索引（不共用会把方差放大，这里量出来）；
  3. series 级与 game 级的标准误真的不同（把两个数摆在一起）。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from lolwp.eval import metrics, power
from lolwp.eval.compare import (
    align_predictions,
    benjamini_hochberg,
    block_bootstrap_ci,
    cluster_rho,
    compare_table,
    diebold_mariano,
    fixed_bin_ece,
    format_compare_table,
    international_leagues,
    loss_diff,
    paired_bootstrap_delta,
    se_comparison,
)

PREDS = "data/preds_ladder.parquet"
needs_preds = pytest.mark.skipif(
    not os.path.exists(PREDS), reason="需先跑 scripts 生成 data/preds_ladder.parquet"
)


# =========================================================================
# 合成场景：真值可解析计算，用来测覆盖率
# =========================================================================
# 设计刻意贴近真实结构：**同一 series 内两队不变 ⇒ 预测 p 在系列内完全相同**，
# 这正是 series 聚类的物理来源。series 长度分布取 {1,1,2,2,3,3,4,5}
# （实测 OOS 是 {1:5995, 2:3036, 3:2363, 4:496, 5:384}，同样以 Bo1/Bo3 为主）。
SX, SU, SHIFT = 0.6, 0.7, 0.15
SIZES = np.array([1, 1, 2, 2, 3, 3, 4, 5])


def _true_logloss(shift: float = 0.0) -> float:
    """总体真值，2D Gauss-Hermite 求积（60×60 节点，机器精度级）。

    DGP：x ~ N(0, SX²) 是 series 级的"预测 logit"，u ~ N(0, SU²) 是 series 级的
    "模型没看到的那部分实力差"，真实胜率 q = σ(x+u)，而模型只报 p = σ(x+shift)。
    于是每场的期望损失 = q·log(1+e^{-z}) + (1−q)·log(1+e^{z})，z = x+shift。
    模型 a（shift=0）与模型 b（shift=0.15）的真实 ΔlogLoss 因此是可算的常数。
    """
    t, w = np.polynomial.hermite.hermgauss(60)
    x_grid, u_grid = np.meshgrid(np.sqrt(2) * SX * t, np.sqrt(2) * SU * t, indexing="ij")
    weight = np.outer(w, w) / np.pi
    q = 1.0 / (1.0 + np.exp(-(x_grid + u_grid)))
    z = x_grid + shift
    loss = q * np.logaddexp(0.0, -z) + (1 - q) * np.logaddexp(0.0, z)
    return float((weight * loss).sum())


TRUE_LL_A = _true_logloss(0.0)
TRUE_LL_B = _true_logloss(SHIFT)
TRUE_DELTA = TRUE_LL_A - TRUE_LL_B


def _synth(seed: int, n_series: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.choice(SIZES, size=n_series)
    groups = np.repeat(np.arange(n_series), sizes)
    xg = rng.normal(0.0, SX, n_series)
    ug = rng.normal(0.0, SU, n_series)
    x, u = xg[groups], ug[groups]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(x + u))))
    p_a = 1.0 / (1.0 + np.exp(-x))
    p_b = 1.0 / (1.0 + np.exp(-(x + SHIFT)))
    return y, p_a, p_b, groups


@pytest.fixture(scope="module")
def coverage_run() -> dict[str, float]:
    """一次跑完 400 次重复，三条覆盖率共用，避免重复生成数据。

    种子全部写死 → 这个数字是确定性的，不会今天过明天挂。
    """
    reps, n_boot = 400, 600
    hit_s = hit_g = hit_d = 0
    narrower = 0
    se_s: list[float] = []
    se_g: list[float] = []
    for r in range(reps):
        y, p_a, p_b, g = _synth(7_600_000 + r)
        s = block_bootstrap_ci(y, p_a, metrics.log_loss, g, n_boot, r)
        gm = block_bootstrap_ci(y, p_a, metrics.log_loss, g, n_boot, r, unit="game")
        d = paired_bootstrap_delta(y, p_a, p_b, metrics.log_loss, g, n_boot, r)
        hit_s += int(s.lo <= TRUE_LL_A <= s.hi)
        hit_g += int(gm.lo <= TRUE_LL_A <= gm.hi)
        hit_d += int(d.lo <= TRUE_DELTA <= d.hi)
        narrower += int(gm.se < s.se)
        se_s.append(s.se)
        se_g.append(gm.se)
    return {
        "reps": reps,
        "cov_series": hit_s / reps,
        "cov_game": hit_g / reps,
        "cov_delta": hit_d / reps,
        "frac_game_narrower": narrower / reps,
        "se_ratio": float(np.mean(se_s) / np.mean(se_g)),
    }


def test_block_bootstrap_covers_95pct(coverage_run: dict[str, float]) -> None:
    """验收：合成真值已知，400 次重复的实际覆盖率必须落在 92%–98%。"""
    cov = coverage_run["cov_series"]
    assert 0.92 <= cov <= 0.98, f"series 级 block bootstrap 覆盖率 {cov:.3f} 越界 {coverage_run}"


def test_paired_delta_ci_covers_true_delta(coverage_run: dict[str, float]) -> None:
    """配对 Δ 的 CI 同样要覆盖 95%。真值 Δ 由 Gauss-Hermite 给出，不是模拟出来的。"""
    cov = coverage_run["cov_delta"]
    assert abs(TRUE_DELTA + 0.002594) < 1e-5, f"真值算错了: {TRUE_DELTA}"
    assert 0.92 <= cov <= 0.98, f"配对 Δ 覆盖率 {cov:.3f} 越界 {coverage_run}"


def test_game_level_bootstrap_is_systematically_narrower(coverage_run: dict[str, float]) -> None:
    """§6.6.1 那句「以 game 为单位重抽会让 CI 系统性偏窄」的实测证明。

    「系统性」= 不是平均偏窄，而是**几乎每一次重复都偏窄**。
    """
    assert coverage_run["frac_game_narrower"] > 0.95, coverage_run
    assert coverage_run["se_ratio"] > 1.05, coverage_run
    # 偏窄的代价直接落在覆盖率上
    assert coverage_run["cov_game"] < coverage_run["cov_series"], coverage_run


def test_paired_bootstrap_must_share_resample_indices() -> None:
    """不共用索引会把 Var(Δ) 从 Var(a−b) 放大成 Var(a)+Var(b)。这里量出放大倍数。

    两个同源模型的逐场损失相关系数极高，共用索引后 Δ 的 SE 应当远小于
    「各抽各的」。若某天有人把共用索引改坏了，这个断言会立刻炸。
    """
    y, p_a, p_b, g = _synth(20260919)
    shared = paired_bootstrap_delta(y, p_a, p_b, metrics.log_loss, g, 2000, 0)
    a = block_bootstrap_ci(y, p_a, metrics.log_loss, g, 2000, 1)
    b = block_bootstrap_ci(y, p_b, metrics.log_loss, g, 2000, 2)
    independent_se = float(np.sqrt(a.se**2 + b.se**2))
    assert shared.se < independent_se / 5, (
        f"共用索引 SE={shared.se:.6f} 应远小于独立重采样 SE={independent_se:.6f}"
    )
    # 同一 seed 两次调用必须逐位相同（可复现三件套，§6.10.3）
    again = paired_bootstrap_delta(y, p_a, p_b, metrics.log_loss, g, 2000, 0)
    assert shared.se == again.se and shared.lo == again.lo and shared.hi == again.hi


def test_point_estimate_is_full_sample_not_boot_mean() -> None:
    """报告里的点估计必须能被别人用一行 log_loss 复算出来，不能带 bootstrap 偏差。"""
    y, p_a, _, g = _synth(7)
    r = block_bootstrap_ci(y, p_a, metrics.log_loss, g, 500, 0)
    assert r.point == pytest.approx(metrics.log_loss(y, p_a), abs=1e-15)
    assert r.lo < r.point < r.hi


def test_works_for_any_metric() -> None:
    """要求 1：log_loss / brier / auc / ece 任意指标都能用。"""
    y, p_a, _, g = _synth(11)
    for fn in (metrics.log_loss, metrics.brier, metrics.auc):
        r = block_bootstrap_ci(y, p_a, fn, g, 400, 0)
        assert r.lo <= r.point <= r.hi, fn
        assert np.isfinite(r.se) and r.se > 0, fn
        assert abs(r.bias_ratio) < 0.4, (fn, r.bias_ratio)
        assert r.n_series == 600 and r.n_games == len(y)


def test_ece_bootstrap_ci_is_not_trustworthy() -> None:
    """**ECE 不能报 bootstrap 置信区间。** 这条测试把原因钉死。

    分箱 ECE 是正偏估计量（E|acc−conf| ≥ |E acc − E conf|），
    而 bootstrap 重采样只有 ≈63% 的唯一观测值、箱内噪声更大，于是 ECE 系统性变大。
    实测 bias/se = +1.88，**percentile CI 整段落在点估计之上**；
    改用 basic CI 也救不回来（偏差比整个下尾还长，区间反而整段掉到点估计下方）。
    冻结分箱边界（fixed_bin_ece）只把 bias/se 从 1.88 压到 1.74，
    能让 CI 重新含住点估计，但偏还在 —— 所以 compare_table 只报 ECE 点估计。
    """
    y, p_a, _, g = _synth(11)
    pct = block_bootstrap_ci(y, p_a, metrics.ece, g, 400, 0)
    basic = block_bootstrap_ci(y, p_a, metrics.ece, g, 400, 0, ci_method="basic")
    assert pct.bias_ratio > 1.5, pct.bias_ratio
    assert pct.lo > pct.point, "percentile CI 本该整段落在点估计之上"
    assert basic.hi < basic.point, "basic CI 也救不回来"
    assert basic.ci_method == "basic" and pct.ci_method == "percentile"

    fixed = block_bootstrap_ci(y, p_a, fixed_bin_ece(p_a), g, 400, 0)
    assert fixed.point == pytest.approx(pct.point, abs=1e-12)   # 全样本上两者同值
    assert fixed.lo <= fixed.point <= fixed.hi                  # 冻结边界后 CI 含住点估计
    assert 1.0 < fixed.bias_ratio < pct.bias_ratio              # 但偏差只小了一点

    with pytest.raises(ValueError, match="ci_method"):
        block_bootstrap_ci(y, p_a, metrics.ece, g, 10, 0, ci_method="bca")


def test_mean_metric_fast_path_matches_generic_path() -> None:
    """log_loss 走 O(n_series) 比值快路径，结果必须与逐次重算指标的慢路径一致。"""
    y, p_a, _, g = _synth(13)

    def slow_log_loss(yy: np.ndarray, pp: np.ndarray) -> float:  # 未登记 → 走慢路径
        return metrics.log_loss(yy, pp)

    fast = block_bootstrap_ci(y, p_a, metrics.log_loss, g, 300, 5)
    slow = block_bootstrap_ci(y, p_a, slow_log_loss, g, 300, 5)
    assert np.allclose(fast.boot, slow.boot, atol=1e-12)


# =========================================================================
# Diebold-Mariano
# =========================================================================
def test_dm_equals_paired_t_when_every_series_is_one_game() -> None:
    """全是单局 series 时，CR1 的修正因子 G/(G−1) 恰好抵消 (n−1)/n，
    聚类 SE 与 iid SE 必须**精确相等**。这是 CR1 实现正确性的解析锚点。"""
    rng = np.random.default_rng(3)
    d = rng.normal(0.01, 0.1, 800)
    r = diebold_mariano(d, np.arange(800))
    assert r.se_cluster == pytest.approx(r.se_iid, rel=1e-12)
    assert r.dm == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(800)), rel=1e-12)
    assert r.df == 799


def test_dm_cluster_se_exceeds_iid_when_within_series_correlated() -> None:
    """系列内正相关时聚类 SE 必须大于 iid SE，且倍数 ≈ √(1+(m̃−1)ρ)。"""
    rng = np.random.default_rng(4)
    n_series, m, rho = 900, 3, 0.4
    g = np.repeat(np.arange(n_series), m)
    a = rng.normal(0, 1, n_series)[g]
    e = rng.normal(0, 1, n_series * m)
    d = np.sqrt(rho) * a + np.sqrt(1 - rho) * e
    r = diebold_mariano(d, g)
    expected = np.sqrt(1 + (m - 1) * rho)
    assert r.se_ratio == pytest.approx(expected, rel=0.08), (r.se_ratio, expected)
    assert r.se_cluster > r.se_iid


def test_dm_harvey_correction_shrinks_stat_and_uses_series_df() -> None:
    """Harvey 因子 = √((m+1−2h+h(h−1)/m)/m)，m 取 **series 数**。

    h=1 时它 <1 且随 m→∞ 趋于 1；h 变大时修正变强。
    df 必须是 n_series−1（独立单位是 series，不是 game）。
    """
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(40), 3)
    d = rng.normal(0.05, 0.2, 120)
    r1 = diebold_mariano(d, g, h=1)
    r3 = diebold_mariano(d, g, h=3)
    assert r1.n_series == 40 and r1.df == 39
    assert r1.harvey_factor == pytest.approx(np.sqrt(39 / 40))
    assert abs(r1.dm_harvey) < abs(r1.dm)
    assert r3.harvey_factor < r1.harvey_factor  # h 越大修正越狠
    assert r3.p_value > r1.p_value
    big = diebold_mariano(rng.normal(0.05, 0.2, 12000), np.repeat(np.arange(4000), 3))
    assert big.harvey_factor > 0.9998  # m=4000 时实测 0.99987，几乎不起作用


def test_dm_null_rejection_rate_is_near_nominal_while_iid_over_rejects() -> None:
    """聚类噪声、真差为 0 时：DM（CR1）的拒绝率应接近 5%，
    而不做聚类的朴素 t 会明显超过 5% —— 这就是必须聚类的全部理由。"""
    from scipy import stats

    rng = np.random.default_rng(99)
    n_series, m, rho, sims = 400, 3, 0.5, 3000
    g = np.repeat(np.arange(n_series), m)
    rej_dm = rej_iid = 0
    for _ in range(sims):
        a = rng.normal(0, 1, n_series)[g]
        e = rng.normal(0, 1, n_series * m)
        d = np.sqrt(rho) * a + np.sqrt(1 - rho) * e
        r = diebold_mariano(d, g)
        rej_dm += int(r.p_value < 0.05)
        t_iid = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        rej_iid += int(2 * stats.t.sf(abs(t_iid), df=len(d) - 1) < 0.05)
    assert 0.02 <= rej_dm / sims <= 0.09, f"DM 拒绝率 {rej_dm / sims:.3f} 偏离名义水平"
    assert rej_iid / sims > 0.12, f"朴素 t 拒绝率 {rej_iid / sims:.3f}，本该严重超发"  # 实测 0.147/0.172


def test_dm_constant_difference_is_infinite_not_noise() -> None:
    """§6.6.4 的边界情况：组间无变异时 t = ±inf，按符号判定，不得当成噪声。"""
    g = np.repeat(np.arange(50), 2)
    r = diebold_mariano(np.full(100, 0.003), g)
    assert np.isposinf(r.dm) and np.isposinf(r.dm_harvey) and r.p_value == 0.0
    r_neg = diebold_mariano(np.full(100, -0.003), g)
    assert np.isneginf(r_neg.dm)
    r_zero = diebold_mariano(np.zeros(100), g)
    assert r_zero.dm == 0.0 and r_zero.p_value == 1.0


def test_dm_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="长度不一致"):
        diebold_mariano(np.zeros(10), np.arange(9))
    with pytest.raises(ValueError, match="样本太小"):
        diebold_mariano(np.zeros(4), np.zeros(4))


# =========================================================================
# Benjamini-Hochberg
# =========================================================================
BH_TEXTBOOK = [
    0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
]


def test_bh_reproduces_textbook_example() -> None:
    """BH(1995) 原文例子：15 个检验、α=0.05 → 拒绝前 4 个。"""
    r = benjamini_hochberg(BH_TEXTBOOK, 0.05)
    assert r.n_reject == 4
    assert r.reject[:4].all() and not r.reject[4:].any()
    assert r.threshold == pytest.approx(0.0095)


def test_bh_is_more_powerful_than_bonferroni() -> None:
    """§6.6.5 选 BH 而不是 Bonferroni 的理由，用同一批 p 值量出来。"""
    r = benjamini_hochberg(BH_TEXTBOOK, 0.05)
    bonf = sum(p <= 0.05 / len(BH_TEXTBOOK) for p in BH_TEXTBOOK)
    assert r.n_reject == 4 > bonf == 3


def test_bh_qvalues_are_monotone_and_never_below_p() -> None:
    rng = np.random.default_rng(8)
    p = rng.uniform(0, 1, 200)
    r = benjamini_hochberg(p, 0.05)
    order = np.argsort(p)
    q = r.qvalues[order]
    assert np.all(np.diff(q) >= -1e-12), "q 值必须随 p 单调不降"
    assert np.all(r.qvalues >= p - 1e-12)
    assert np.all(r.qvalues <= 1.0)
    # reject 与 q<=alpha 必须是同一件事
    assert np.array_equal(r.reject, r.qvalues <= 0.05)


def test_bh_is_invariant_to_input_order() -> None:
    rng = np.random.default_rng(9)
    p = rng.uniform(0, 0.2, 50)
    perm = rng.permutation(50)
    a = benjamini_hochberg(p, 0.05)
    b = benjamini_hochberg(p[perm], 0.05)
    assert a.n_reject == b.n_reject
    assert np.allclose(a.qvalues[perm], b.qvalues)
    assert np.array_equal(a.reject[perm], b.reject)


def test_bh_controls_fdr_under_simulation() -> None:
    """900 个真 null + 100 个真效应，重复 200 次，实际 FDR 必须 ≤ α（含蒙卡余量）。"""
    from scipy import stats

    rng = np.random.default_rng(10)
    fdrs = []
    for _ in range(200):
        p_null = rng.uniform(0, 1, 900)
        z = rng.normal(3.2, 1.0, 100)
        p_alt = 2 * stats.norm.sf(np.abs(z))
        p = np.concatenate([p_null, p_alt])
        r = benjamini_hochberg(p, 0.10)
        if r.n_reject:
            fdrs.append(float(r.reject[:900].sum()) / r.n_reject)
        else:
            fdrs.append(0.0)
    assert np.mean(fdrs) <= 0.11, f"实际 FDR {np.mean(fdrs):.4f} 超过 α=0.10"


def test_bh_edge_cases() -> None:
    assert benjamini_hochberg([], 0.05).n_reject == 0
    assert benjamini_hochberg([0.4, 0.9], 1.0).reject.all()
    assert benjamini_hochberg([0.9, 0.99], 0.05).n_reject == 0
    with pytest.raises(ValueError, match="NaN"):
        benjamini_hochberg([0.1, np.nan])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.1, 1.5])


# =========================================================================
# 组内相关的三种口径（ICC(1) 在本数据上会骗人）
# =========================================================================
def test_cluster_rho_recovers_known_correlation() -> None:
    """等相关、同方差的合成数据上，三种口径应当一致地还原真 ρ。"""
    rng = np.random.default_rng(12)
    n_series, m, rho = 4000, 3, 0.35
    g = np.repeat(np.arange(n_series), m)
    d = np.sqrt(rho) * rng.normal(0, 1, n_series)[g] + np.sqrt(1 - rho) * rng.normal(
        0, 1, n_series * m
    )
    r = cluster_rho(d, g)
    assert r["rho_pairwise"] == pytest.approx(rho, abs=0.03)
    assert r["rho_cr1"] == pytest.approx(rho, abs=0.03)
    assert r["icc_anova"] == pytest.approx(rho, abs=0.03)
    assert r["m_tilde"] == pytest.approx(3.0)


def test_icc_anova_is_fooled_by_singleton_heteroskedasticity() -> None:
    """**真 ρ = 0，但 ICC(1) 报 0.2+。** 这是 compare.cluster_rho 文档里那条实测的可执行版本。

    机制：单局 series 对 SSB 贡献完整的 (d−d̄)²、对 SSW 贡献 0，
    只要单局的方差比多局大，(MSB−MSW) 就把异方差当成了组间方差分量。
    实测 OOS 上单局 var(d) 是多局的 2.04 倍，正好落在这个陷阱里。
    """
    rng = np.random.default_rng(14)
    sizes = np.repeat([1, 2, 3, 4, 5], [5995, 3036, 2363, 496, 384])
    g = np.repeat(np.arange(len(sizes)), sizes)
    singleton = np.repeat(sizes == 1, sizes)
    d = rng.normal(0, 1, len(g)) * np.where(singleton, np.sqrt(2.04), 1.0)  # 完全独立
    r = cluster_rho(d, g)
    assert r["rho_pairwise"] == pytest.approx(0.0, abs=0.02), "真 ρ 就是 0"
    assert r["icc_anova"] > 0.18, f"ICC(1) 本该被骗，实测 {r['icc_anova']:.4f}"  # 实测 0.2088
    assert r["deff_anova"] > 1.15 and r["deff_pairwise"] < 1.05  # 实测 1.183 vs 1.001


# =========================================================================
# 真数据：data/preds_ladder.parquet
# =========================================================================
@pytest.fixture(scope="module")
def preds() -> pd.DataFrame:
    return pd.read_parquet(PREDS)


@needs_preds
def test_reproduces_measured_l2_vs_l3_delta(preds: pd.DataFrame) -> None:
    """验收：实跑 L2 vs L3，复现 ΔLL = +0.00160，并给出 CI 与 DM p。"""
    spine, p_mat = align_predictions(preds, ["L2", "L3"])
    y = spine.y.to_numpy()
    g = spine.series_id.to_numpy()
    p2, p3 = p_mat["L2"].to_numpy(), p_mat["L3"].to_numpy()

    assert metrics.log_loss(y, p2) == pytest.approx(0.6339, abs=5e-5)
    assert metrics.log_loss(y, p3) == pytest.approx(0.6323, abs=5e-5)

    b = paired_bootstrap_delta(y, p2, p3, metrics.log_loss, g, 2000, 0)
    assert b.point == pytest.approx(0.00160, abs=5e-6)
    assert b.lo > 0 and b.hi > b.lo          # CI 不跨 0
    assert b.lo == pytest.approx(0.00042, abs=2e-4)
    assert b.hi == pytest.approx(0.00282, abs=2e-4)

    dm = diebold_mariano(loss_diff(y, p2, p3), g)
    assert dm.p_value == pytest.approx(0.0083, abs=1e-3)
    assert dm.n_series == 12274 and dm.n_games == 23060

    # |Δ| 低于 MDE（阶段 3 的裁决），但 DM p<0.05 —— 两条门槛不是同一件事，
    # 手册 §6.3（MDE 用 t*=2.8）比 §6.6.4（|t|>2.5）严，这里两者恰好分居两侧。
    assert abs(b.point) < b.mde
    assert abs(dm.dm_harvey) > 2.5


@needs_preds
def test_series_and_game_standard_errors_really_differ(preds: pd.DataFrame) -> None:
    """验收：把 series 级与 game 级的 SE 都算出来对比（真数据）。"""
    spine, p_mat = align_predictions(preds, ["L2", "L3"])
    y, g = spine.y.to_numpy(), spine.series_id.to_numpy()
    cmp = se_comparison(y, p_mat["L2"].to_numpy(), p_mat["L3"].to_numpy(), g, n_boot=2000)
    assert cmp["se_series"] > cmp["se_game"]
    assert 1.01 < cmp["se_ratio"] < 1.10, cmp   # 实测 1.035（n_boot=2000）
    assert cmp["ci_width_series"] > cmp["ci_width_game"]
    # bootstrap 的 SD 自己带 ~1/√(2B) ≈ 1.6% 的蒙卡噪声，
    # 所以再钉一个完全确定性的锚点：CR1 / iid（实测 1.0194）
    dm = diebold_mariano(loss_diff(y, p_mat["L2"].to_numpy(), p_mat["L3"].to_numpy()), g)
    assert dm.se_ratio == pytest.approx(1.0194, abs=1e-3)
    # 两条路线必须都摆出来，因为它们的误差方向会随 ρ 估计器翻面：
    # 2026-09-19 更正：power.paired_power 的 deff 改用稳健的 pairwise ρ。
    # 旧值 1.129 来自 ANOVA 版 ICC=0.3120，而那个数主要是异方差伪影——
    # OOS 里 48.8% 的 series 是单局，单局 var(d) 是多局的 2.04 倍，
    # 单局组只进组间平方和不进组内平方和，异方差被整段记到"组间"。
    # 反证：真 ρ=0 配上单局方差×2.04，icc_oneway 报 0.4454。
    # 稳健 ρ=0.0266 → sqrt_deff≈1.012。裁决不翻（ΔLL 0.00160 仍 < MDE 0.00169）。
    assert cmp["sqrt_deff"] == pytest.approx(1.012, abs=0.01)
    # **误差方向翻面了**：用旧的 ANOVA ICC=0.312 时 √deff=1.129 > 经验 1.035（过度保守）；
    # 换成稳健 ρ=0.0266 后 √deff=1.012 < 经验 1.042（反而不够保守）。
    # 结论：参数化 deff 路线的偏差方向依赖于 ρ 估计器的选择，不可靠。
    # **MDE 的主判据一律用 bootstrap**，它不对相关结构做参数假设。
    # 参数化路线只作为对照打进表脚，供发现失真。
    assert cmp["sqrt_deff"] < cmp["se_ratio"], (
        "若这条翻回来，说明 ρ 估计器又变了，请重新审视 MDE 主判据的选择"
    )


@needs_preds
def test_single_model_logloss_ci_is_wider_at_series_level(preds: pd.DataFrame) -> None:
    spine, p_mat = align_predictions(preds, ["L3"])
    y, g = spine.y.to_numpy(), spine.series_id.to_numpy()
    p3 = p_mat["L3"].to_numpy()
    s = block_bootstrap_ci(y, p3, metrics.log_loss, g, 2000, 0)
    gm = block_bootstrap_ci(y, p3, metrics.log_loss, g, 2000, 0, unit="game")
    assert s.point == pytest.approx(0.6323, abs=5e-5)
    assert (s.hi - s.lo) > (gm.hi - gm.lo)
    assert 1.01 < s.se / gm.se < 1.10  # 实测 1.03–1.05（随 n_boot 浮动）


@needs_preds
def test_compare_table_has_13_columns_and_a_power_footer(preds: pd.DataFrame) -> None:
    """验收⑤：表脚必须自动带上 power.footer() 的功效信息。"""
    tbl = compare_table(preds, baseline="L3", n_boot=400, seed=0, echo=False)
    assert len(tbl.columns) == 13, list(tbl.columns)
    assert tbl.index.name == "model" and "L3" in tbl.index
    foot = tbl.attrs["footer"]
    # 2026-09-19：footer 的组内相关改用稳健 ρ（旧的 ICC= 是 ANOVA 版，会被异方差抬高）
    for token in ("n_games=", "n_series=", "m̄=", "ρ=", "deff=", "n_eff=", "σ_d=", "MDE="):
        assert token in foot, token
    assert "MDE_boot" in foot and "低于可检出下限" in foot
    # footer 必须与 power.footer() 逐字一致，不许自己另起炉灶
    pw = tbl.attrs["power"]["L2"]
    assert power.footer(pw) in foot
    assert power.footer(pw) in format_compare_table(tbl)
    # 拿不到的三列留占位
    assert tbl.loc["L3", "训练耗时 T_state/T_refit/T_fit"].startswith("—")
    assert tbl.loc["L3", "推理延迟 p50/p99"].startswith("—")


@needs_preds
def test_compare_table_reproduces_stage3_numbers(preds: pd.DataFrame) -> None:
    """对比表里的数必须和阶段 3 实测的一致（0.6323 / 0.2207 / 0.6958 / 0.0154 / 0.946）。"""
    tbl = compare_table(preds, baseline="L3", n_boot=400, seed=0, echo=False)
    raw = tbl.attrs["raw"]
    assert raw.loc["L0", "log_loss"] == pytest.approx(0.6931, abs=5e-5)
    assert raw.loc["L1", "log_loss"] == pytest.approx(0.6916, abs=5e-5)
    assert raw.loc["L1b", "log_loss"] == pytest.approx(0.6918, abs=5e-5)
    assert raw.loc["L2", "log_loss"] == pytest.approx(0.6339, abs=5e-5)
    assert raw.loc["L3", "log_loss"] == pytest.approx(0.6323, abs=5e-5)
    assert raw.loc["L3", "brier"] == pytest.approx(0.2207, abs=5e-5)
    assert raw.loc["L3", "auc"] == pytest.approx(0.6958, abs=5e-5)
    assert raw.loc["L3", "ece"] == pytest.approx(0.0154, abs=5e-5)
    assert raw.loc["L3", "calib_slope"] == pytest.approx(0.946, abs=5e-4)
    # Murphy 恒等式在表里也要成立
    r = raw.loc["L3"]
    assert r["brier"] == pytest.approx(
        r["reliability"] - r["resolution"] + r["uncertainty"], abs=2e-3
    )
    # L2 vs L3 低于 MDE；baseline 行不参与判定
    assert not bool(raw.loc["L2", "exceeds_mde"])
    assert raw.loc["L2", "delta_ll"] == pytest.approx(0.00160, abs=5e-6)
    assert np.isnan(raw.loc["L3", "dm_p"])
    # exploratory 比较全部带 BH q 值
    assert "bh_q" in raw.columns
    assert raw.loc["L2", "bh_q"] >= raw.loc["L2", "dm_p"]


@needs_preds
def test_compare_table_subsets(preds: pd.DataFrame) -> None:
    """国际赛子集按 eval/tier1_leagues.yaml 取；rating 列缺失时必须诚实说缺。"""
    tbl = compare_table(preds, baseline="L3", n_boot=200, seed=0, echo=False)
    assert "无 rating 列" in tbl.loc["L3", "|Δrating|<50 LL"]
    assert tbl.attrs["raw"].loc["L3", "n_intl"] > 0
    # 补一列评分差进去，那一列就应当算得出来
    with_rating = preds.copy()
    rng = np.random.default_rng(0)
    per_game = dict(
        zip(preds.game_id.unique(), rng.normal(0, 120, preds.game_id.nunique()), strict=True)
    )
    with_rating["team_elo_diff"] = with_rating.game_id.map(per_game)
    tbl2 = compare_table(with_rating, baseline="L3", n_boot=200, seed=0, echo=False)
    assert "n=" in tbl2.loc["L3", "|Δrating|<50 LL"]
    assert 0 < tbl2.attrs["raw"].loc["L3", "n_close"] < len(preds.game_id.unique())


def test_international_list_follows_prereg_not_the_manual() -> None:
    """手册 §6.4.1 写的 IC 已被预注册修订 A1 剔除，WLDS 的正确写法是 WLDs。"""
    intl = international_leagues()
    assert "WLDs" in intl and "MSI" in intl
    assert "IC" not in intl, "IC = Iberian Cup，二级队杯赛，不是国际赛（预注册 A1）"
    assert "WLDS" not in intl
    assert international_leagues("不存在的路径.yaml") == international_leagues.__globals__[
        "_INTERNATIONAL_FALLBACK"
    ]


# =========================================================================
# 对齐与防呆
# =========================================================================
def _toy_long() -> pd.DataFrame:
    rows = []
    for m, base in (("A", 0.55), ("B", 0.60)):
        for i in range(12):
            rows.append(
                {
                    "game_id": f"g{i}",
                    "series_id": f"s{i // 3}",
                    "league_canon": "WLDs" if i < 4 else "LCK",
                    "model": m,
                    "y": i % 2,
                    "p": base if i % 2 else 1 - base,
                }
            )
    return pd.DataFrame(rows)


def test_align_rejects_misaligned_or_inconsistent_inputs() -> None:
    long = _toy_long()
    bad = pd.concat([long, long[long.model == "A"].head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="重复 game_id"):
        align_predictions(bad, ["A", "B"])
    short = long[~((long.model == "B") & (long.game_id == "g0"))]
    with pytest.raises(ValueError, match="game_id 集合不同"):
        align_predictions(short, ["A", "B"])
    flipped = long.copy()
    flipped.loc[(flipped.model == "B") & (flipped.game_id == "g0"), "y"] = 1
    with pytest.raises(ValueError, match="y 与"):
        align_predictions(flipped, ["A", "B"])
    with pytest.raises(ValueError, match="缺列"):
        align_predictions(long.drop(columns=["series_id"]), ["A", "B"])


def test_compare_table_rejects_unknown_baseline() -> None:
    with pytest.raises(ValueError, match="baseline"):
        compare_table(_toy_long(), baseline="L3", echo=False)


def test_format_requires_footer() -> None:
    with pytest.raises(ValueError, match="功效脚注"):
        format_compare_table(pd.DataFrame({"a": [1]}))


def test_bootstrap_rejects_length_mismatch_and_bad_unit() -> None:
    y, p_a, _, g = _synth(1, n_series=20)
    with pytest.raises(ValueError, match="长度不一致"):
        block_bootstrap_ci(y[:-1], p_a, metrics.log_loss, g, 10, 0)
    with pytest.raises(ValueError, match="unit"):
        block_bootstrap_ci(y, p_a, metrics.log_loss, g, 10, 0, unit="row")
    with pytest.raises(ValueError, match="长度必须一致"):
        paired_bootstrap_delta(y, p_a, p_a[:-1], metrics.log_loss, g, 10, 0)
