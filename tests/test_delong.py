"""DeLong AUC 检验的数值验收（PLAN.md §6.6.3）。

这个模块唯一真正的验收是 **解析方差 vs bootstrap 经验方差**：DeLong 的全部价值
就在于"不用 bootstrap 也能拿到 AUC 差的标准误"，方差错了它就一文不值。
AUC 点估计的一致性是前置条件，聚类偏差是使用限制，分别在下面两组里测。
"""
import os

import numpy as np
import pandas as pd
import pytest

from lolwp.eval.delong import (
    _fast_delong,
    _midrank,
    delong_roc_test,
    delong_roc_variance,
    paired_auc_block_bootstrap,
    pairwise_delong_table,
)
from lolwp.eval.metrics import auc as metrics_auc

PREDS = "data/preds_ladder.parquet"
needs_preds = pytest.mark.skipif(not os.path.exists(PREDS), reason="需先跑基线阶梯")


# ---------------- midrank ----------------

def test_midrank_sums_to_triangular_number():
    """任何并列结构下，秩之和都必须是 n(n+1)/2——midrank 只重新分配、不创造秩。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 500))
        x = np.round(rng.normal(0, 1, n), 1)      # 故意大量并列
        assert _midrank(x).sum() == pytest.approx(n * (n + 1) / 2)


def test_midrank_all_tied_gives_mean_rank():
    n = 17
    assert np.allclose(_midrank(np.full(n, 3.0)), (n + 1) / 2)


def test_midrank_is_permutation_invariant():
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(0, 1, 200), 1)
    perm = rng.permutation(200)
    assert np.allclose(_midrank(x)[perm], _midrank(x[perm]))


# ---------------- AUC 点估计必须与 metrics.auc 一致 ----------------

def test_auc_matches_metrics_including_ties():
    """验收①：max|diff| < 1e-12，含并列。"""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(30, 4000))
        y = rng.binomial(1, float(rng.uniform(0.15, 0.85)), n)
        if y.min() == y.max():
            continue
        # 小数位数越少并列越多；0 位 = 只有几个不同分数
        p = np.round(rng.normal(0.6 * y, 1.0, n), int(rng.integers(0, 3)))
        got, _ = delong_roc_variance(y, p)
        worst = max(worst, abs(got - metrics_auc(y, p)))
    assert worst < 1e-12, f"max|diff| = {worst:g}"


def test_auc_extremes():
    y = np.array([0] * 300 + [1] * 300)
    assert delong_roc_variance(y, y.astype(float))[0] == 1.0
    assert delong_roc_variance(y, 1.0 - y)[0] == 0.0
    # 常数预测：AUC = 0.5 且方差恒为 0（全部并列，结构分量没有波动）
    a, v = delong_roc_variance(y, np.full(600, 0.5))
    assert a == 0.5
    assert v == 0.0


def test_single_class_raises():
    with pytest.raises(ValueError, match="正负样本"):
        delong_roc_variance(np.ones(50, dtype=int), np.random.random(50))


# ---------------- 核心验收：解析方差 vs bootstrap ----------------

def test_variance_matches_bootstrap():
    """验收②（本模块唯一真正的验收）：2000 样本 / 2000 次 bootstrap，相对误差 < 10%。

    实测 rel_err ≈ 2.5%（seed=7）。bootstrap 自身在 B=2000 下的抽样误差约
    ±3%（方差的方差 ~ 2/B），所以 10% 是一个既能抓住实现错误、又不会随机失败的门槛。
    """
    rng = np.random.default_rng(7)
    n = 2000
    y = rng.binomial(1, 0.5, n)
    p = 1.0 / (1.0 + np.exp(-rng.normal(0.9 * y - 0.45, 1.0)))
    _, var_analytic = delong_roc_variance(y, p)

    boot = np.empty(2000)
    rb = np.random.default_rng(123)
    for i in range(len(boot)):
        idx = rb.integers(0, n, n)
        boot[i] = _fast_delong(y[idx], p[idx][None, :])[0][0]
    var_boot = float(np.var(boot, ddof=1))

    rel = abs(var_boot - var_analytic) / var_analytic
    assert rel < 0.10, f"var_delong={var_analytic:.4e} var_boot={var_boot:.4e} rel={rel:.2%}"


def test_paired_variance_matches_bootstrap():
    """配对差的方差也要对——协方差项写错时单模型方差仍然是对的，只有配对能抓到。

    这里用 B=4000 而不是 2000：配对差的方差比单模型小一个量级，bootstrap 自身的
    Monte Carlo 误差（~sqrt(2/B)）就会吃掉大半个门槛。实测 B=2000 时 6 个随机种子
    的相对误差分布在 −3.5% ~ +7.5%，B=4000 时收到 −3.5% ~ +6.2%，都无系统性偏向。
    """
    rng = np.random.default_rng(11)
    n = 2000
    y = rng.binomial(1, 0.5, n)
    z = rng.normal(0, 1, n)
    pa = 1.0 / (1.0 + np.exp(-(1.0 * y + 0.9 * z)))
    pb = 1.0 / (1.0 + np.exp(-(0.6 * y + 0.9 * z)))   # 与 pa 强相关但更弱
    res = delong_roc_test(y, pa, pb)

    boot = np.empty(4000)
    rb = np.random.default_rng(1011)
    for i in range(len(boot)):
        idx = rb.integers(0, n, n)
        aucs, _ = _fast_delong(y[idx], np.vstack([pa[idx], pb[idx]]))
        boot[i] = aucs[0] - aucs[1]
    var_boot = float(np.var(boot, ddof=1))

    rel = abs(var_boot - res.se**2) / res.se**2
    assert rel < 0.10, f"se_delong={res.se:.5f} se_boot={var_boot**0.5:.5f} rel={rel:.2%}"


def test_covariance_is_positive_semidefinite():
    rng = np.random.default_rng(13)
    y = rng.binomial(1, 0.5, 800)
    scores = np.vstack([rng.normal(0.5 * y, 1, 800) for _ in range(3)])
    _, cov = _fast_delong(y, scores)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-15


# ---------------- 退化与对称性 ----------------

def test_self_comparison_is_exactly_null():
    """验收③：自己和自己比 → Δ=0, z=0, p=1，不能除零崩溃。"""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.5, 500)
    p = rng.random(500)
    r = delong_roc_test(y, p, p)
    assert r.delta == 0.0
    assert r.se == 0.0
    assert r.z == 0.0
    assert r.p_value == 1.0


def test_self_comparison_of_constant_predictor():
    """L0（全常数 0.5）自比：Δ 和 var 都是 0，同样不能炸。"""
    y = np.array([0, 1] * 400)
    r = delong_roc_test(y, np.full(800, 0.5), np.full(800, 0.5))
    assert (r.z, r.p_value) == (0.0, 1.0)


def test_test_is_antisymmetric():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.5, 1000)
    a, b = rng.random(1000), rng.random(1000)
    ab, ba = delong_roc_test(y, a, b), delong_roc_test(y, b, a)
    assert ab.delta == pytest.approx(-ba.delta)
    assert ab.se == pytest.approx(ba.se)
    assert ab.p_value == pytest.approx(ba.p_value)


def test_no_difference_in_distribution_gives_uniformish_p():
    """两个同分布的无信息预测：p 不应系统性地小。粗测 30 次里显著的 ≤ 4 次。"""
    rng = np.random.default_rng(6)
    n_sig = 0
    for _ in range(30):
        y = rng.binomial(1, 0.5, 600)
        n_sig += delong_roc_test(y, rng.random(600), rng.random(600)).p_value < 0.05
    assert n_sig <= 4, f"30 次里 {n_sig} 次显著，方差被低估了"


def test_ci_contains_delta_and_scales_with_level():
    rng = np.random.default_rng(8)
    y = rng.binomial(1, 0.5, 1500)
    r = delong_roc_test(y, rng.normal(0.5 * y, 1), rng.normal(0.2 * y, 1))
    lo95, hi95 = r.ci(0.95)
    lo99, hi99 = r.ci(0.99)
    assert lo95 < r.delta < hi95
    assert lo99 < lo95 and hi95 < hi99


# ---------------- 聚类：DeLong 偏乐观 ----------------

def _clustered_fixture(seed: int, per: int = 5, n_series: int = 600):
    """强聚类合成数据：同一 series 内共享一个隐变量，组内相关很高。"""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_series), per)
    z = np.repeat(rng.normal(0, 2.0, n_series), per) + rng.normal(0, 0.2, n_series * per)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-z)))
    pa = 1.0 / (1.0 + np.exp(-(z + rng.normal(0, 0.3, len(z)))))
    pb = 1.0 / (1.0 + np.exp(-(0.8 * z + rng.normal(0, 0.3, len(z)))))
    return y, pa, pb, groups


def test_single_auc_se_is_understated_by_delong_under_clustering():
    """聚类下 DeLong 会低估**单个 AUC** 的标准误——docstring 那条限制的可执行版本。

    技巧：拿常数预测（方差恒为 0）当对照，则 var(ΔAUC) 恰好等于 var(AUC_a)，
    于是可以用同一个配对 bootstrap 函数量出单模型 AUC 的聚类惩罚。
    实测 se_block/se_delong ≈ 1.23-1.27（seed 21/22/23，B=800）。
    """
    y, pa, _, groups = _clustered_fixture(21)
    r = paired_auc_block_bootstrap(y, pa, np.full(len(y), 0.5), groups,
                                   n_boot=800, seed=0)
    assert r["se_ratio"] > 1.15, f"se_ratio={r['se_ratio']:.3f} 没体现出聚类惩罚"


def test_cluster_penalty_largely_cancels_in_the_paired_difference():
    """**实测发现**：聚类惩罚在配对差上基本抵消，这与 log loss 侧的行为不同。

    同一份强聚类数据上，单模型 AUC 的 se_ratio ≈ 1.25，而 ΔAUC 的 se_ratio ≈ 1.0：
    series 层的共享效应同时抬高两个模型的 AUC，相减时消掉了。真实数据同向——
    L2 vs L3 的 ΔAUC se_ratio 只有 1.02（B=10,000），而配对 log loss 的 deff 是 1.274。
    所以"DeLong 偏乐观"在**比较两个模型**时幅度有限，但方向永远是乐观的。
    """
    y, pa, pb, groups = _clustered_fixture(21)
    single = paired_auc_block_bootstrap(y, pa, np.full(len(y), 0.5), groups,
                                        n_boot=800, seed=0)["se_ratio"]
    paired = paired_auc_block_bootstrap(y, pa, pb, groups, n_boot=800, seed=0)["se_ratio"]
    assert paired < single - 0.1, f"single={single:.3f} paired={paired:.3f}"
    assert paired < 1.15


def test_block_bootstrap_matches_delong_when_groups_are_singletons():
    """每个 series 只有一局 = 真独立，此时两者应该接近（比值落在 0.85-1.15）。"""
    rng = np.random.default_rng(22)
    n = 2000
    y = rng.binomial(1, 0.5, n)
    pa = 1.0 / (1.0 + np.exp(-rng.normal(0.9 * y - 0.45, 1.0)))
    pb = 1.0 / (1.0 + np.exp(-rng.normal(0.5 * y - 0.25, 1.0)))
    r = paired_auc_block_bootstrap(y, pa, pb, np.arange(n), n_boot=600, seed=0)
    assert 0.85 < r["se_ratio"] < 1.15, r["se_ratio"]


# ---------------- 真实数据 ----------------

@pytest.fixture(scope="module")
def ladder() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    df = pd.read_parquet(PREDS)
    base = df[df.model == "L0"].sort_values("game_id").reset_index(drop=True)
    preds = {m: g.sort_values("game_id").p.to_numpy() for m, g in df.groupby("model")}
    for m, g in df.groupby("model"):
        assert (g.sort_values("game_id").game_id.to_numpy()
                == base.game_id.to_numpy()).all(), f"{m} 与 L0 的 game_id 不对齐"
    return base.y.to_numpy(), preds, base.series_id.to_numpy()


@needs_preds
def test_real_auc_matches_metrics(ladder):
    y, preds, _ = ladder
    worst = max(abs(delong_roc_variance(y, p)[0] - metrics_auc(y, p))
                for p in preds.values())
    assert worst < 1e-12, f"max|diff| = {worst:g}"


@needs_preds
def test_real_l3_auc_reproduces_stage3(ladder):
    """阶段 3 报的 L3 AUC = 0.6958，这里必须复现。"""
    y, preds, _ = ladder
    assert delong_roc_variance(y, preds["L3"])[0] == pytest.approx(0.6958, abs=5e-5)


@needs_preds
def test_real_l2_vs_l3_delong(ladder):
    """实测 ΔAUC = +0.003402, se = 0.000975, z = 3.490, p = 4.83e-4。"""
    y, preds, _ = ladder
    r = delong_roc_test(y, preds["L3"], preds["L2"])
    assert r.delta == pytest.approx(0.003402, abs=5e-6)
    assert r.se == pytest.approx(0.0009746, rel=1e-3)
    assert r.z == pytest.approx(3.490, abs=0.005)
    assert r.p_value < 0.001


@needs_preds
def test_real_l0_has_zero_variance(ladder):
    """L0 是常数 0.5：AUC 恰为 0.5、方差恰为 0，与 L1 比时 var 全部来自 L1。"""
    y, preds, _ = ladder
    a, v = delong_roc_variance(y, preds["L0"])
    assert (a, v) == (0.5, 0.0)
    r = delong_roc_test(y, preds["L1"], preds["L0"])
    assert r.se == pytest.approx(delong_roc_variance(y, preds["L1"])[1] ** 0.5)


@needs_preds
def test_real_pairwise_table_shape(ladder):
    y, preds, _ = ladder
    pairs = [("L1", "L0"), ("L1b", "L1"), ("L2", "L1b"), ("L3", "L2")]
    t = pairwise_delong_table(y, preds, pairs)
    assert len(t) == 4
    assert {"model_a", "model_b", "delta_auc", "z", "p_delong"} <= set(t.columns)
    # L1b→L2 是阶梯上唯一的巨大跳跃，其余都是小数
    assert t.set_index("model_a").loc["L2", "delta_auc"] > 0.15
