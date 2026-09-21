"""阶段 3：评估基础设施的验收测试。

「先有裁判再有选手」——这些测试保证裁判本身是可信的。
"""
import os

import numpy as np
import pandas as pd
import pytest

from lolwp.eval.metrics import (
    auc,
    brier_decomposition,
    calibration_slope,
    ece,
    log_loss,
)
from lolwp.eval.power import design_effect, icc_oneway, paired_power
from lolwp.eval.walkforward import (
    WFConfig,
    assert_embargo_respected,
    assert_no_series_straddle,
    make_folds,
    oos_frame,
)

SILVER = "data/silver/games.parquet"
needs_data = pytest.mark.skipif(not os.path.exists(SILVER), reason="需先跑 silver.build()")


# ---------------- 指标的解析性质 ----------------

def test_log_loss_matches_closed_form():
    y = np.array([1] * 530 + [0] * 470)
    p = np.full(1000, 0.53)
    expected = -(0.53 * np.log(0.53) + 0.47 * np.log(0.47))
    assert abs(log_loss(y, p) - expected) < 1e-9


def test_auc_is_half_for_constant():
    y = np.array([0, 1] * 500)
    assert auc(y, np.full(1000, 0.5)) == 0.5


def test_auc_is_one_for_perfect():
    y = np.array([0] * 500 + [1] * 500)
    assert auc(y, y.astype(float)) == 1.0


def test_brier_decomposition_identity():
    """Brier = Reliability − Resolution + Uncertainty，恒等式必须成立。"""
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 5000)
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    d = brier_decomposition(y, p)
    lhs = float(np.mean((p - y) ** 2))
    rhs = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert abs(lhs - rhs) < 0.002


def test_calibration_slope_is_one_for_true_model():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 20000)
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    slope, _ = calibration_slope(y, p)
    assert 0.94 < slope < 1.06


def test_ece_zero_for_perfectly_calibrated_constant():
    y = np.array([1] * 530 + [0] * 470)
    assert ece(y, np.full(1000, 0.53)) < 0.01


# ---------------- 功效分析 ----------------

def test_icc_zero_for_independent_data():
    rng = np.random.default_rng(3)
    v = rng.normal(0, 1, 6000)
    g = np.repeat(np.arange(3000), 2)
    r = icc_oneway(v, g)
    assert abs(r["icc"]) < 0.06


def test_icc_one_for_fully_correlated_groups():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 1, 3000)
    v = np.repeat(base, 2)          # 组内完全相同
    g = np.repeat(np.arange(3000), 2)
    assert icc_oneway(v, g)["icc"] > 0.95


def test_design_effect_formula():
    assert design_effect(2.14, 1.0) == pytest.approx(2.14)
    assert design_effect(2.14, 0.0) == pytest.approx(1.0)


def test_mde_shrinks_with_sample_size():
    rng = np.random.default_rng(5)
    small = paired_power(rng.normal(0.001, 0.1, 1000), np.repeat(np.arange(500), 2))
    large = paired_power(rng.normal(0.001, 0.1, 100000), np.repeat(np.arange(50000), 2))
    assert large["mde"] < small["mde"]
    assert large["mde"] < small["mde"] / 5      # 样本 ×100 → MDE 约 ÷10


def test_icc_undefined_for_singleton_groups():
    """每组只有一局时 ICC 无定义，必须返回 NaN 而不是悄悄给 0。"""
    rng = np.random.default_rng(6)
    r = icc_oneway(rng.normal(0, 1, 100), np.arange(100))
    assert np.isnan(r["icc"])


# ---------------- walk-forward 切分 ----------------

@pytest.fixture(scope="module")
def dev():
    if not os.path.exists(SILVER):
        pytest.skip("需先跑 silver.build()")
    from lolwp.features.loader import load_dev
    return load_dev(min_tier=1, since="2019-01-01").reset_index(drop=True)


@needs_data
def test_loader_excludes_holdout(dev):
    from lolwp.features.loader import holdout_start
    assert dev.kickoff_utc.max() < holdout_start()


@needs_data
def test_holdout_loader_is_guarded():
    from lolwp.features.loader import load_holdout
    os.environ.pop("LOLWP_ALLOW_HOLDOUT", None)
    with pytest.raises(RuntimeError, match="拒绝加载"):
        load_holdout()


@needs_data
def test_folds_meet_acceptance(dev):
    """阶段 3 验收④：≥10 折、OOS ≥3,000 场、无 series 跨界。"""
    cfg = WFConfig()
    folds = make_folds(dev, cfg)
    assert len(folds) >= 10
    oos = oos_frame(dev, folds)
    assert len(oos) >= 3000
    assert_no_series_straddle(dev, folds, cfg)
    assert_embargo_respected(dev, folds, cfg)


@needs_data
def test_no_game_counted_twice(dev):
    folds = make_folds(dev, WFConfig())
    oos = oos_frame(dev, folds)
    assert not oos.game_id.duplicated().any()


@needs_data
def test_train_always_precedes_valid(dev):
    cfg = WFConfig()
    for f in make_folds(dev, cfg):
        assert dev.loc[f.train_idx, "kickoff_utc"].max() < dev.loc[f.valid_idx, "kickoff_utc"].min()


# ---------------- 红线 ----------------

@pytest.mark.skipif(not os.path.exists("data/preds_ladder.parquet"), reason="需先跑基线")
def test_no_red_line_tripped():
    """红线（预注册修订 A3 后）：accuracy > 67% / log loss < 0.60 / **合并** AUC > 0.78。

    A3 把原来的「任一 fold AUC > 0.78」换掉了，因为它是坏的：
    90 折里 8 折不足 50 场、最小的只有 3 场；纯随机预测下最大 fold AUC 的
    中位数就是 1.0000，超过 0.78 的概率 63%。即便限制到 n≥200，
    一个合并 AUC 0.7148 的健康模型仍有 38.4% 概率触发。
    一条 38% 时间在报警的告警器等于没有告警器。
    """
    from lolwp.eval.metrics import evaluate
    preds = pd.read_parquet("data/preds_ladder.parquet")
    for name, s in preds.groupby("model"):
        e = evaluate(s.y.to_numpy(), s.p.to_numpy())
        assert e["accuracy"] <= 0.67, f"{name} accuracy {e['accuracy']:.4f} 触及红线"
        assert e["log_loss"] >= 0.60, f"{name} log loss {e['log_loss']:.4f} 触及红线"
        assert e["auc"] <= 0.78, f"{name} 合并 AUC {e['auc']:.4f} 触及红线"


def test_per_fold_auc_red_line_would_false_alarm():
    """锁住 A3 的依据：旧判据在小折上必然误报。若这条失败，说明切分变了，
    A3 的论证需要重做。"""
    from lolwp.eval.walkforward import WFConfig, make_folds
    from lolwp.features.loader import load_dev
    if not os.path.exists(SILVER):
        pytest.skip("需先跑 silver.build()")
    d = load_dev(min_tier=1, since="2019-01-01").reset_index(drop=True)
    sizes = np.array([f.n_valid for f in make_folds(d, WFConfig())])
    assert sizes.min() < 50, "最小折已不再是小折，A3 的第一条论证需重新验证"
    assert (sizes < 200).sum() > 10, "小折数量大幅下降，A3 的论证需重新验证"


@pytest.mark.skipif(not os.path.exists("data/preds_ladder.parquet"), reason="需先跑基线")
def test_baselines_land_in_predicted_range():
    """L3 必须落在手册 §1.2 预注册的区间内。落在区间外说明管道有问题。"""
    from lolwp.eval.metrics import evaluate
    preds = pd.read_parquet("data/preds_ladder.parquet")
    s = preds[preds.model == "L3"]
    e = evaluate(s.y.to_numpy(), s.p.to_numpy())
    assert 0.615 <= e["log_loss"] <= 0.645, f"log loss {e['log_loss']:.4f} 超出预注册区间"
    assert 0.216 <= e["brier"] <= 0.228, f"Brier {e['brier']:.4f} 超出预注册区间"
    assert e["ece"] <= 0.03


@pytest.mark.skipif(not os.path.exists("data/preds_ladder.parquet"), reason="需先跑基线")
def test_ladder_is_monotone_improving():
    """基线阶梯的 log loss 应逐级下降（L1b 与 L1 可以持平）。"""
    from lolwp.eval.metrics import log_loss as ll
    preds = pd.read_parquet("data/preds_ladder.parquet")
    v = {n: ll(s.y.to_numpy(), s.p.to_numpy()) for n, s in preds.groupby("model")}
    assert v["L0"] > v["L2"] > v["L3"]
    assert v["L1"] < v["L0"]


def test_series_context_stays_out_of_production():
    """阶段后实验：BO 内序列上下文未过门禁，不得悄悄进生产特征集。

    实验见 reports/exp_series_context.md：
    S_score 在全样本 ΔLL +0.00073（MDE 0.00084）、非首局 +0.00168（MDE 0.00179），
    两段留出期都未超 MDE。是全项目最接近通过的一次近失，但仍未过。
    """
    from lolwp.models.artifact import PROD_FEATURES
    banned = {"game_in_series", "game_is_even", "series_score_diff",
              "is_match_point", "is_first_game"}
    leaked = banned & set(PROD_FEATURES)
    assert not leaked, f"未过门禁的序列特征进了生产模型：{leaked}"


def test_series_length_is_never_computed():
    """series_length 是不安全的：打到第 5 局意味着知道了前 4 局是 2-2，
    而那在第 1 局的 BP 时点未知。features/series_ctx.py 不得产出它。"""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "lolwp" / "features" / "series_ctx.py")
    if not src.exists():
        pytest.skip("无 series_ctx 模块")

    from lolwp.features.loader import load_dev
    from lolwp.features.series_ctx import build
    d = load_dev(min_tier=1, since="2025-01-01").head(2000)
    f = build(d)
    assert "series_length" not in f.columns
    assert not any("length" in c for c in f.columns)
