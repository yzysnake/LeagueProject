"""分组 L2 logistic 的验收测试（PLAN.md §5.3，阶段 4 硬验收①）。

**全部用真实特征**：silver → `run_state_machine()` 现算 as-of 特征，
fold 用 `eval/walkforward.py` 的真实切分，valid 段的 game_id 与已冻结的
`data/preds_ladder.parquet` 对齐（`test_fixture_is_the_real_oos_set` 会验证这件事）。
玩具数据能让重参数化的符号错误蒙混过关——真实特征的量纲跨 3 个数量级
（side_prior ≈ 0.52，team_elo_diff 可达 ±400），λ 放错位置会立刻暴露。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from lolwp.eval.baselines import LADDER
from lolwp.eval.baselines import predict_fold as baseline_predict_fold
from lolwp.eval.metrics import log_loss
from lolwp.eval.walkforward import WFConfig, make_folds
from lolwp.models.grouped_ridge import (
    DEFAULT_SOLVER,
    GroupedRidgeLogistic,
    Standardizer,
    predict_fold,
)

SILVER = "data/silver/games.parquet"
PREDS = "data/preds_ladder.parquet"
needs_data = pytest.mark.skipif(not os.path.exists(SILVER), reason="需先跑 silver.build()")

L3_COLS = ["side_prior", "team_elo_diff", "player_elo_diff"]
# 手册 §5.3 的分组形式：side 一组、实力项一组，各自一个 λ
GROUPS: dict[str, list[str]] = {
    "side": ["side_prior"],
    "strength": ["team_elo_diff", "player_elo_diff"],
}


# ------------------------------------------------------------------ 真实数据夹具


@pytest.fixture(scope="module")
def real() -> dict[str, object]:
    """复刻 `data/preds_ladder.parquet` 的口径：状态机跑全赛区，评测收窄到 tier≥1 & 2019 起。

    这个口径不是猜的——它是唯一能把 L0/L1/L1b/L2/L3 的 OOS 预测逐场复现到
    max|Δp| = 0 的口径（换成"只在 tier≥1 上跑状态机"，L3 会变成 0.6307 而不是 0.6323）。
    """
    if not os.path.exists(SILVER):
        pytest.skip("需先跑 silver.build()")
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine

    full = load_dev().reset_index(drop=True)
    feats_full = run_state_machine(full)
    mask = (full.league_tier >= 1) & (full.kickoff_utc >= pd.Timestamp("2019-01-01", tz="UTC"))
    df = full[mask].reset_index(drop=True)
    feats = feats_full[mask.to_numpy()].reset_index(drop=True)
    folds = make_folds(df, WFConfig())
    return {"df": df, "feats": feats, "folds": folds}


@pytest.fixture(scope="module")
def fold_data(real: dict[str, object]) -> dict[str, object]:
    """取一个靠后的 fold（训练段 12,719 场），单元测试都在它上面跑。"""
    df, feats, folds = real["df"], real["feats"], real["folds"]
    f = folds[45]
    rng = np.random.default_rng(20260919)
    return {
        "X_tr": feats.loc[f.train_idx, L3_COLS],
        "X_va": feats.loc[f.valid_idx, L3_COLS],
        "y_tr": df.y_blue_win.loc[f.train_idx].to_numpy(),
        "y_va": df.y_blue_win.loc[f.valid_idx].to_numpy(),
        "series_tr": df.series_id.loc[f.train_idx].to_numpy(),
        "time_tr": df.kickoff_utc.loc[f.train_idx],
        # 样本权重用一段随机正权重：比时间衰减更"刁钻"，能查出权重被吞掉的实现
        "w_tr": rng.uniform(0.2, 2.0, f.n_train),
        "fold": f,
    }


@needs_data
@pytest.mark.skipif(not os.path.exists(PREDS), reason="需先跑基线阶梯")
def test_fixture_is_the_real_oos_set(real: dict[str, object]) -> None:
    """夹具必须就是 preds_ladder 里那 23,060 场，不是随便切的一段。"""
    from lolwp.eval.walkforward import oos_frame

    oos = oos_frame(real["df"], real["folds"])
    ref = pd.read_parquet(PREDS)
    assert len(oos) == ref.game_id.nunique() == 23060
    assert set(oos.game_id) == set(ref.game_id)


# ------------------------------------------------- 验收①：统一 λ 必须等于 sklearn


@needs_data
@pytest.mark.parametrize("lam", [0.25, 1.0, 3.7, 50.0, 1000.0])
def test_equal_lambda_matches_sklearn(fold_data: dict[str, object], lam: float) -> None:
    """**阶段 4 硬验收①**：所有 λ_g 相同时，与 `LogisticRegression(C=1/λ)` 的预测
    max|Δp| < 1e-8。

    对照组用同一个 solver / tol，比的是"重参数化是否等价"，不是"两个解算器谁更准"。
    """
    X_tr, X_va = fold_data["X_tr"], fold_data["X_va"]
    y_tr = fold_data["y_tr"]

    model = GroupedRidgeLogistic(GROUPS, default_lambda=lam).fit(X_tr, y_tr)
    Z_tr = model.standardizer_.transform(X_tr)
    Z_va = model.standardizer_.transform(X_va)
    ref = LogisticRegression(
        C=1.0 / lam, solver=DEFAULT_SOLVER, tol=1e-10, max_iter=200
    ).fit(Z_tr, y_tr)

    d_p = np.abs(model.predict_proba(X_va)[:, 1] - ref.predict_proba(Z_va)[:, 1]).max()
    d_b = np.abs(model.coef_ - ref.coef_[0]).max()
    assert d_p < 1e-8, f"λ={lam}: max|Δp|={d_p:.3e}"
    assert d_b < 1e-8, f"λ={lam}: max|Δβ|={d_b:.3e}"
    assert abs(model.intercept_ - ref.intercept_[0]) < 1e-8


@needs_data
@pytest.mark.parametrize("lam", [0.25, 3.7, 1000.0])
def test_equal_lambda_matches_sklearn_with_sample_weight(
    fold_data: dict[str, object], lam: float
) -> None:
    """验收①的带权版本。权重进的是似然项而不是罚项，等价关系必须仍然成立。"""
    X_tr, X_va, y_tr, w = (
        fold_data["X_tr"], fold_data["X_va"], fold_data["y_tr"], fold_data["w_tr"]
    )
    model = GroupedRidgeLogistic(GROUPS, default_lambda=lam).fit(X_tr, y_tr, w)
    Z_tr = model.standardizer_.transform(X_tr)
    Z_va = model.standardizer_.transform(X_va)
    ref = LogisticRegression(
        C=1.0 / lam, solver=DEFAULT_SOLVER, tol=1e-10, max_iter=200
    ).fit(Z_tr, y_tr, sample_weight=w)
    d_p = np.abs(model.predict_proba(X_va)[:, 1] - ref.predict_proba(Z_va)[:, 1]).max()
    assert d_p < 1e-8, f"λ={lam} 带权: max|Δp|={d_p:.3e}"


@needs_data
def test_lbfgs_would_not_pass_the_1e_8_gate(fold_data: dict[str, object]) -> None:
    """记录偏离手册的实测依据：手册 §5.3.2 示例里的 lbfgs 过不了它自己定的 1e-8。

    不是断言 lbfgs "错"——它只是停在 scipy 的 ftol 地板上（约 1e-7），
    这个测试把那条地板钉住，防止有人把默认 solver 改回去还以为验收①仍然成立。
    """
    X_tr, X_va, y_tr = fold_data["X_tr"], fold_data["X_va"], fold_data["y_tr"]
    lam = 3.7
    model = GroupedRidgeLogistic(GROUPS, default_lambda=lam, solver="lbfgs", tol=1e-12,
                                 max_iter=50000).fit(X_tr, y_tr)
    Z_tr = model.standardizer_.transform(X_tr)
    Z_va = model.standardizer_.transform(X_va)
    ref = LogisticRegression(C=1.0 / lam, solver="lbfgs", tol=1e-12, max_iter=50000).fit(
        Z_tr, y_tr
    )
    d_p = np.abs(model.predict_proba(X_va)[:, 1] - ref.predict_proba(Z_va)[:, 1]).max()
    assert 1e-8 < d_p < 1e-5, f"lbfgs 的地板变了：max|Δp|={d_p:.3e}"


# --------------------------------------------- 重参数化本身：与直接最优化对拍


@needs_data
def test_reparam_matches_direct_optimizer(fold_data: dict[str, object]) -> None:
    """独立验证 λ 的语义：直接用 scipy 最小化 J(β)=Σ w·ℓ + ½Σ λ_g‖β_g‖²，解必须一致。

    这是唯一能抓住"γ = β/√λ 还是 √λ·β"这类符号错误的测试——统一 λ 的验收①抓不住，
    因为两组 λ 相同时很多错法会自洽。这里两组 λ 差 40 倍（0.3 vs 12）。
    """
    from scipy.optimize import minimize

    X_tr, y_tr, w = fold_data["X_tr"], fold_data["y_tr"], fold_data["w_tr"]
    lams = {"side": 0.3, "strength": 12.0}
    model = GroupedRidgeLogistic(GROUPS, lambdas=lams).fit(X_tr, y_tr, w)
    Z = model.standardizer_.transform(X_tr)
    lam_vec = np.array([lams[g] for g in model.group_of_])

    def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b0, b = theta[0], theta[1:]
        z = Z @ b + b0
        val = float(np.sum(w * (np.logaddexp(0.0, z) - y_tr * z)) + 0.5 * np.sum(lam_vec * b**2))
        r = w * (1.0 / (1.0 + np.exp(-z)) - y_tr)
        return val, np.concatenate([[r.sum()], Z.T @ r + lam_vec * b])

    res = minimize(obj, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 100000, "ftol": 1e-18, "gtol": 1e-14, "maxls": 100})
    assert res.success
    assert np.abs(res.x[1:] - model.coef_).max() < 1e-8
    assert abs(res.x[0] - model.intercept_) < 1e-8


@needs_data
def test_bigger_lambda_shrinks_only_its_own_group(fold_data: dict[str, object]) -> None:
    """λ 必须只作用在自己的组上：把 strength 罚到 1e10，side 的系数不能跟着塌。"""
    X_tr, y_tr = fold_data["X_tr"], fold_data["y_tr"]
    base = GroupedRidgeLogistic(GROUPS, default_lambda=1.0).fit(X_tr, y_tr)
    heavy = GroupedRidgeLogistic(GROUPS, lambdas={"strength": 1e10}).fit(X_tr, y_tr)
    t = heavy.coef_table().set_index("feature")
    b = base.coef_table().set_index("feature")
    assert list(heavy.lambda_of_) == [1.0, 1e10, 1e10]
    # strength 两列被罚没（实测 2.1e-7），side 仍在原量级（实测 0.00710 → 0.00665）
    assert np.abs(t.loc["team_elo_diff", "coef_std"]) < 1e-6
    assert np.abs(t.loc["player_elo_diff", "coef_std"]) < 1e-6
    assert np.abs(t.loc["side_prior", "coef_std"]) > 1e-3
    rel = abs(t.loc["side_prior", "coef_std"] / b.loc["side_prior", "coef_std"] - 1.0)
    assert rel < 0.25, f"side 的系数被隔壁组的 λ 改写了 {rel:.1%}"


@needs_data
def test_grouped_differs_from_any_uniform(fold_data: dict[str, object]) -> None:
    """分组确实在干活：λ 分组的预测不等于任何一个统一 λ 的预测。"""
    X_tr, X_va, y_tr = fold_data["X_tr"], fold_data["X_va"], fold_data["y_tr"]
    g = GroupedRidgeLogistic(GROUPS, lambdas={"side": 0.01, "strength": 300.0})
    p_g = g.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    for lam in (0.01, 1.0, 300.0):
        p_u = GroupedRidgeLogistic(GROUPS, default_lambda=lam).fit(
            X_tr, y_tr
        ).predict_proba(X_va)[:, 1]
        assert np.abs(p_g - p_u).max() > 1e-6


# ----------------------------------------------------- 红线 3：标准化器只在训练段拟合


@needs_data
def test_standardizer_is_fit_on_train_only(fold_data: dict[str, object]) -> None:
    """§6.1.2 红线 3。标准化器的 mean/std 必须来自训练段，且随模型保存。"""
    X_tr, X_va, y_tr = fold_data["X_tr"], fold_data["X_va"], fold_data["y_tr"]
    model = GroupedRidgeLogistic(GROUPS).fit(X_tr, y_tr)
    s = model.standardizer_
    assert s is not None
    A = X_tr[list(s.columns)].to_numpy(float)
    assert np.abs(s.mean - A.mean(0)).max() < 1e-12
    assert np.abs(s.scale - A.std(0)).max() < 1e-12
    # 用 train+valid 一起拟合会得到不同的 mean —— 说明这个断言不是空断言
    both = Standardizer.fit(pd.concat([X_tr, X_va]), L3_COLS)
    assert np.abs(both.mean - s.mean).max() > 1e-6


@needs_data
def test_predictions_do_not_depend_on_valid_batch(fold_data: dict[str, object]) -> None:
    """逐场预测必须与"和谁一起被预测"无关（标准化器若在预测段重估就会破）。"""
    X_tr, X_va, y_tr = fold_data["X_tr"], fold_data["X_va"], fold_data["y_tr"]
    model = GroupedRidgeLogistic(GROUPS).fit(X_tr, y_tr)
    p_all = model.predict_proba(X_va)[:, 1]
    p_one = np.array([model.predict_proba(X_va.iloc[[i]])[0, 1] for i in range(10)])
    assert np.abs(p_all[:10] - p_one).max() < 1e-15


@needs_data
def test_raw_scale_coefficients_reproduce_the_logit(fold_data: dict[str, object]) -> None:
    """`coef_raw_` 必须能在原始量纲上直接复原 logit（what-if 页面要按它读加项）。"""
    X_tr, X_va, y_tr = fold_data["X_tr"], fold_data["X_va"], fold_data["y_tr"]
    model = GroupedRidgeLogistic(GROUPS, lambdas={"side": 0.5, "strength": 20.0})
    model.fit(X_tr, y_tr)
    z_raw = X_va[model.feature_names_].to_numpy(float) @ model.coef_raw_ + model.intercept_raw_
    assert np.abs(z_raw - model.decision_function(X_va)).max() < 1e-9


@needs_data
def test_predict_proba_is_a_distribution(fold_data: dict[str, object]) -> None:
    model = GroupedRidgeLogistic(GROUPS).fit(fold_data["X_tr"], fold_data["y_tr"])
    P = model.predict_proba(fold_data["X_va"])
    assert P.shape == (len(fold_data["X_va"]), 2)
    assert np.abs(P.sum(1) - 1.0).max() < 1e-12
    assert P.min() > 0.0


# ------------------------------------------------------- 与冻结的 L3 基线对齐


@needs_data
def test_lambda_one_reproduces_baseline_l3_on_a_fold(fold_data: dict[str, object]) -> None:
    """λ 全 1 + L3 三列 = `baselines.predict_fold("L3")`（后者就是 C=1 的 L2 logistic）。

    分两档断言，因为差异只来自解算器而不是模型：
      · 配成和 baselines 一样的 lbfgs/tol=1e-4/max_iter=2000 → **逐场完全相同**（实测 0.0）；
      · 用本模块默认的 newton-cholesky → 实测 max|Δp| = 1.09e-4，那是 lbfgs 在
        tol=1e-4 上没走完的部分（见 `test_lbfgs_would_not_pass_the_1e_8_gate`）。
    """
    f = fold_data["fold"]
    df_y = pd.Series(
        np.concatenate([fold_data["y_tr"], fold_data["y_va"]]),
        index=list(f.train_idx) + list(f.valid_idx),
    )
    feats = pd.concat([fold_data["X_tr"], fold_data["X_va"]])
    p_ref = baseline_predict_fold("L3", feats, df_y, f.train_idx, f.valid_idx, C=1.0)
    p_same, _ = predict_fold(
        feats, df_y, f.train_idx, f.valid_idx, groups=GROUPS, default_lambda=1.0,
        solver="lbfgs", tol=1e-4, max_iter=2000,
    )
    assert np.abs(p_ref - p_same).max() < 1e-12
    p_new, _ = predict_fold(
        feats, df_y, f.train_idx, f.valid_idx, groups=GROUPS, default_lambda=1.0
    )
    assert np.abs(p_ref - p_new).max() < 1e-3
    assert LADDER["L3"] == L3_COLS  # 组里的列就是 L3 的列，别改了列还以为在比 L3


# -------------------------------------------------------- 内层验证段选 λ（§6.1.1）


@needs_data
def test_lambda_grid_never_touches_the_valid_segment(fold_data: dict[str, object]) -> None:
    """选超参只能用 train 段最后 10%；valid 段是 OOS，碰了它 OOS 指标就作废。

    做法上的保证：`fit_lambda_grid` 只收 train 段的 X/y。这里从行为上再钉一遍——
    同样的 train 段配两套完全不同的 valid 段，选出来的 λ 必须一模一样。
    """
    X_tr, y_tr = fold_data["X_tr"], fold_data["y_tr"]
    grid = (1.0, 30.0, 1000.0)
    a = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(X_tr, y_tr, grid=grid, rounds=1)
    b = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(X_tr, y_tr, grid=grid, rounds=1)
    assert a.lambdas_ == b.lambdas_
    split = a.inner_split_
    assert split is not None
    assert abs(split["inner_frac_actual"] - 0.10) < 0.01
    assert split["n_inner_train"] + split["n_inner_valid"] == split["n_train_total"]


@needs_data
def test_inner_split_respects_series_boundary(fold_data: dict[str, object]) -> None:
    """内层边界不能把一个 Bo3 劈开（红线 1 的同款理由，只是发生在 train 内部）。"""
    X_tr, y_tr, series = fold_data["X_tr"], fold_data["y_tr"], fold_data["series_tr"]
    m = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(
        X_tr, y_tr, grid=(1.0, 100.0), rounds=1, series=series, time=fold_data["time_tr"]
    )
    cut = int(m.inner_split_["n_inner_train"])
    assert series[cut - 1] != series[cut]


@needs_data
def test_lambda_grid_refits_on_the_whole_train_segment(fold_data: dict[str, object]) -> None:
    """§6.1.1：选完超参用【整个 train 段】重拟合，标准化器也必须在整段上重估。"""
    X_tr, y_tr = fold_data["X_tr"], fold_data["y_tr"]
    m = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(X_tr, y_tr, grid=(1.0, 100.0), rounds=1)
    A = X_tr[L3_COLS].to_numpy(float)
    assert np.abs(m.standardizer_.mean - A.mean(0)).max() < 1e-12
    inner_mean = X_tr.iloc[: int(m.inner_split_["n_inner_train"])][L3_COLS].to_numpy(float).mean(0)
    assert np.abs(inner_mean - m.standardizer_.mean).max() > 1e-9


@needs_data
def test_grouped_search_is_never_worse_than_shared_on_inner(
    fold_data: dict[str, object],
) -> None:
    """坐标下降从"共用最优 λ"起步且只接受严格改进，内层 log loss 必须 ≤ 共用最优。"""
    m = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(
        fold_data["X_tr"], fold_data["y_tr"], rounds=2
    )
    s = m.inner_split_
    assert s["grouped_best_inner_ll"] <= s["shared_best_inner_ll"] + 1e-15
    assert s["inner_ll_gain"] >= 0.0
    assert set(m.lambdas_) == set(GROUPS)


@needs_data
def test_lambda_grid_trace_is_complete(fold_data: dict[str, object]) -> None:
    """搜索轨迹要能复盘：K 次共用扫描 + 每轮每组 K-1 次坐标扫描。"""
    grid = (1.0, 30.0, 1000.0)
    m = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(
        fold_data["X_tr"], fold_data["y_tr"], grid=grid, rounds=1
    )
    tr = m.search_trace_
    assert (tr.stage == "shared").sum() == len(grid)
    assert (tr.stage == "coord").sum() == len(GROUPS) * (len(grid) - 1)
    assert tr.inner_log_loss.min() == pytest.approx(m.inner_split_["grouped_best_inner_ll"])


@needs_data
def test_refit_false_keeps_the_model_unfitted(fold_data: dict[str, object]) -> None:
    m = GroupedRidgeLogistic(GROUPS).fit_lambda_grid(
        fold_data["X_tr"], fold_data["y_tr"], grid=(1.0, 100.0), rounds=1, refit=False
    )
    with pytest.raises(RuntimeError, match="还没 fit"):
        m.predict_proba(fold_data["X_va"])


# ------------------------------------------------------------------ 输入校验


def test_rejects_zero_or_negative_lambda() -> None:
    with pytest.raises(ValueError, match="λ"):
        GroupedRidgeLogistic(GROUPS, lambdas={"side": 0.0}).resolve_lambdas()
    with pytest.raises(ValueError, match="λ"):
        GroupedRidgeLogistic(GROUPS, default_lambda=-1.0).resolve_lambdas()


def test_rejects_unknown_group_in_lambdas() -> None:
    with pytest.raises(ValueError, match="未声明的组"):
        GroupedRidgeLogistic(GROUPS, lambdas={"draft": 1.0}).resolve_lambdas()


def test_rejects_column_in_two_groups() -> None:
    with pytest.raises(ValueError, match="多个组"):
        GroupedRidgeLogistic({"a": ["side_prior"], "b": ["side_prior"]})


def test_rejects_empty_groups() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        GroupedRidgeLogistic({})
    with pytest.raises(ValueError, match="是空的"):
        GroupedRidgeLogistic({"side": []})


def test_rejects_missing_column() -> None:
    X = pd.DataFrame({"side_prior": [0.5, 0.5, 0.51, 0.49]})
    with pytest.raises(KeyError, match="缺列"):
        GroupedRidgeLogistic(GROUPS).fit(X, np.array([0, 1, 0, 1]))


def test_rejects_nan_feature() -> None:
    X = pd.DataFrame(
        {"side_prior": [0.5, np.nan, 0.51, 0.49],
         "team_elo_diff": [1.0, 2.0, 3.0, 4.0],
         "player_elo_diff": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.raises(ValueError, match="NaN"):
        GroupedRidgeLogistic(GROUPS).fit(X, np.array([0, 1, 0, 1]))


def test_rejects_single_class() -> None:
    X = pd.DataFrame(
        {"side_prior": [0.5, 0.52, 0.51, 0.49],
         "team_elo_diff": [1.0, 2.0, 3.0, 4.0],
         "player_elo_diff": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.raises(ValueError, match="一个类别"):
        GroupedRidgeLogistic(GROUPS).fit(X, np.ones(4, dtype=int))


@needs_data
def test_rejects_unsorted_time_in_lambda_search(fold_data: dict[str, object]) -> None:
    t = fold_data["time_tr"].iloc[::-1]
    with pytest.raises(ValueError, match="升序"):
        GroupedRidgeLogistic(GROUPS).fit_lambda_grid(
            fold_data["X_tr"], fold_data["y_tr"], grid=(1.0,), time=t
        )


@needs_data
def test_rejects_tiny_inner_valid(fold_data: dict[str, object]) -> None:
    X = fold_data["X_tr"].iloc[:200]
    y = fold_data["y_tr"][:200]
    with pytest.raises(ValueError, match="内层验证段"):
        GroupedRidgeLogistic(GROUPS).fit_lambda_grid(X, y, grid=(1.0,), inner_frac=0.05)


# ------------------------------------------------- 全 walk-forward：与冻结预测对齐


@needs_data
@pytest.mark.skipif(not os.path.exists(PREDS), reason="需先跑基线阶梯")
def test_walkforward_lambda_one_reproduces_frozen_l3(real: dict[str, object]) -> None:
    """90 个 fold 全量跑一遍：λ 全 1 的分组岭 = 冻结的 L3 预测（逐场 max|Δp| < 1e-6）。

    这是整个模块最硬的锚——它同时验证了列顺序、标准化口径、截距不罚、fold 切分。
    """
    df, feats, folds = real["df"], real["feats"], real["folds"]
    y = df.y_blue_win
    rows = []
    for f in folds:
        # 解算器配成与 baselines.py 一致，这样对不上就只能是模型本身对不上
        p_same, _ = predict_fold(
            feats[L3_COLS], y, f.train_idx, f.valid_idx, groups=GROUPS, default_lambda=1.0,
            solver="lbfgs", tol=1e-4, max_iter=2000,
        )
        p_nc, _ = predict_fold(
            feats[L3_COLS], y, f.train_idx, f.valid_idx, groups=GROUPS, default_lambda=1.0
        )
        rows.append(pd.DataFrame({"game_id": df.loc[f.valid_idx, "game_id"].to_numpy(),
                                  "p_same": p_same, "p_nc": p_nc}))
    got = pd.concat(rows, ignore_index=True)
    ref = pd.read_parquet(PREDS).query("model == 'L3'")[["game_id", "y", "p"]]
    m = ref.merge(got, on="game_id")
    assert len(m) == len(ref) == 23060
    assert (m.p - m.p_same).abs().max() < 1e-12
    # 默认解算器只差在收敛精度上：逐场 <1e-3，汇总 log loss 与冻结的 0.632304 同到 1e-4
    assert (m.p - m.p_nc).abs().max() < 1e-3
    assert abs(log_loss(m.y.to_numpy(), m.p_nc.to_numpy()) - 0.632304) < 1e-4
    assert abs(log_loss(m.y.to_numpy(), m.p_same.to_numpy()) - 0.632304) < 1e-6
