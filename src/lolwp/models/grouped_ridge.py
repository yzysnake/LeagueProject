"""分组 L2 正则化 logistic regression（PLAN.md §5.3，主力模型形式）。

为什么是"分组"：不同特征组承载的信息量差别很大。side 先验只有 1 维、信噪比高、
本来就该几乎不收缩；draft 侧的 champ_meta / synergy / counter 是 zero-init 的
弱信号，必须重罚才不会把 15,000 场的样本吃干。用一个全局 C 去同时服务这两类，
要么放纵弱信号过拟合，要么把强信号压死。

=========================== 重参数化的推导（自己推一遍） ===========================
本实现定义的目标函数（w_i 为样本权重，ℓ 为二分类交叉熵，截距不进罚项）：

    J(β, β₀) = Σ_i w_i · ℓ(y_i, x_iᵀβ + β₀) + ½ Σ_g λ_g ‖β_g‖²

令  γ_g = √λ_g · β_g ，同时把设计矩阵按组缩放 X̃_g = X_g / √λ_g ，则

    X̃γ = Σ_g (X_g/√λ_g)(√λ_g β_g) = Σ_g X_g β_g = Xβ        （线性预测子逐点不变）
    ½ Σ_g λ_g‖β_g‖² = ½ Σ_g λ_g ‖γ_g‖²/λ_g = ½‖γ‖²            （罚项变成统一 L2）

于是 J = Σ_i w_i ℓ(·) + ½‖γ‖²，**正好是 sklearn `LogisticRegression(C=1.0)` 的目标**
（sklearn 的目标是 ½‖w‖² + C·Σ w_i ℓ）。拟合完再按 β_g = γ_g/√λ_g 还原。

⚠️ 手册 §5.3.2 的注释写的是「令 γ_g = β_g/√λ_g」，这与它自己下一行的还原式
`coef = clf.coef_[0][sl] / np.sqrt(lam_g)` 自相矛盾——若 γ=β/√λ，还原应当是 β=γ·√λ。
正确的是 **γ_g = √λ_g·β_g**；手册那两行"缩放 X"和"还原系数"的代码都是对的，
错的只有中间那句符号定义。本模块按推导实现，并在
`tests/test_grouped_ridge.py::test_reparam_matches_direct_optimizer` 里
用 scipy 直接最小化 J(β) 做了独立验证（λ_side=0.3 / λ_strength=12，实测 max|Δβ| = 1.1e-11）。

统一 λ 的特例：J = Σ w_i ℓ + (λ/2)‖β‖² = λ·[ (1/λ)Σ w_i ℓ + ½‖β‖² ]，
与 sklearn `LogisticRegression(C=1/λ)` 同解 —— 这就是手册要求的验收①。

=========================== 为什么默认用 newton-cholesky ===========================
手册 §5.3.2 的示例代码用 `solver='lbfgs'`，**但那样过不了它自己定的 1e-8 验收**：
实测（真实 L3 特征，fold 45，n_train=12,719）lbfgs 把 tol 从 1e-4 一路收到 1e-12，
两侧预测概率的 max|Δp| 卡在 **1.7e-7** 不再下降——scipy 的 L-BFGS-B 还有一个
ftol 地板（sklearn 固定为 64·eps），gtol 再小也没用，迭代数根本不再增加。
换 `newton-cholesky`（同样的 λ 与样本权重网格）max|Δp| ≤ **2.2e-16**，即机器精度，
且 3 特征的问题上耗时 <10 ms。所以默认解算器偏离手册，改用 newton-cholesky；
lbfgs 仍可通过 `solver=` 选用（特征上千或稀疏矩阵时它更省内存）。

=========================== 红线 3（§6.1.2）===========================
标准化器只在**训练段**内拟合，拟合结果随模型一起保存（`standardizer_`），
预测时复用；`fit_lambda_grid()` 搜超参时，标准化器只在**内层训练段**拟合。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lolwp.eval.metrics import log_loss

FeatureGroups = Mapping[str, Sequence[str]]
Solver = Literal["newton-cholesky", "lbfgs", "newton-cg"]

DEFAULT_SOLVER: Solver = "newton-cholesky"
DEFAULT_TOL = 1e-10
DEFAULT_MAX_ITER = 200
# λ 网格：跨 6 个数量级。λ=1 与 baselines.py 的 C=1 对齐，便于对照。
# 上界取到 3000 是实测定的——真实 L3 特征上内层最优常落在 100-1000（fold 45 实测
# 内层 log loss 在 λ=300 取到最小 0.613946），网格封顶在 1000 会把最优点顶在边界上。
DEFAULT_GRID: tuple[float, ...] = (
    0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
)


@dataclass(frozen=True)
class Standardizer:
    """z-score 标准化器。**必须在训练段内拟合**（§6.1.2 红线 3），故做成可保存的值对象。

    std 用 ddof=0，与 `eval/baselines.py` 的 `X_tr.std(0)` 口径一致——不一致的话
    λ=1 时就复现不出 L3 的 0.6323。
    """

    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame, columns: Sequence[str]) -> Standardizer:
        A = _as_matrix(X, columns)
        mean = A.mean(axis=0)
        scale = A.std(axis=0)
        # 常数列（例如某 fold 内 is_playoffs 全 0）：除以 1，系数自然为 0，不要产生 inf
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        return cls(tuple(columns), mean, scale)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return (_as_matrix(X, self.columns) - self.mean) / self.scale


class GroupedRidgeLogistic:
    """分组 L2 logistic：每个特征组一个 λ_g，靠列缩放重参数化后交给 sklearn。

    参数
    ----
    groups : {组名: [列名, ...]}，例如
        {"side": ["side_prior"], "strength": ["team_elo_diff", "player_elo_diff"]}
        列的顺序 = 组的声明顺序 × 组内声明顺序，`coef_` 与之对齐。
    lambdas : {组名: λ}，缺席的组用 `default_lambda`（手册要求"未指定的组用默认 λ"）。
        λ 必须 > 0：λ=0 在重参数化里是 1/√0，不支持；要"几乎不罚"请给 1e-6。
    standardize : 是否在 fit 内做 z-score。关掉的话 λ 的含义依赖特征量纲，
        除非调用方自己已经标准化过，否则不要关。
    """

    def __init__(
        self,
        groups: FeatureGroups,
        *,
        lambdas: Mapping[str, float] | None = None,
        default_lambda: float = 1.0,
        standardize: bool = True,
        fit_intercept: bool = True,
        solver: Solver = DEFAULT_SOLVER,
        tol: float = DEFAULT_TOL,
        max_iter: int = DEFAULT_MAX_ITER,
    ) -> None:
        self.groups: dict[str, list[str]] = {g: list(c) for g, c in groups.items()}
        _validate_groups(self.groups)
        self.lambdas: dict[str, float] = dict(lambdas or {})
        self.default_lambda = float(default_lambda)
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.solver: Solver = solver
        self.tol = tol
        self.max_iter = max_iter

        self.feature_names_: list[str] = [c for cols in self.groups.values() for c in cols]
        self.group_of_: list[str] = [g for g, cols in self.groups.items() for _ in cols]
        # fit 后才有的东西
        self.standardizer_: Standardizer | None = None
        self.lambdas_: dict[str, float] = {}
        self.lambda_of_: np.ndarray = np.empty(0)
        self.gamma_: np.ndarray = np.empty(0)      # 重参数化空间里的系数（sklearn 直接给的）
        self.coef_: np.ndarray = np.empty(0)       # 还原到"标准化后特征"尺度的 β
        self.intercept_: float = 0.0
        self.coef_raw_: np.ndarray = np.empty(0)   # 再还原到原始特征量纲的 β
        self.intercept_raw_: float = 0.0
        self.n_iter_: int = 0
        self.search_trace_: pd.DataFrame | None = None
        self.inner_split_: dict[str, float] | None = None

    # ------------------------------------------------------------------ 拟合

    def resolve_lambdas(self) -> dict[str, float]:
        """把 {组名: λ} 补全成每个组都有值，并校验。"""
        out: dict[str, float] = {}
        unknown = set(self.lambdas) - set(self.groups)
        if unknown:
            raise ValueError(f"lambdas 里有未声明的组：{sorted(unknown)}")
        for g in self.groups:
            lam = float(self.lambdas.get(g, self.default_lambda))
            if not np.isfinite(lam) or lam <= 0:
                raise ValueError(f"组 {g!r} 的 λ={lam} 非法：必须是有限正数（λ=0 请用 1e-6 近似）")
            out[g] = lam
        return out

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        sample_weight: np.ndarray | pd.Series | None = None,
    ) -> GroupedRidgeLogistic:
        """在训练段上拟合。X 必须含 groups 里声明的全部列；多余列会被忽略。"""
        y_arr = np.asarray(y).astype(int).ravel()
        if len(np.unique(y_arr)) < 2:
            raise ValueError("训练段只有一个类别，logistic 无解")
        if len(X) != len(y_arr):
            raise ValueError(f"X 有 {len(X)} 行而 y 有 {len(y_arr)} 个")

        self.lambdas_ = self.resolve_lambdas()
        self.lambda_of_ = np.array([self.lambdas_[g] for g in self.group_of_], dtype=float)

        if self.standardize:
            self.standardizer_ = Standardizer.fit(X, self.feature_names_)
            Z = self.standardizer_.transform(X)
        else:
            self.standardizer_ = None
            Z = _as_matrix(X, self.feature_names_)

        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        if w is not None and len(w) != len(y_arr):
            raise ValueError(f"sample_weight 长度 {len(w)} 与样本数 {len(y_arr)} 不符")

        est = self._fit_reparam(Z, y_arr, w, self.lambda_of_)
        self.gamma_ = est.coef_[0].copy()
        self.coef_ = self.gamma_ / np.sqrt(self.lambda_of_)
        self.intercept_ = float(est.intercept_[0]) if self.fit_intercept else 0.0
        self.n_iter_ = int(np.asarray(est.n_iter_).ravel()[0])
        self._restore_raw_scale()
        return self

    def _fit_reparam(
        self, Z: np.ndarray, y: np.ndarray, w: np.ndarray | None, lam: np.ndarray
    ) -> LogisticRegression:
        """核心三行：列缩放 → 统一强度 L2（C=1）→ 由调用方还原。"""
        est = LogisticRegression(
            # sklearn 1.8 起 penalty= 已弃用，纯 L2 用 l1_ratio=0 表达（metrics.py 同款处理）
            l1_ratio=0.0,
            C=1.0,
            solver=self.solver,
            tol=self.tol,
            max_iter=self.max_iter,
            fit_intercept=self.fit_intercept,
        )
        est.fit(Z / np.sqrt(lam), y, sample_weight=w)
        return est

    def _restore_raw_scale(self) -> None:
        """把标准化也还原掉，给出原始量纲下的 β（what-if 页面要直接读它做 log-odds 加项）。"""
        if self.standardizer_ is None:
            self.coef_raw_ = self.coef_.copy()
            self.intercept_raw_ = self.intercept_
            return
        s = self.standardizer_
        self.coef_raw_ = self.coef_ / s.scale
        self.intercept_raw_ = float(self.intercept_ - np.dot(self.coef_raw_, s.mean))

    # ------------------------------------------------------------------ 预测

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if self.standardizer_ is not None:
            Z = self.standardizer_.transform(X)
        else:
            Z = _as_matrix(X, self.feature_names_)
        return Z @ self.coef_ + self.intercept_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回 (n, 2)，列顺序与 sklearn 一致：[P(y=0), P(y=1)]。"""
        p1 = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def coef_table(self) -> pd.DataFrame:
        """系数表：组、λ、标准化尺度系数、原尺度系数。sanity check 对照手册 §5.3.4。"""
        self._check_fitted()
        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "group": self.group_of_,
                "lambda": self.lambda_of_,
                "coef_std": self.coef_,
                "coef_raw": self.coef_raw_,
            }
        )

    def _check_fitted(self) -> None:
        if not len(self.coef_):
            raise RuntimeError("模型还没 fit")

    # ------------------------------------------------------- 内层验证段搜 λ

    def fit_lambda_grid(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        grid: Sequence[float] = DEFAULT_GRID,
        sample_weight: np.ndarray | pd.Series | None = None,
        inner_frac: float = 0.10,
        series: Sequence[str] | np.ndarray | None = None,
        time: pd.Series | None = None,
        rounds: int = 2,
        min_inner_valid: int = 50,
        refit: bool = True,
    ) -> GroupedRidgeLogistic:
        """在 **train 段最后 inner_frac** 上搜每组的 λ（§6.1.1 的内层结构）。

        **绝不能用 walk-forward 的 valid 段选超参**——那是 OOS，用它调参等于把
        测试集看了一遍，报出来的 OOS 指标就不再是 OOS 了。这里只切训练段内部。

        X / y 必须按时间升序（传 `time` 会硬断言这一点）。传 `series` 时，
        内层边界会往前挪到 series 边界上，避免同一个 Bo3 劈在内层 train/valid 两侧
        （红线 1 的同款理由；内层泄漏虽然只影响选超参，但它会系统性地偏向小 λ）。

        搜索策略：先在"所有组共用一个 λ"的一维网格上定起点，再按组坐标下降
        `rounds` 轮。组数 G、网格 K 时，代价是 K + rounds·G·(K−1) 次拟合，
        而全网格是 K^G —— 4 组 8 格就是 4,096 次 × 90 个 origin，没必要。
        坐标下降每一步只在严格更优时才接受，因此最终 inner log loss 一定 ≤ 共用 λ 的最优值。
        """
        y_arr = np.asarray(y).astype(int).ravel()
        grid_arr = np.asarray([float(g) for g in grid], dtype=float)
        if grid_arr.size == 0 or np.any(grid_arr <= 0) or not np.all(np.isfinite(grid_arr)):
            raise ValueError("grid 必须非空且全为有限正数")
        if time is not None and not pd.Series(time).is_monotonic_increasing:
            raise ValueError("X/y 必须按 kickoff_utc 升序，内层切分依赖这个顺序")

        n = len(X)
        cut = _inner_cut(n, inner_frac, series)
        n_valid = n - cut
        if n_valid < min_inner_valid:
            raise ValueError(
                f"内层验证段只有 {n_valid} 场（< {min_inner_valid}），选不动超参；"
                "要么放大 inner_frac，要么在这个 fold 上退回默认 λ"
            )
        if len(np.unique(y_arr[cut:])) < 2 or len(np.unique(y_arr[:cut])) < 2:
            raise ValueError("内层 train 或 valid 只有一个类别，无法选超参")

        X_in, X_va = X.iloc[:cut], X.iloc[cut:]
        y_in, y_va = y_arr[:cut], y_arr[cut:]
        w_all = None if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        w_in = None if w_all is None else w_all[:cut]

        # 标准化器只在内层训练段拟合（红线 3 在内层同样成立）
        if self.standardize:
            std = Standardizer.fit(X_in, self.feature_names_)
            Z_in, Z_va = std.transform(X_in), std.transform(X_va)
        else:
            Z_in = _as_matrix(X_in, self.feature_names_)
            Z_va = _as_matrix(X_va, self.feature_names_)

        names = list(self.groups)
        trace: list[dict[str, object]] = []

        def score(lam_map: dict[str, float]) -> float:
            lam_vec = np.array([lam_map[g] for g in self.group_of_], dtype=float)
            est = self._fit_reparam(Z_in, y_in, w_in, lam_vec)
            beta = est.coef_[0] / np.sqrt(lam_vec)
            p = _sigmoid(Z_va @ beta + (est.intercept_[0] if self.fit_intercept else 0.0))
            return log_loss(y_va, p)

        # 阶段一：共用 λ 的一维扫描，给坐标下降一个像样的起点
        best_shared, best_ll = float(grid_arr[0]), np.inf
        for lam in grid_arr:
            ll = score(dict.fromkeys(names, float(lam)))
            trace.append({"stage": "shared", "round": 0, "group": "*", "lam": float(lam),
                          "inner_log_loss": ll})
            if ll < best_ll:
                best_shared, best_ll = float(lam), ll
        shared_ll = best_ll
        cur = dict.fromkeys(names, best_shared)

        # 阶段二：按组坐标下降
        for r in range(1, rounds + 1):
            improved = False
            for g in names:
                keep = cur[g]
                for lam in grid_arr:
                    if float(lam) == keep:
                        continue
                    cand = dict(cur, **{g: float(lam)})
                    ll = score(cand)
                    trace.append({"stage": "coord", "round": r, "group": g,
                                  "lam": float(lam), "inner_log_loss": ll})
                    if ll < best_ll:
                        best_ll, cur, improved = ll, cand, True
            if not improved:
                break

        self.lambdas = dict(cur)
        self.search_trace_ = pd.DataFrame(trace)
        self.inner_split_ = {
            "n_train_total": float(n),
            "n_inner_train": float(cut),
            "n_inner_valid": float(n_valid),
            "inner_frac_actual": n_valid / n,
            "shared_best_lambda": best_shared,
            "shared_best_inner_ll": shared_ll,
            "grouped_best_inner_ll": best_ll,
            "inner_ll_gain": shared_ll - best_ll,
        }
        if refit:
            # §6.1.1：选完超参后用【整个 train 段】重拟合（标准化器也在整段上重估）
            self.fit(X, y_arr, sample_weight)
        return self


# ---------------------------------------------------------------- fold 级便利函数


def predict_fold(
    feats: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    *,
    groups: FeatureGroups,
    lambdas: Mapping[str, float] | None = None,
    default_lambda: float = 1.0,
    grid: Sequence[float] | None = None,
    sample_weight: pd.Series | None = None,
    series: pd.Series | None = None,
    time: pd.Series | None = None,
    inner_frac: float = 0.10,
    **kwargs: object,
) -> tuple[np.ndarray, GroupedRidgeLogistic]:
    """单 fold 训练 + 预测，签名对齐 `eval/baselines.py::predict_fold`，方便进对比表。

    `grid` 为 None 时用给定的 λ 直接拟合；给了 grid 就在 train 段内层选 λ 再整段重拟合。
    """
    model = GroupedRidgeLogistic(
        groups, lambdas=lambdas, default_lambda=default_lambda, **kwargs  # type: ignore[arg-type]
    )
    X_tr, X_va = feats.loc[train_idx], feats.loc[valid_idx]
    y_tr = y.loc[train_idx]
    w_tr = None if sample_weight is None else sample_weight.loc[train_idx]
    if grid is None:
        model.fit(X_tr, y_tr, w_tr)
    else:
        model.fit_lambda_grid(
            X_tr,
            y_tr,
            grid=grid,
            sample_weight=w_tr,
            inner_frac=inner_frac,
            series=None if series is None else series.loc[train_idx].to_numpy(),
            time=None if time is None else time.loc[train_idx],
        )
    return model.predict_proba(X_va)[:, 1], model


# ---------------------------------------------------------------------- 内部工具


def _validate_groups(groups: Mapping[str, Sequence[str]]) -> None:
    if not groups:
        raise ValueError("groups 不能为空")
    seen: set[str] = set()
    for g, cols in groups.items():
        if not len(cols):
            raise ValueError(f"组 {g!r} 是空的")
        dup = seen & set(cols)
        if dup:
            raise ValueError(f"列 {sorted(dup)} 出现在多个组里，λ 会有歧义")
        seen |= set(cols)


def _as_matrix(X: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise KeyError(f"特征表缺列：{missing}")
    A = X.loc[:, list(columns)].to_numpy(dtype=float)
    if not np.all(np.isfinite(A)):
        bad = [c for c, ok in zip(columns, np.all(np.isfinite(A), axis=0)) if not ok]
        raise ValueError(f"特征含 NaN/inf：{bad}")
    return A


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _inner_cut(n: int, frac: float, series: Sequence[str] | np.ndarray | None) -> int:
    """内层验证段的起始位置：最后 frac 比例，必要时往前挪到 series 边界。"""
    if not 0.0 < frac < 1.0:
        raise ValueError(f"inner_frac 必须在 (0,1) 内，给的是 {frac}")
    cut = n - max(1, round(n * frac))
    if series is None:
        return max(cut, 1)
    s = np.asarray(series)
    while 1 < cut < n and s[cut - 1] == s[cut]:
        cut -= 1
    return max(cut, 1)
