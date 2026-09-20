"""概率校准：测它、监控它，而不是无脑加一层（PLAN.md §5.6 / §9.7 ①）。

为什么这个模块存在而不是直接 `CalibratedClassifierCV`
--------------------------------------------------------
1. **方法选择由校准集样本量决定，不由方法优劣决定**（§5.6.2）。
   手册记录的方法-样本量交互是决定性的：同一个 isotonic vs Platt 的改动，
   在 ~35,000 个 in-game 样本上 isotonic 赢（配对 t = +6.91），
   在 1,716 场 pre/post-draft 样本上 Platt 赢（t = −4.74）。
   本项目的校准集是"每个 fold 训练段的内层留出片"，量级几千场 → 属于后者。
   `choose_method()` 把这条规则写成代码，而不是留给调用方凭印象选。
2. **校准器必须在每个 fold 的训练段内拟合**（§6.1.2 红线 3）。
   用 OOS 拟合再评自己是自证式泄漏：它一定让 ECE 变好，且完全不可复现到线上。
   `fit_in_fold()` 的签名强制调用方传入"内层拟合段/内层校准段"两段索引，
   让"拿 valid 拟合校准器"这件事写起来别扭到不会顺手发生。
3. **主力模型很可能根本不需要校准层**（§5.6.1）。用 log loss 直接训练的 logistic
   天生就是校准的。所以这个模块的主产品其实是 `calibration_gate()` 与
   `stratified_report()`——**监控**，不是**修正**。`Identity` 是一等公民，
   不是占位符；`choose_method()` 在"现状已达标"时返回它。
4. **ECE 不能报 bootstrap 置信区间**（`eval/compare.py` 实测 bias/se = +1.88，
   百分位区间整段落在点估计之上）。本模块只给 ECE 点估计，并额外给出
   冻结分箱版（`eval.compare.fixed_bin_ece`）以便需要时做有偏但一致的比较。

分层为什么比全局重要（§5.6.5）
--------------------------------
跨区 rating 锚定不准会在国际赛上产生系统性偏差，而全局 ECE 会把它平均掉：
一组 +0.03、另一组 −0.03，总体看起来完美。`stratified_report()` 因此是
本模块唯一的"主判据"入口，`calibration_gate()` 把总体与分层阈值一起判。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from lolwp.eval.metrics import brier, calibration_slope, ece, log_loss

EPS = 1e-12

Method = Literal["identity", "temperature", "platt", "isotonic"]

# 验收③ 的阈值（阶段 8 任务书）：总体 ECE ≤ 0.02，分层各组 ECE ≤ 0.03，slope ∈ [0.90, 1.10]
ECE_OVERALL_MAX = 0.02
ECE_GROUP_MAX = 0.03
SLOPE_LO, SLOPE_HI = 0.90, 1.10
# 分层单元小于这个样本量时，ECE/slope 的抽样噪声已经大于阈值本身，只报不判。
MIN_GROUP_N = 300


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)


def logit(p: np.ndarray) -> np.ndarray:
    q = _clip(p)
    return np.log(q / (1 - q))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ---------------------------------------------------------------------------
# 1. 校准器
# ---------------------------------------------------------------------------
class Calibrator(Protocol):
    """全部校准器的统一接口：在 (p, y) 上拟合，对 p 做单调变换。

    约定：`transform` 必须是**单调不减**的，否则 AUC 会被改变——
    校准层只允许改概率的刻度，不允许改排序。`assert_monotone()` 用来验这件事。
    """

    name: str
    n_params: int

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator: ...

    def transform(self, p: np.ndarray) -> np.ndarray: ...


@dataclass
class Identity:
    """不校准。**§5.6.1 的默认答案**，不是占位符。

    主力 logistic 用 log loss 训练，天生近似校准；加一层校准器要额外切掉
    一段训练数据去拟合它，这个代价是真实的（见 stage8 报告里的三方对比）。
    """

    name: str = "identity"
    n_params: int = 0

    def fit(self, p: np.ndarray, y: np.ndarray) -> Identity:
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)


@dataclass
class Temperature:
    """单参数温度标定：p' = σ(logit(p) / T)。**最省数据的校准器**。

    只能修"整体过度自信/过度保守"（slope ≠ 1），修不了系统性偏移（intercept ≠ 0）。
    在本项目里这恰好是主要失效模式（L3 的 slope 0.946、intercept +0.013），
    所以它的代价-收益比通常比 Platt 好。

    拟合方式：对 1/T 做无截距 logistic 回归的等价闭式——直接用一维牛顿法最小化
    NLL，比调 sklearn 快且不引入正则（sklearn 的 C 默认不是无穷）。
    """

    T: float = 1.0
    name: str = "temperature"
    n_params: int = 1

    def fit(self, p: np.ndarray, y: np.ndarray, *, max_iter: int = 100) -> Temperature:
        z = logit(p)
        yy = np.asarray(y, dtype=float)
        b = 1.0  # b = 1/T
        for _ in range(max_iter):
            q = sigmoid(b * z)
            g = float(np.sum((q - yy) * z))
            h = float(np.sum(q * (1 - q) * z * z))
            if h <= EPS:
                break
            step = g / h
            b -= step
            if abs(step) < 1e-10:
                break
        self.T = float(1.0 / b) if abs(b) > EPS else float("inf")
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return sigmoid(logit(p) / self.T)


@dataclass
class Platt:
    """两参数 Platt：p' = σ(a · logit(p) + b)。§5.6.2 在几千场校准集上的默认选择。

    注意 §5.6.4：Platt 会把输出压缩到一个**真实存在的截断区间**
    （手册给的参考值约 [0.18, 0.84]，具体值必须由训练时实测写入 `calib_range`）。
    这不是 bug，是 Platt 的性质，UI 必须诚实表达。
    **但反过来也成立：不加 Platt 的模型 `calib_range = null`，不许假装有一个边界。**
    """

    a: float = 1.0
    b: float = 0.0
    name: str = "platt"
    n_params: int = 2

    def fit(self, p: np.ndarray, y: np.ndarray) -> Platt:
        from sklearn.linear_model import LogisticRegression

        z = logit(p).reshape(-1, 1)
        # C=inf：校准器本身不能再被正则收缩，否则 a 会被压向 0（= 把信号抹平）
        m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(z, np.asarray(y))
        self.a = float(m.coef_[0][0])
        self.b = float(m.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * logit(p) + self.b)


@dataclass
class Isotonic:
    """非参保序回归。**在本项目的校准集规模上是错误答案**（§5.6.2），实现它是为了能实测这一点。

    它有 O(n) 个有效参数，在几千场上会把噪声拟合成台阶，且输出被硬夹在
    训练集见过的 [min, max] 之间——比 Platt 的软压缩更粗暴。
    """

    x: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    yhat: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    name: str = "isotonic"
    n_params: int = -1  # 非参：有效参数量随 n 增长，用 -1 标记

    def fit(self, p: np.ndarray, y: np.ndarray) -> Isotonic:
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
        pp = np.asarray(p, dtype=float)
        ir.fit(pp, np.asarray(y, dtype=float))
        grid = np.unique(pp)
        self.x = grid
        self.yhat = ir.predict(grid)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(p, dtype=float), self.x, self.yhat)


_CTORS: dict[str, type] = {
    "identity": Identity,
    "temperature": Temperature,
    "platt": Platt,
    "isotonic": Isotonic,
}


def make_calibrator(method: Method) -> Calibrator:
    if method not in _CTORS:
        raise ValueError(f"未知校准方法 {method!r}，可选 {sorted(_CTORS)}")
    return _CTORS[method]()  # type: ignore[return-value]


# 手册 §5.6.2 的两个实测锚点：1,716 场 → Platt 赢；~35,000 样本 → isotonic 赢。
# 阈值取两个锚点之间偏保守的一侧：非参方法要到 20,000 才允许考虑。
ISOTONIC_MIN_N = 20_000
PLATT_MIN_N = 1_000


def choose_method(
    n_calib: int, *, needs_layer: bool = True, fix_offset: bool = True
) -> Method:
    """按**校准集样本量**选方法（§5.6.2），不按方法优劣选。

    参数
    ----
    n_calib :
        校准器实际能看到的样本数（= 每个 fold 内层校准段的场数，不是 OOS 总数）。
    needs_layer :
        现状是否已经不达标。§5.6.1：主力 logistic 天生校准，达标时正确答案是
        `identity`——加层是有代价的（要切走一段训练数据）。
    fix_offset :
        是否需要同时修偏移。只需要修 slope 时 temperature 更省数据。
    """
    if not needs_layer:
        return "identity"
    if n_calib >= ISOTONIC_MIN_N:
        return "isotonic"
    if n_calib >= PLATT_MIN_N and fix_offset:
        return "platt"
    return "temperature"


def assert_monotone(cal: Calibrator, lo: float = 1e-4, hi: float = 1 - 1e-4) -> None:
    """校准层不许改排序：在网格上验单调不减。改了排序 = AUC 被动过 = 实现有 bug。"""
    grid = np.linspace(lo, hi, 2001)
    out = cal.transform(grid)
    d = np.diff(out)
    if float(d.min()) < -1e-9:
        raise AssertionError(f"{cal.name} 的变换不单调，最小步长 {d.min():.3e}")


def output_range(cal: Calibrator, p: np.ndarray | None = None) -> tuple[float, float] | None:
    """§5.6.4 的 `calib_range`：校准后**实际可达**的概率区间。

    - `Identity` 返回 `None` —— 明确表示"不存在截断区间"。
      UI 在 `calib_range == null` 时不得渲染死区、API 不得返回触顶 warning。
    - 其余校准器返回在 `p`（不给就用 [1e-4, 1-1e-4] 全域网格）上的实际像区间。
      **必须实测写入，不许把手册的 [0.18, 0.84] 当常数抄。**
    """
    if isinstance(cal, Identity):
        return None
    grid = np.linspace(1e-4, 1 - 1e-4, 2001) if p is None else np.asarray(p, dtype=float)
    out = cal.transform(grid)
    return float(np.min(out)), float(np.max(out))


# ---------------------------------------------------------------------------
# 2. 在 fold 训练段内拟合（§6.1.2 红线 3）
# ---------------------------------------------------------------------------
def inner_split(
    time: pd.Series | np.ndarray,
    series: Sequence[object] | np.ndarray,
    *,
    calib_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """把一段训练索引按**时间**切成 (内层拟合段, 内层校准段)，且不劈开 series。

    为什么按时间切而不是随机切：校准层要修的是"模型在**它没见过的、更晚的**数据上
    过不过度自信"。随机切出来的校准段与拟合段同期，模型对它的过度自信程度
    比真正的 OOS 低，校准器会被低估 → 上线后仍然过度自信。

    为什么不许劈开 series：同一 series 内多局高度相关（实测 ρ≈0.16），
    劈开等于把校准段的有效样本量灌水。
    """
    t = pd.Series(pd.to_datetime(np.asarray(time), utc=True)).reset_index(drop=True)
    if not t.is_monotonic_increasing:
        raise ValueError("训练段必须按时间升序传入")
    s = np.asarray(series, dtype=object)
    n = len(t)
    cut = round(n * (1.0 - calib_frac))
    cut = max(1, min(n - 1, cut))
    # 往后推到 series 边界，保证切点不落在一个 series 内部
    while cut < n and s[cut] == s[cut - 1]:
        cut += 1
    if cut >= n:
        raise ValueError("训练段只有一个 series，无法切出内层校准段")
    return np.arange(cut), np.arange(cut, n)


def cross_fit_predictions(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    series: Sequence[object] | np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """在**训练段内部**做 series 分组 K 折交叉拟合，给每一行一个样本外预测。

    为什么要它，而不是只用 `inner_split` 的最后 20%
    -------------------------------------------------
    分层校准的瓶颈永远是**小组的校准样本量**：国际赛在整个 2019+ 评测段里只有
    1,698 场，切 20% 只剩几百场（还要再按 fold 拆），Platt 的两个参数拟不稳。
    交叉拟合把整个训练段都变成校准样本（5 倍），代价是这些预测与 valid 段
    **不存在时间落差**——它测不出"模型对更晚的数据更过度自信"这一部分。
    两种口径都要跑一遍，差多少要写进报告，不要只挑好看的那个。

    分组必须按 series：同 series 的多局实测 ρ≈0.16，劈开 series 会让
    校准集的有效样本量灌水，选出来的方法偏向更复杂的那个（isotonic）。
    """
    s = np.asarray(series, dtype=object)
    n = len(s)
    uniq = pd.unique(s)
    rng = np.random.default_rng(seed)
    bucket = dict(zip(uniq, rng.integers(0, k, len(uniq)), strict=True))
    fold_of = np.array([bucket[v] for v in s])
    out = np.full(n, np.nan)
    for j in range(k):
        te = np.flatnonzero(fold_of == j)
        tr = np.flatnonzero(fold_of != j)
        if len(te) == 0 or len(tr) < 50:
            continue
        out[te] = fit_predict(tr, te)
    if np.isnan(out).any():
        raise RuntimeError("交叉拟合有行没拿到预测，检查 k 与 series 数量")
    return out


@dataclass
class GroupwiseCalibrator:
    """分层校准器：每个分层单元一条校准曲线，样本不够的单元退回全局曲线。

    §5.6.5 要求"分别检查并单独校准"，但**单独校准是有前提的**：
    组内校准样本量决定了能拟几个参数。本类把这条前提写成 `min_n` 与
    `fallback`——组内样本不足时**不是不校准，而是用全局校准器**，
    否则小组会被自己组里的噪声反向推坏（实测见 stage8 报告 §4）。

    `method_of_group` 记录每个组实际用了哪个方法与多少校准样本，
    报告里必须原样打印：分层校准最容易出的错就是"以为分层了，其实全在退回"。
    """

    method: Method = "temperature"
    min_n: int = 500
    fallback: Method = "temperature"
    global_cal: Calibrator | None = None
    per_group: dict[object, Calibrator] = field(default_factory=dict)
    method_of_group: dict[object, tuple[str, int]] = field(default_factory=dict)

    def fit(self, p: np.ndarray, y: np.ndarray, groups: Sequence[object] | np.ndarray) -> GroupwiseCalibrator:
        p = np.asarray(p, dtype=float)
        y = np.asarray(y)
        g = np.asarray(groups, dtype=object)
        self.global_cal = make_calibrator(self.fallback)
        self.global_cal.fit(p, y)
        self.per_group, self.method_of_group = {}, {}
        for key in pd.unique(g):
            m = g == key
            n = int(m.sum())
            if n >= self.min_n and 0 < y[m].mean() < 1:
                c = make_calibrator(self.method)
                c.fit(p[m], y[m])
                self.per_group[key] = c
                self.method_of_group[key] = (self.method, n)
            else:
                self.method_of_group[key] = (f"{self.fallback}(全局·n={n}不足)", n)
        return self

    def transform(self, p: np.ndarray, groups: Sequence[object] | np.ndarray) -> np.ndarray:
        if self.global_cal is None:
            raise RuntimeError("未拟合")
        p = np.asarray(p, dtype=float)
        g = np.asarray(groups, dtype=object)
        out = self.global_cal.transform(p)
        for key, c in self.per_group.items():
            m = g == key
            if m.any():
                out[m] = c.transform(p[m])
        return out


@dataclass(frozen=True)
class FoldCalibration:
    """单个 fold 的校准结果。`p_valid_raw` / `p_valid_cal` 供上层拼成 OOS 长表。"""

    fold: int
    method: str
    n_inner_fit: int
    n_inner_calib: int
    params: dict[str, float]
    calib_range: tuple[float, float] | None
    p_valid_raw: np.ndarray
    p_valid_cal: np.ndarray


def fit_in_fold(
    p_calib: np.ndarray,
    y_calib: np.ndarray,
    p_valid: np.ndarray,
    *,
    method: Method | None = None,
    fold: int = -1,
    needs_layer: bool = True,
) -> FoldCalibration:
    """在**内层校准段**上拟合校准器，再作用到 valid 段（§6.1.2 红线 3）。

    这个函数故意**不接收 y_valid**：拿不到标签就不可能用 OOS 拟合校准器。
    这是本模块防泄漏的主要手段——把红线做成签名，而不是做成注释。
    """
    m: Method = method or choose_method(len(p_calib), needs_layer=needs_layer)
    cal = make_calibrator(m)
    cal.fit(np.asarray(p_calib, dtype=float), np.asarray(y_calib))
    assert_monotone(cal)
    params: dict[str, float] = {}
    if isinstance(cal, Platt):
        params = {"a": cal.a, "b": cal.b}
    elif isinstance(cal, Temperature):
        params = {"T": cal.T}
    return FoldCalibration(
        fold=fold,
        method=m,
        n_inner_fit=0,
        n_inner_calib=len(p_calib),
        params=params,
        calib_range=output_range(cal, np.asarray(p_valid, dtype=float)),
        p_valid_raw=np.asarray(p_valid, dtype=float),
        p_valid_cal=cal.transform(np.asarray(p_valid, dtype=float)),
    )


# ---------------------------------------------------------------------------
# 3. 诊断：可靠性表 / 分层表 / 闸门
# ---------------------------------------------------------------------------
def reliability_table(
    y: np.ndarray, p: np.ndarray, *, edges: Sequence[float] | None = None
) -> pd.DataFrame:
    """§5.6.1 那张等宽分箱的可靠性表（给人看，不作判据）。

    判据用等频分箱的 ECE：本问题的预测集中在 0.35–0.75，等宽分箱两端会空掉，
    n=31 的箱上 ±0.04 的偏差毫无统计意义却会被眼睛过度解读。
    """
    e = np.asarray(edges if edges is not None else np.linspace(0, 1, 11), dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    rows = []
    for b in range(len(e) - 1):
        m = idx == b
        k = int(m.sum())
        if not k:
            continue
        rows.append(
            {
                "bin": f"{e[b]:.1f}–{e[b + 1]:.1f}",
                "n": k,
                "mean_p": float(p[m].mean()),
                "freq": float(y[m].mean()),
                "gap": float(y[m].mean() - p[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def _n_series(series: Sequence[object] | np.ndarray | None, mask: np.ndarray) -> int:
    if series is None:
        return -1
    return int(pd.unique(np.asarray(series, dtype=object)[mask]).size)


def group_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    series: Sequence[object] | np.ndarray | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """单个分层单元的校准画像。**ECE 只给点估计**（compare.py 实测 bootstrap 上偏 +1.88 se）。"""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    slope, intercept = (float("nan"), float("nan"))
    if n >= 30 and 0 < y.mean() < 1:
        slope, intercept = calibration_slope(y, p)
    k = min(n_bins, max(2, n // 50))  # 小样本上 15 箱等于每箱几十场，噪声压过信号
    return {
        "n": float(n),
        "n_series": float(_n_series(series, np.ones(n, dtype=bool))),
        "mean_p": float(p.mean()),
        "freq": float(y.mean()),
        "bias": float(y.mean() - p.mean()),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p, n_bins=k),
        "n_bins": float(k),
        "calib_slope": slope,
        "calib_intercept": intercept,
    }


def stratified_report(
    y: np.ndarray,
    p: np.ndarray,
    groups: pd.Series | np.ndarray,
    *,
    series: Sequence[object] | np.ndarray | None = None,
    name: str = "group",
    n_bins: int = 15,
) -> pd.DataFrame:
    """按一个维度分层出校准表（§5.6.5 / §6.8）。

    返回列：n / n_series / mean_p / freq / bias / log_loss / brier / ece /
    calib_slope / calib_intercept / verdict。
    `verdict` 对 n < MIN_GROUP_N 的单元给 "n小·只报不判"——
    在 200 场上 ECE 的抽样噪声本身就有 0.02 量级，判了也是判噪声。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    g = pd.Series(np.asarray(groups)).reset_index(drop=True)
    ser = None if series is None else np.asarray(series, dtype=object)
    rows = []
    for key in pd.unique(g):
        m = (g == key).to_numpy()
        r = group_metrics(y[m], p[m], series=None if ser is None else ser[m], n_bins=n_bins)
        r[name] = key
        r["verdict"] = _verdict(r, group=True)
        rows.append(r)
    cols = [name, "n", "n_series", "mean_p", "freq", "bias", "log_loss", "brier",
            "ece", "calib_slope", "calib_intercept", "verdict"]
    return pd.DataFrame(rows)[cols].sort_values("n", ascending=False).reset_index(drop=True)


def _verdict(r: dict[str, float], *, group: bool) -> str:
    lim = ECE_GROUP_MAX if group else ECE_OVERALL_MAX
    if group and r["n"] < MIN_GROUP_N:
        return "n小·只报不判"
    bad = []
    if r["ece"] > lim:
        bad.append(f"ECE>{lim}")
    s = r["calib_slope"]
    if not (np.isnan(s) or SLOPE_LO <= s <= SLOPE_HI):
        bad.append("slope出界")
    return "通过" if not bad else "不通过(" + ",".join(bad) + ")"


@dataclass(frozen=True)
class GateResult:
    """§9.7 ① 说校准这一项"几乎零成本"，前提是它被**持续监控**。这就是那个监控点。"""

    passed: bool
    overall: dict[str, float]
    strata: dict[str, pd.DataFrame]
    failures: list[str]

    def summary(self) -> str:
        o = self.overall
        head = (
            f"总体 n={o['n']:,.0f}  ECE={o['ece']:.4f}(≤{ECE_OVERALL_MAX})  "
            f"slope={o['calib_slope']:.4f}∈[{SLOPE_LO},{SLOPE_HI}]  "
            f"bias={o['bias']:+.4f}  LL={o['log_loss']:.5f}"
        )
        tail = "  裁决：通过" if self.passed else "  裁决：不通过 → " + "; ".join(self.failures)
        return head + tail


def calibration_gate(
    y: np.ndarray,
    p: np.ndarray,
    strata: dict[str, pd.Series | np.ndarray] | None = None,
    *,
    series: Sequence[object] | np.ndarray | None = None,
) -> GateResult:
    """验收③ 的闸门：总体 ECE ≤ 0.02，各分层组 ECE ≤ 0.03 且 slope ∈ [0.90, 1.10]。

    分层组里 n < MIN_GROUP_N 的单元不参与判定（只报），否则闸门会被小样本噪声
    随机拉红，变成一个没人再看的警报。
    """
    overall = group_metrics(y, p, series=series)
    failures: list[str] = []
    v = _verdict(overall, group=False)
    if v != "通过":
        failures.append(f"总体 {v}")
    tables: dict[str, pd.DataFrame] = {}
    for dim, g in (strata or {}).items():
        tbl = stratified_report(y, p, g, series=series, name=dim)
        tables[dim] = tbl
        for _, row in tbl.iterrows():
            if row["verdict"] not in ("通过", "n小·只报不判"):
                failures.append(f"{dim}={row[dim]} {row['verdict']}")
    return GateResult(passed=not failures, overall=overall, strata=tables, failures=failures)


# ---------------------------------------------------------------------------
# 4. 本项目约定的三个分层维度（阶段 8 验收③）
# ---------------------------------------------------------------------------
def strata_tier(league_tier: pd.Series | np.ndarray) -> pd.Series:
    """赛区层级：tier-1（主流大区）vs tier-2。跨区 rating 锚定不准最先在这里暴露。"""
    t = pd.Series(np.asarray(league_tier)).astype(int)
    return t.map(lambda v: "tier1" if v <= 1 else "tier2").astype("object")


def strata_rating_gap(rating_diff: pd.Series | np.ndarray, *, cut: float = 50.0) -> pd.Series:
    """实力差档：|Δrating| < 50 与其余。§6.8 把 <50 那一档单列为 KPI——

    强弱悬殊时 log loss 低是应该的，很多模型在势均力敌那一档退化到 0.69 等于没用；
    而校准上正相反：<50 档的预测全挤在 0.5 附近，分箱几乎没有跨度，
    是最容易出现"看起来 ECE 很小其实什么都没说"的一档，必须连同 slope 一起看。
    """
    d = np.abs(np.asarray(rating_diff, dtype=float))
    return pd.Series(np.where(d < cut, f"|Δ|<{cut:.0f}", f"|Δ|>={cut:.0f}"))


def strata_patch_age(patch_age_days: pd.Series | np.ndarray, *, cut: float = 14.0) -> pd.Series:
    """patch 新旧：新版本头两周 vs 之后。这条直接量化版本漂移对校准的成本。

    §6.8 用 0-7/8-14/15-30/30+ 四档看 log loss；校准上用两档（以 14 天为界）是因为
    四档会把 tier-2 × 新 patch 这类交叉单元切到几百场，落进"只报不判"区。
    """
    a = np.asarray(patch_age_days, dtype=float)
    return pd.Series(np.where(a <= cut, f"patch≤{cut:.0f}d", f"patch>{cut:.0f}d"))


def default_strata(df: pd.DataFrame, *, rating_col: str = "team_elo_diff") -> dict[str, pd.Series]:
    """阶段 8 约定的三个维度。df 需含 league_tier / patch_age_days / rating_col。"""
    return {
        "tier": strata_tier(df["league_tier"]),
        "rating_gap": strata_rating_gap(df[rating_col]),
        "patch_age": strata_patch_age(df["patch_age_days"]),
    }
