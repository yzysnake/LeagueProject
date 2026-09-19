"""评估指标（PLAN.md §6.2）。

主指标是 log loss。accuracy 只展示不决策——它把 0.51 和 0.99 的预测同等对待，
而本项目大部分预测会落在 0.45-0.65 区间，accuracy 对这段几乎没有分辨力。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def per_game_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """逐场 log loss，配对检验与 block bootstrap 都要用它。"""
    p = _clip(p)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    """Murphy 分解：Brier = Reliability − Resolution + Uncertainty。

    Reliability 越小越好（校准），Resolution 越大越好（分辨力）。
    版本漂移时 Reliability 会先涨，比总 Brier 早 1-2 周预警（§6.11）。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    obar = y.mean()
    unc = obar * (1 - obar)
    # 等频分箱
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.digitize(p, qs[1:-1])
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        rel += nb * (p[m].mean() - y[m].mean()) ** 2
        res += nb * (y[m].mean() - obar) ** 2
    n = len(y)
    return {"reliability": rel / n, "resolution": res / n, "uncertainty": unc}


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U，含并列的正确处理。"""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """等频分箱。等宽分箱在本问题上尾部会空掉（预测集中在 0.4-0.7）。"""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.digitize(p, qs[1:-1])
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """把 logit(p̂) 当唯一自变量重做 logistic。slope<1 = 过度自信 = 过拟合指纹。"""
    from sklearn.linear_model import LogisticRegression

    z = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
    # sklearn 1.8 起 penalty=None 已弃用，改用 C=inf 表达"无正则"
    m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    d = brier_decomposition(y, p)
    slope, intercept = calibration_slope(y, p)
    return {
        "n": len(y),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "reliability": d["reliability"],
        "resolution": d["resolution"],
        "auc": auc(y, p),
        "ece": ece(y, p),
        "calib_slope": slope,
        "calib_intercept": intercept,
        "accuracy": float(((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).mean()),
    }


def table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).T[
        ["n", "log_loss", "brier", "reliability", "resolution", "auc", "ece",
         "calib_slope", "accuracy"]
    ].round(4)
