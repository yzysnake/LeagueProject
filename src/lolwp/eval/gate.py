"""上线门禁的公用件：BT4_LR 的 walk-forward、第二层的走前向堆叠、§6.7 的配对比较。

给 scripts/exp_*.py 用，口径与阶段 8 的 BT4_LR 逐字一致：每折只在训练段上标准化 + L2 logistic(C=1)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lolwp.eval.metrics import evaluate, per_game_log_loss
from lolwp.eval.power import paired_power


def bt4_walk_forward(dev: pd.DataFrame, F: pd.DataFrame, folds: list, cols: list[str]) -> np.ndarray:
    """返回与 dev 行对齐的 OOS 概率（不在任何 valid 段的行为 NaN）。"""
    y = dev.y_blue_win.to_numpy().astype(int)
    p = np.full(len(dev), np.nan)
    for f in folds:
        tr, va = f.train_idx.to_numpy(), f.valid_idx.to_numpy()
        Xtr, Xva = F.loc[tr, cols].to_numpy(float), F.loc[va, cols].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, y[tr])
        p[va] = lr.predict_proba((Xva - mu) / sd)[:, 1]
    return p


def fold_ids(n: int, folds: list) -> np.ndarray:
    """每行属于哪个 valid fold（-1 = 不在任何 valid 段）。"""
    out = np.full(n, -1)
    for f in folds:
        out[f.valid_idx.to_numpy()] = f.idx
    return out


def stack_walk_forward(p: np.ndarray, f: np.ndarray, y: np.ndarray, fold: np.ndarray,
                       fit, min_train: int = 2000) -> tuple[np.ndarray, list[float]]:
    """第二层：logit q = logit p + β·f，β 只用更早 fold 的行拟合（扩张窗）；训练行不足时 q = p。
    p / f / y / fold 是同序的一维数组（只含 OOS 行）；fit(logit_p, f, y) → β。"""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    q, betas = p.copy(), []
    for k in np.unique(fold):
        tr, va = fold < k, fold == k
        if tr.sum() < min_train:
            continue
        b = fit(z[tr], f[tr], y[tr])
        betas.append(b)
        q[va] = 1 / (1 + np.exp(-(z[va] + b * f[va])))
    return q, betas


def paired_gate(y: np.ndarray, p_base: np.ndarray, p_cand: np.ndarray, series: np.ndarray) -> dict:
    """逐场 log loss 之差（base − cand，正 = 候选更好）。行须按时间升序（两半按时间切，边界挪到 series 边界）。"""
    d = per_game_log_loss(y, p_base) - per_game_log_loss(y, p_cand)
    pw = paired_power(d, series)
    half = len(d) // 2
    while 0 < half < len(d) and series[half] == series[half - 1]:
        half += 1
    h1, h2 = float(d[:half].mean()), float(d[half:].mean())
    eb, ec = evaluate(y, p_base), evaluate(y, p_cand)
    return {
        "n_games": len(d), "n_series": int(pw["n_series"]),
        "log_loss_base": eb["log_loss"], "log_loss_cand": ec["log_loss"],
        "delta": float(pw["mean_delta"]), "mde": float(pw["mde"]), "t_clustered": float(pw["t_clustered"]),
        "half_1": h1, "half_2": h2,
        "passed": bool(np.sign(h1) == np.sign(h2) and abs(pw["t_clustered"]) > 2.5
                       and abs(pw["mean_delta"]) > pw["mde"]),
        "ece_base": eb["ece"], "ece_cand": ec["ece"],
        "slope_base": eb["calib_slope"], "slope_cand": ec["calib_slope"],
    }
