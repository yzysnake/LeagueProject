"""基线阶梯 L0-L3（PLAN.md §5.2）。

每一级都必须真的跑出来并进对比表。没有诚实的基线，后面任何"提升"都无从判断。
  L0  常数 0.5
  L1  训练窗口内的蓝方基准率（as-of）
  L1b 按赛区收缩的 side 先验
  L2  + team Elo
  L3  + player Elo        ← primary comparison 的对照组
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LADDER: dict[str, list[str]] = {
    "L0": [],
    "L1": [],
    "L1b": ["side_prior"],
    "L2": ["side_prior", "team_elo_diff"],
    "L3": ["side_prior", "team_elo_diff", "player_elo_diff"],
}


def _fit_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, *, C: float = 1.0
) -> np.ndarray:
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1]


def predict_fold(
    name: str,
    feats: pd.DataFrame,
    y: pd.Series,
    train_idx: pd.Index,
    valid_idx: pd.Index,
    *,
    C: float = 1.0,
) -> np.ndarray:
    """在单个 fold 上训练并预测。标准化器只在训练段拟合（§6.1.2 红线 3）。"""
    n = len(valid_idx)
    if name == "L0":
        return np.full(n, 0.5)
    if name == "L1":
        # as-of 蓝方基准率：只用训练段
        return np.full(n, float(y.loc[train_idx].mean()))

    cols = LADDER[name]
    X_tr = feats.loc[train_idx, cols].to_numpy(dtype=float)
    X_va = feats.loc[valid_idx, cols].to_numpy(dtype=float)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1.0
    return _fit_predict((X_tr - mu) / sd, y.loc[train_idx].to_numpy(), (X_va - mu) / sd, C=C)


def run_ladder(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    folds: list,
    names: list[str] | None = None,
    *,
    C: float = 1.0,
) -> pd.DataFrame:
    """跑完整阶梯，返回长表：一行 = 一场 × 一个模型。"""
    names = names or list(LADDER)
    y = df.y_blue_win
    recs = []
    for f in folds:
        for nm in names:
            p = predict_fold(nm, feats, y, f.train_idx, f.valid_idx, C=C)
            recs.append(
                pd.DataFrame(
                    {
                        "game_id": df.loc[f.valid_idx, "game_id"].to_numpy(),
                        "series_id": df.loc[f.valid_idx, "series_id"].to_numpy(),
                        "kickoff_utc": df.loc[f.valid_idx, "kickoff_utc"].to_numpy(),
                        "league_canon": df.loc[f.valid_idx, "league_canon"].to_numpy(),
                        "league_tier": df.loc[f.valid_idx, "league_tier"].to_numpy(),
                        "fold": f.idx,
                        "model": nm,
                        "y": y.loc[f.valid_idx].to_numpy(),
                        "p": p,
                    }
                )
            )
    return pd.concat(recs, ignore_index=True)
