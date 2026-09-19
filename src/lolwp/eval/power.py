"""功效分析：ICC → design effect → MDE（PLAN.md §6.3）。

手册 v1.0 用的 `n_eff = n_games / 2.5` 从未测量也从未推导，等价于假设
series 内 ICC = 1（同系列多局完全相关），是最保守的一端。v1.1 已删掉它，
改为实测。本模块给出那个实测。

deff = 1 + (m̄ − 1)·ICC
n_eff = n_games / deff
MDE  = t* · σ_d / sqrt(n_eff)      (t* ≈ 2.8 对应双侧 α=0.005 的保守取值)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """单因素随机效应 ICC(1)，用方差分量法。

    values 通常是两个模型的 per-game log loss 之差 d_i，
    groups 是 series_id。ICC 高 = 同一系列内的 d_i 高度相关 = 有效样本少。
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": groups})
    df = df.dropna()
    k = df.groupby("g").v.size()
    n_groups = len(k)
    n_total = len(df)
    if n_groups < 2 or n_total <= n_groups:
        return {"icc": float("nan"), "m_bar": float(k.mean()) if n_groups else float("nan"),
                "n_groups": n_groups, "n": n_total}

    grand = df.v.mean()
    gm = df.groupby("g").v.mean()
    # 组间 / 组内均方
    ssb = float((k * (gm - grand) ** 2).sum())
    ssw = float(((df.v - df.g.map(gm)) ** 2).sum())
    msb = ssb / (n_groups - 1)
    msw = ssw / (n_total - n_groups)
    # 不等组大小的调整系数
    m0 = (n_total - (k**2).sum() / n_total) / (n_groups - 1)
    var_b = max((msb - msw) / m0, 0.0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else 0.0
    return {
        "icc": float(icc),
        "m_bar": float(n_total / n_groups),
        "m0": float(m0),
        "n_groups": int(n_groups),
        "n": int(n_total),
        "msb": msb,
        "msw": msw,
    }


def design_effect(m_bar: float, icc: float) -> float:
    return 1.0 + (m_bar - 1.0) * icc


def mde(sigma_d: float, n_eff: float, t_star: float = 2.8) -> float:
    return t_star * sigma_d / np.sqrt(n_eff)


def paired_power(
    d: np.ndarray, groups: np.ndarray, *, t_star: float = 2.8
) -> dict[str, float]:
    """给一组配对差（per-game log loss 之差），返回完整的功效画像。"""
    d = np.asarray(d, dtype=float)
    stats = icc_oneway(d, groups)
    sigma = float(np.std(d, ddof=1))
    n = len(d)
    deff = design_effect(stats["m_bar"], stats["icc"])
    n_eff = n / deff
    mean = float(np.mean(d))
    return {
        "mean_delta": mean,
        "sigma_d": sigma,
        "n_games": n,
        "n_series": stats["n_groups"],
        "m_bar": stats["m_bar"],
        "icc": stats["icc"],
        "deff": deff,
        "n_eff": n_eff,
        "t_naive": mean / (sigma / np.sqrt(n)) if sigma else float("nan"),
        "t_clustered": mean / (sigma / np.sqrt(n_eff)) if sigma else float("nan"),
        "mde": mde(sigma, n_eff, t_star),
        "exceeds_mde": abs(mean) > mde(sigma, n_eff, t_star),
    }


def footer(p: dict[str, float]) -> str:
    """每张表脚都要打印这一行（阶段 3 验收⑤）。"""
    return (
        f"n_games={p['n_games']:,}  n_series={p['n_series']:,}  m̄={p['m_bar']:.3f}  "
        f"ICC={p['icc']:.4f}  deff={p['deff']:.3f}  n_eff={p['n_eff']:,.0f}  "
        f"σ_d={p['sigma_d']:.4f}  MDE={p['mde']:.5f}"
    )
