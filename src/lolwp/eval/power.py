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


def pairwise_rho(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """组内相关的稳健估计：只用**多元素组内部的真实配对**。

    为什么需要它（2026-09-19 实测）：
    ANOVA 版 ICC(1) 在「单元素组的方差与多元素组不同」时会严重高估。
    本项目的 OOS 里 **48.8% 的 series 是单局**，且单局的 var(d) 是多局的 **2.04 倍**
    （0.013124 vs 0.006421）。单局组只进组间平方和、不进组内平方和，
    于是这个异方差被整段记到"组间"头上。

    反证（真 ρ = 0，只把单局组的方差放大）：
        ×1.0 → icc_oneway 报 0.0051   ×1.5 → 0.2301
        ×2.04 → 0.4454                ×3.0 → 0.6788

    实测对照：L2→L3 的 icc_oneway = 0.3120，而本函数给 **0.0266**。
    阶段 3 报告里那个 0.3120 主要是伪影，已在 reports/stage3_power_and_icc.md 更正。
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": groups}).dropna()
    mu, sd = df.v.mean(), df.v.std()
    if sd == 0:
        return {"rho": 0.0, "n_pairs": 0}
    num = cnt = 0.0
    for _, s in df.groupby("g"):
        v = s.v.to_numpy()
        k = len(v)
        if k < 2:
            continue
        c = v - mu
        # Σ_{i<j} c_i c_j = ((Σc)² − Σc²) / 2
        num += (c.sum() ** 2 - (c**2).sum()) / 2.0
        cnt += k * (k - 1) / 2.0
    if cnt == 0:
        return {"rho": 0.0, "n_pairs": 0}
    return {"rho": float(num / cnt / sd**2), "n_pairs": int(cnt)}


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """单因素随机效应 ICC(1)，方差分量法。

    ⚠️ **在单元素组与多元素组存在异方差时会严重高估**，本项目正是这种情况。
    判定 MDE 时请用 paired_power() 返回的 rho（稳健版），
    本函数保留是为了与文献口径对照，以及暴露失真程度。
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
    rob = pairwise_rho(d, groups)
    sigma = float(np.std(d, ddof=1))
    n = len(d)
    # 用稳健的 ρ 算 deff；ANOVA 版一并返回以便发现失真
    deff = design_effect(stats["m_bar"], rob["rho"])
    n_eff = n / deff
    mean = float(np.mean(d))
    return {
        "mean_delta": mean,
        "sigma_d": sigma,
        "n_games": n,
        "n_series": stats["n_groups"],
        "m_bar": stats["m_bar"],
        "icc": rob["rho"],
        "icc_anova": stats["icc"],
        "icc_distortion": stats["icc"] - rob["rho"],
        "n_pairs": rob["n_pairs"],
        "deff": deff,
        "n_eff": n_eff,
        "t_naive": mean / (sigma / np.sqrt(n)) if sigma else float("nan"),
        "t_clustered": mean / (sigma / np.sqrt(n_eff)) if sigma else float("nan"),
        "mde": mde(sigma, n_eff, t_star),
        "exceeds_mde": abs(mean) > mde(sigma, n_eff, t_star),
    }


def footer(p: dict[str, float]) -> str:
    """每张表脚都要打印这一行（阶段 3 验收⑤）。"""
    warn = ""
    if abs(p.get("icc_distortion", 0.0)) > 0.05:
        warn = f"  ⚠ANOVA版={p['icc_anova']:.4f}(失真{p['icc_distortion']:+.4f})"
    return (
        f"n_games={p['n_games']:,}  n_series={p['n_series']:,}  m̄={p['m_bar']:.3f}  "
        f"ρ={p['icc']:.4f}  deff={p['deff']:.3f}  n_eff={p['n_eff']:,.0f}  "
        f"σ_d={p['sigma_d']:.4f}  MDE={p['mde']:.5f}{warn}"
    )
