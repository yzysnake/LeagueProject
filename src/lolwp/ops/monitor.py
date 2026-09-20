"""漂移监控与重训门禁（PLAN.md §6.11 / §7.9）。

**监控挂在 calibration slope 与 Murphy 分解的 reliability 分量上，不挂在 log loss 上。**
版本漂移的顺序是：reliability 先涨、slope 从 1 偏离，总 log loss 最后才动。
盯 slope 能早 1-2 周预警，而且 slope 出界时先重跑校准比全量重训便宜得多。

⚠️ **不要用 ECE 做监控指标**：它有随样本缩小而增大的系统性正偏
（实测全样本 0.0105 的模型，在 989 场的窗口上期望就是 0.0455），
按固定阈值告警会在小窗口上永远报警。斜率无此偏差。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lolwp.eval.metrics import brier_decomposition, calibration_slope, log_loss
from lolwp.eval.power import paired_power

LEADERBOARD = "models/leaderboard.jsonl"


def calibrate_thresholds(
    y: np.ndarray, p: np.ndarray, *, window: int, n_sim: int = 400, seed: int = 0,
    q: float = 0.99,
) -> dict[str, float]:
    """由抽样分布推出告警阈值，而不是拍一个常数。

    **为什么必须这么做**（本项目第三次撞上同一个坑）：
      1. 预注册红线「任一 fold AUC > 0.78」：零信号下误报 63%（修订 A3 已替换）；
      2. 预注册「ECE ≤ 0.03」：在 989 场的 holdout 上只有 1.9% 概率能满足；
      3. 本模块第一版「reliability > 3×基线」「slope ∈ [0.90,1.10]」：
         **23/23 个窗口全部告警**。
    共同病根：阈值按全样本尺度定，却拿去判小窗口。
    reliability 与 ECE 一样有随样本缩小而增大的系统性正偏。

    做法：从模型自身的样本外预测里随机抽同样大小的窗口 n_sim 次，
    取 q 分位作为阈值。这样「健康模型」的误报率就是 1−q。
    """
    rng = np.random.default_rng(seed)
    rels, slopes = [], []
    for _ in range(n_sim):
        i = rng.choice(len(y), window, replace=False)
        rels.append(brier_decomposition(y[i], p[i])["reliability"])
        slopes.append(calibration_slope(y[i], p[i])[0])
    rels, slopes = np.array(rels), np.array(slopes)
    lo_q, hi_q = (1 - q) / 2 * 100, (1 - (1 - q) / 2) * 100
    return {
        "window": window, "n_sim": n_sim, "q": q,
        "rel_max": float(np.quantile(rels, q)),
        "slope_lo": float(np.percentile(slopes, lo_q)),
        "slope_hi": float(np.percentile(slopes, hi_q)),
        "rel_median": float(np.median(rels)),
        "slope_median": float(np.median(slopes)),
    }


@dataclass
class MonitorPoint:
    window_end: str
    n: int
    log_loss: float
    calib_slope: float
    reliability: float
    alert: str


def rolling_monitor(
    y: np.ndarray, p: np.ndarray, t: pd.Series, *, window: int = 500, step: int = 100,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """滚动窗口的三条监控曲线。t 必须已按时间升序。

    阈值默认由 `calibrate_thresholds()` 在同样窗口大小上推出，
    使健康模型的误报率等于 1−q（默认 1%）。
    """
    order = np.argsort(t.to_numpy(), kind="mergesort")
    y, p, tt = y[order], p[order], t.to_numpy()[order]
    th = thresholds or calibrate_thresholds(y, p, window=window)
    rows = []
    for end in range(window, len(y) + 1, step):
        sl = slice(end - window, end)
        yy, pp = y[sl], p[sl]
        slope, _ = calibration_slope(yy, pp)
        rel = brier_decomposition(yy, pp)["reliability"]
        alerts = []
        if not (th["slope_lo"] <= slope <= th["slope_hi"]):
            alerts.append(f"slope={slope:.3f} 出界[{th['slope_lo']:.3f},{th['slope_hi']:.3f}]")
        if rel > th["rel_max"]:
            alerts.append(f"reliability={rel:.5f} > {th['rel_max']:.5f}")
        rows.append(asdict(MonitorPoint(
            window_end=str(tt[end - 1])[:10], n=window,
            log_loss=log_loss(yy, pp), calib_slope=slope, reliability=rel,
            alert="; ".join(alerts) or "-",
        )))
    out = pd.DataFrame(rows)
    out.attrs["thresholds"] = th
    return out


def promotion_gate(
    y: np.ndarray, p_champion: np.ndarray, p_challenger: np.ndarray,
    series_id: np.ndarray, *, t_threshold: float = 2.5,
) -> dict[str, object]:
    """上线门禁：挑战者必须**显著更好**才准替换现役（§7.9）。

    判据三条同时满足：ΔlogLoss > 0、series 级聚类 t > 阈值、|Δ| > 该比较自测的 MDE。
    只要有一条不满足就拒绝——**默认保留现役**，而不是默认上线新模型。
    """
    from lolwp.eval.metrics import per_game_log_loss
    d = per_game_log_loss(y, p_champion) - per_game_log_loss(y, p_challenger)
    pw = paired_power(d, series_id)
    reasons = []
    if pw["mean_delta"] <= 0:
        reasons.append(f"ΔlogLoss {pw['mean_delta']:+.5f} ≤ 0，挑战者不更好")
    if pw["t_clustered"] <= t_threshold:
        reasons.append(f"聚类 t {pw['t_clustered']:.2f} ≤ {t_threshold}")
    if not pw["exceeds_mde"]:
        reasons.append(f"|Δ| {abs(pw['mean_delta']):.5f} 未超过 MDE {pw['mde']:.5f}")
    return {"promote": not reasons, "reasons": reasons, **{k: pw[k] for k in
            ("mean_delta", "t_clustered", "mde", "n_games", "n_series")}}


def append_leaderboard(entry: dict, path: str = LEADERBOARD) -> None:
    """append-only 的模型履历。每次训练或门禁裁决都记一条，便于回滚时查。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def read_leaderboard(path: str = LEADERBOARD) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as fh:
        return pd.DataFrame([json.loads(x) for x in fh if x.strip()])
