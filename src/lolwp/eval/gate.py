"""上线门禁的公用件：BT4_LR 的 walk-forward、第二层的走前向堆叠、逐折选设定、§6.7 的配对比较。

给 scripts/exp_*.py 与 ops/backlog 用，口径与阶段 8 的 BT4_LR 逐字一致：每折只在训练段上标准化 + L2 logistic(C=1)。

**选设定的默认办法是逐折滚动选（`adaptive_select`，预注册修订 A8）**：每个 4 周的 fold 开打之前，
只用它之前 2 年的 OOS 表现挑设定。挑选永远只看过去、贴着当前版本，又从不看被打分的那一折，所以仍是样本外的。
原先「在调参段 2016–2018 上选一次、验收段 2019 起确认」会被版本演变带偏（只会偏向保守，不会造出假阳性），
A8 实测它与逐折选几乎总是选中同一个设定，见 experiments/preregistration.md。
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

SELECT_WINDOW = pd.Timedelta(days=730)   # 逐折选设定时往回看的长度
SELECT_MIN_ROWS = 1000                   # 窗口里的 OOS 行不足这么多时，用第一个（事先最合理的）设定
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


def adaptive_select(preds: dict[str, np.ndarray], y: np.ndarray, fold: np.ndarray, kick: np.ndarray, *,
                    window: pd.Timedelta = SELECT_WINDOW,
                    min_rows: int = SELECT_MIN_ROWS) -> tuple[np.ndarray, list[str]]:
    """逐折选设定（默认协议，修订 A8）：对每个 fold，在「该折开始 − window」到该折开始之间、
    更早的 fold 的 OOS 行上比各设定的 log loss，挑最低的那个用于这一折。

    preds：设定名 → 与 y 同序的 OOS 概率（每个设定各自已是走前向的）。**放第一个的是事先最合理的设定**：
    窗口里历史不足 min_rows 时用它。返回 (逐行选中后的预测, 每折选中的设定名)。
    """
    from lolwp.eval.metrics import per_game_log_loss

    names = list(preds)
    if not names:
        raise ValueError("至少要给一个设定")
    y = np.asarray(y, dtype=float)
    kick = pd.DatetimeIndex(pd.to_datetime(kick, utc=True))
    ll = {n: per_game_log_loss(y, np.clip(np.asarray(preds[n], dtype=float), 1e-6, 1 - 1e-6)) for n in names}
    out = np.full(len(y), np.nan)
    picks: list[str] = []
    for k in np.unique(fold):
        va = fold == k
        tr = (fold < k) & (kick >= kick[va].min() - window)
        best = names[0] if tr.sum() < min_rows else min(names, key=lambda n: ll[n][tr].mean())
        out[va] = np.asarray(preds[best], dtype=float)[va]
        picks.append(best)
    return out, picks


def select_and_gate(preds: dict[str, np.ndarray], base: np.ndarray, y: np.ndarray, fold: np.ndarray,
                    kick: np.ndarray, series: np.ndarray, **kw) -> dict:
    """默认协议下的一次门禁：逐折选设定，再与 base 配对比较。结果里多两项：各设定被选中的折数、协议名。"""
    q, picks = adaptive_select(preds, y, fold, kick, **kw)
    return {**paired_gate(y, base, q, series), "picks": dict(Counter(picks)), "protocol": "per-window"}


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


def production_oos(segment: str = "accept") -> dict:
    """当前生产规格（软标签 BT4_LR + 近期状态第二层）在开发期一段上的 walk-forward OOS 预测。

    给「在生产规格之上再叠一层」的门禁用（ops/backlog、scripts/exp_*）：返回同序的
    oos（比赛表，按开赛升序）、p（外层的 OOS 概率，残差的基准）、q（再加近期状态之后的概率，比较的 base）、
    fold、y、series。只走 load_dev()；分层 BT 命中 data/cache/bt 的缓存。
    """
    from lolwp.eval.walkforward import WFConfig, make_folds
    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine
    from lolwp.models.artifact import PROD_FEATURES
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.form import DEFAULT, fit_beta, form_diff
    from lolwp.models.hier_bt import HierBTConfig
    from lolwp.models.margin import TARGET, with_margin_target

    lo, hi = {"tune": ("2016-01-01", "2019-01-01"), "accept": ("2019-01-01", None)}[segment]
    full = with_margin_target(load_dev())
    F_all = pd.concat([run_state_machine(full), run_draft_state_machine(full),
                       rolling_ratings_cached(full, HierBTConfig(refit_every_days=3), target=TARGET)], axis=1)
    m = (full.league_tier >= 1) & (full.kickoff_utc >= lo)
    if hi:
        m &= full.kickoff_utc < hi
    dev = full[m].reset_index(drop=True)
    folds = make_folds(dev, WFConfig())
    p = bt4_walk_forward(dev, F_all.loc[full[m].index].reset_index(drop=True), folds, PROD_FEATURES)
    ok = np.isfinite(p)
    oos, po, fo = dev[ok].reset_index(drop=True), p[ok], fold_ids(len(dev), folds)[ok]
    y = oos.y_blue_win.to_numpy(dtype=float)
    q, _ = stack_walk_forward(po, form_diff(oos, po, DEFAULT), y, fo, fit_beta)
    return {"oos": oos, "p": po, "q": q, "fold": fo, "y": y, "series": oos.series_id.to_numpy()}
