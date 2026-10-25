#!/usr/bin/env python3
"""复现预注册修订 A6 的依据：分层 BT 的训练标签用胜负 vs 赛后经济差软标签（reports/exp_model_improvements.md §2）。

口径与阶段 8 的 BT4_LR 一致：tier≥1、walk-forward（WFConfig 默认）、每折标准化 + L2 logistic(C=1)，
外层特征 = PROD_FEATURES；状态机（side 先验、英雄版本强度）与 BT 都在全赛区全时段上跑一遍再取子集（训宽评窄）。
两段不重叠的数据：调参段 2016–2018（在这里选的设定）、验收段 2019 起（只用来确认）。只走 load_dev()，holdout 不碰。

比较量 = 逐场 log loss 之差（胜负标签 − 软标签，正 = 软标签更好）：series 分组的 MDE（eval.power.paired_power）、
clustered t、时间上前后两半各自的均值。§6.7 门禁三条：两半同号、|t| > 2.5、|Δ| > MDE。

第一次跑要算软标签那条 3 天网格的滚动快照（15–40 分钟），之后命中 data/cache/bt；胜负那条一般已在缓存里。
结果写 reports/exp_margin_bt.json。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lolwp.eval.metrics import evaluate, per_game_log_loss
from lolwp.eval.power import paired_power
from lolwp.eval.walkforward import WFConfig, make_folds
from lolwp.features.draft_priors import run_draft_state_machine
from lolwp.features.loader import load_dev
from lolwp.features.priors import run_state_machine
from lolwp.models.artifact import PROD_FEATURES
from lolwp.models.bt_cache import rolling_ratings_cached
from lolwp.models.hier_bt import HierBTConfig
from lolwp.models.margin import MARGIN_KAPPA, TARGET, WIN, with_margin_target

SEGMENTS = {"tune": ("2016-01-01", "2019-01-01"), "accept": ("2019-01-01", None)}
LABEL = {"tune": "调参段 2016–2018", "accept": "验收段 2019 起"}
OUT = "reports/exp_margin_bt.json"


def walk_forward(dev: pd.DataFrame, F: pd.DataFrame, folds: list) -> np.ndarray:
    """BT4_LR 的 walk-forward：返回与 dev 行对齐的 OOS 概率（不在任何 valid 段的行为 NaN）。"""
    y = dev.y_blue_win.to_numpy().astype(int)
    p = np.full(len(dev), np.nan)
    for f in folds:
        tr, va = f.train_idx.to_numpy(), f.valid_idx.to_numpy()
        Xtr = F.loc[tr, PROD_FEATURES].to_numpy(float)
        Xva = F.loc[va, PROD_FEATURES].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=1.0, max_iter=3000).fit((Xtr - mu) / sd, y[tr])
        p[va] = lr.predict_proba((Xva - mu) / sd)[:, 1]
    return p


def compare(dev: pd.DataFrame, p_win: np.ndarray, p_margin: np.ndarray) -> dict:
    m = np.isfinite(p_win) & np.isfinite(p_margin)
    y, ser = dev.y_blue_win.to_numpy()[m], dev.series_id.to_numpy()[m]
    d = per_game_log_loss(y, p_win[m]) - per_game_log_loss(y, p_margin[m])
    pw = paired_power(d, ser)
    half = len(d) // 2
    while 0 < half < len(d) and ser[half] == ser[half - 1]:   # 两半的边界挪到 series 边界上
        half += 1
    h1, h2 = float(d[:half].mean()), float(d[half:].mean())
    ew, em = evaluate(y, p_win[m]), evaluate(y, p_margin[m])
    return {
        "n_games": int(m.sum()), "n_series": int(pw["n_series"]),
        "log_loss_win": ew["log_loss"], "log_loss_margin": em["log_loss"],
        "delta": float(pw["mean_delta"]), "mde": float(pw["mde"]), "t_clustered": float(pw["t_clustered"]),
        "half_1": h1, "half_2": h2,
        "passed": bool(np.sign(h1) == np.sign(h2) and abs(pw["t_clustered"]) > 2.5
                       and abs(pw["mean_delta"]) > pw["mde"]),
        "ece_win": ew["ece"], "ece_margin": em["ece"],
        "slope_win": ew["calib_slope"], "slope_margin": em["calib_slope"],
    }


def main() -> int:
    t0 = time.time()
    full = with_margin_target(load_dev())
    sm = pd.concat([run_state_machine(full), run_draft_state_machine(full)], axis=1)
    cfg = HierBTConfig(refit_every_days=3)
    bt = {t: rolling_ratings_cached(full, cfg, target=t) for t in (WIN, TARGET)}
    print(f"特征就绪（{time.time() - t0:.0f} 秒）", flush=True)

    out = {"kappa": MARGIN_KAPPA, "grid_days": cfg.refit_every_days, "segments": {}}
    for key, (lo, hi) in SEGMENTS.items():
        m = (full.league_tier >= 1) & (full.kickoff_utc >= lo)
        if hi:
            m &= full.kickoff_utc < hi
        dev = full[m].reset_index(drop=True)
        folds = make_folds(dev, WFConfig())
        p = {t: walk_forward(dev, pd.concat([sm, bt[t]], axis=1).loc[full[m].index].reset_index(drop=True),
                             folds) for t in (WIN, TARGET)}
        out["segments"][key] = compare(dev, p[WIN], p[TARGET])

    print(f"\n{'':<16}{'场次':>8}{'胜负标签':>10}{'软标签':>10}{'Δ':>10}{'MDE':>9}{'余量':>7}{'t':>7}"
          f"{'前半':>10}{'后半':>10}  门禁")
    for key, r in out["segments"].items():
        print(f"{LABEL[key]:<16}{r['n_games']:>8,}{r['log_loss_win']:>10.5f}{r['log_loss_margin']:>10.5f}"
              f"{r['delta']:>+10.5f}{r['mde']:>9.5f}{r['delta'] / r['mde']:>6.2f}×{r['t_clustered']:>7.2f}"
              f"{r['half_1']:>+10.5f}{r['half_2']:>+10.5f}  {'✓' if r['passed'] else '✗'}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT}（{time.time() - t0:.0f} 秒）")
    return 0 if all(r["passed"] for r in out["segments"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
