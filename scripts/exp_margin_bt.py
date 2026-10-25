#!/usr/bin/env python3
"""复现预注册修订 A6 的依据：分层 BT 的训练标签用胜负 vs 赛后经济差软标签（reports/exp_model_improvements.md §2）。

口径与阶段 8 的 BT4_LR 一致：tier≥1、walk-forward（WFConfig 默认）、每折标准化 + L2 logistic(C=1)，
外层特征 = PROD_FEATURES；状态机（side 先验、英雄版本强度）与 BT 都在全赛区全时段上跑一遍再取子集（训宽评窄）。
两段不重叠的数据：调参段 2016–2018、验收段 2019 起。只走 load_dev()，holdout 不碰。

设定是 (λ, κ)：软标签 = (1 − λ)·胜负 + λ·σ(κ·经济差)。两种选设定的办法都报：
- **默认（修订 A8 起）：逐折滚动选**——验收段的每个 fold 之前只看近 2 年的 OOS 表现，在 5 个设定里挑
  （eval.gate.adaptive_select）；
- 修订 A6 当时的做法：调参段上选出 λ=1、κ=0.5，两段分别与胜负标签比（下表）。
退出码按默认协议。

比较量 = 逐场 log loss 之差（胜负标签 − 软标签，正 = 软标签更好）：series 分组的 MDE（eval.power.paired_power）、
clustered t、时间上前后两半各自的均值。§6.7 门禁三条：两半同号、|t| > 2.5、|Δ| > MDE。

第一次跑要算 5 条软标签的 3 天网格滚动快照（各 15–40 分钟，可先并行预热），之后命中 data/cache/bt。
结果写 reports/exp_margin_bt.json。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from lolwp.eval.gate import bt4_walk_forward, fold_ids, paired_gate, select_and_gate
from lolwp.eval.walkforward import WFConfig, make_folds
from lolwp.features.draft_priors import run_draft_state_machine
from lolwp.features.loader import load_dev
from lolwp.features.priors import run_state_machine
from lolwp.models.artifact import PROD_FEATURES
from lolwp.models.bt_cache import rolling_ratings_cached
from lolwp.models.hier_bt import HierBTConfig
from lolwp.models.margin import MARGIN_KAPPA, TARGET, WIN, with_margin_target

SEGMENTS = {"tune": ("2016-01-01", "2019-01-01"), "accept": ("2019-01-01", None)}
# (λ, κ) → 软标签列；第一个是生产的设定（λ=1 只用软标签）。逐折选时历史不足就用它
GRID = {(1.0, 0.5): TARGET, (0.75, 0.5): "y_margin_l0.75_k0.5", (0.5, 0.5): "y_margin_l0.5_k0.5",
        (0.5, 1.0): "y_margin_l0.5_k1", (0.5, 0.25): "y_margin_l0.5_k0.25"}
LABEL = {"tune": "调参段 2016–2018", "accept": "验收段 2019 起"}
OUT = "reports/exp_margin_bt.json"


def main() -> int:
    t0 = time.time()
    full = load_dev()
    for (lam, kappa), col in GRID.items():
        full = with_margin_target(full, lam=lam, kappa=kappa, column=col)
    sm = pd.concat([run_state_machine(full), run_draft_state_machine(full)], axis=1)
    cfg = HierBTConfig(refit_every_days=3)
    bt = {t: rolling_ratings_cached(full, cfg, target=t) for t in (WIN, *GRID.values())}
    print(f"特征就绪（{time.time() - t0:.0f} 秒）", flush=True)

    out = {"kappa": MARGIN_KAPPA, "grid_days": cfg.refit_every_days, "segments": {}}
    for key, (lo, hi) in SEGMENTS.items():
        m = (full.league_tier >= 1) & (full.kickoff_utc >= lo)
        if hi:
            m &= full.kickoff_utc < hi
        dev = full[m].reset_index(drop=True)
        folds = make_folds(dev, WFConfig())
        p = {t: bt4_walk_forward(dev, pd.concat([sm, bt[t]], axis=1).loc[full[m].index].reset_index(drop=True),
                                 folds, PROD_FEATURES) for t in (WIN, *GRID.values())}
        ok = np.isfinite(p[WIN])
        y, ser = dev.y_blue_win.to_numpy()[ok], dev.series_id.to_numpy()[ok]
        out["segments"][key] = paired_gate(y, p[WIN][ok], p[TARGET][ok], ser)
        if key == "accept":
            out["per_window"] = select_and_gate(
                {f"λ{lam:g} κ{kappa:g}": p[col][ok] for (lam, kappa), col in GRID.items()}, p[WIN][ok], y,
                fold_ids(len(dev), folds)[ok], dev.kickoff_utc.to_numpy()[ok], ser)

    print(f"\n{'':<16}{'场次':>8}{'胜负标签':>10}{'软标签':>10}{'Δ':>10}{'MDE':>9}{'余量':>7}{'t':>7}"
          f"{'前半':>10}{'后半':>10}  门禁")
    for key, r in out["segments"].items():
        print(f"{LABEL[key]:<16}{r['n_games']:>8,}{r['log_loss_base']:>10.5f}{r['log_loss_cand']:>10.5f}"
              f"{r['delta']:>+10.5f}{r['mde']:>9.5f}{r['delta'] / r['mde']:>6.2f}×{r['t_clustered']:>7.2f}"
              f"{r['half_1']:>+10.5f}{r['half_2']:>+10.5f}  {'✓' if r['passed'] else '✗'}")
    pw = out["per_window"]
    print(f"\n默认（逐折滚动选 5 个设定，近 2 年）：验收段 {pw['log_loss_base']:.5f} → {pw['log_loss_cand']:.5f}，"
          f"Δ {pw['delta']:+.5f}（{pw['delta'] / pw['mde']:.2f}× MDE，t={pw['t_clustered']:.2f}）"
          f"{'✓ 过门禁' if pw['passed'] else '✗ 未过门禁'}；各折选中 {pw['picks']}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n→ {OUT}（{time.time() - t0:.0f} 秒）")
    return 0 if pw["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
