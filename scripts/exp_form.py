#!/usr/bin/env python3
"""近期状态（form）第二层的门禁（reports/exp_model_improvements.md §3）。

base = 当前生产规格（BT4_LR + 赛后经济差软标签，修订 A6）的 walk-forward OOS 预测。
form 用 base 在更早开赛的 OOS 局上的残差算（models.form），β 只用更早的 fold 拟合（扩张窗）。
6 个设定 = 半衰期 {5, 15, 40} 局 × 本系列已打完的局 {算, 不算}。§6.7 三条门禁。

两种选设定的办法都报：
- **默认（修订 A8 起）：逐折滚动选**——每个 fold 之前只看近 2 年的 OOS 表现挑设定（eval.gate.adaptive_select）；
- 修订 A7 当时的做法：在调参段（2016–2018）上选一次，选中的那一个拿到验收段（2019 起）确认。
退出码按默认协议。只走 load_dev()。结果写 reports/exp_form.json。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from lolwp.eval.gate import (
    bt4_walk_forward,
    fold_ids,
    paired_gate,
    select_and_gate,
    stack_walk_forward,
)
from lolwp.eval.walkforward import WFConfig, make_folds
from lolwp.features.draft_priors import run_draft_state_machine
from lolwp.features.loader import load_dev
from lolwp.features.priors import run_state_machine
from lolwp.models.artifact import PROD_FEATURES
from lolwp.models.bt_cache import rolling_ratings_cached
from lolwp.models.form import FormConfig, fit_beta, form_diff
from lolwp.models.hier_bt import HierBTConfig
from lolwp.models.margin import TARGET, with_margin_target

SEGMENTS = {"tune": ("2016-01-01", "2019-01-01"), "accept": ("2019-01-01", None)}
VARIANTS = [FormConfig(half_life_games=h, include_series=s) for h in (5.0, 15.0, 40.0) for s in (True, False)]
OUT = "reports/exp_form.json"


def _name(cfg: FormConfig) -> str:
    return f"hl{cfg.half_life_games:g}_{'series' if cfg.include_series else 'noseries'}"


def main() -> int:
    t0 = time.time()
    full = with_margin_target(load_dev())
    F_all = pd.concat([run_state_machine(full), run_draft_state_machine(full),
                       rolling_ratings_cached(full, HierBTConfig(refit_every_days=3), target=TARGET)], axis=1)
    print(f"特征就绪（{time.time() - t0:.0f} 秒）", flush=True)

    res: dict[str, dict[str, dict]] = {}
    preds: dict[str, np.ndarray] = {}
    for key, (lo, hi) in SEGMENTS.items():
        m = (full.league_tier >= 1) & (full.kickoff_utc >= lo)
        if hi:
            m &= full.kickoff_utc < hi
        dev = full[m].reset_index(drop=True)
        folds = make_folds(dev, WFConfig())
        p = bt4_walk_forward(dev, F_all.loc[full[m].index].reset_index(drop=True), folds, PROD_FEATURES)
        ok = np.isfinite(p)
        oos, po, fo = dev[ok].reset_index(drop=True), p[ok], fold_ids(len(dev), folds)[ok]
        y, ser = oos.y_blue_win.to_numpy(dtype=float), oos.series_id.to_numpy()
        res[key] = {}
        for cfg in VARIANTS:
            q, betas = stack_walk_forward(po, form_diff(oos, po, cfg), y, fo, fit_beta)
            res[key][_name(cfg)] = {**paired_gate(y, po, q, ser), "beta_last": betas[-1] if betas else 0.0,
                                    "config": asdict(cfg)}
            if key == "accept":
                preds[_name(cfg)] = q
        if key == "accept":
            # 默认协议：逐折滚动选；首选（历史不足时用）放 A7 选中的那一个
            order = ["hl5_series"] + [n for n in preds if n != "hl5_series"]
            per_window = select_and_gate({n: preds[n] for n in order}, po, y, fo, oos.kickoff_utc.to_numpy(), ser)

    pick = max(res["tune"], key=lambda k: res["tune"][k]["delta"])
    dec = res["accept"][pick]
    print(f"\n{'设定':<18}{'调参 Δ':>10}{'余量':>7}{'验收 Δ':>10}{'余量':>7}{'t':>7}{'前半':>10}{'后半':>10}  β")
    for k in res["tune"]:
        a, b = res["tune"][k], res["accept"][k]
        print(f"{k + (' ←选中' if k == pick else ''):<18}{a['delta']:>+10.5f}{a['delta'] / a['mde']:>6.2f}×"
              f"{b['delta']:>+10.5f}{b['delta'] / b['mde']:>6.2f}×{b['t_clustered']:>7.2f}"
              f"{b['half_1']:>+10.5f}{b['half_2']:>+10.5f}  {b['beta_last']:.3f}")
    ok = "✓ 过门禁"
    no = "✗ 未过门禁"
    pw = per_window
    print(f"\n默认（逐折滚动选，近 2 年）：验收段 {pw['log_loss_base']:.5f} → {pw['log_loss_cand']:.5f}，"
          f"Δ {pw['delta']:+.5f}（{pw['delta'] / pw['mde']:.2f}× MDE，t={pw['t_clustered']:.2f}，"
          f"两半 {pw['half_1']:+.5f} / {pw['half_2']:+.5f}）{ok if pw['passed'] else no}；各折选中 {pw['picks']}")
    print(f"A7 当时（调参段选中 {pick}）：验收段 {dec['log_loss_base']:.5f} → {dec['log_loss_cand']:.5f}，"
          f"Δ {dec['delta']:+.5f}（{dec['delta'] / dec['mde']:.2f}× MDE，t={dec['t_clustered']:.2f}）"
          f"{ok if dec['passed'] else no}")
    with open(OUT, "w") as fh:
        json.dump({"per_window": pw, "passed": pw["passed"], "tune_selected": pick,
                   "tune_selected_passed": dec["passed"], "segments": res}, fh, ensure_ascii=False, indent=2)
    print(f"→ {OUT}（{time.time() - t0:.0f} 秒）")
    return 0 if pw["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
