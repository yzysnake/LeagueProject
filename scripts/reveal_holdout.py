#!/usr/bin/env python3
"""final holdout 一次性揭盲（PLAN.md §8 阶段 9）。

**这个脚本在整个项目里只应该运行一次。**
运行前请确认 reports/stage9_pre_reveal_check.md 的清单全部通过。

运行后：
- 结果写入 reports/final_holdout_<日期>.md，并记录当时的 git sha 与数据快照 sha；
- 之后任何基于 holdout 结果的模型调整都会污染这个估计，
  若确有必要，必须在 preregistration.md 里记为新的修订并说明。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd


def main() -> int:
    if os.environ.get("LOLWP_ALLOW_HOLDOUT") != "yes":
        print("拒绝执行：这是 final holdout 的一次性揭盲。\n"
              "确认要揭盲请设 LOLWP_ALLOW_HOLDOUT=yes 再运行。", file=sys.stderr)
        return 2

    from sklearn.linear_model import LogisticRegression

    from lolwp.eval.metrics import evaluate
    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import holdout_start, load_holdout
    from lolwp.features.priors import run_state_machine
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.hier_bt import HierBTConfig

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    ho = load_holdout(min_tier=1)
    print(f"holdout：{len(ho):,} 场  {ho.kickoff_utc.min()} → {ho.kickoff_utc.max()}")

    # 状态机必须在全量（dev + holdout）上单遍跑，因为 rating 是递归的；
    # 但**拟合只用 dev**，holdout 只被预测、不参与任何参数估计。
    silver = pd.read_parquet("data/silver/games.parquet")
    silver = silver.sort_values("kickoff_utc").reset_index(drop=True)
    F = pd.concat([run_state_machine(silver), run_draft_state_machine(silver),
                   rolling_ratings_cached(silver, HierBTConfig(refit_every_days=3))], axis=1)
    cut = holdout_start()
    is_dev = (silver.kickoff_utc < cut) & (silver.league_tier >= 1) & \
             (silver.kickoff_utc >= "2019-01-01")
    is_ho = silver.game_id.isin(set(ho.game_id))

    COLS = ["side_prior", "bt_player_diff", "bt_chem_diff", "bt_league_diff", "champ_meta_diff"]
    Xtr = F.loc[is_dev, COLS].to_numpy(float)
    ytr = silver.loc[is_dev, "y_blue_win"].to_numpy()
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000).fit((Xtr - mu) / sd, ytr)

    Xho = F.loc[is_ho, COLS].to_numpy(float)
    yho = silver.loc[is_ho, "y_blue_win"].to_numpy()
    p = clf.predict_proba((Xho - mu) / sd)[:, 1]
    sid = silver.loc[is_ho, "series_id"].to_numpy()

    e = evaluate(yho, p)
    sizes = pd.Series(sid).value_counts()
    out = {
        "git_sha": sha, "n_games": len(yho), "n_series": int(sizes.size),
        "m_bar": float(len(yho) / sizes.size),
        "date_range": [str(ho.kickoff_utc.min()), str(ho.kickoff_utc.max())],
        "n_train": int(is_dev.sum()), **e,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=float))

    # 预注册区间核对
    checks = {
        "log_loss ∈ [0.615, 0.645]": 0.615 <= e["log_loss"] <= 0.645,
        "brier ∈ [0.216, 0.228]": 0.216 <= e["brier"] <= 0.228,
        "ece ≤ 0.03（注意小样本正偏）": e["ece"] <= 0.03,
        "slope ∈ [0.93, 1.07]": 0.93 <= e["calib_slope"] <= 1.07,
        "红线 accuracy ≤ 0.67": e["accuracy"] <= 0.67,
        "红线 log_loss ≥ 0.60": e["log_loss"] >= 0.60,
        "红线 合并 AUC ≤ 0.78": e["auc"] <= 0.78,
    }
    print("\n=== 预注册核对 ===")
    for k, v in checks.items():
        print(f"  {'✓' if v else '✗'} {k}")
    out["checks"] = checks
    os.makedirs("reports", exist_ok=True)
    with open("reports/final_holdout_result.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print("\n→ reports/final_holdout_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
