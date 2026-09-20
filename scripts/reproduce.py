#!/usr/bin/env python3
"""可复现性核对（PLAN.md §8 阶段 9 验收⑦）。

从当前数据快照重建关键数字，与归档报告里的值逐条 diff。
**OE 会持续回填修订历史年份的文件**，所以"重跑相同代码得到不同结果"是常态；
这个脚本的作用是把差异**显式**摆出来，而不是让它悄悄发生。
"""
from __future__ import annotations

import argparse
import json
import sys

# 归档值。来源：reports/stage8_comparison.md 与 reports/final_technical_report.md
ARCHIVED = {
    "n_games_total": 102084,
    "n_games_silver": 102042,
    "n_manifest_years": 13,
    "n_champions": 173,
    "holdout_games": 989,
    "oos_log_loss_BT4_LR": 0.61917,
    "holdout_log_loss": 0.61653,
}
TOL = {"oos_log_loss_BT4_LR": 5e-4, "holdout_log_loss": 5e-4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="latest")
    ap.parse_args()

    import pandas as pd

    got: dict[str, float] = {}
    with open("data/manifest.jsonl") as fh:
        man = [json.loads(x) for x in fh if x.strip()]
    got["n_manifest_years"] = len({m["year"] for m in man})
    got["n_games_total"] = sum(m["n_games"] for m in
                               {m["year"]: m for m in man}.values())
    s = pd.read_parquet("data/silver/games.parquet", columns=["game_id", "kickoff_utc",
                                                              "league_tier"])
    got["n_games_silver"] = len(s)
    with open("data/ddragon/champion_map.json") as fh:
        got["n_champions"] = json.load(fh)["n_champions"]
    got["holdout_games"] = int(((s.kickoff_utc >= "2026-06-01") & (s.league_tier >= 1)).sum())
    try:
        p = pd.read_parquet("data/cache/stage8_preds.parquet")
        from lolwp.eval.metrics import log_loss
        m = p[p.model == "BT4_LR"]
        got["oos_log_loss_BT4_LR"] = log_loss(m.y.to_numpy(), m.p.to_numpy())
    except Exception as e:  # noqa: BLE001
        print(f"  （跳过 OOS：{type(e).__name__}）", file=sys.stderr)
    try:
        with open("reports/final_holdout_result.json") as fh:
            got["holdout_log_loss"] = json.load(fh)["log_loss"]
    except Exception as e:  # noqa: BLE001
        print(f"  （跳过 holdout：{type(e).__name__}，尚未揭盲或结果文件缺失）",
              file=sys.stderr)

    print(f"{'指标':<26}{'归档值':>14}{'当前值':>14}  差异")
    bad = 0
    for k, v in ARCHIVED.items():
        g = got.get(k)
        if g is None:
            print(f"{k:<26}{v:>14}{'(未产出)':>14}  跳过")
            continue
        tol = TOL.get(k, 0)
        ok = abs(g - v) <= tol if tol else g == v
        mark = "一致 ✓" if ok else f"**不一致** Δ={g - v:+.6g}"
        bad += 0 if ok else 1
        print(f"{k:<26}{v:>14}{g:>14.5f}  {mark}" if isinstance(g, float)
              else f"{k:<26}{v:>14}{g:>14}  {mark}")
    print(f"\n不一致 {bad} 项")
    if bad:
        print("提示：OE 会回填修订历史文件，差异不一定是 bug；\n"
              "      但必须显式确认过再更新归档值。")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
