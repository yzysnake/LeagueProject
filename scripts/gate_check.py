#!/usr/bin/env python3
"""上线门禁验证（PLAN.md §8 阶段 9 验收⑤）。

**默认保留现役**：挑战者必须三条同时满足才准替换——
ΔlogLoss > 0、series 级聚类 t > 2.5、|Δ| > 该比较自测的 MDE。

自检模式 `--selftest` 会人为注入一个退化模型（把实力特征置零），
门禁必须拒绝它。**一个从不拒绝的门禁等于没有门禁。**
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="BT4_LR")
    ap.add_argument("--challenger", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="注入退化模型，验证门禁真的会拒绝")
    a = ap.parse_args()

    from lolwp.ops.monitor import append_leaderboard, promotion_gate

    p = pd.read_parquet("data/cache/stage8_preds.parquet")
    w = p.pivot_table(index=["game_id", "series_id"], columns="model",
                      values="p", aggfunc="first").reset_index()
    y = p.drop_duplicates("game_id").set_index("game_id").y
    w["y"] = w.game_id.map(y)
    yy = w.y.to_numpy()
    sid = w.series_id.to_numpy()
    champ = w[a.champion].to_numpy()

    cases = []
    if a.selftest:
        rng = np.random.default_rng(0)
        # 退化模型 1：整体退回常数（相当于实力特征被置零）
        cases.append(("退化·常数基准率", np.full(len(yy), float(yy.mean()))))
        # 退化模型 2：在现役上叠噪声（相当于 rating 服务挂了一半）
        noisy = np.clip(champ + rng.normal(0, 0.08, len(champ)), 0.01, 0.99)
        cases.append(("退化·rating 加噪", noisy))
        # 对照：现役自己（应当被拒，因为 Δ = 0）
        cases.append(("对照·现役自己", champ.copy()))
    if a.challenger:
        cases.append((a.challenger, w[a.challenger].to_numpy()))

    rc = 0
    for name, chal in cases:
        r = promotion_gate(yy, champ, chal, sid)
        verdict = "PROMOTE" if r["promote"] else "REJECT"
        print(f"\n[{verdict}] {a.champion} → {name}")
        print(f"  ΔlogLoss {r['mean_delta']:+.5f}  t聚类 {r['t_clustered']:+.2f}  "
              f"MDE {r['mde']:.5f}  n={r['n_games']:,}/{r['n_series']:,}")
        for why in r["reasons"]:
            print(f"    拒绝理由：{why}")
        append_leaderboard({"champion": a.champion, "challenger": name,
                            "promote": r["promote"], "delta_ll": r["mean_delta"],
                            "t": r["t_clustered"], "mde": r["mde"]})
        if a.selftest and name.startswith("退化") and r["promote"]:
            print("  ✗✗ 门禁放过了退化模型，门禁本身有问题")
            rc = 1
    if a.selftest and rc == 0:
        print("\n✓ 自检通过：门禁拒绝了全部退化模型")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
