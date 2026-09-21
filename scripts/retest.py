#!/usr/bin/env python3
"""定期重测被搁置的假设，并打印趋势。

    python scripts/retest.py                # 跑全部
    python scripts/retest.py --only counter_reliability
    python scripts/retest.py --history      # 只看历史趋势，不重跑

设计意图：本项目有一批假设是「现在测不出来」而非「已被证伪」，
它们应当随数据积累自动复查。没有这个机制，就只能靠人凭感觉想起来——
而人通常想不起来。
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--history", action="store_true", help="只看历史，不重跑")
    a = ap.parse_args()

    from lolwp.ops.backlog import REGISTRY, history, run_all

    if not a.history:
        print(f"重测 {len(a.only or REGISTRY)} 个被搁置的假设…\n", flush=True)
        df = run_all(a.only)
        if df.empty:
            print("没有产出任何结果", file=sys.stderr)
            return 1
        print(f"{'假设':<24}{'样本':>9}{'统计量':>11}{'门槛':>11}{'余量':>8}  裁决")
        for r in df.itertuples(index=False):
            v = "**通过**" if r.passed else "未过"
            print(f"{r.key:<24}{r.n_games:>9,}{r.statistic:>11.5f}"
                  f"{r.threshold:>11.5f}{r.margin:>7.2f}×  {v}")

    h = history()
    if h.empty:
        return 0
    print("\n=== 历史趋势（台账 append-only，趋势比单次结论重要）===")
    for key, g in h.groupby("key"):
        g = g.sort_values("run_date")
        meta = next((x for x in REGISTRY if x.key == key), None)
        print(f"\n  {key}　{meta.title if meta else ''}")
        print(f"    门禁：{meta.gate if meta else '?'}")
        for r in g.itertuples(index=False):
            arrow = "✓" if r.passed else " "
            print(f"    {r.run_date}  n={r.n_games:>7,}  "
                  f"统计量 {r.statistic:>9.5f}  余量 {r.margin:>5.2f}×  {arrow}")
        if len(g) >= 2:
            d = g.statistic.iloc[-1] - g.statistic.iloc[0]
            dn = g.n_games.iloc[-1] - g.n_games.iloc[0]
            print(f"    趋势：样本 {dn:+,} → 统计量 {d:+.5f}")
    newly = h.sort_values("run_date").groupby("key").tail(1)
    hit = newly[newly.passed]
    if len(hit):
        print("\n⚠ 以下假设本次**已越过门禁**，应当重新评估是否加入生产模型：")
        for r in hit.itertuples(index=False):
            print(f"    {r.key}（余量 {r.margin:.2f}×）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
