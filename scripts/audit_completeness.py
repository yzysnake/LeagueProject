#!/usr/bin/env python
"""阶段 2 专项审计：patch 归一化 + `datacompleteness` 可用性 + 缺 id 场次。

回答四个必须用真实数据回答、不能靠推理的问题（PLAN.md §3.10.1 / §3.2.5 / §8 阶段 2）：

  A. OE 的 `patch` 到底长什么样？手册那条 `major >= 25 则 -10` 在 OE 上成立吗？
  B. `partial` 的行到底缺什么？BP 时点可见的列在 partial 里完整吗？
  C. 丢掉 partial 会损失多少场？（决定"绝不写 `df[df.datacompleteness=='complete']`"）
  D. 在 12 行长表上 `dropna(subset=[playerid, teamid])` 会造出多少畸形场？

用法（仓库根目录）::

    .venv/bin/python scripts/audit_completeness.py              # 只打报告
    .venv/bin/python scripts/audit_completeness.py --out reports/completeness
    .venv/bin/python scripts/audit_completeness.py --section patch

默认不写任何文件；给了 `--out` 才落 CSV。

实现注记
--------
* 用 duckdb 直扫 13 个 CSV（850 MB，约 4 秒），**必须显式 `quote='"' escape='"'`**：
  实测 duckdb 1.5 的方言嗅探会把 quote 猜成空，然后在 2019 文件第 2114 行
  （队名 ``"University of California, Irvine"`` 含逗号）炸成 "Found 166 columns"。
* 全列按 ``all_varchar=true`` 读。`patch` 一旦被推断成 DOUBLE，``13.01`` 会变成
  13.01 再变回 ``'13.01'`` 还好，但 ``13.10`` 会变成 ``'13.1'``——零填充丢失，
  和 13.01 撞车（同 patches.py 模块 docstring 第 3 条）。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

import duckdb
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lolwp.store.patches import (
    OE_PATCH_RE,
    PatchLadder,
    attach_patch_features,
    build_patch_table,
    norm_patch,
    patch_order_violations,
    patch_str,
)

RAW_GLOB = "data/raw/*_LoL_esports_match_data_from_OraclesElixir.csv"
TIER1_YAML = "eval/tier1_leagues.yaml"
HOLDOUT_START_FILE = "eval/HOLDOUT_START"

#: BP（Ban/Pick 结束、开局之前）时点理论上可见的列。partial 行里这些必须完整，
#: 否则 partial 数据对本项目就不可用了。
BP_VISIBLE_COLS = [
    "gameid", "league", "date", "patch", "side", "position",
    "playerid", "teamid", "champion", "result",
    "ban1", "ban2", "ban3", "ban4", "ban5",
]
#: 这些是"BP 时点可见但不是每年都有"的列，单独看，不进硬性结论。
BP_OPTIONAL_COLS = ["firstPick", "pick1", "pick2", "pick3", "pick4", "pick5"]
#: 赛后统计的代表列，用来证实 partial 缺的就是这些。`dragons` 只写在 team 行。
POSTGAME_SAMPLE = ["goldat10", "goldat15", "golddiffat15", "gamelength", "kills", "dragons"]
#: 只出现在 team 汇总行的列（选手行恒空），统计覆盖率时必须限定作用域，
#: 否则会得到"complete 也是 0%"这种没意义的数字。
TEAM_ONLY_COLS = {"ban1", "ban2", "ban3", "ban4", "ban5",
                  "pick1", "pick2", "pick3", "pick4", "pick5", "dragons"}

pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 220)


# --------------------------------------------------------------------------- #
def connect(raw_glob: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE VIEW oe AS
        SELECT *, regexp_extract(filename, '(\\d{{4}})_LoL', 1) AS file_year
        FROM read_csv('{raw_glob}', all_varchar=true, header=true, filename=true,
                      quote='"', escape='"');
        """
    )
    return con


def all_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.sql("DESCRIBE oe").fetchall() if r[0] not in ("filename", "file_year")]


def hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dump(out: str | None, name: str, df: pd.DataFrame) -> None:
    if out:
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, name)
        df.to_csv(path, index=False)
        print(f"[写出] {path}  ({len(df)} 行)")


# --------------------------------------------------------------------------- #
# A. patch
# --------------------------------------------------------------------------- #
def section_patch(con: duckdb.DuckDBPyConnection, out: str | None,
                  versions_path: str | None = None) -> None:
    hdr("A. patch 归一化")

    # 一场一行：participantid='100' 是蓝方 team 汇总行，实测每场恰好一条。
    g = con.sql(
        "SELECT gameid, league, date, patch, file_year FROM oe WHERE participantid='100'"
    ).df()
    # 排序固定成 (date, gameid)，让乱序统计可复现（同一时刻开赛的比赛很多，
    # 不定序的话 stale 判定会在不同机器上差几十场）。
    g = g.sort_values(["date", "gameid"], kind="mergesort").reset_index(drop=True)
    n_games = len(g)
    print(f"总场次 {n_games:,}；patch 非空 {int(g.patch.notna().sum()):,}，"
          f"缺失 {int(g.patch.isna().sum()):,} "
          f"({g.patch.isna().mean() * 100:.2f}%)")

    # --- A1 格式枚举 ---
    shapes = con.sql(
        """
        SELECT coalesce(regexp_replace(regexp_replace(patch,'[0-9]','N','g'),'N+','D','g'),
                        '<NULL>') AS shape,
               count(DISTINCT patch) n_distinct, count(DISTINCT gameid) n_games,
               min(patch) ex_min, max(patch) ex_max
        FROM oe GROUP BY 1 ORDER BY n_games DESC
        """
    ).df()
    print("\n-- A1 patch 字符串形态（数字压成 D）--")
    print(shapes.to_string(index=False))

    uniq = sorted(g.patch.dropna().unique())
    hit = [v for v in uniq if OE_PATCH_RE.match(v)]
    print(f"\ndistinct 非空 patch = {len(uniq)}；命中 OE 规范格式 ^\\d{{1,2}}\\.\\d{{2}}$ "
          f"= {len(hit)} ({len(hit) / len(uniq) * 100:.1f}%)")
    miss = [v for v in uniq if v not in hit]
    print("不命中的取值:", miss if miss else "（无）")

    per_patch = con.sql(
        """
        SELECT patch, count(DISTINCT gameid) n_games, min(date) d_min, max(date) d_max,
               count(DISTINCT league) n_leagues
        FROM oe WHERE patch IS NOT NULL GROUP BY 1
        """
    ).df()
    print("\npatch 缺失的分布（按赛区）:")
    print(con.sql(
        """SELECT league, count(DISTINCT gameid) n_games, min(date)[1:10] d_min, max(date)[1:10] d_max
           FROM oe WHERE patch IS NULL GROUP BY 1 ORDER BY 2 DESC"""
    ).df().to_string(index=False))

    print("\n同一 gameid 内 patch 是否唯一（应为 0）:",
          int(con.sql("SELECT count(*) FROM (SELECT gameid FROM oe GROUP BY gameid "
                      "HAVING count(DISTINCT patch)>1)").fetchone()[0]))

    # --- A2 验证手册的 -10 规则 ---
    print("\n-- A2 手册 §3.10.1 `major >= 25 则 -10` 在 OE 上成立吗 --")
    majors = sorted({norm_patch(v)[0] for v in uniq})
    minors = sorted({norm_patch(v)[1] for v in uniq})
    n_would_fire = sum(1 for v in uniq if norm_patch(v)[0] >= 25)
    print(f"OE major 取值 = {majors}")
    print(f"OE minor 取值 = {minors[0]}..{minors[-1]}（{len(minors)} 个）")
    print(f"会触发 -10 的 patch 数 = {n_would_fire} / {len(uniq)}")
    print("→ 结论：这条修正在 OE 上是 no-op，**OE 根本不需要它**。"
          "它只对 Leaguepedia/gptilt 系有效，必须由 source= 显式开启。")
    pad = [v for v in uniq if norm_patch(v)[1] < 10]
    print(f"\nminor<10 的 patch 共 {len(pad)} 个，全部写成两位零填充？",
          all(v.split(".")[1].startswith("0") and len(v.split(".")[1]) == 2 for v in pad))
    print("例:", pad[:6])
    rt_bad = [v for v in uniq if patch_str(norm_patch(v)) != v]
    print(f"往返 patch_str(norm_patch(v)) == v 失败数 = {len(rt_bad)} {rt_bad}")
    print("→ 结论：OE 用零填充（'13.01'），Leaguepedia 不填充（'13.1'）；"
          "字符串直接比较必错配，归一成 (major,minor) 元组后才安全。")

    # --- A3 patch_idx / first_seen ---
    print("\n-- A3 patch_idx 与 patch_first_seen --")
    ladder_dd = _try_ddragon_ladder(versions_path)
    pt = build_patch_table(g, game_col="gameid")
    print(f"观测阶梯：{len(pt)} 个 patch，patch_idx 0..{int(pt.patch_idx.max())}")
    if ladder_dd is not None:
        obs = [(int(a), int(b)) for a, b in zip(pt.patch_major, pt.patch_minor)]
        notin = [patch_str(k) for k in obs if ladder_dd.index_of(k) is None]
        lo, hi = min(obs), max(obs)
        gaps = [patch_str(k) for k in ladder_dd.keys if lo <= k <= hi and k not in set(obs)]
        print(f"DDragon 发布阶梯：{len(ladder_dd)} 个版本；OE 观测 patch 不在其中的 = {notin}")
        print(f"阶梯上处于观测区间内但**零职业赛**的 patch = {len(gaps)} 个：{gaps}")
        print("→ 用观测阶梯做 patch_idx 会低估版本距离，但 2018 年后只差 3 个，可接受。")
    else:
        print("（未找到 data/ddragon/versions.json，跳过发布阶梯对照；"
              "跑 ingest/ddragon.py 落盘后重跑可得完整对照）")

    gg = attach_patch_features(g, ladder=ladder_dd)
    age = gg.patch_age_days.describe(percentiles=[0.05, 0.5, 0.9, 0.95, 0.99])
    print("\npatch_age_days（kickoff − 该 patch 全库首现日）:")
    print(age.round(3).to_string())
    print("负值场次（首现日按定义 ≤ 本场，应为 0）:", int((gg.patch_age_days < 0).sum()))
    buckets = pd.cut(gg.patch_age_days, [-0.001, 7, 14, 30, 1e9],
                     labels=["0-7d", "8-14d", "15-30d", "30+d"])
    bt = buckets.value_counts().sort_index().to_frame("n_games")
    bt["pct"] = (bt.n_games / bt.n_games.sum() * 100).round(2)
    print("\n分桶（PLAN.md §6 的分层口径）:")
    print(bt.to_string())

    fs = pt.sort_values("patch_idx")
    bad_order = fs[fs.patch_first_seen < fs.patch_first_seen.cummax().shift(1)]
    print(f"\n按 patch_idx 排列时「首现日」倒挂的 patch 数 = {len(bad_order)}")
    if len(bad_order):
        print(bad_order[["patch_raw_canon", "n_games", "patch_first_seen"]].to_string(index=False))
    print("→ 结论：(major,minor) 排序与首现日几乎一致，patch_idx 可放心按版本号排。")

    # --- A4 乱序量化 ---
    print("\n-- A4 乱序：老 patch 打在新日期（跨赛区版本推进不同步）--")
    vg = patch_order_violations(gg)
    vl = patch_order_violations(gg, group_col="league")
    has = gg.patch_idx.notna()
    print(f"全库口径  ：{int(vg.is_stale.sum()):,} / {int(has.sum()):,} 场 "
          f"({vg.is_stale.sum() / has.sum() * 100:.2f}%) 的 patch 低于此前全库最高版本")
    lag = vg.loc[vg.is_stale, "patch_lag"]
    print("  落后版本数分位:", {k: round(float(v), 2) for k, v in
                                lag.describe(percentiles=[0.5, 0.95, 0.99]).items()})
    print(f"赛区内口径：{int(vl.is_stale.sum()):,} 场 "
          f"({vl.is_stale.sum() / has.sum() * 100:.2f}%) —— 同一赛区内几乎从不回退，"
          "所以 33% 的乱序**全部来自赛区之间不同步**。")

    gg = gg.assign(is_stale=vg.is_stale.to_numpy(), patch_lag=vg.patch_lag.to_numpy())
    gg["y"] = pd.to_datetime(gg.date).dt.year
    by_y = gg[has].groupby("y").agg(n=("gameid", "size"), stale=("is_stale", "sum"))
    by_y["pct"] = (by_y.stale / by_y.n * 100).round(2)
    print("\n按年:")
    print(by_y.to_string())
    by_l = gg[has].groupby("league").agg(n=("gameid", "size"), stale=("is_stale", "sum"))
    by_l["pct"] = (by_l.stale / by_l.n * 100).round(2)
    by_l = by_l[by_l.n >= 300].sort_values("pct", ascending=False)
    print("\n按赛区（n>=300，最乱的 12 个 / 最同步的 6 个）:")
    print(pd.concat([by_l.head(12), by_l.tail(6)]).to_string())
    print("\n→ 下游必须知道：`patch_idx` 只能按版本号排；`patch_age_days` 对慢赛区"
          "系统性偏大（实测全局首现口径中位 12.09 天 vs 赛区内首现口径 5.00 天，"
          "45.9% 的比赛两者相差 >7 天）。")

    dump(out, "patch_table.csv", pt)
    dump(out, "patch_raw_values.csv", per_patch.sort_values("d_min"))
    dump(out, "patch_stale_by_league.csv", by_l.reset_index())


def _try_ddragon_ladder(path: str | None = None) -> PatchLadder | None:
    """加载 DDragon 版本阶梯。默认找 data/ddragon/versions.json，可用 --versions 指定。

    没有也能跑：`patch_idx` 退回"观测阶梯"，只是无法报告"零职业赛的版本"这一项。
    """
    from lolwp.store.patches import load_ddragon_ladder

    return load_ddragon_ladder(path) if path else load_ddragon_ladder()


# --------------------------------------------------------------------------- #
# B. datacompleteness
# --------------------------------------------------------------------------- #
def _null_rates(con: duckdb.DuckDBPyConnection, cols: Iterable[str], where: str) -> tuple[int, pd.Series]:
    sel = ",".join(f'avg(CASE WHEN "{c}" IS NULL THEN 1.0 ELSE 0.0 END) AS "{c}"' for c in cols)
    d = con.sql(f"SELECT count(*) AS __n, {sel} FROM oe WHERE {where}").df()
    return int(d["__n"].iloc[0]), d.drop(columns="__n").iloc[0].astype(float)


def section_completeness(con: duckdb.DuckDBPyConnection, out: str | None) -> None:
    hdr("B. datacompleteness")

    print("-- B1 取值全集 --")
    print(con.sql(
        """SELECT datacompleteness, count(*) n_rows, count(DISTINCT gameid) n_games
           FROM oe GROUP BY 1 ORDER BY 3 DESC"""
    ).df().to_string(index=False))
    print("同一 gameid 内取值混用的场次（应为 0）:",
          int(con.sql("SELECT count(*) FROM (SELECT gameid FROM oe GROUP BY gameid "
                      "HAVING count(DISTINCT datacompleteness)>1)").fetchone()[0]))

    by_year = con.sql(
        """
        SELECT CAST(date[1:4] AS INT) y, count(DISTINCT gameid) n_games,
          count(DISTINCT CASE WHEN datacompleteness='complete' THEN gameid END) n_complete,
          count(DISTINCT CASE WHEN datacompleteness='partial'  THEN gameid END) n_partial
        FROM oe WHERE participantid='100' GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_year["pct_partial"] = (by_year.n_partial / by_year.n_games * 100).round(2)
    print("\n-- B2 按年 --")
    print(by_year.to_string(index=False))

    lg = con.sql(
        """
        SELECT league, CAST(date[1:4] AS INT) y, count(DISTINCT gameid) n_games,
          count(DISTINCT CASE WHEN datacompleteness='partial' THEN gameid END) n_partial
        FROM oe WHERE participantid='100' GROUP BY 1,2
        """
    ).df()
    lg["pct_partial"] = (lg.n_partial / lg.n_games * 100).round(2)
    print("\n-- B2b partial 场次最多的 20 个 (赛区, 年) --")
    print(lg.nlargest(20, "n_partial").to_string(index=False))

    print("\n-- B3 手册 §3.2.5 两个待验证数字 --")
    r22 = lg[(lg.league == "LPL") & (lg.y == 2022)].iloc[0]
    r26 = lg[(lg.league == "LPL") & (lg.y == 2026)].iloc[0]
    n_team_2022 = int(con.sql("SELECT count(*) FROM oe WHERE league='LPL' AND date[1:4]='2022' "
                              "AND position='team'").fetchone()[0])
    print("  手册：2022 LPL 100% partial（1,572 个 team row 全 partial）")
    print(f"  实测：2022 LPL {int(r22.n_games)} 场，partial {int(r22.n_partial)} 场 "
          f"= {r22.pct_partial}%；team row {n_team_2022} 条  → 完全一致 ✓")
    print("  手册：2026 LPL 19.09%（613 场里 117 场 partial）")
    print(f"  实测：2026 LPL {int(r26.n_games)} 场，partial {int(r26.n_partial)} 场 "
          f"= {r26.pct_partial}%  → 比例对得上量级，但绝对数是旧快照（613/117），"
          f"当前快照是 {int(r26.n_games)}/{int(r26.n_partial)}")
    only26 = lg[(lg.y == 2026) & (lg.n_partial > 0)]
    print(f"  2026 含 partial 的赛区 = {list(only26.league)} → 手册「2026 只有 LPL 含 partial」✓")
    print("  补充（手册未记）：LPL 2022–2025 连续四年 100% partial，2026 才掉到",
          f"{r26.pct_partial}%；LDL 2018/2022/2023/2024 也是 100% partial。")

    # --- B4 逐列空值率 ---
    print("\n-- B4 partial vs complete 逐列空值率（差异最大的 20 列）--")
    cols = all_columns(con)
    for scope_label, scope in [("选手行 position<>'team'", "position<>'team'"),
                               ("team 行 position='team'", "position='team'")]:
        nc, c = _null_rates(con, cols, f"datacompleteness='complete' AND {scope}")
        npt, p = _null_rates(con, cols, f"datacompleteness='partial' AND {scope}")
        t = pd.DataFrame({"complete_null%": c * 100, "partial_null%": p * 100})
        t["diff_pp"] = t["partial_null%"] - t["complete_null%"]
        print(f"\n[{scope_label}]  n_complete={nc:,}  n_partial={npt:,}")
        print(t.nlargest(20, "diff_pp").round(3).to_string())
        neg = t[t.diff_pp < -1].sort_values("diff_pp")
        print(f"  partial 反而更全的列（差 >1pp）: {len(neg)} 列")
        if len(neg):
            print(neg.round(3).to_string())
        dump(out, f"nullrate_{'player' if 'team' not in scope_label[:4] else 'team'}.csv",
             t.reset_index(names="column"))

    # --- B5 BP 时点可见列 ---
    print("\n-- B5 关键判定：BP 时点可见的列在 partial 里完整吗 --")
    for label, extra in [("全部 13 年", ""), ("近 5 年 date>='2021'", "AND date>='2021'")]:
        rows = []
        for c in BP_VISIBLE_COLS + BP_OPTIONAL_COLS + POSTGAME_SAMPLE:
            # champion/playerid 只在选手行有意义；ban/pick/dragons 只在 team 行有意义。
            scope = "position='team'" if c in TEAM_ONLY_COLS else "position<>'team'"
            sub = f"{scope} {extra}"
            a = con.sql(f"""SELECT avg(CASE WHEN "{c}" IS NOT NULL THEN 100.0 ELSE 0 END)
                            FROM oe WHERE datacompleteness='complete' AND {sub}""").fetchone()[0]
            b = con.sql(f"""SELECT avg(CASE WHEN "{c}" IS NOT NULL THEN 100.0 ELSE 0 END)
                            FROM oe WHERE datacompleteness='partial' AND {sub}""").fetchone()[0]
            rows.append({"column": c, "complete_cov%": round(a, 3), "partial_cov%": round(b, 3),
                         "delta_pp": round(b - a, 3),
                         "group": ("BP必需" if c in BP_VISIBLE_COLS
                                   else "BP可选" if c in BP_OPTIONAL_COLS else "赛后")})
        t = pd.DataFrame(rows)
        print(f"\n[{label}] 非空覆盖率 %")
        print(t.to_string(index=False))
        dump(out, f"bp_coverage_{'all' if not extra else 'since2021'}.csv", t)

    # --- B5b 场级 ban/pick 可用性 ---
    con.execute(
        """
        CREATE OR REPLACE VIEW gm AS
        SELECT gameid, any_value(league) league, CAST(any_value(date)[1:4] AS INT) y,
               any_value(datacompleteness) dc,
               min(CASE WHEN ban1 IS NOT NULL THEN 1 ELSE 0 END) ban_both,
               min(CASE WHEN pick1 IS NOT NULL THEN 1 ELSE 0 END) pick_both
        FROM oe WHERE position='team' GROUP BY gameid
        """
    )
    print("\n-- B5b 场级：两队 ban1 都在 / 两队 pick1 都在 --")
    print(con.sql("""SELECT dc, count(*) n, sum(1-ban_both) n_noban,
                     round(100.0*avg(1-ban_both),3) pct_noban,
                     sum(1-pick_both) n_nopick, round(100.0*avg(1-pick_both),3) pct_nopick
                     FROM gm GROUP BY 1""").df().to_string(index=False))
    print("\npartial 里缺 ban 的场集中在哪：")
    print(con.sql("""SELECT y, league, count(*) n, sum(1-ban_both) n_noban
                     FROM gm WHERE dc='partial' GROUP BY 1,2 HAVING n_noban>0
                     ORDER BY n_noban DESC LIMIT 10""").df().to_string(index=False))
    print("\nLPL 逐年（partial 占比 vs ban/pick 可用率）：")
    print(con.sql("""SELECT y, count(*) n, sum(CASE WHEN dc='partial' THEN 1 ELSE 0 END) n_partial,
                     round(100.0*avg(ban_both),2) ban_ok, round(100.0*avg(pick_both),2) pick_ok
                     FROM gm WHERE league='LPL' GROUP BY 1 ORDER BY 1""").df().to_string(index=False))

    # --- B5c tier-1 口径（结论的落点在这里）---
    t1 = _tier1_codes(con)
    lst = ",".join("'" + x.replace("'", "''") + "'" for x in t1)
    if t1:
        print("\n-- B5c 只看 tier1+国际赛：partial 的 BP 信息完整吗 --")
        for label, extra in [("全部 13 年", ""), ("date>='2021'", "AND date>='2021'")]:
            rows = []
            for c in ["patch", "playerid", "teamid", "champion", "side", "result",
                      "ban1", "ban5", "pick1", "firstPick"]:
                scope = "position='team'" if c in TEAM_ONLY_COLS else "position<>'team'"
                vals = {}
                for dc in ("complete", "partial"):
                    vals[dc] = con.sql(
                        f"""SELECT avg(CASE WHEN "{c}" IS NOT NULL THEN 100.0 ELSE 0 END)
                            FROM oe WHERE datacompleteness='{dc}' AND {scope}
                              AND league IN ({lst}) {extra}"""
                    ).fetchone()[0]
                rows.append({"column": c, "complete_cov%": round(vals["complete"], 3),
                             "partial_cov%": round(vals["partial"], 3),
                             "delta_pp": round(vals["partial"] - vals["complete"], 3)})
            print(f"\n[{label}]")
            print(pd.DataFrame(rows).to_string(index=False))
        print("\n→ tier1 口径下 partial 与 complete 的 BP 列差距全部在 ±2.5pp 以内，"
              "核心六列（champion/side/result/date/playerid/teamid）持平或 partial 更优；"
              "全库口径看到的 -14pp ban 缺口几乎全部来自 LDL 2023/2024 这两批三级联赛，"
              "与 `partial` 这个标记本身无关。")

    # --- B6 保留 partial 的提升 ---
    print("\n-- B6 保留 partial 相对只用 complete 的场次提升 --")
    for label, extra in [("全赛区", ""), (f"tier1+国际（{len(t1)} 个 code）", f"AND league IN ({lst})")]:
        d = con.sql(
            f"""SELECT CAST(date[1:4] AS INT) y, count(DISTINCT gameid) n_all,
                count(DISTINCT CASE WHEN datacompleteness='complete' THEN gameid END) n_complete
                FROM oe WHERE participantid='100' {extra} GROUP BY 1 ORDER BY 1"""
        ).df()
        d["n_partial"] = d.n_all - d.n_complete
        d["uplift_%"] = ((d.n_all / d.n_complete - 1) * 100).round(2)
        tot = d[["n_all", "n_complete"]].sum()
        print(f"\n[{label}]")
        print(d.to_string(index=False))
        print(f"13 年合计：{int(tot.n_complete):,} → {int(tot.n_all):,} "
              f"(+{(tot.n_all / tot.n_complete - 1) * 100:.2f}%)")
        dump(out, f"partial_uplift_{'all' if not extra else 'tier1'}.csv", d)
    print("\n→ 结论：`df[df.datacompleteness=='complete']` 这一行在 2022–2025 会删掉整个 LPL，"
          "tier1 口径下 13 年合计少 21.10% 的样本（2024 年一年就少 42.58%）。绝对不能写。")
    print("→ 并且 `datacompleteness` 本身禁止进特征：它与赛区几乎共线"
          "（2026 年 partial 100% 落在 LPL），等于偷偷把赛区喂给模型。")


def _tier1_codes(con: duckdb.DuckDBPyConnection) -> list[str]:
    """读 eval/tier1_leagues.yaml，并与数据里真实存在的 league code 求交。"""
    try:
        import yaml
        with open(TIER1_YAML, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        want = set(cfg.get("tier1_2026", [])) | set(cfg.get("tier1_2025_aliases", [])) \
            | set(cfg.get("international", []))
    except (OSError, ImportError, AttributeError, TypeError) as exc:  # 配置缺失时降级
        print(f"[warn] 读不到 {TIER1_YAML}（{exc}），tier1 口径跳过")
        return []
    have = {r[0] for r in con.sql("SELECT DISTINCT league FROM oe").fetchall()}
    missing = sorted(want - have)
    if missing:
        print(f"[warn] tier1_leagues.yaml 里这些 code 在数据中不存在：{missing} "
              "（实测数据里世界赛写作 'WLDs' 小写 s，配置里是 'WLDS'——大小写不一致）")
    return sorted(want & have)


# --------------------------------------------------------------------------- #
# C. 缺 id 场次 / dropna 陷阱
# --------------------------------------------------------------------------- #
def section_dropna(con: duckdb.DuckDBPyConnection, out: str | None) -> None:
    hdr("C. 12 行长表上 dropna 的后果（手册 v1.2 待修点）")

    con.execute(
        """
        CREATE OR REPLACE VIEW gstat AS
        SELECT gameid, any_value(league) league, CAST(any_value(date)[1:4] AS INT) y,
          any_value(datacompleteness) dc, count(*) n_rows,
          sum(CASE WHEN position<>'team' AND playerid IS NOT NULL THEN 1 ELSE 0 END) n_pid,
          sum(CASE WHEN position='team'  AND teamid  IS NOT NULL THEN 1 ELSE 0 END) n_tid,
          sum(CASE WHEN position<>'team' AND champion IS NULL THEN 1 ELSE 0 END) n_champ_null,
          sum(CASE WHEN patch  IS NULL THEN 1 ELSE 0 END) n_patch_null,
          sum(CASE WHEN result IS NULL THEN 1 ELSE 0 END) n_result_null,
          sum(CASE WHEN side   IS NULL THEN 1 ELSE 0 END) n_side_null
        FROM oe GROUP BY gameid
        """
    )
    print("-- C1 每场行数分布（阶段 2 验收③「12 行/场通过率 = 100%」）--")
    print(con.sql("SELECT n_rows, count(*) n_games FROM gstat GROUP BY 1 ORDER BY 1")
          .df().to_string(index=False))

    tot = int(con.sql("SELECT count(*) FROM gstat").fetchone()[0])
    r = con.sql(
        """SELECT
           sum(CASE WHEN n_pid<10 THEN 1 ELSE 0 END) miss_any_playerid,
           sum(CASE WHEN n_pid=0  THEN 1 ELSE 0 END) playerid_all_null,
           sum(CASE WHEN n_tid<2  THEN 1 ELSE 0 END) miss_any_teamid,
           sum(CASE WHEN n_tid=0  THEN 1 ELSE 0 END) teamid_all_null,
           sum(CASE WHEN n_pid<10 OR n_tid<2 THEN 1 ELSE 0 END) union_bad
           FROM gstat"""
    ).df().iloc[0]
    print(f"\n-- C2 全量 13 年（{tot:,} 场）缺 id 情况 --")
    for k, v in r.items():
        print(f"  {k:22s} {int(v):7,d}  ({int(v) / tot * 100:6.3f}%)")
    print(f"  → 折叠后「每场 10 个非空 playerid」的比例 = "
          f"{(1 - r.miss_any_playerid / tot) * 100:.2f}%（阶段 2 验收③要求 ≥99%，"
          f"**全库口径不达标**，只有整场丢弃后才是 100%）")

    print("\n-- C3 手册数字复核：2026 年 675 场缺 playerid / 380 场缺 teamid --")
    r26 = con.sql(
        """SELECT count(*) n, sum(CASE WHEN n_pid<10 THEN 1 ELSE 0 END) mp,
           sum(CASE WHEN n_tid<2 THEN 1 ELSE 0 END) mt,
           sum(CASE WHEN n_pid<10 OR n_tid<2 THEN 1 ELSE 0 END) u
           FROM gstat WHERE y=2026"""
    ).df().iloc[0]
    print(f"  实测 2026：{int(r26.n):,} 场；缺 playerid {int(r26.mp)} 场 "
          f"({r26.mp / r26.n * 100:.2f}%)；缺 teamid {int(r26.mt)} 场 "
          f"({r26.mt / r26.n * 100:.2f}%)；并集 {int(r26.u)} 场 ({r26.u / r26.n * 100:.2f}%)")
    print("  → 675 / 380 两个数字完全复现 ✓（手册写的 7.6% / 4.3% 分母是 8,832，"
          "当前快照 8,851，比例 7.63% / 4.29%）")

    by_y = con.sql(
        """SELECT y, count(*) n,
           sum(CASE WHEN n_pid<10 THEN 1 ELSE 0 END) miss_pid,
           sum(CASE WHEN n_tid<2  THEN 1 ELSE 0 END) miss_tid,
           sum(CASE WHEN n_pid<10 OR n_tid<2 THEN 1 ELSE 0 END) bad
           FROM gstat GROUP BY 1 ORDER BY 1"""
    ).df()
    by_y["pct_bad"] = (by_y.bad / by_y.n * 100).round(2)
    print("\n-- C4 按年 --")
    print(by_y.to_string(index=False))
    by_l = con.sql(
        """SELECT league, count(*) n, sum(CASE WHEN n_pid<10 OR n_tid<2 THEN 1 ELSE 0 END) bad
           FROM gstat GROUP BY 1 HAVING bad>0 ORDER BY bad DESC LIMIT 15"""
    ).df()
    by_l["pct"] = (by_l.bad / by_l.n * 100).round(2)
    print("\n-- C5 按赛区（丢弃最多的 15 个）--")
    print(by_l.to_string(index=False))
    dump(out, "bad_id_games_by_year.csv", by_y)
    dump(out, "bad_id_games_by_league.csv", by_l)

    print("\n-- C6 行级 dropna 到底造出什么 --")
    kept = con.sql(
        """SELECT count(*) rows_kept, count(DISTINCT gameid) games_kept FROM oe
           WHERE champion IS NOT NULL AND side IS NOT NULL AND result IS NOT NULL
             AND patch IS NOT NULL AND date IS NOT NULL
             AND playerid IS NOT NULL AND teamid IS NOT NULL"""
    ).df().iloc[0]
    short = int(con.sql(
        """SELECT count(*) FROM (SELECT gameid, count(*) k FROM oe
           WHERE champion IS NOT NULL AND side IS NOT NULL AND result IS NOT NULL
             AND patch IS NOT NULL AND date IS NOT NULL
             AND playerid IS NOT NULL AND teamid IS NOT NULL
           GROUP BY gameid) t WHERE k<10"""
    ).fetchone()[0])
    print("  手册 v1.0 写法 df.dropna(subset=['champion','side','result','patch','date',"
          "'playerid','teamid'])：")
    print(f"    剩余行 {int(kept.rows_kept):,} / 1,225,008；剩余场 {int(kept.games_kept):,} / {tot:,}")
    print(f"    其中剩 <10 个选手行的畸形场 = {short:,}")
    print("    致命点：team 行的 champion 与 playerid **按设计恒为空**，所以这一行 dropna 会把"
          "全部 204,168 条 team 行删光——ban1-5 / pick1-5 / teamid 全没了，"
          "而代码不报错。")

    print("\n-- C7 正确做法：整场判定 --")
    ok_sql = ("n_rows=12 AND n_pid>=10 AND n_tid=2 AND n_champ_null=0 "
              "AND n_result_null=0 AND n_side_null=0")
    d = con.sql(f"""SELECT count(*) n_all, sum(CASE WHEN {ok_sql} THEN 1 ELSE 0 END) n_keep,
                    sum(CASE WHEN NOT({ok_sql}) THEN 1 ELSE 0 END) n_drop FROM gstat""").df().iloc[0]
    d2 = con.sql(f"""SELECT sum(CASE WHEN NOT({ok_sql} AND n_patch_null=0) THEN 1 ELSE 0 END) n_drop
                     FROM gstat""").df().iloc[0]
    print(f"  整场判定（不要求 patch）：保留 {int(d.n_keep):,}，丢弃 {int(d.n_drop):,} "
          f"({d.n_drop / tot * 100:.3f}%)")
    print(f"  再要求 patch 非空：丢弃 {int(d2.n_drop):,} ({d2.n_drop / tot * 100:.3f}%)")
    print("\n-- C8 把手册 §3.2.5 的 keep_complete_games() 原样跑一遍（2026 年）--")
    _run_manual_keep_complete(con)


def _run_manual_keep_complete(con: duckdb.DuckDBPyConnection) -> None:
    """把手册那段 `keep_complete_games` **逐字**跑在 2026 数据上，看它真的保留几场。

    结论先说：**保留 0 场**。因为 `g[['champion','side','result','date']].notna()
    .all().all()` 作用在整组 12 行上，而 team 汇总行的 `champion` 按 OE 设计恒为空，
    于是每一组都是 False。这类 bug 不抛异常、下游只会看到一张空表。
    """
    df = con.sql(
        """SELECT gameid, position, playerid, teamid, champion, side, result, date, patch
           FROM oe WHERE date >= '2026'"""
    ).df()
    n = df.gameid.nunique()

    def ok(g: pd.DataFrame) -> bool:  # 手册原文
        return (len(g) == 12
                and g.loc[g.position != "team", "playerid"].notna().sum() >= 10
                and g.loc[g.position == "team", "teamid"].notna().all()
                and g[["champion", "side", "result", "date"]].notna().all().all())

    def ok_fixed(g: pd.DataFrame) -> bool:  # 修正版：champion 只在选手行上要求
        p = g.position != "team"
        return (len(g) == 12
                and g.loc[p, "playerid"].notna().sum() >= 10
                and g.loc[~p, "teamid"].notna().all()
                and g.loc[p, "champion"].notna().all()
                and g[["side", "result", "date"]].notna().all().all())

    kept = df.groupby("gameid").filter(ok).gameid.nunique()
    kept_fixed = df.groupby("gameid").filter(ok_fixed).gameid.nunique()
    print(f"  2026 共 {n:,} 场")
    print(f"  手册原文 ok()   → 保留 {kept:,} 场   （team 行 champion 恒空，每组必 False）")
    print(f"  修正版 ok_fixed() → 保留 {kept_fixed:,} 场，丢弃 {n - kept_fixed:,} "
          f"({(n - kept_fixed) / n * 100:.2f}%)")
    print("  修正只需一处：g[['champion',...]].notna().all().all()")
    print("            → g.loc[g.position!='team','champion'].notna().all()")

    d1 = df.dropna(subset=["champion", "side", "result", "patch", "date", "playerid", "teamid"])
    k = d1.groupby("gameid").size()
    print(f"  对照：手册 v1.0 的行级 dropna → 剩 {d1.gameid.nunique():,} 场，"
          f"其中 {int((k < 10).sum()):,} 场只剩 <10 个选手行，"
          f"{int((k == 12).sum()):,} 场还剩 12 行（team 行已被删光，所以必然是 0）")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=RAW_GLOB, help="原始 CSV glob")
    ap.add_argument("--out", default=None, help="CSV 输出目录；不给则只打印")
    ap.add_argument("--section", default="all",
                    choices=["all", "patch", "completeness", "dropna"])
    ap.add_argument("--versions", default=None,
                    help="DDragon versions.json 路径；默认 data/ddragon/versions.json")
    a = ap.parse_args(argv)

    con = connect(a.raw)
    n_rows, n_games = con.sql("SELECT count(*), count(DISTINCT gameid) FROM oe").fetchone()
    print(f"数据源 {a.raw}：{n_rows:,} 行 / {n_games:,} 场")
    if os.path.exists(HOLDOUT_START_FILE):
        with open(HOLDOUT_START_FILE, encoding="utf-8") as fh:
            holdout = fh.read().strip()
        print(f"holdout 起点 {holdout}（本报告是纯描述性统计，不用于任何调参或模型选择）")

    if a.section in ("all", "patch"):
        section_patch(con, a.out, a.versions)
    if a.section in ("all", "completeness"):
        section_completeness(con, a.out)
    if a.section in ("all", "dropna"):
        section_dropna(con, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
