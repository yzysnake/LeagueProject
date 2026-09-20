#!/usr/bin/env python3
"""阶段 0：Oracle's Elixir 年度 CSV 的五重校验 + manifest 登记。

相对手册 v1.1 的修正（均有实测依据，见 docs/PLAN.md §3.2.3）：
  1. 新鲜度阈值 3 天 → 45 天硬失败 / 10 天告警，且冬歇窗口（12-15 ~ 01-31）豁免硬失败。
     实测全量 2014-2026 的最大无比赛空档是 **34 天**（2024-12-08 → 2025-01-11），
     其次 28 天（2023-11-27 → 2023-12-25）、26 天（2014）、25 天（2015）。
     手册原本的 3 天每年固定误报数周；v1.2 建议的 35 天对 34 天只剩 1 天余量，仍偏紧。
     ⚠️ 本闸的根本局限：它测的是"最后一场比赛离今天多久"，而我们真正关心的是
     "文件离上次更新多久"。后者要靠 Drive 的 modifiedTime，尚未接入（见 blockers）。
  2. datetime.utcnow() 已在 Python 3.12 废弃 → datetime.now(timezone.utc)。
  3. sha256 改分块读取（单文件最大 109 MB，13 个合计约 700 MB）。
  4. manifest.jsonl 是 append-only 日志，读取时按 year 取最后一条，不是盲读第一条。
  5. 当年文件在年初可能为空，此时跳过新鲜度校验而不是失败。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, date, datetime

RAW = "data/raw"
MANIFEST = "data/manifest.jsonl"
EXPECTED_PREFIX = (
    "gameid,datacompleteness,url,league,year,split,playoffs,date,game,patch"
)
STALE_FAIL_DAYS = 45
STALE_WARN_DAYS = 10
ROWS_PER_GAME = 12

csv.field_size_limit(10_000_000)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prev(path: str) -> dict[int, dict]:
    """manifest 是 append-only 日志；同一 year 取最后一条。"""
    prev: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prev[rec["year"]] = rec
    return prev


def validate(path: str, year: int, prev: dict, today: date) -> tuple[dict, list[str]]:
    warns: list[str] = []

    # 闸 1：不是 Drive 配额错误页（HTTP 200 + HTML，csv 读它得到 0 行且不报错）
    with open(path, "rb") as fh:
        head = fh.read(4096)
    if b"Quota exceeded" in head or head.lstrip()[:1] == b"<":
        raise AssertionError("闸1 失败：拿到的是 HTML（Drive 配额错误页），不是 CSV")

    # 闸 2：体积下界
    size = os.path.getsize(path)
    if size < 1_000_000:
        raise AssertionError(f"闸2 失败：文件过小 {size} bytes")

    # 闸 3：表头前缀（防列被重排；注意 OE 会在中间插列，故只校验前 10 列）
    first_line = head.decode("utf-8", "replace").split("\n")[0].strip().lstrip("﻿")
    if not first_line.startswith(EXPECTED_PREFIX):
        raise AssertionError(f"闸3 失败：表头前缀异常 {first_line[:100]!r}")

    # 闸 4：12 行/场，且场次 > 0（0 % 12 == 0 会放行，必须显式查 n_games > 0）
    per_game: Counter[str] = Counter()
    n_rows = 0
    max_date = ""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        n_cols = len(rd.fieldnames or [])
        for row in rd:
            n_rows += 1
            per_game[row["gameid"]] += 1
            d = row.get("date") or ""
            max_date = max(max_date, d)
    n_games = len(per_game)
    if n_games == 0:
        raise AssertionError("闸4 失败：解析出 0 场比赛")
    bad = {g: c for g, c in per_game.items() if c != ROWS_PER_GAME}
    if bad:
        raise AssertionError(
            f"闸4 失败：{len(bad)} 场不是 {ROWS_PER_GAME} 行/场，例 {list(bad.items())[:3]}"
        )
    if n_rows != ROWS_PER_GAME * n_games:
        raise AssertionError(f"闸4 失败：行数 {n_rows} != 12 × {n_games}")

    # 闸 5a：相对上次不得倒退
    if prev:
        if n_games < prev["n_games"] * 0.98:
            raise AssertionError(f"闸5a 失败：场次倒退 {prev['n_games']} → {n_games}")
        if max_date < prev["max_date"]:
            raise AssertionError(f"闸5a 失败：max_date 倒退 {prev['max_date']} → {max_date}")

    # 闸 5b：当年文件的新鲜度（年初可能尚无比赛，此时跳过）
    if year == today.year and max_date:
        age = (today - date.fromisoformat(max_date[:10])).days
        in_winter_break = (today.month == 12 and today.day >= 15) or today.month == 1
        if age > STALE_FAIL_DAYS and not in_winter_break:
            raise AssertionError(f"闸5b 失败：数据陈旧 {age} 天，max_date={max_date[:10]}")
        if age > STALE_WARN_DAYS:
            warns.append(
                f"新鲜度告警：{age} 天未更新"
                + ("（冬歇窗口，已豁免硬失败）" if in_winter_break else f"（阈值 {STALE_WARN_DAYS}）")
            )

    rec = {
        "year": year,
        "n_rows": n_rows,
        "n_games": n_games,
        "n_cols": n_cols,
        "max_date": max_date[:10],
        "bytes": size,
        "sha256": sha256_of(path),
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    return rec, warns


def main() -> int:
    today = datetime.now(UTC).date()
    prev = load_prev(MANIFEST)
    files = sorted(
        f for f in os.listdir(RAW) if f.endswith("_LoL_esports_match_data_from_OraclesElixir.csv")
    )
    if not files:
        print("FAIL: data/raw 下没有年度 CSV", file=sys.stderr)
        return 1

    recs, failures, all_warns = [], [], []
    for fn in files:
        year = int(fn[:4])
        try:
            rec, warns = validate(os.path.join(RAW, fn), year, prev.get(year, {}), today)
        except AssertionError as e:
            failures.append(f"{year}: {e}")
            print(f"  {year}  FAIL  {e}")
            continue
        recs.append(rec)
        all_warns += [f"{year}: {w}" for w in warns]
        print(
            f"  {rec['year']}  ok  {rec['n_games']:>6} 场  {rec['n_cols']} 列  "
            f"max_date={rec['max_date']}  {rec['bytes']/1048576:6.1f} MB  {rec['sha256'][:12]}"
        )

    # 按 (year, sha256) 去重：内容没变就不重复追加，否则重跑会把 manifest 撑爆，
    # 且阶段 0 验收⑤「manifest 恰好 13 条」永远不可能满足。
    seen = {(r["year"], r["sha256"]) for r in prev.values()}
    if os.path.exists(MANIFEST):
        seen = set()
        with open(MANIFEST) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    seen.add((r["year"], r["sha256"]))
    new = [r for r in recs if (r["year"], r["sha256"]) not in seen]
    if new:
        tmp = MANIFEST + ".tmp"
        with open(tmp, "w") as mf:
            if os.path.exists(MANIFEST):
                with open(MANIFEST) as old:
                    mf.write(old.read())
            for r in new:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, MANIFEST)
    print(f"manifest：新增 {len(new)} 条，已存在 {len(recs) - len(new)} 条未重复写入")

    print(f"\n通过 {len(recs)} / {len(files)}，合计 {sum(r['n_games'] for r in recs)} 场")
    for w in all_warns:
        print("  WARN", w)
    for f in failures:
        print("  FAIL", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
