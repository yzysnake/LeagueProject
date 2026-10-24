"""把盯盘台账 reports/live_watch.jsonl 整理成「一局一行」的复盘表。

和站点「历史回看」的区别：那一页是事后拿模型重算历史比赛（含 holdout，只能肉眼看）；
这里是盯盘读到 BP 后阵容时落盘的预测——只用阵容（feed 的 gameMetadata）和冻结的模型，
不读任何赛中数据，是唯一干净的线上成绩。落盘一般在开局后几分钟内，但盯盘休眠 / 断线恢复后
会补抓，个别局落盘时已经打完（`late`）：概率和开局时算的一样，只是不算严格的实时预测。

几个坑：
- `teams` 是赛程里的队伍顺序，**不是蓝红方**（换边后会反过来）。蓝红方优先用对账写进 result
  的 blue_team / red_team（来自 feed 的 esportsTeamId）；没有时按选手名的队伍 tag 投票——
  "DK Jaehyuk" 和不带空格的 "JDGXiaoxu" 都要认，替补可能挂别队的 tag，所以取多数。
  认不出就显示 "?"，绝不按赛程顺序猜。
- 同一局可能有多条 result（判不出 → 之后重对账补上；补蓝红方与时间），取最后一条。
- `source: "backfill"` 的预测是盯盘停摆 / 机器睡眠时整局错过、事后从赛程补出来的
  （scripts/live_watch.py --backfill）：一律算「赛后补录」，开局时间在对账前就有（started_at）。
- `model` 是算这条预测的 artifact（「BT4_LR@训练截止日」，2026-10-24 起才记）。在那之前落盘的
  预测全部出自训练截至 2026-05-31 的那一个模型，没有这个字段的按它算（LEGACY_MODEL）。
- 台账由另一个进程追加，单行可能损坏（磁盘写满、手改）。坏行跳过并计数（df.attrs），
  不能让一行坏数据把整页带崩。
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import pandas as pd

LEDGER = "reports/live_watch.jsonl"
LEGACY_MODEL = "BT4_LR@2026-05-31"
COLUMNS = ["game_time", "recorded_at", "started_at", "ended_at", "late", "backfill", "model", "game_id",
           "match_id", "league", "blue", "red", "p_blue", "blue_won", "status", "hit"]


def _tag(name: object, teams: list[str]) -> str | None:
    """单个选手名的队伍 tag：先认赛程里的队伍代码（不分大小写，最长优先："T1A" 先于 "T1"），
    再认 "TAG 名字" 的空格前缀。名字缺失（resolver 对缺位 / 空名给 None）→ None。"""
    if not isinstance(name, str) or not name.strip():
        return None
    up = name.upper()
    for t in sorted(teams, key=len, reverse=True):
        if up.startswith(t.upper()):
            return t
    return name.split(" ", 1)[0] if " " in name else None


def _side_code(names: object, teams: list[str]) -> str | None:
    """一方的队伍代码：过半名字挂同一个 tag 才算数，替补挂别队 tag 不影响。"""
    votes = Counter(t for t in (_tag(n, teams) for n in (names if isinstance(names, list) else []))
                    if t)
    if not votes:
        return None
    code, n = votes.most_common(1)[0]
    return code if n * 2 > sum(votes.values()) else None


def _sides(pred: dict, res: dict) -> tuple[str, str]:
    teams = [t for t in dict.fromkeys(pred.get("teams") or []) if isinstance(t, str) and t]
    b = res.get("blue_team") or _side_code(pred.get("blue_names"), teams)
    r = res.get("red_team") or _side_code(pred.get("red_names"), teams)
    if b and b == r:
        return "?", "?"  # 两边认成同一队：数据有问题，不硬配
    if len(teams) == 2:  # 只认出一边时，另一边是赛程里的另一队
        if b in teams and not r:
            r = teams[1 - teams.index(b)]
        elif r in teams and not b:
            b = teams[1 - teams.index(r)]
    return b or "?", r or "?"


def _utc(x: object) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce") if x else pd.NaT


def load_review(path: str = LEDGER) -> pd.DataFrame:
    """一局一行，按比赛时间从新到旧。status ∈ {命中, 未中, 待定}；待定 = 没结果或结果判不出。

    game_time 是开局时间（对账写入的 started_at；补录的局预测里自带）；
    两处都没有的局只有落盘时间，就用它。
    late = 落盘晚于终局（True / False；终局时间未知时 None）；补录的局恒为 True。
    """
    preds, results, bad = {}, {}, 0
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not isinstance(r, dict) or not r.get("game_id"):
                    bad += 1
                    continue
                if r.get("kind") == "prediction":
                    preds[r["game_id"]] = r
                elif r.get("kind") == "result":
                    results[r["game_id"]] = r
    rows = []
    for gid, p in preds.items():
        res = results.get(gid) or {}
        blue, red = _sides(p, res)
        pb = p.get("p_blue")
        pb = float(pb) if isinstance(pb, (int, float)) and not isinstance(pb, bool) else None
        bw = res.get("blue_won")
        bw = bw if isinstance(bw, bool) else None
        hit = None if bw is None or pb is None else (pb >= 0.5) == bw
        rec, end = _utc(p.get("recorded_at")), _utc(res.get("ended_at"))
        start = _utc(res.get("started_at") or p.get("started_at"))
        backfill = p.get("source") == "backfill"
        late = True if backfill else (bool(rec > end) if pd.notna(rec) and pd.notna(end) else None)
        rows.append({
            "game_time": start if pd.notna(start) else rec, "recorded_at": rec,
            "started_at": start, "ended_at": end, "late": late, "backfill": backfill,
            "model": p.get("model") or LEGACY_MODEL,
            "game_id": str(gid), "match_id": p.get("match_id"), "league": p.get("league") or "?",
            "blue": blue, "red": red, "p_blue": pb, "blue_won": bw,
            "status": "待定" if hit is None else ("命中" if hit else "未中"), "hit": hit,
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("game_time", "recorded_at", "started_at", "ended_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    # 同一时刻落盘的几局（盯盘补抓时常见）按 game_id 定序：同一系列里 id 随局数递增
    df = df.sort_values(["game_time", "game_id"], ascending=False, ignore_index=True)
    df.attrs["n_bad_lines"] = bad
    return df


def summarize(df: pd.DataFrame) -> dict:
    """只对已判出胜负的局算指标；待定的局只计数。"""
    done = df[df["hit"].notna()]
    out = {"n": len(df), "n_done": len(done), "n_pending": len(df) - len(done),
           "hit_rate": None, "log_loss": None, "brier": None}
    if len(done):
        p = done["p_blue"].astype(float).clip(1e-9, 1 - 1e-9)
        y = done["blue_won"].astype(bool).astype(float)
        out["hit_rate"] = float(done["hit"].astype(bool).mean())
        out["log_loss"] = float(-(y * p.map(math.log) + (1 - y) * (1 - p).map(math.log)).mean())
        out["brier"] = float(((p - y) ** 2).mean())
    return out
