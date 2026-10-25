"""把盯盘台账 reports/live_watch.jsonl 整理成「一局一行」的复盘表；顺带给实力快照供料。

和站点「历史回看」的区别：那一页是事后拿模型重算历史比赛（含 holdout，只能肉眼看）；
这里是盯盘读到 BP 后阵容时落盘的预测——只用阵容（feed 的 gameMetadata）和当时的模型，
不读任何赛中数据，是唯一干净的线上成绩。落盘一般在开局后几分钟内，但盯盘休眠 / 断线恢复后
会补抓，个别局落盘时已经打完（`late`）：概率和开局时算的一样，只是不算严格的实时预测。
「当时的模型」= 外层系数冻结、实力快照随赛果重解（`models.artifact.refresh`，见 live_games）。

几个坑：
- `teams` 是赛程里的队伍顺序，**不是蓝红方**（换边后会反过来）。蓝红方优先用对账写进 result
  的 blue_team / red_team（来自 feed 的 esportsTeamId）；没有时按选手名的队伍 tag 投票——
  "DK Jaehyuk" 和不带空格的 "JDGXiaoxu" 都要认，替补可能挂别队的 tag，所以取多数。
  认不出就显示 "?"，绝不按赛程顺序猜。
- 同一局可能有多条 result（判不出 → 之后重对账补上；补蓝红方与时间），取最后一条。
- `source: "backfill"` 的预测是盯盘停摆 / 机器睡眠时整局错过、事后从赛程补出来的
  （scripts/live_watch.py --backfill）：一律算「赛后补录」，开局时间在对账前就有（started_at）。
- `model` 是算这条预测的 artifact（「规格@训练截止日」，2026-10-24 起才记；实力快照用赛后经济差软标签的
  规格叫 BT4_LR+MOV，修订 A6）。在那之前落盘的预测全部出自训练截至 2026-05-31 的那一个模型，
  没有这个字段的按它算（LEGACY_MODEL）。
- 台账由另一个进程追加，单行可能损坏（磁盘写满、手改）。坏行跳过并计数（df.attrs），
  不能让一行坏数据把整页带崩。
- `rating_only: true` 的 result 没有对应的预测：盯盘漏掉、开局又早于实力快照截止的局补不了预测，
  只补赛果（scripts/live_watch.py 的 backfill_results），给实力快照用；「近期战绩」只看有预测的局，不看它。
- 对账时会把这一局的阵容解析成 OE 的队伍 / 选手 id 写进 result（`blue_oe_team` …），
  `live_games` 据此把 OE 还没收录的局拼成和 silver 同形的行，给实力快照重解用。
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

LIVE_PREFIX = "live:"
# OE 与 feed 记的开赛时间差几分钟；同系列相邻两局开赛至少隔 32 分钟（2019 起 12,707 个系列实测）。
# 30 分钟的护栏挡得住同一局算两遍，又不会误伤同系列的下一局。
LIVE_GUARD = pd.Timedelta(minutes=30)
LIVE_COLUMNS = ["game_id", "kickoff_utc", "ended_utc", "league_canon", "league_tier", "blue_team_id",
                "red_team_id", "blue_player_ids", "red_player_ids", "y_blue_win", "y_margin", "p_pre"]
_FALLBACK = "fallback:"


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


def _read(path: str) -> tuple[dict, dict, int]:
    """(预测 by game_id, 最后一条结果 by game_id, 坏行数)。"""
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
    return preds, results, bad


def load_review(path: str = LEDGER) -> pd.DataFrame:
    """一局一行，按比赛时间从新到旧。status ∈ {命中, 未中, 待定}；待定 = 没结果或结果判不出。

    game_time 是开局时间（对账写入的 started_at；补录的局预测里自带）；
    两处都没有的局只有落盘时间，就用它。
    late = 落盘晚于终局（True / False；终局时间未知时 None）；补录的局恒为 True。
    """
    preds, results, bad = _read(path)
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


def _players(res: dict, side: str) -> list[str | None]:
    """5 个位置的 OE 选手 id，解析不出的位置是 None（快照按该队普通选手计价，和 silver 的缺位一样）。"""
    ps = list(res.get(f"{side}_oe_players") or [])[:5]
    ps += [None] * (5 - len(ps))
    return [p if isinstance(p, str) and p and not p.startswith(_FALLBACK) else None for p in ps]


def _num(x: object) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else float("nan")


def _team(res: dict, side: str, last_team: dict[str, str]) -> str | None:
    """对账解析出的 OE 队伍 id。primary 对照直接认；contested 的（同一个 esports 队历史上对过
    多个 OE 队，例如改名换壳）要阵容作证：至少 3 名解析出的选手在 OE 里最近一年最后效力的就是它。
    实测 2026 世界赛的 MVK 就是这种：对照可信度 contested，但 5 个人上一场都在这支 OE 队。"""
    t = res.get(f"{side}_oe_team")
    if not isinstance(t, str) or not t or t.startswith(_FALLBACK):
        return None
    if res.get(f"{side}_team_conf") == "primary":
        return t
    return t if sum(last_team.get(p) == t for p in _players(res, side) if p) >= 3 else None


def _last_team(silver: pd.DataFrame, days: int = 365) -> dict[str, str]:
    """选手 → silver 里最近一年最后效力的 OE 队。"""
    recent = silver[silver.kickoff_utc >= silver.kickoff_utc.max() - pd.Timedelta(days=days)]
    out: dict[str, str] = {}
    for side in ("blue", "red"):
        for t, ps, k in zip(recent[f"{side}_team_id"], recent[f"{side}_player_ids"], recent.kickoff_utc):
            for p in ps if ps is not None else ():
                if isinstance(p, str):
                    out.setdefault(p, (k, t))
                    if k > out[p][0]:
                        out[p] = (k, t)
    return {p: t for p, (_, t) in out.items()}


def live_games(silver: pd.DataFrame, path: str = LEDGER) -> pd.DataFrame:
    """台账里判出胜负、OE 还没收录的局 → 与 silver 同形的行（只含实力快照要用的列）。

    给 `models.artifact.refresh` 重解实力快照用：快照冻结一个月 log loss 就差 0.012
    （reports/exp_model_improvements.md §1.2），而 OE 会断更（2026-09-18 起 Drive 配额），
    这时快照只能靠盯盘自己记下的赛果保持新鲜。只收：
    - 对账时解析出了阵容、两队对照可信的局（primary，或 contested 但阵容作证，见 _team；
      选手解析不出的位置留 None）；
    - 胜负判得出（blue_won 是 bool）；
    - OE 还没有的局：开局晚于 silver 里**该赛区**最后一场 + 30 分钟，且 silver 里没有同一对队伍
      在前后 30 分钟内开的局（防赛区标签两边不一致时同一局算两遍）。

    game_id 加 `live:` 前缀，和 OE 的 id 不会撞；league_tier 取 silver 里该赛区最近的口径，
    没见过的赛区按 0（不参与「主场赛区」的推断以外的任何口径）。
    y_margin = 赛后经济差软标签（models.margin），终局经济取对账记下的 feed 终局帧，时长 = 终局 − 首帧；
    缺了就退回胜负，和 silver 一样。
    p_pre = 开赛前落盘的蓝方胜率，给近期状态（models.form）算残差：有 p_base（加近期状态之前的概率）用它，
    更早的记录没有这一项、当时也还没有近期状态这一层，就用 p_blue；没有预测的局为 NaN。
    """
    from lolwp.models.margin import margin_target

    preds, results, _ = _read(path)
    if not results:
        return pd.DataFrame(columns=LIVE_COLUMNS)
    last = silver.groupby("league_canon").kickoff_utc.max()
    tier = silver.groupby("league_canon").league_tier.last()
    last_team = _last_team(silver) if any(
        r.get(f"{s}_team_conf") == "contested" for r in results.values() for s in ("blue", "red")) else {}
    pairs: dict[frozenset, list[pd.Timestamp]] = {}
    for b, r, t in zip(silver.blue_team_id.astype(str), silver.red_team_id.astype(str),
                       silver.kickoff_utc):
        pairs.setdefault(frozenset((b, r)), []).append(t)
    rows = []
    for gid, res in results.items():
        if not isinstance(res.get("blue_won"), bool):
            continue
        bt, rt = _team(res, "blue", last_team), _team(res, "red", last_team)
        if not bt or not rt or bt == rt:
            continue
        p = preds.get(gid) or {}
        start = _utc(res.get("started_at") or p.get("started_at") or p.get("recorded_at"))
        if pd.isna(start):
            continue
        lg = res.get("league_canon") or "UNKNOWN"
        if lg in last.index and start <= last[lg] + LIVE_GUARD:
            continue
        if any(abs(start - t) <= LIVE_GUARD for t in pairs.get(frozenset((bt, rt)), ())):
            continue
        end = _utc(res.get("ended_at"))
        secs = (end - start).total_seconds() if pd.notna(end) else float("nan")
        y_margin = margin_target([float(res["blue_won"])], [_num(res.get("blue_gold"))],
                                 [_num(res.get("red_gold"))], [secs])[0]
        rows.append({
            "game_id": f"{LIVE_PREFIX}{gid}", "kickoff_utc": start, "ended_utc": end,
            "league_canon": lg, "league_tier": int(tier.get(lg, 0)),
            "blue_team_id": bt, "red_team_id": rt,
            "blue_player_ids": _players(res, "blue"), "red_player_ids": _players(res, "red"),
            "y_blue_win": int(res["blue_won"]), "y_margin": float(y_margin),
            "p_pre": _num(p.get("p_base", p.get("p_blue"))),
        })
    df = pd.DataFrame(rows, columns=LIVE_COLUMNS)
    for c in ("kickoff_utc", "ended_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df.sort_values(["kickoff_utc", "game_id"], kind="mergesort", ignore_index=True)


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
