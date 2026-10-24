#!/usr/bin/env python3
"""实时盯盘 + 赛后对账。

流程：轮询 getLive → 对每个 gameId 等 livestats 的 204→200 跳变 →
解析阵容 → 出 BP 后的正式预测 → 落盘 → 赛后拉结果对账。

为什么值得单独写：赛前我只能用占位英雄给「实力层估计」，
而且队伍不在历史数据里就只能拒绝预测。**开打后阵容出来，情况会变好**——
即使队伍名陌生，选手可能通过 crosswalk 解析出来，再由选手反推队伍。
所以「赛前不可预测」不等于「赛后不可预测」。

产出 reports/live_watch.jsonl（append-only），每条是一个 gameId 的完整记录。

**赛后补录（--backfill）**：盯盘没在跑（launchd 停了）或机器在睡（合盖），
整局开打又打完都不在 getLive 里出现，实时轮询就永远看不到它。但 feed 对打完的局
仍然给 gameMetadata（BP 后阵容），所以可以事后从赛程找出这些局、读阵容、用同一个
冻结模型补出预测——和实时算的是同一个数（只用阵容，不读任何赛中数据），只是落盘晚，
记录带 `source: "backfill"`，在「近期战绩」里归入「赛后补录」，可以一键排除。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta

LEDGER = "reports/live_watch.jsonl"
API = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
WAKE_GAP_S = 120          # 一轮间隔超过这么久 = 机器刚从睡眠里醒来
SCHEDULE_MAX_PAGES = 12   # getSchedule 每页 80 场，12 页约覆盖一个月


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _append(rec: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _records():
    """逐条读台账。坏行（磁盘写满时的半截行、手改）跳过：盯盘启动和对账都要读台账，
    一行坏数据不能让 launchd 每次重启都崩在这里、从此再也盯不了盘。"""
    if not os.path.exists(LEDGER):
        return
    with open(LEDGER) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  台账有一行无法解析，已跳过：{line[:80]!r}", file=sys.stderr)
                continue
            if isinstance(r, dict) and r.get("game_id"):
                yield r


def _seen_games() -> set[str]:
    return {r["game_id"] for r in _records() if r.get("kind") == "prediction"}


def _ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _feed_game(s, gid: str) -> dict | None:
    """从 livestats 读这一局：蓝红方队伍、开局 / 终局时间、终局帧判出的胜负。拿不到 window → None。

    **蓝红方以 feed 为准**：预测的 p_blue 是 feed 的 blueTeamMetadata 那一方的胜率，
    而 getEventDetails 的 games[].teams[].side 常常和 feed 对不上（LPL、DCGI 淘汰赛、
    Worlds 2025 T1-AL 五局全反）。拿 side 当蓝方，系列比分推出的结果会整个翻过来。

    胜负：推掉水晶必须先推掉至少一座水晶塔（inhibitor），所以 gameState=finished 那一帧里
    **恰好一方 inhibitors>0** 时胜方无歧义。双方都>0 或都为 0（投降）→ blue_won=None，不猜。
    window 每次只返回 ~10 秒的帧，从首帧起每 5 分钟往后跳；跳过终局后 feed 一直回最后几帧，
    所以帧不再前进 = 这局 feed 停在了对局中途、没有 finished 帧，不必再跳。
    """
    from datetime import timedelta

    r = s.get(f"{FEED}/window/{gid}", timeout=20)
    if r.status_code != 200 or not r.content:
        return None
    first = r.json()
    md = first.get("gameMetadata") or {}
    frames = first.get("frames") or []
    out = {"blue_id": (md.get("blueTeamMetadata") or {}).get("esportsTeamId"),
           "red_id": (md.get("redTeamMetadata") or {}).get("esportsTeamId"),
           "started_at": frames[0].get("rfc460Timestamp") if frames else None,
           "ended_at": None, "blue_won": None}
    if not frames:
        return out
    t0 = datetime.fromisoformat(frames[0]["rfc460Timestamp"])
    t0 = t0.replace(second=t0.second // 10 * 10, microsecond=0)  # startingTime 须是 10 秒整
    last = None
    for mins in range(5, 125, 5):
        r = s.get(f"{FEED}/window/{gid}", params={"startingTime": _ts(t0 + timedelta(minutes=mins))},
                  timeout=20)
        frames = (r.json().get("frames") or []) if r.status_code == 200 and r.content else []
        if not frames:
            break
        fin = next((f for f in frames if f.get("gameState") == "finished"), None)
        if fin:
            out["ended_at"] = fin.get("rfc460Timestamp")
            b = (fin.get("blueTeam") or {}).get("inhibitors") or 0
            rd = (fin.get("redTeam") or {}).get("inhibitors") or 0
            if (b > 0) != (rd > 0):
                out["blue_won"] = b > 0
            break
        if frames[-1].get("rfc460Timestamp") == last:
            break
        last = frames[-1].get("rfc460Timestamp")
    return out


def _series_winner_id(games: list, teams: list) -> str | None:
    """系列已打完时返回胜方 team id，否则 None。"""
    if any(g.get("state") not in ("completed", "unneeded") for g in games):
        return None
    w = {t["id"]: ((t.get("result") or {}).get("gameWins") or 0) for t in teams}
    if len(w) != 2:
        return None
    (a, x), (b, y) = w.items()
    return a if x > y else b if y > x else None


def _series_blue_won(gm: dict, games: list, teams: list, blue_id: str | None) -> bool | None:
    """仅凭系列比分能确定的逐局胜负：系列最后一局 = 系列胜方；零封时每局都是胜方。

    blue_id 必须是 **feed 的**蓝方队伍 id（_feed_game），不能取 getEventDetails 的 side。
    """
    win_id = _series_winner_id(games, teams)
    if win_id is None or blue_id not in {t["id"] for t in teams}:
        return None
    done = [g for g in games if g.get("state") == "completed"]
    loser_wins = next(((t.get("result") or {}).get("gameWins") or 0
                       for t in teams if t["id"] != win_id), 0)
    is_last = bool(done) and max(done, key=lambda g: g.get("number") or 0).get("id") == gm.get("id")
    if is_last or loser_wins == 0:
        return blue_id == win_id
    return None


def _load_ledger() -> tuple[dict, dict]:
    """返回 (预测 by game_id, 最新一条结果 by game_id)。结果可能被重对账追加多次，取最后一条。"""
    preds, results = {}, {}
    for r in _records():
        if r.get("kind") == "prediction":
            preds[r["game_id"]] = r
        elif r.get("kind") == "result":
            results[r["game_id"]] = r
    return preds, results


def reconcile() -> None:
    """拉赛程结果，与已记录的预测对账。"""
    import requests

    preds, results = _load_ledger()
    if not preds:
        print("尚无预测记录")
        return

    s = requests.Session()
    s.headers.update({"x-api-key": KEY, "User-Agent": "lolwp/1.0"})
    events: dict[str, dict] = {}
    n_new = n_pending = n_fill = 0
    for gid, p in preds.items():
        prev = results.get(gid)
        # 判出胜负、且带了 feed 的蓝红方与开局/终局时间（ended_at 这个键在）才算对完。
        # blue_won=None 的会重试；早期判出胜负但没带这些字段的只补字段，胜负不动。
        refill = prev is not None and prev.get("blue_won") is not None
        if refill and "ended_at" in prev:
            continue
        mid = p.get("match_id")
        if not mid:
            continue
        try:
            if mid not in events:
                d = s.get(f"{API}/getEventDetails", params={"hl": "en-US", "id": mid},
                          timeout=20).json()
                events[mid] = (d.get("data") or {}).get("event") or {}
            ev = events[mid]
            games = ((ev.get("match") or {}).get("games") or [])
            gm = next((x for x in games if str(x.get("id")) == gid), None)
            if not gm or gm.get("state") != "completed":
                n_pending += 1
                continue
            fg = _feed_game(s, gid)
            if fg is None:
                # feed 暂时拿不到：蓝红方无从和预测对齐，下次再试
                n_pending += 0 if refill else 1
                continue
            # ⚠️ getEventDetails 的 games[].teams[] **只有 id 和 side，没有逐局胜负**，side 也不可信。
            #    两路独立来源：livestats 终局帧（_feed_game）与系列比分（_series_blue_won），
            #    蓝红方一律用 feed 的 esportsTeamId；都有答案却打架 → 留空，不猜。
            teams = (ev.get("match") or {}).get("teams") or []
            ids = {t["id"] for t in teams}
            blue_id = fg["blue_id"] if fg["blue_id"] in ids else None
            red_id = fg["red_id"] if fg["red_id"] in ids else None
            code = {t["id"]: t.get("code") for t in teams}
            extra = {"blue_team": code.get(blue_id), "red_team": code.get(red_id),
                     "started_at": fg["started_at"], "ended_at": fg["ended_at"]}
            if refill:
                if fg["blue_won"] is not None and fg["blue_won"] != prev["blue_won"]:
                    print(f"  {gid} 终局帧与已记录的胜负矛盾，未补字段", file=sys.stderr)
                    continue
                _append({**prev, "recorded_at": _now(), **extra})
                n_fill += 1
                continue
            by_feed = fg["blue_won"]
            by_series = _series_blue_won(gm, games, teams, blue_id)
            if by_feed is not None and by_series is not None and by_feed != by_series:
                print(f"  {gid} 终局帧与系列比分矛盾，留空", file=sys.stderr)
                blue_won, source = None, "conflict"
            elif by_feed is not None:
                blue_won, source = by_feed, "feed+series" if by_series is not None else "feed"
            else:
                blue_won, source = by_series, "series" if by_series is not None else None
            if blue_won is None and prev is not None:
                n_pending += 1
                continue  # 仍判不出，不重复追加空记录
            win_id = None if blue_won is None else (blue_id if blue_won else red_id)
            winner = next((t.get("name") for t in teams if t["id"] == win_id), None)
            _append({"kind": "result", "game_id": gid, "recorded_at": _now(),
                     "game_winner": winner, "blue_won": blue_won, "source": source,
                     "p_blue": p.get("p_blue"), "teams": p.get("teams"), **extra})
            n_new += 1
            if blue_won is None:
                n_pending += 1
        except Exception as e:  # noqa: BLE001
            print(f"  对账 {gid} 失败：{type(e).__name__}: {e}", file=sys.stderr)
    print(f"新对账 {n_new} 条，补字段 {n_fill} 条，仍待定 {n_pending} 条")

    # 汇总：和网站「近期战绩」同一套逻辑（蓝红方、赛后补录、指标），两边数字不会对不上
    from lolwp.live.ledger import load_review, summarize

    df = load_review(LEDGER)
    rows = df[df["hit"].notna()].iloc[::-1]  # 按比赛时间从早到晚
    if rows.empty:
        print("尚无可对账的完赛记录")
        return
    print(f"\n=== 对账（{len(rows)} 局）===")
    print(f"{'对阵（蓝 vs 红）':<36}{'预测蓝胜':>9}{'实际':>7}  命中")
    for r in rows.itertuples():
        t = f"{r.blue} vs {r.red}" + ("（赛后补录）" if r.late is True else "")
        print(f"{t:<36}{r.p_blue:>9.1%}{'蓝胜' if r.blue_won else '红胜':>7}  {'✓' if r.hit else '✗'}")
    for label, part in (("", df), ("排除赛后补录：", df[~df["late"].eq(True)])):
        sm = summarize(part)
        if sm["n_done"]:
            hits = round(sm["hit_rate"] * sm["n_done"])
            print(f"\n  {label}命中 {hits}/{sm['n_done']} = {sm['hit_rate']:.1%}   "
                  f"log loss {sm['log_loss']:.4f}   Brier {sm['brier']:.4f}", end="")
    print()


def _record(g: dict, pr: dict, model: str | None = None, **extra) -> dict:
    """台账里的一条预测。实时捕获与赛后补录共用，两边字段必须一模一样。

    model = 算这条预测的 artifact（「BT4_LR@训练截止日」）。生产模型会重训，
    不记下来的话，台账里新旧模型的预测就混在一起分不开了。
    """
    return {"kind": "prediction", "game_id": str(g.get("game_id")), "model": model,
            "match_id": g.get("match_id"), "recorded_at": _now(),
            "league": g.get("league"), "teams": g.get("teams"),
            "p_blue": pr.get("p_blue"), "layers": pr.get("layers"),
            "blue_champ_keys": pr.get("blue_champ_keys"),
            "red_champ_keys": pr.get("red_champ_keys"),
            "blue_names": pr.get("blue_names"), "red_names": pr.get("red_names"),
            "warnings": g.get("warnings"),
            "latency_200_to_pred_s": g.get("latency_200_to_pred_s"), **extra}


def _model_id(pol) -> str | None:
    meta = getattr(getattr(pol, "artifact", None), "meta", None) or {}
    return f"{meta.get('model', '?')}@{str(meta.get('trained_on', '?'))[:10]}" if meta else None


def _schedule(s, since: datetime) -> list[dict]:
    """getSchedule（不带 leagueId = 全部赛区）从最新一页往回翻，翻到早于 since 为止。

    每页按开赛时间升序，第一条最早；`pages.older` 指向更早的一页。
    """
    out: list[dict] = []
    token = None
    for _ in range(SCHEDULE_MAX_PAGES):
        params = {"hl": "en-US", **({"pageToken": token} if token else {})}
        d = ((s.get(f"{API}/getSchedule", params=params, timeout=20).json().get("data") or {})
             .get("schedule") or {})
        ev = d.get("events") or []
        out += ev
        token = (d.get("pages") or {}).get("older")
        if not token or not ev or _utc(ev[0].get("startTime")) < since:
            break
    return out


def _utc(x: str | None) -> datetime:
    """lolesports 的时间戳（'...Z' 或带偏移）→ aware datetime；缺失视为极早。"""
    if not x:
        return datetime.min.replace(tzinfo=UTC)
    return datetime.fromisoformat(x)


# 本进程里已经确定补不出来的局（没有 livestats、或开局早于训练截止）与已经清完的 match。
# 机器合盖时每 15 分钟 DarkWake 一次、每次都会触发补录，不记下来就会反复去问同一批局。
_SKIP_GAMES: set[str] = set()
_SETTLED_MATCHES: set[str] = set()


def backfill(pol, s, since: datetime, *, league: str | None = None,
             seen: set[str] | None = None, pause_s: float = 0.5) -> int:
    """赛后补录：赛程里已打完、台账里没有的局，从 feed 读 BP 后阵容补出预测。返回补了几局。

    三条纪律：
    - **只补模型没见过的局**：开局早于 artifact 训练截止的局，模型训练时可能见过它，
      补出来的「预测」是样本内的，直接跳过。
    - **站在开局时刻预测**：now = feed 首帧时间，与实时捕获的衰减一致。
    - **不和实时盯盘抢**：只补 state=completed 的局；落盘前再读一次台账，
      另一个进程刚记下的局不重复写。

    赛程翻页失败往上抛（调用方下一圈重试）；单场 / 单局失败只跳过它，不拖垮整轮。
    """
    from lolwp.live.lolesports import FeedState

    seen = _seen_games() if seen is None else seen
    trained_on = _utc(str(pol.artifact.meta.get("trained_on")).replace(" ", "T"))
    n = 0
    for ev in _schedule(s, since):
        m = ev.get("match") or {}
        mid = str(m.get("id") or "")
        if (not mid or mid in _SETTLED_MATCHES
                or ev.get("state") not in ("completed", "inProgress")
                or _utc(ev.get("startTime")) < since
                or (league and (ev.get("league") or {}).get("name") != league)):
            continue
        try:
            d = s.get(f"{API}/getEventDetails", params={"hl": "en-US", "id": mid},
                      timeout=20).json()
        except Exception as e:  # noqa: BLE001
            print(f"  补录：getEventDetails({mid}) 失败：{type(e).__name__}: {e}", file=sys.stderr)
            continue
        det = (d.get("data") or {}).get("event") or {}
        name = (det.get("league") or {}).get("name") or (ev.get("league") or {}).get("name")
        codes = [t.get("code") for t in ((det.get("match") or m).get("teams") or [])
                 if t.get("code")]
        open_games = False
        for gm in (det.get("match") or {}).get("games") or []:
            gid = str(gm.get("id") or "")
            if not gid or gid in seen or gid in _SKIP_GAMES or gm.get("state") == "unneeded":
                continue
            if gm.get("state") != "completed":
                open_games = True      # 还没打完的局归实时盯盘
                continue
            try:
                res = pol.client.window(gid)
                time.sleep(pause_s)
                if res.state is not FeedState.READY or not (res.payload or {}).get("gameMetadata"):
                    if res.state is FeedState.NOT_STARTED:
                        _SKIP_GAMES.add(gid)   # 打完了还是 204：该局没有 livestats 覆盖
                    else:
                        open_games = True      # 403 / 5xx / 超时：下次再试
                    continue
                frames = res.payload.get("frames") or []
                started = frames[0].get("rfc460Timestamp") if frames else None
                if started is None or _utc(started) <= trained_on:
                    _SKIP_GAMES.add(gid)
                    continue
                pr, warns = pol.predict_window(res.payload, name, now=_utc(started))
                if pr.get("p_blue") is None:
                    _SKIP_GAMES.add(gid)
                    continue
                seen |= _seen_games()
                if gid in seen:
                    continue
                g = {"game_id": gid, "match_id": mid, "league": name, "teams": codes,
                     "warnings": warns, "latency_200_to_pred_s": None}
                _append(_record(g, pr, _model_id(pol), source="backfill", started_at=started))
                seen.add(gid)
                n += 1
                print(f"[{_now()}] 补录 {name} {' vs '.join(codes or ['?', '?'])}"
                      f"（开局 {started[:16].replace('T', ' ')} UTC）→ 蓝方 {pr['p_blue']:.1%}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                open_games = True
                print(f"  补录 {gid} 失败：{type(e).__name__}: {e}", file=sys.stderr)
        if ev.get("state") == "completed" and not open_games:
            _SETTLED_MATCHES.add(mid)
    return n


def _session():
    import requests

    s = requests.Session()
    s.headers.update({"x-api-key": KEY, "User-Agent": "lolwp/1.0"})
    return s


def _nap(seconds: float) -> float:
    """睡 seconds 秒墙钟时间，返回实际过去的墙钟秒数（远大于 seconds = 机器睡过一觉）。

    按 1 秒切片睡：Mac 合盖时 time.sleep 跟着冻结，醒来后会把剩下的半截睡完；
    而 DarkWake 常常只有 2 秒，等它睡完机器已经又睡着了。切片后醒来 1 秒内就接着干活。
    """
    t0 = time.time()
    end = t0 + seconds
    while (left := end - time.time()) > 0:
        time.sleep(min(1.0, left))
    return time.time() - t0


def watch(hours: float) -> int:
    from lolwp.live.poller import build_default_poller

    deadline = time.time() + hours * 3600
    try:
        pol = build_default_poller()
    except Exception as e:  # noqa: BLE001
        print(f"轮询器构建失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    seen = _seen_games()
    print(f"[{_now()}] 开始盯盘 {hours:.1f} 小时，已记录 {len(seen)} 局，模型 {_model_id(pol)}", flush=True)
    s = _session()
    # 启动先补一次：launchd 重启、或上一轮结束到这一轮开始之间打完的局。
    # 只在补录成功后清空：赛程接口一时拿不到就下一圈重来。
    backfill_from: datetime | None = datetime.now(UTC) - timedelta(days=2)
    last_log = 0.0
    while time.time() < deadline:
        try:
            pol.tick()
            snap = pol.snapshot()
            for g in snap.get("games", []):
                gid = str(g.get("game_id"))
                if g.get("status") != "ready" or gid in seen:
                    continue
                pr = g.get("prediction") or {}
                if pr.get("p_blue") is None:
                    continue
                seen |= _seen_games()   # 补录可能由另一个进程（make backfill）刚写过
                if gid in seen:
                    continue
                _append(_record(g, pr, _model_id(pol)))
                seen.add(gid)
                print(f"[{_now()}] 捕获 {g.get('league')} "
                      f"{' vs '.join(g.get('teams') or ['?','?'])} → 蓝方 {pr['p_blue']:.1%}",
                      flush=True)
            if backfill_from is not None:
                backfill(pol, s, backfill_from, seen=seen)
                backfill_from = None
            if time.time() - last_log > 1800:
                h = pol.health()
                print(f"[{_now()}] 心跳：监视 {h['n_watched']} 场，已捕获 {len(seen)} 局",
                      flush=True)
                last_log = time.time()
        except Exception as e:  # noqa: BLE001
            print(f"[{_now()}] tick 异常（继续）：{type(e).__name__}: {e}", flush=True)
        gap = _nap(30)
        if gap > WAKE_GAP_S:
            # 刚醒：下一圈先实时轮询（轮询器用墙钟，节流不会挡住），再补录睡着时打完的局。
            # 往前多看 6 小时：BO5 的赛程时间是系列开赛时间，可能远早于入睡。
            since = datetime.now(UTC) - timedelta(seconds=gap, hours=6)
            backfill_from = since if backfill_from is None else min(backfill_from, since)
    print(f"[{_now()}] 盯盘结束，共捕获 {len(seen)} 局", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=16.0)
    ap.add_argument("--reconcile", action="store_true", help="只对账，不盯盘")
    ap.add_argument("--backfill", action="store_true",
                    help="只补录：赛程里已打完、台账里没有的局（之后自动对账）")
    ap.add_argument("--days", type=float, default=14.0, help="补录往回看几天")
    ap.add_argument("--league", default=None, help="补录只看这个赛事（feed 名，如 Worlds）")
    a = ap.parse_args()
    if a.reconcile:
        reconcile()
        return 0
    if a.backfill:
        from lolwp.live.poller import build_default_poller

        since = datetime.now(UTC) - timedelta(days=a.days)
        n = backfill(build_default_poller(), _session(), since, league=a.league)
        print(f"补录 {n} 局" + ("，执行一次对账：" if n else ""), flush=True)
        if n:
            reconcile()
        return 0
    rc = watch(a.hours)
    print("\n盯盘结束，执行一次对账：", flush=True)
    reconcile()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
