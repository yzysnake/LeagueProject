#!/usr/bin/env python3
"""实时盯盘 + 赛后对账。

流程：轮询 getLive → 对每个 gameId 等 livestats 的 204→200 跳变 →
解析阵容 → 出 BP 后的正式预测 → 落盘 → 赛后拉结果对账。

为什么值得单独写：赛前我只能用占位英雄给「实力层估计」，
而且队伍不在历史数据里就只能拒绝预测。**开打后阵容出来，情况会变好**——
即使队伍名陌生，选手可能通过 crosswalk 解析出来，再由选手反推队伍。
所以「赛前不可预测」不等于「赛后不可预测」。

产出 reports/live_watch.jsonl（append-only），每条是一个 gameId 的完整记录。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime

LEDGER = "reports/live_watch.jsonl"
API = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _append(rec: dict) -> None:
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _seen_games() -> set[str]:
    if not os.path.exists(LEDGER):
        return set()
    out = set()
    with open(LEDGER) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "prediction":
                    out.add(r["game_id"])
    return out


def _ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _feed_blue_won(s, gid: str) -> bool | None:
    """从 livestats window 的终局帧判胜负。

    推掉水晶必须先推掉至少一座水晶塔（inhibitor），所以 gameState=finished 那一帧里
    **恰好一方 inhibitors>0** 时胜方无歧义。双方都>0 或都为 0（投降）→ 返回 None，不猜。
    window 每次只返回 ~10 秒的帧，从首帧起每 5 分钟往后跳，直到找到 finished 帧。
    """
    from datetime import timedelta

    r = s.get(f"{FEED}/window/{gid}", timeout=20)
    if r.status_code != 200 or not r.content:
        return None
    frames = r.json().get("frames") or []
    if not frames:
        return None
    t0 = datetime.fromisoformat(frames[0]["rfc460Timestamp"].replace("Z", "+00:00"))
    t0 = t0.replace(second=t0.second // 10 * 10, microsecond=0)  # startingTime 须是 10 秒整
    for mins in range(5, 125, 5):
        r = s.get(f"{FEED}/window/{gid}", params={"startingTime": _ts(t0 + timedelta(minutes=mins))},
                  timeout=20)
        frames = (r.json().get("frames") or []) if r.status_code == 200 and r.content else []
        if not frames:
            return None
        fin = next((f for f in frames if f.get("gameState") == "finished"), None)
        if fin:
            b = (fin.get("blueTeam") or {}).get("inhibitors") or 0
            rd = (fin.get("redTeam") or {}).get("inhibitors") or 0
            if (b > 0) != (rd > 0):
                return b > 0
            return None
    return None


def _series_winner_id(games: list, teams: list) -> str | None:
    """系列已打完时返回胜方 team id，否则 None。"""
    if any(g.get("state") not in ("completed", "unneeded") for g in games):
        return None
    w = {t["id"]: ((t.get("result") or {}).get("gameWins") or 0) for t in teams}
    if len(w) != 2:
        return None
    (a, x), (b, y) = w.items()
    return a if x > y else b if y > x else None


def _series_blue_won(gm: dict, games: list, teams: list) -> bool | None:
    """仅凭系列比分能确定的逐局胜负：系列最后一局 = 系列胜方；零封时每局都是胜方。"""
    win_id = _series_winner_id(games, teams)
    if win_id is None:
        return None
    blue_id = next((t.get("id") for t in gm.get("teams") or [] if t.get("side") == "blue"), None)
    if blue_id is None:
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
    if not os.path.exists(LEDGER):
        return preds, results
    with open(LEDGER) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
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
    # 只有判出胜负的才算对完；blue_won=None 的旧记录会重试（ledger 仍 append-only）
    done = {g for g, r in results.items() if r.get("blue_won") is not None}

    s = requests.Session()
    s.headers.update({"x-api-key": KEY, "User-Agent": "lolwp/1.0"})
    events: dict[str, dict] = {}
    n_new = n_pending = 0
    for gid, p in preds.items():
        if gid in done:
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
            # ⚠️ getEventDetails 的 games[].teams[] **只有 id 和 side，没有逐局胜负**。
            #    两路独立来源：livestats 终局帧（_feed_blue_won）与系列比分（_series_blue_won），
            #    都有答案却打架 → 留空，不猜。
            teams = (ev.get("match") or {}).get("teams") or []
            by_feed = _feed_blue_won(s, gid)
            by_series = _series_blue_won(gm, games, teams)
            if by_feed is not None and by_series is not None and by_feed != by_series:
                print(f"  {gid} 终局帧与系列比分矛盾，留空", file=sys.stderr)
                blue_won, source = None, "conflict"
            elif by_feed is not None:
                blue_won, source = by_feed, "feed+series" if by_series is not None else "feed"
            else:
                blue_won, source = by_series, "series" if by_series is not None else None
            if blue_won is None and gid in results:
                n_pending += 1
                continue  # 仍判不出，不重复追加空记录
            gteams = gm.get("teams") or []
            side_id = {t.get("side"): t.get("id") for t in gteams}
            win_id = None if blue_won is None else side_id.get("blue" if blue_won else "red")
            winner = next((t.get("name") for t in teams if t["id"] == win_id), None)
            _append({"kind": "result", "game_id": gid, "recorded_at": _now(),
                     "game_winner": winner, "blue_won": blue_won, "source": source,
                     "p_blue": p.get("p_blue"), "teams": p.get("teams")})
            n_new += 1
            if blue_won is None:
                n_pending += 1
        except Exception as e:  # noqa: BLE001
            print(f"  对账 {gid} 失败：{type(e).__name__}: {e}", file=sys.stderr)
    print(f"新对账 {n_new} 条，仍待定 {n_pending} 条")

    # 汇总
    _, results = _load_ledger()
    rows = [r for r in results.values()
            if r.get("blue_won") is not None and r.get("p_blue") is not None]
    if not rows:
        print("尚无可对账的完赛记录")
        return
    import numpy as np
    p = np.array([r["p_blue"] for r in rows])
    y = np.array([1 if r["blue_won"] else 0 for r in rows])
    hit = ((p >= 0.5).astype(int) == y)
    ll = float(-np.mean(y * np.log(np.clip(p, 1e-9, 1 - 1e-9))
                        + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9))))
    print(f"\n=== 对账（{len(rows)} 局）===")
    print(f"{'对阵':<42}{'预测蓝胜':>9}{'实际':>7}  命中")
    for r, h in zip(rows, hit):
        t = " vs ".join(r.get("teams") or ["?", "?"])
        print(f"{t:<42}{r['p_blue']:>9.1%}{'蓝胜' if r['blue_won'] else '红胜':>7}  {'✓' if h else '✗'}")
    print(f"\n  命中 {hit.sum()}/{len(rows)} = {hit.mean():.1%}   log loss {ll:.4f}   Brier {np.mean((p-y)**2):.4f}")


def watch(hours: float) -> int:
    from lolwp.live.poller import build_default_poller

    deadline = time.time() + hours * 3600
    try:
        pol = build_default_poller()
    except Exception as e:  # noqa: BLE001
        print(f"轮询器构建失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    seen = _seen_games()
    print(f"[{_now()}] 开始盯盘 {hours:.1f} 小时，已记录 {len(seen)} 局", flush=True)
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
                rec = {"kind": "prediction", "game_id": gid,
                       "match_id": g.get("match_id"), "recorded_at": _now(),
                       "league": g.get("league"), "teams": g.get("teams"),
                       "p_blue": pr.get("p_blue"), "layers": pr.get("layers"),
                       "blue_champ_keys": pr.get("blue_champ_keys"),
                       "red_champ_keys": pr.get("red_champ_keys"),
                       "blue_names": pr.get("blue_names"), "red_names": pr.get("red_names"),
                       "warnings": g.get("warnings"),
                       "latency_200_to_pred_s": g.get("latency_200_to_pred_s")}
                _append(rec)
                seen.add(gid)
                print(f"[{_now()}] 捕获 {g.get('league')} "
                      f"{' vs '.join(g.get('teams') or ['?','?'])} → 蓝方 {pr['p_blue']:.1%}",
                      flush=True)
            if time.time() - last_log > 1800:
                h = pol.health()
                print(f"[{_now()}] 心跳：监视 {h['n_watched']} 场，已捕获 {len(seen)} 局",
                      flush=True)
                last_log = time.time()
        except Exception as e:  # noqa: BLE001
            print(f"[{_now()}] tick 异常（继续）：{type(e).__name__}: {e}", flush=True)
        time.sleep(30)
    print(f"[{_now()}] 盯盘结束，共捕获 {len(seen)} 局", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=16.0)
    ap.add_argument("--reconcile", action="store_true", help="只对账，不盯盘")
    a = ap.parse_args()
    if a.reconcile:
        reconcile()
        return 0
    rc = watch(a.hours)
    print("\n盯盘结束，执行一次对账：", flush=True)
    reconcile()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
