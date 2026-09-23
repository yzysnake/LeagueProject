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


def reconcile() -> None:
    """拉赛程结果，与已记录的预测对账。"""
    import requests

    if not os.path.exists(LEDGER):
        print("尚无预测记录")
        return
    preds = {}
    done = set()
    with open(LEDGER) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "prediction":
                preds[r["game_id"]] = r
            elif r.get("kind") == "result":
                done.add(r["game_id"])
    if not preds:
        print("尚无预测记录")
        return

    s = requests.Session()
    s.headers.update({"x-api-key": KEY, "User-Agent": "lolwp/1.0"})
    n_new = 0
    for gid, p in preds.items():
        if gid in done:
            continue
        mid = p.get("match_id")
        if not mid:
            continue
        try:
            d = s.get(f"{API}/getEventDetails", params={"hl": "en-US", "id": mid},
                      timeout=20).json()
            ev = (d.get("data") or {}).get("event") or {}
            games = ((ev.get("match") or {}).get("games") or [])
            gm = next((x for x in games if str(x.get("id")) == gid), None)
            if not gm or gm.get("state") != "completed":
                continue
            # ⚠️ getEventDetails 的 games[].teams[] **只有 id 和 side，没有逐局胜负**。
            #    逐局结果得从系列比分反推：
            #    · Bo1：系列比分就是本局胜负，无歧义
            #    · 多局系列：单靠 gameWins 推不出「第 k 局谁赢」，标记为待定
            teams = (ev.get("match") or {}).get("teams") or []
            winner = next((t.get("name") for t in teams
                           if (t.get("result") or {}).get("outcome") == "win"), None)
            gteams = gm.get("teams") or []
            blue_id = next((t.get("id") for t in gteams if t.get("side") == "blue"), None)
            score = {t["id"]: ((t.get("result") or {}).get("gameWins") or 0) for t in teams}
            blue_won = None
            if len(games) == 1 and blue_id in score:
                other = [v for k, v in score.items() if k != blue_id]
                if other:
                    blue_won = score[blue_id] > other[0]
                    if winner is None:
                        winner = next((t.get("name") for t in teams
                                       if t["id"] == (blue_id if blue_won else
                                                      next(k for k in score if k != blue_id))),
                                      None)
            _append({"kind": "result", "game_id": gid, "recorded_at": _now(),
                     "match_winner": winner, "blue_won": blue_won,
                     "p_blue": p.get("p_blue"), "teams": p.get("teams")})
            n_new += 1
        except Exception as e:  # noqa: BLE001
            print(f"  对账 {gid} 失败：{type(e).__name__}: {e}", file=sys.stderr)
    print(f"新对账 {n_new} 条")

    # 汇总
    rows = []
    with open(LEDGER) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "result" and r.get("blue_won") is not None \
                        and r.get("p_blue") is not None:
                    rows.append(r)
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
