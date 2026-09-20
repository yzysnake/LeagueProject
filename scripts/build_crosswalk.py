#!/usr/bin/env python
"""构建 OE playerid ↔ lolesports esportsPlayerId 对照表（PLAN.md §3.3.5 / §3.11.3）。

分四个阶段，每阶段落盘一个 jsonl，可断点续跑（这很重要：全量要打 3 万+ 请求，
中途挂掉重来一遍既慢又不礼貌）：

    schedule  : 遍历 getLeagues -> 每个联赛 getSchedule 向过去翻页   -> matches.jsonl
    games     : 对 completed match 调 getEventDetails 取 gameId      -> games.jsonl
    windows   : 对 completed game 调 livestats window 取 gameMetadata -> windows.jsonl
    join      : 与 OE CSV 硬 join，产出 crosswalk + 覆盖率报告

────────────────────────────────────────────────────────────────────────
join 键的选择（**偏离手册 §3.3.5 / §3.11.6，有实测依据**）
────────────────────────────────────────────────────────────────────────
手册说：有 Riot platformGameId 的赛区可以 `OE.gameid ↔ lolesports gameId` 确定性对照，
LPL 因为用腾讯格式必须走 `(date, team1, team2, game_number)` 模糊匹配。

实测发现手册这条路**两头都走不通、也没必要走**：

1. OE 的 `gameid` 对非 LPL 赛区确实是 Riot platformGameId（`LOLTMNT01_123456`），
   但 lolesports 的 gameId 是雪花号（`117155436343202197`），**API 的任何端点都不返回
   platformGameId**。要做这个对照必须引入第三方（Leaguepedia / gptilt），
   而手册自己把它们列为 L3 非硬依赖。
2. 根本不需要。**「同日 + 该局 10 个英雄的多重集」本身就是一个近乎无碰撞的主键**：
       2026: 8,851 局 -> 0 局碰撞
       2025: 10,041 局 -> 0 局碰撞
       2024: 10,195 局 -> 6 局碰撞（3 个键）
       2023: 11,106 局 -> 54 局碰撞（24 个键）
   碰撞的极少数用 (蓝方 5 英雄集合, 红方 5 英雄集合) 再切一刀基本就散了。
   这个键对 LPL 和非 LPL **完全一视同仁**，所以手册说的「LPL 专用模糊脚本」
   在本实现里不存在 —— LPL 走的是和别人一模一样的确定性路径。
3. 日期用 ±1 天窗口，因为 OE 的 date 是比赛开始的当地/UTC 时刻，
   lolesports 的 startTime 是整点排期，跨 UTC 零点时两边会差一天。

匹配上之后，(局, 英雄) -> 选手 的对应关系是**确定性**的：同一局里 10 个英雄互不重复，
所以「同场 + 同英雄」唯一确定一名选手。position/role 只用来做一致性校验，
不参与键 —— 因为 OE 的 position 是按 OE 自己的判定给的，feed 的 role 偶尔不一致
（摇摆位），拿它当键会白白丢样本。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lolwp.ingest.lolesports import (
    EsportsApiClient,
    LivestatsClient,
    LolesportsError,
    NoContentError,
    NotFoundError,
    parse_window_metadata,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(REPO, "data", "raw")
OUT_DIR = os.path.join(REPO, "data", "crosswalk")
CACHE_DIR = os.path.join(OUT_DIR, "_cache")


# ── jsonl 小工具 ────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


# ── 阶段 1：赛程 ────────────────────────────────────────────────────────
def stage_schedule(stop_before: str, rate: float) -> None:
    api = EsportsApiClient(rate_per_sec=rate)
    path = os.path.join(CACHE_DIR, "matches.jsonl")
    seen = {m["match_id"] for m in load_jsonl(path)}
    leagues = api.get_leagues()
    print(f"[schedule] {len(leagues)} 个联赛，回溯到 {stop_before}，已有 {len(seen)} 条")
    total_new = 0
    for i, lg in enumerate(leagues, 1):
        rows = []
        try:
            for ev in api.iter_schedule_backwards(lg["id"], stop_before=stop_before):
                if ev.get("type") != "match" or ev.get("state") != "completed":
                    continue
                if ev["startTime"] < stop_before:
                    continue
                m = ev["match"]
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                rows.append(
                    {
                        "match_id": m["id"],
                        "start_time": ev["startTime"],
                        "league_name": (ev.get("league") or {}).get("name"),
                        "league_slug": lg.get("slug"),
                        "league_id": lg["id"],
                        "block": ev.get("blockName"),
                        "best_of": (m.get("strategy") or {}).get("count"),
                        "teams": [
                            {"name": t.get("name"), "code": t.get("code")}
                            for t in m.get("teams", [])
                        ],
                    }
                )
        except LolesportsError as exc:
            print(f"  ! {lg['slug']}: {type(exc).__name__} {exc}")
        append_jsonl(path, rows)
        total_new += len(rows)
        print(f"  [{i:2d}/{len(leagues)}] {lg['slug']:<24} +{len(rows):4d}  (累计 {total_new})")
    print(f"[schedule] 新增 {total_new}，合计 {len(seen)}；HTTP {dict(api.counts)}")


# ── 阶段 2：match -> gameId ────────────────────────────────────────────
def stage_games(limit: int | None, rate: float, workers: int) -> None:
    api = EsportsApiClient(rate_per_sec=rate)
    matches = load_jsonl(os.path.join(CACHE_DIR, "matches.jsonl"))
    path = os.path.join(CACHE_DIR, "games.jsonl")
    done = {g["match_id"] for g in load_jsonl(path)}
    todo = [m for m in matches if m["match_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[games] 待办 {len(todo)} / 总 {len(matches)}（已有 {len(done)}）")

    def work(m: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            games = api.game_ids_of_match(m["match_id"])
        except LolesportsError as exc:
            return [{"match_id": m["match_id"], "error": type(exc).__name__}]
        return [
            {
                "match_id": m["match_id"],
                "game_id": g["id"],
                "number": g.get("number"),
                "state": g.get("state"),
                "league_name": m.get("league_name"),
                "start_time": m.get("start_time"),
            }
            for g in games
        ]

    buf: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(work, todo), 1):
            buf.extend(rows)
            if len(buf) >= 500:
                append_jsonl(path, buf)
                buf = []
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s  HTTP {dict(api.counts)}")
    append_jsonl(path, buf)
    print(f"[games] 完成，用时 {time.time()-t0:.0f}s，HTTP {dict(api.counts)}")


# ── 阶段 3：gameId -> window gameMetadata ──────────────────────────────
def stage_windows(limit: int | None, rate: float, workers: int) -> None:
    feed = LivestatsClient(rate_per_sec=rate)
    games = load_jsonl(os.path.join(CACHE_DIR, "games.jsonl"))
    path = os.path.join(CACHE_DIR, "windows.jsonl")
    done = {w["game_id"] for w in load_jsonl(path)}
    todo = [
        g
        for g in games
        if g.get("state") == "completed" and g.get("game_id") not in done
    ]
    if limit:
        todo = todo[:limit]
    print(f"[windows] 待办 {len(todo)}（completed 总 {sum(1 for g in games if g.get('state')=='completed')}，已有 {len(done)}）")

    def work(g: dict[str, Any]) -> dict[str, Any]:
        gid = g["game_id"]
        base = {
            "game_id": gid,
            "match_id": g.get("match_id"),
            "number": g.get("number"),
            "league_name": g.get("league_name"),
            "start_time": g.get("start_time"),
        }
        try:
            payload = feed.window(gid)
        except NoContentError:
            return {**base, "status": "no_content"}
        except NotFoundError:
            return {**base, "status": "not_found"}
        except LolesportsError as exc:
            return {**base, "status": f"error:{type(exc).__name__}"}
        try:
            ps = parse_window_metadata(payload)
        except Exception as exc:  # noqa: BLE001 - 结构异常五花八门，绝不能让一局炸掉整批
            return {**base, "status": f"bad_metadata:{type(exc).__name__}:{exc}"}
        return {
            **base,
            "status": "ok",
            "patch": ps[0].patch_version,
            "blue_team_id": next(p.esports_team_id for p in ps if p.side == "blue"),
            "red_team_id": next(p.esports_team_id for p in ps if p.side == "red"),
            "n_missing_player_id": sum(1 for p in ps if not p.esports_player_id),
            "participants": [
                {
                    "pid": p.participant_id,
                    "side": p.side,
                    "player_id": p.esports_player_id,
                    "name": p.summoner_name,
                    "champ": p.champion_id,
                    "role": p.role,
                    "pos": p.oe_position,
                    "team_id": p.esports_team_id,
                }
                for p in ps
            ],
        }

    buf: list[dict[str, Any]] = []
    t0 = time.time()
    stat: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(work, todo), 1):
            stat[row["status"].split(":")[0]] += 1
            buf.append(row)
            if len(buf) >= 500:
                append_jsonl(path, buf)
                buf = []
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s  {dict(stat)}")
    append_jsonl(path, buf)
    print(f"[windows] 完成 {time.time()-t0:.0f}s  状态 {dict(stat)}  HTTP {dict(feed.counts)}")


# ── 阶段 4：与 OE 硬 join ───────────────────────────────────────────────
def _load_oe(years: list[str]):
    import pandas as pd

    parts = []
    for y in years:
        f = os.path.join(RAW_DIR, f"{y}_LoL_esports_match_data_from_OraclesElixir.csv")
        d = pd.read_csv(
            f,
            usecols=[
                "gameid", "date", "league", "position", "champion", "side",
                "playerid", "playername", "teamid", "teamname", "game",
            ],
            low_memory=False,
        )
        parts.append(d[d.position != "team"])
    return pd.concat(parts, ignore_index=True)


def _champ_to_dd_id() -> dict[str, str]:
    """OE 展示名 -> DDragon id。§3.4 实测两边零失配，所以不做任何 fuzzy。"""
    p = os.path.join(REPO, "data", "ddragon", "champion_map.json")
    with open(p) as fh:
        m = json.load(fh)
    return {c["name"]: c["dd_id"] for c in m["champions"]}


def stage_join(years: list[str], holdout_start: str | None = None) -> None:
    """window gameMetadata × OE 长表 -> 选手级观测 -> crosswalk + 覆盖率报告。

    ``holdout_start``：仅用于在报告里把 holdout 段单列出来。crosswalk 是**纯描述性
    的 ID 对照**，不含任何标签信息，所以 holdout 段的数据可以用来建表（§0 纪律允许
    纯描述性统计）；但报告里仍然分开列，方便人工复核。
    """
    import pandas as pd

    oe = _load_oe(years)
    name2dd = _champ_to_dd_id()
    oe["dd_id"] = oe.champion.map(name2dd)
    miss = oe.dd_id.isna() & oe.champion.notna()
    if miss.any():
        print(f"  ! OE 英雄名未映射到 DDragon: {sorted(oe.loc[miss,'champion'].unique())[:10]}")
    oe["day"] = oe.date.str[:10]

    g = (
        oe.groupby("gameid")
        .agg(
            day=("day", "first"),
            league=("league", "first"),
            n=("dd_id", "size"),
            champs=("dd_id", lambda s: tuple(sorted(s.dropna()))),
        )
        .reset_index()
    )
    joinable = g[(g.n == 10) & (g.champs.map(len) == 10)]
    by_champs: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for r in joinable.itertuples(index=False):
        by_champs[r.champs].append((r.day, r.gameid))

    windows = load_jsonl(os.path.join(CACHE_DIR, "windows.jsonl"))
    wok = [w for w in windows if w.get("status") == "ok"]
    print(f"[join] OE 局 {len(g)}（可 join {len(joinable)}）；window ok {len(wok)} / {len(windows)}")

    from datetime import date

    def near(a: str, b: str, days: int = 1) -> bool:
        ya, ma, da = (int(x) for x in a.split("-"))
        yb, mb, db = (int(x) for x in b.split("-"))
        return abs((date(ya, ma, da) - date(yb, mb, db)).days) <= days

    matched: list[dict[str, Any]] = []
    game_link: list[dict[str, Any]] = []
    stat: Counter[str] = Counter()
    oe_by_game = {k: v for k, v in oe.groupby("gameid")}

    for w in wok:
        wchamps = tuple(sorted(p["champ"] for p in w["participants"]))
        day = w["start_time"][:10]
        cands = [(d, gid) for d, gid in by_champs.get(wchamps, []) if near(d, day)]
        if len(cands) != 1:
            stat["no_oe_game" if not cands else "ambiguous"] += 1
            game_link.append(
                {
                    "esports_game_id": w["game_id"],
                    "oe_gameid": None,
                    "league_feed": w.get("league_name"),
                    "day": day,
                    "status": "no_oe_game" if not cands else f"ambiguous:{len(cands)}",
                }
            )
            continue
        stat["matched_game"] += 1
        oe_gid = cands[0][1]
        game_link.append(
            {
                "esports_game_id": w["game_id"],
                "oe_gameid": oe_gid,
                "league_feed": w.get("league_name"),
                "day": cands[0][0],
                "status": "ok",
            }
        )
        rows = oe_by_game[oe_gid]
        champ2row = {r.dd_id: r for r in rows.itertuples(index=False)}
        for p in w["participants"]:
            r = champ2row.get(p["champ"])
            if r is None:
                stat["champ_no_oe_row"] += 1
                continue
            if not isinstance(r.playerid, str):
                stat["oe_playerid_null"] += 1
                continue
            if not p.get("player_id"):
                stat["feed_playerid_missing"] += 1
                continue
            matched.append(
                {
                    "oe_playerid": r.playerid,
                    "esports_player_id": p["player_id"],
                    "oe_playername": r.playername,
                    "feed_summoner_name": p["name"],
                    "oe_teamid": r.teamid if isinstance(r.teamid, str) else None,
                    "esports_team_id": p["team_id"],
                    "oe_gameid": oe_gid,
                    "esports_game_id": w["game_id"],
                    "league_oe": r.league,
                    "league_feed": w.get("league_name"),
                    "day": cands[0][0],
                    "oe_position": r.position,
                    "feed_role": p["role"],
                    "pos_agree": bool(r.position == p["pos"]),
                    "champ": p["champ"],
                    "side_agree": bool(str(r.side).lower() == p["side"]),
                }
            )
    print(f"[join] 局级 {dict(stat)}，选手级配对 {len(matched)}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    obs = pd.DataFrame(matched)
    obs.to_parquet(os.path.join(CACHE_DIR, "observations.parquet"), index=False)
    pd.DataFrame(game_link).to_parquet(
        os.path.join(CACHE_DIR, "game_link.parquet"), index=False
    )
    cw = build_crosswalk_table(obs)
    write_coverage_report(oe, g, obs, cw, windows, game_link, years, holdout_start)


def build_crosswalk_table(obs):
    """从选手级观测汇总出 (oe_playerid, esportsPlayerId) 对照表。

    **这张表必须是多对多的，不能强行压成一对一** —— 这是实测出来的，不是设计选择：

      · 一个 oe_playerid 对应多个 esportsPlayerId：294 / 3,413（8.6%）。
        其中 n_games≥3 的次要 pair 有 136 条，**80% 的时间区间与主 pair 完全不重叠**，
        例如 `Scenari0` 2026-03~06 在 G3V 用 116096153885590485，
        2026-07~09 在 GLE 换成 107582813961833026。
        即 **feed 会在选手转会时重新签发 esportsPlayerId**，旧号不再出现。
        压成一对一 = 转会后的选手在 live 链路上查不到历史，正是 §9.4 那条静默退化。
      · 一个 esportsPlayerId 对应多个 oe_playerid：177 / 3,547（5.0%）。
        绝大多数（190/226 条次要 pair）只有 1 场证据，来自 **feed 侧的单局元数据错乱**
        （实测样例：LPL 2024-03-01 `10788-10788_game_1`，该局 10 个 summonerName 全空，
        Kanavi 与 Beichuan 的 esportsPlayerId 互换）。这类必须靠证据量过滤掉。

    据此给每条 pair 打 confidence：

      primary : 该 oe_playerid 下证据最多的 pair
      alias   : 次要 pair 且 n_games >= MIN_ALIAS_GAMES（默认 3）—— 当作真·换号收下
      weak    : 次要 pair 且证据不足 —— 保留在表里供人工复核，**live 链路必须忽略**
    """
    import pandas as pd

    MIN_ALIAS_GAMES = 3
    if obs.empty:
        print("[crosswalk] 无观测，跳过")
        return pd.DataFrame()
    obs = obs.copy()
    obs["feed_name_blank"] = obs.feed_summoner_name.fillna("").str.strip() == ""
    pair = (
        obs.groupby(["oe_playerid", "esports_player_id"])
        .agg(
            n_games=("oe_gameid", "nunique"),
            oe_name=("oe_playername", "last"),
            feed_name=("feed_summoner_name", "max"),
            first_day=("day", "min"),
            last_day=("day", "max"),
            leagues=("league_oe", lambda s: ",".join(sorted(set(s.dropna()))[:4])),
            pos_agree_rate=("pos_agree", "mean"),
            side_agree_rate=("side_agree", "mean"),
            blank_name_rate=("feed_name_blank", "mean"),
        )
        .reset_index()
    )
    pair["rk"] = pair.groupby("oe_playerid").n_games.rank(
        method="first", ascending=False
    )
    pair["confidence"] = "weak"
    pair.loc[pair.rk == 1, "confidence"] = "primary"
    pair.loc[(pair.rk > 1) & (pair.n_games >= MIN_ALIAS_GAMES), "confidence"] = "alias"
    # 反向（live 链路真正用的方向）：同一 esportsPlayerId 下也只认证据最多的那条
    pair["rk_rev"] = pair.groupby("esports_player_id").n_games.rank(
        method="first", ascending=False
    )
    pair["live_usable"] = (pair.confidence != "weak") & (pair.rk_rev == 1)
    pair = pair.drop(columns=["rk", "rk_rev"]).sort_values(
        ["oe_playerid", "n_games"], ascending=[True, False]
    ).reset_index(drop=True)

    n_multi_oe = int((pair.groupby("oe_playerid").size() > 1).sum())
    n_multi_es = int((pair.groupby("esports_player_id").size() > 1).sum())
    usable = pair[pair.live_usable]
    print(
        f"[crosswalk] pair {len(pair)}（primary {int((pair.confidence=='primary').sum())} / "
        f"alias {int((pair.confidence=='alias').sum())} / weak {int((pair.confidence=='weak').sum())}）；"
        f"oe_playerid {pair.oe_playerid.nunique()}（{n_multi_oe} 个有多候选）；"
        f"esportsPlayerId {pair.esports_player_id.nunique()}（{n_multi_es} 个有多候选）"
    )
    print(
        f"[crosswalk] live 可用（esportsPlayerId -> oe_playerid 唯一）"
        f"{len(usable)} 条，覆盖 {usable.esports_player_id.nunique()} 个 esportsPlayerId"
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    pair.to_parquet(os.path.join(OUT_DIR, "player_crosswalk.parquet"), index=False)
    pair.to_json(
        os.path.join(OUT_DIR, "player_crosswalk.json"),
        orient="records",
        force_ascii=False,
        indent=1,
    )
    print(f"[crosswalk] 写出 {len(pair)} 条 -> data/crosswalk/player_crosswalk.parquet")
    _write_oe_alias_candidates(pair, MIN_ALIAS_GAMES)
    # 覆盖率口径统一用 primary+alias（weak 不算覆盖）
    return pair[pair.confidence != "weak"]


def _write_oe_alias_candidates(pair, min_games: int) -> None:
    """**副产物，但可能是本次最重要的发现**：`oe:player:<hash>` 不是稳定的人的 ID。

    同一个 esportsPlayerId 在 OE 侧对应两个 playerid，且两边都有足够证据时，
    说明 OE 把同一个人拆成了两个实体。[实测] 15 组，例如：

        esportsPlayerId 105501797931408936 = "Taeyoon"
          oe:player:952237fce714ed1eca8ded17523bee9  2024-01-19~2025-12-11  220 行
          oe:player:4047b7fecd736a6924aa13b05f42897  2026-01-15~2026-09-03  108 行
        两段都打过 Nongshim RedForce，是同一个人。

    后果：player rating 状态机若以 oe_playerid 为键，这个人在 2026 年会从先验重新起步，
    **不报错、指标只轻微下滑** —— 正是 §9.4 描述的那种静默退化，只是来源方向
    手册没有预料到（手册假设 OE 的 hash 是稳定主键）。

    这里只产出**候选**清单，不自动合并：feed 侧也可能复用 ID，需人工过一遍
    再落 configs/player_overrides.yaml。
    """

    strong = pair[pair.n_games >= min_games]
    grp = strong.groupby("esports_player_id").filter(lambda d: len(d) > 1)
    if grp.empty:
        return
    out = []
    for eid, d in grp.groupby("esports_player_id"):
        d = d.sort_values("first_day")
        out.append(
            {
                "esports_player_id": eid,
                "feed_names": sorted(set(d.feed_name.dropna())),
                "oe_playerids": [
                    {
                        "oe_playerid": r.oe_playerid,
                        "oe_name": r.oe_name,
                        "n_games": int(r.n_games),
                        "first_day": r.first_day,
                        "last_day": r.last_day,
                    }
                    for r in d.itertuples(index=False)
                ],
            }
        )
    path = os.path.join(OUT_DIR, "oe_player_alias_candidates.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[crosswalk] OE playerid 疑似重复实体 {len(out)} 组 -> {os.path.basename(path)}")


LPL_LIKE = {"LPL"}


def write_coverage_report(
    oe, oe_games, obs, cw, windows, game_link, years, holdout_start
) -> None:
    """三个层次的覆盖率，全部分赛区。§3.11.4：只做 gauge，不做 assert。"""
    import pandas as pd

    os.makedirs(OUT_DIR, exist_ok=True)
    oe = oe.copy()
    oe["year"] = oe.date.str[:4]
    gl = pd.DataFrame(game_link)
    wdf = pd.DataFrame(windows)

    # ① livestats 覆盖率：分母 = lolesports 标 completed 的 game
    liv = (
        wdf.assign(ok=lambda d: d.status == "ok", year=lambda d: d.start_time.str[:4])
        .groupby("league_name")
        .agg(n=("ok", "size"), ok=("ok", "sum"))
    )
    liv["rate"] = (liv.ok / liv.n).round(4)

    # ② OE 局级对齐率：分母 = OE 该赛区该年的全部 game
    linked_oe = set(gl[gl.status == "ok"].oe_gameid.dropna())
    og = oe_games.copy()
    og["year"] = og.day.str[:4]
    og["linked"] = og.gameid.isin(linked_oe)
    by_lg = og.groupby("league").agg(n_games=("gameid", "size"), linked=("linked", "sum"))
    by_lg["rate"] = (by_lg.linked / by_lg.n_games).round(4)
    by_lg = by_lg.sort_values("n_games", ascending=False)

    # ③ 选手级覆盖率：分母 = OE 的选手行（每场 10 行）
    cw_ids = set(cw.oe_playerid) if len(cw) else set()
    oe["covered"] = oe.playerid.isin(cw_ids)
    pl_by_lg = oe.groupby("league").agg(n_rows=("covered", "size"), n_cov=("covered", "sum"))
    pl_by_lg["rate"] = (pl_by_lg["n_cov"] / pl_by_lg["n_rows"]).round(4)
    pl_by_year = oe.groupby("year").agg(n_rows=("covered", "size"), n_cov=("covered", "sum"))
    pl_by_year["rate"] = (pl_by_year["n_cov"] / pl_by_year["n_rows"]).round(4)

    tier1 = _tier1_codes()
    def agg(mask_g, mask_p):
        a = og[mask_g]
        b = oe[mask_p]
        return (
            len(a), int(a.linked.sum()), round(a.linked.mean(), 4) if len(a) else None,
            len(b), int(b.covered.sum()), round(b.covered.mean(), 4) if len(b) else None,
        )

    scopes = {
        "全部 OE 赛区": (og.gameid.notna(), oe.gameid.notna()),
        "tier-1 + 国际赛": (og.league.isin(tier1), oe.league.isin(tier1)),
        "LPL": (og.league.isin(LPL_LIKE), oe.league.isin(LPL_LIKE)),
        "非 LPL（全部）": (~og.league.isin(LPL_LIKE), ~oe.league.isin(LPL_LIKE)),
        "非 LPL（tier-1+国际）": (
            og.league.isin(tier1 - LPL_LIKE),
            oe.league.isin(tier1 - LPL_LIKE),
        ),
    }
    lines = [
        "# player_crosswalk 覆盖率报告",
        "",
        f"- 生成时间（UTC 日期）：{time.strftime('%Y-%m-%d', time.gmtime())}",
        f"- OE 年份范围：{','.join(years)}",
        f"- lolesports completed game 数：{len(wdf)}；window 返回 200 的：{int((wdf.status=='ok').sum())}",
        f"- OE 局数：{len(og)}；其中与 lolesports 局对上的：{int(og.linked.sum())}",
        f"- crosswalk 条目：{len(cw)}（唯一 oe_playerid）",
        "",
        "## 汇总（口径 = 2024-2026）",
        "",
        "| 口径 | OE 局数 | 局已对齐 | 局对齐率 | OE 选手行 | 行已覆盖 | 选手行覆盖率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, (mg, mp) in scopes.items():
        ng, lg_, rg, nr, cr, rr = agg(mg, mp)
        lines.append(f"| {name} | {ng} | {lg_} | {rg} | {nr} | {cr} | {rr} |")
    lines += [
        "",
        "## ① livestats 覆盖率（分母 = lolesports 标 completed 的 game）",
        "",
        "| 赛区(feed) | completed | window 200 | 覆盖率 |",
        "|---|---:|---:|---:|",
    ]
    for lgname, r in liv.sort_values("n", ascending=False).iterrows():
        lines.append(f"| {lgname} | {int(r.n)} | {int(r.ok)} | {r.rate} |")
    lines += [
        "",
        "## ② 分赛区：OE 局对齐率 + 选手行覆盖率（分母 = OE）",
        "",
        "| 赛区(OE) | OE 局数 | 已对齐 | 局对齐率 | 选手行覆盖率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for lgname, r in by_lg.iterrows():
        pr = pl_by_lg["rate"].get(lgname)
        lines.append(f"| {lgname} | {int(r.n_games)} | {int(r.linked)} | {r.rate} | {pr} |")
    lines += ["", "## ③ 按年", "", "| 年 | OE 选手行 | 已覆盖 | 覆盖率 |", "|---|---:|---:|---:|"]
    for y, r in pl_by_year.iterrows():
        lines.append(f"| {y} | {int(r.n_rows)} | {int(r.n_cov)} | {r.rate} |")
    if len(obs):
        lines += [
            "",
            "## ④ 一致性自检（不参与 join，只用于验证键是否正确）",
            "",
            f"- position 一致率（OE position vs feed role 经 ROLE_MAP）：{obs.pos_agree.mean():.4f}（n={len(obs)}）",
            f"- side 一致率（OE side vs feed blue/red）：{obs.side_agree.mean():.4f}",
        ]
    lines += _temporal_holdout_section(obs, holdout_start or "2026-06-01")
    path = os.path.join(OUT_DIR, "coverage_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    by_lg.join(pl_by_lg[["n_rows", "n_cov", "rate"]], rsuffix="_player").to_csv(
        os.path.join(OUT_DIR, "coverage_by_league.csv")
    )
    print(f"[report] -> {path}")
    for name, (mg, mp) in scopes.items():
        ng, lg_, rg, nr, cr, rr = agg(mg, mp)
        print(f"  {name:<24} 局对齐 {rg}  选手行覆盖 {rr}")


def _temporal_holdout_section(obs, cut: str) -> list[str]:
    """时序泛化检验：把 crosswalk 冻结在 ``cut``，看之后的真实比赛能查到多少。

    这是 live 链路唯一真正重要的数字 —— 全量自 join 出来的覆盖率是"考完再出卷"，
    必然虚高；真实场景是"表是上周建的，今天来了一场新比赛"。
    只用 ID 对照、不碰任何标签，属 §0 允许的纯描述性统计。
    """

    if obs.empty:
        return []
    tr, te = obs[obs.day < cut], obs[obs.day >= cut]
    if tr.empty or te.empty:
        return []
    p = tr.groupby(["oe_playerid", "esports_player_id"]).oe_gameid.nunique().reset_index(
        name="n"
    )
    p["rk"] = p.groupby("oe_playerid").n.rank(ascending=False, method="first")
    keep = (p.rk == 1) | (p.n >= 3)
    p = p[keep]
    p["rkr"] = p.groupby("esports_player_id").n.rank(ascending=False, method="first")
    lut = dict(zip(p[p.rkr == 1].esports_player_id, p[p.rkr == 1].oe_playerid))
    te = te.copy()
    te["hit"] = te.esports_player_id.map(lut).notna()
    te["right"] = te.esports_player_id.map(lut) == te.oe_playerid
    t1 = _tier1_codes()
    rows = [
        ("全部赛区", te),
        ("tier-1 + 国际赛", te[te.league_oe.isin(t1)]),
        ("LPL", te[te.league_oe == "LPL"]),
        ("非 LPL（tier-1+国际）", te[te.league_oe.isin(t1 - {"LPL"})]),
    ]
    out = [
        "",
        f"## ⑤ 时序泛化检验（crosswalk 冻结在 {cut}，在其后的真实比赛上评测）",
        "",
        f"- 训练段观测 {len(tr)}；冻结表 {len(lut)} 个 esportsPlayerId",
        f"- 评测段 {te.oe_gameid.nunique()} 局 / {len(te)} 个选手坑位（{te.day.min()} ~ {te.day.max()}）",
        "",
        "| 口径 | 坑位数 | 命中率 | 命中且正确 | 命中内正确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, sub in rows:
        if not len(sub):
            continue
        acc = sub[sub.hit].right.mean() if sub.hit.any() else float("nan")
        out.append(
            f"| {name} | {len(sub)} | {sub.hit.mean():.4f} | {sub.right.mean():.4f} | {acc:.4f} |"
        )
    return out


def _tier1_codes() -> set[str]:
    """从 eval/tier1_leagues.yaml 读 tier-1 + 国际赛 code，别在代码里再抄一份。

    yaml 里写的是 ``WLDS``，而 OE CSV 里的实际写法是 ``WLDs``（[实测] 2024-2026
    共 236 场）。大小写不一致会让这个赛区整体从分母里消失、覆盖率虚高，
    所以这里按大小写不敏感地把 OE 侧的写法补齐。
    """
    import yaml

    with open(os.path.join(REPO, "eval", "tier1_leagues.yaml")) as fh:
        y = yaml.safe_load(fh)
    out: set[str] = set()
    for k in ("tier1_2026", "tier1_2025_aliases", "international"):
        out |= {str(v).strip() for v in (y.get(k) or [])}
    out.add("WLDs")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["schedule", "games", "windows", "join", "all"])
    ap.add_argument("--stop-before", default="2024-01-01")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--years", default="2024,2025,2026")
    ap.add_argument("--holdout-start", default=None)
    a = ap.parse_args()
    years = a.years.split(",")
    if a.stage in ("schedule", "all"):
        stage_schedule(a.stop_before, a.rate)
    if a.stage in ("games", "all"):
        stage_games(a.limit, a.rate, a.workers)
    if a.stage in ("windows", "all"):
        stage_windows(a.limit, a.rate, a.workers)
    if a.stage in ("join", "all"):
        stage_join(years, a.holdout_start)


if __name__ == "__main__":
    main()
