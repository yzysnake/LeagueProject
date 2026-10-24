"""站点的选择目录：英雄（按位置分组）、队伍、各队各位置的选手名单。

纯 pandas，不碰 Streamlit——站点（app/data.py 套缓存）和测试都直接调这里。

**这些只是展示和选择用的目录，不进任何模型特征。** 位置分组、队伍与名单取自
silver / bronze 的最近一年（含 holdout 期的比赛，和「历史回看」一样只供肉眼用）；
模型推理仍然只读冻结的 artifact，选手实力分位也只读 artifact 里的分层 BT 快照。

几个约定：
- 一个队一个位置可以有多名选手（替补、轮换、赛季中换人）。名单按「最近一次出场」
  排序，默认选该队最近一场的首发——和旧版「只取最近一场」的结果一致，但可以换人。
- 队伍的「所属赛区」取它最近一年在非国际赛里打得最多的赛区；只打过国际赛的队取最常见的。
- 位置池：最近一年 tier≥1 里在该位置出场 ≥3 次、且占该英雄全部出场 ≥5% 的英雄，
  按该位置出场次数降序。没进任何池的冷门英雄只出现在「全部」里。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

ROLES = ("top", "jng", "mid", "bot", "sup")
ROLE_CN = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}
# 国际赛：configs/league_canonical.yaml 的 intl_canon（tier 2）
INTL = ("ASI", "EWC", "FST", "IEM", "IWCI", "MSC", "MSI", "Riot", "WLDs")
LEAGUE_CN = {"WLDs": "全球总决赛", "MSI": "季中冠军赛", "EWC": "电竞世界杯", "FST": "First Stand",
             "ASI": "亚洲邀请赛", "IEM": "IEM", "IWCI": "国际外卡邀请赛", "MSC": "季中杯"}
WINDOW_DAYS = 365


def league_label(code: str) -> str:
    """「WLDs」→「全球总决赛」，常规赛区原样。"""
    return LEAGUE_CN.get(code, code)


# ── 英雄 ────────────────────────────────────────────────────────────────


def role_counts(games: pd.DataFrame, days: int = WINDOW_DAYS) -> dict[str, Counter]:
    """最近 days 天每个位置每个英雄的出场次数（双方都算）。"""
    if games.empty:
        return {r: Counter() for r in ROLES}
    recent = games[games["kickoff_utc"] >= games["kickoff_utc"].max() - pd.Timedelta(days=days)]
    out = {r: Counter() for r in ROLES}
    for side in ("blue", "red"):
        for keys in recent[f"{side}_champ_keys"]:
            if keys is None:
                continue
            for role, k in zip(ROLES, keys):
                if k is not None and not pd.isna(k):
                    out[role][int(k)] += 1
    return out


def build_champions(cmap: dict, zh: dict | None, games: pd.DataFrame, *,
                    min_games: int = 3, min_share: float = 0.05) -> dict[str, Any]:
    """英雄目录。zh 是 data/ddragon/champion_zh.json（可选，版本对不上就不用）。"""
    version = cmap["version"]
    zh_by = zh["champions"] if zh and zh.get("version") == version else {}
    counts = role_counts(games)
    champs = []
    for c in cmap["champions"]:
        k, z = int(c["dd_key"]), zh_by.get(c["dd_id"]) or {}
        n = {r: counts[r].get(k, 0) for r in ROLES}
        tot = sum(n.values())
        champs.append({
            "key": k, "id": c["dd_id"], "en": c["name"],
            "zh": z.get("name") or c["name"], "epithet": z.get("epithet") or "",
            "n": n, "roles": [r for r in ROLES if n[r] >= min_games and n[r] >= min_share * tot],
        })
    champs.sort(key=lambda c: c["en"])
    pool = {r: [c["key"] for c in sorted((c for c in champs if r in c["roles"]),
                                         key=lambda c: (-c["n"][r], c["en"]))]
            for r in ROLES}
    return {"version": version, "list": champs, "by_key": {c["key"]: c for c in champs},
            "pool": pool}


# ── 队伍 ────────────────────────────────────────────────────────────────


def _team_rows(games: pd.DataFrame) -> pd.DataFrame:
    cols = ["kickoff_utc", "league_canon"]
    b = games[cols + ["blue_team_id", "blue_team_name"]].set_axis(cols + ["team_id", "name"], axis=1)
    r = games[cols + ["red_team_id", "red_team_name"]].set_axis(cols + ["team_id", "name"], axis=1)
    return pd.concat([b, r], ignore_index=True).dropna(subset=["team_id"])


def _home_league(leagues: pd.Series) -> str:
    c = Counter(leagues)
    home = Counter({k: v for k, v in c.items() if k not in INTL})
    return (home or c).most_common(1)[0][0]


def _short_code(name: str) -> str:
    """没有官方代码时的兜底：取首字母（"Bilibili Gaming" → "BG"），单词就截前 4 个字符。"""
    words = [w for w in str(name).replace(".", " ").split() if w[:1].isalnum()]
    if len(words) >= 2:
        return "".join(w[0] for w in words[:4]).upper()
    return str(name)[:4].upper()


def build_teams(games: pd.DataFrame, codes: dict[str, dict] | None = None) -> pd.DataFrame:
    """一队一行：team_id / name / code / league / last_game / n_recent / active。

    games 传 tier≥1（含国际赛）。codes 是 oe_team_id → {"code", "esports_id"}，
    来自 lolesports 的队伍表 + team_crosswalk（可选）。active = 最近一年打过比赛。
    """
    cols = ["team_id", "name", "code", "esports_id", "league", "last_game", "n_recent", "active"]
    if games.empty:
        return pd.DataFrame(columns=cols)
    rows = _team_rows(games).sort_values("kickoff_utc")
    cut = rows["kickoff_utc"].max() - pd.Timedelta(days=WINDOW_DAYS)
    codes = codes or {}
    out = []
    for tid, g in rows.groupby("team_id", sort=False):
        last = g["kickoff_utc"].iloc[-1]
        recent = g[g["kickoff_utc"] >= last - pd.Timedelta(days=WINDOW_DAYS)]
        name = str(g["name"].iloc[-1])
        meta = codes.get(str(tid)) or {}
        out.append({"team_id": str(tid), "name": name, "code": meta.get("code") or _short_code(name),
                    "esports_id": meta.get("esports_id"), "league": _home_league(recent["league_canon"]),
                    "last_game": last, "n_recent": len(recent), "active": bool(last >= cut)})
    df = pd.DataFrame(out, columns=cols)
    return df.sort_values(["active", "last_game"], ascending=False, ignore_index=True)


def team_codes(crosswalk: pd.DataFrame, teams_json: list[dict]) -> dict[str, dict]:
    """oe_team_id → {"code", "esports_id"}。同一个 OE 队对上多个 lolesports 队时取场次最多的。"""
    by_esports = {str(t.get("id")): t for t in teams_json if t.get("id")}
    out: dict[str, dict] = {}
    for r in crosswalk.sort_values("n_games").itertuples(index=False):
        t = by_esports.get(str(r.esports_team_id))
        if t and t.get("code"):
            out[str(r.oe_teamid)] = {"code": t["code"], "esports_id": str(r.esports_team_id)}
    return out


# ── 选手名单 ──────────────────────────────────────────────────────────────


def build_rosters(players: pd.DataFrame, teams: pd.DataFrame) -> dict[str, dict]:
    """team_id → {"roles": {role: [选手…]}, "default": [5 个 player_id]}。

    players 是 bronze 的选手行（gameid / date / teamid / playerid / playername / position）。
    每队只看它自己最后一场之前一年的出场，所以退役队伍也能摆出当年的阵容。
    每名选手：{"id", "name", "games", "last"}，按最近出场、出场数排序。
    """
    p = players[players["position"].isin(ROLES) & players["playerid"].notna()].copy()
    p["date"] = pd.to_datetime(p["date"], utc=True, errors="coerce")
    last = teams.set_index("team_id")["last_game"]
    p = p[p["teamid"].isin(last.index)]
    p = p[p["date"] >= p["teamid"].map(last) - pd.Timedelta(days=WINDOW_DAYS)]
    p = p.sort_values("date")
    agg = (p.groupby(["teamid", "position", "playerid"], sort=False)
           .agg(name=("playername", "last"), games=("gameid", "nunique"), last=("date", "max"))
           .reset_index().sort_values(["last", "games"], ascending=False))
    out: dict[str, dict] = {}
    for tid, g in agg.groupby("teamid", sort=False):
        roles = {r: [] for r in ROLES}
        for r in g.itertuples(index=False):
            roles[r.position].append({"id": str(r.playerid), "name": str(r.name),
                                      "games": int(r.games), "last": str(r.last.date())})
        out[str(tid)] = {"roles": roles, "default": [None] * 5}
    # 默认首发：该队最后一场的五个人（缺位时取该位置最近出场的那位）
    for tid, g in p.groupby("teamid", sort=False):
        lastgame = g[g["gameid"] == g["gameid"].iloc[-1]]
        by_role = dict(zip(lastgame["position"], lastgame["playerid"].astype(str)))
        roles = out[str(tid)]["roles"]
        out[str(tid)]["default"] = [by_role.get(r) or (roles[r][0]["id"] if roles[r] else None)
                                    for r in ROLES]
    return out


def strength_percentiles(r_player: dict[str, float]) -> dict[str, int]:
    """分层 BT 选手 rating → 全体已知选手里的分位（0–100）。只用于名单里的「实力」提示。"""
    if not r_player:
        return {}
    s = pd.Series(r_player, dtype=float)
    return (s.rank(pct=True) * 100).round().astype(int).to_dict()
