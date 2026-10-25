"""站点的数据入口：artifact、英雄 / 队伍 / 名单目录、历史比赛，全部带 Streamlit 缓存。

**完全离线**：英雄图标与队标在 build 时落到 frontend/public/（`make icons`），
经 Streamlit 静态服务（.streamlit/config.toml 的 server.enableStaticServing，
`src/lolwp/app/static` 是指向 frontend/public 的软链）以 /app/static/... 提供，运行时零外部请求。
图标或队标缺失时页面用文字徽章兜底，不报错。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import pandas as pd
import streamlit as st

from lolwp.app import catalog
from lolwp.models.artifact import Artifact

ART = "models/prod.json"
CMAP = "data/ddragon/champion_map.json"
CZH = "data/ddragon/champion_zh.json"
SILVER = "data/silver/games.parquet"
BRONZE = "data/bronze/all.parquet"
TEAM_XW = "data/crosswalk/team_crosswalk.parquet"
TEAMS_JSON = "data/crosswalk/_cache/teams.json"
PUBLIC = "frontend/public"
STATIC_URL = "/app/static"
# 设了 LOLWP_API 就显示「经 API」（docker compose 的 web 容器）；推理本身在两条路径上
# 共用 app.predict 的同一套函数，不会 skew。
API_BASE = os.environ.get("LOLWP_API", "").rstrip("/")


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_resource(max_entries=1)
def _artifact_at(mtime: float) -> Artifact:
    return Artifact.load(ART)


def artifact() -> Artifact:
    """当前的模型。文件一换（盯盘每有新赛果就重解实力快照，`artifact.refresh`）就自动重新加载：
    按修改时间分缓存，不用重启站点。"""
    return _artifact_at(_mtime(ART))


# 每个已加载的 lolwp 模块对象第一次被看到时的文件修改时间。按模块对象记：
# 万一 Streamlit 自己重载了某个模块，那是个新对象，不算过时
_seen: dict[object, float] = {}


def stale() -> list[str]:
    """站点进程里已经过时的东西：加载之后磁盘上又改过的 lolwp 模块与比赛数据（silver）。

    Streamlit 每次运行都重读页面脚本（views/*.py），但 import 过的库模块留在进程里不动。
    改了代码不重启，就是「新页面 + 旧模块」：缺列报 KeyError（2026-10-24 实测撞上过）。
    页面顶上据此提示重启。模型文件不在此列：它按修改时间自动重载（见 artifact）。
    """
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not (name.startswith("lolwp.") and f):
            continue
        try:
            m = os.path.getmtime(f)
        except OSError:
            continue
        if _seen.setdefault((name, id(mod)), m) != m:
            out.append(name)
    for path, label in ((SILVER, "比赛数据"),):
        if path in _seen and os.path.exists(path) and os.path.getmtime(path) != _seen[path]:
            out.append(f"{label} {path}")
    return out


def bt_snapshot():
    """分层 BT 快照（选手实力分位用）。与 app.predict 用的是同一份 artifact，
    跟着它换（app.predict 按 artifact 对象缓存重建后的快照）。"""
    from lolwp.app.predict import _bt_snapshot

    return _bt_snapshot(artifact())


@st.cache_data(show_spinner=False)
def games() -> pd.DataFrame:
    """tier≥1（含国际赛）全部比赛，新的在前。"""
    _seen[SILVER] = os.path.getmtime(SILVER)
    g = pd.read_parquet(SILVER)
    return g[g.league_tier >= 1].sort_values("kickoff_utc", ascending=False)


@st.cache_resource(show_spinner=False)
def champions() -> dict[str, Any]:
    cm = _read_json(CMAP)
    ch = catalog.build_champions(cm, _read_json(CZH), games())
    v = ch["version"]
    for c in ch["list"]:
        local = f"{PUBLIC}/dd/{v}/{c['id']}.png"
        c["icon"] = f"{STATIC_URL}/dd/{v}/{c['id']}.png" if os.path.exists(local) else None
    return ch


@st.cache_data(show_spinner=False)
def teams() -> pd.DataFrame:
    xw = pd.read_parquet(TEAM_XW) if os.path.exists(TEAM_XW) else pd.DataFrame()
    tj = _read_json(TEAMS_JSON) or []
    codes = catalog.team_codes(xw, tj) if len(xw) else {}
    t = catalog.build_teams(games(), codes)
    t["logo"] = [_logo_url(e) for e in t["esports_id"]]
    return t


def _logo_url(esports_id: str | None) -> str | None:
    if not esports_id:
        return None
    for ext in ("png", "webp", "jpg", "svg"):
        if os.path.exists(f"{PUBLIC}/teams/{esports_id}.{ext}"):
            return f"{STATIC_URL}/teams/{esports_id}.{ext}"
    return None


def rosters() -> dict[str, dict]:
    """team_id → 各位置选手（含实力分位 / 模型是否见过）与默认首发。实力分位跟着模型文件换。"""
    return _rosters_at(_mtime(ART))


@st.cache_resource(show_spinner=False, max_entries=1)
def _rosters_at(art_mtime: float) -> dict[str, dict]:
    p = pd.read_parquet(BRONZE, columns=["gameid", "date", "teamid", "playerid",
                                         "playername", "position"])
    ro = catalog.build_rosters(p, teams())
    snap = bt_snapshot()
    pct = catalog.strength_percentiles(snap.r_player)
    for r in ro.values():
        for lst in r["roles"].values():
            for pl in lst:
                pl["pct"] = pct.get(pl["id"])
    return ro


def poller():
    """轮询器做成 cache_resource，才能跨 rerun 保持 watchlist 与计时状态；
    模型文件换过就换上新 artifact（新快照只用于之后开打的局）。

    构建失败（网络不通、crosswalk 缺失）时返回 {"error": ...}，实时页整体降级为提示，
    **手动摆阵容不受任何影响**——实时是增强功能不是依赖。
    """
    pol = _poller()
    reload = getattr(pol, "reload_artifact_if_changed", None)
    if reload is not None:
        try:
            reload()
        except Exception as e:  # noqa: BLE001  换不上就接着用旧的，实时页照常
            if note := getattr(pol, "_note_error", None):
                note(f"artifact reload: {type(e).__name__}: {e}")
    return pol


@st.cache_resource
def _poller():
    try:
        from lolwp.live.poller import build_default_poller

        return build_default_poller()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# app.predict 的通用告警；页面用 uncovered() 逐个点名英雄替换掉它
COVERAGE_WARN = "有英雄在当前版本的一线赛事中缺少出场记录"


def uncovered(blue_keys: list, red_keys: list) -> list[str]:
    """已选英雄里，模型在该位置没有一线赛事出场记录的（其版本强度按位置基准计）。"""
    from lolwp.app.catalog import ROLE_CN, ROLES

    champ_t1 = artifact().draft_state["champ_t1"]
    by_key = champions()["by_key"]
    out = []
    for keys in (blue_keys, red_keys):
        for role, k in zip(ROLES, keys or []):
            if k is None or int(k) not in by_key:
                continue
            if (champ_t1.get(f"{int(k)}|{role}") or [0, 0])[1] <= 0:
                out.append(f"{by_key[int(k)]['zh']}（{ROLE_CN[role]}）")
    return out


def team_row(team_id: str) -> dict[str, Any]:
    t = teams()
    m = t[t["team_id"] == team_id]
    if m.empty:
        return {"team_id": team_id, "name": team_id, "code": "?", "league": "?", "logo": None}
    return m.iloc[0].to_dict()


def api_alive() -> bool:
    """API 可达才显示「经 API」；不可达不影响任何页面。"""
    if not API_BASE:
        return False
    try:
        import requests

        return requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except Exception:  # noqa: BLE001
        return False
