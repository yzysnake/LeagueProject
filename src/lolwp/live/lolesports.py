"""lolesports 非官方 API 客户端。

**唯一权威的开赛触发器是 window 端点的 204 → 200 跳变，不是 schedule 的 state 字段。**
[实测 2026-09-19 10:02] LEC Movistar KOI vs Karmine Corp 的 game 1 在 getEventDetails
里 state=inProgress，而 feed.lolesports.com 的 window 返回 204。
schedule 的 inProgress 标的是「局间休息 + BP 开始」，不是「已开打」。
按 state 过滤写轮询逻辑，前端会先空转很久然后错过真正的开赛时刻。

这是非官方端点，无 SLA，随时可能变。所有调用都必须能优雅降级到手动模式。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

API = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
# 公开常量，不是密钥；Riot 把它硬编码在自家前端里
PUBLIC_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
UA = "lolwp/1.0 (qxing@uchicago.edu)"


class FeedState(str, Enum):
    """window 端点的四种结果，每一种都要有明确处理，不能只处理 200。"""
    READY = "ready"            # 200：有数据，可以预测
    NOT_STARTED = "not_started"  # 204：还没发布第一帧
    FORBIDDEN = "forbidden"    # 403：key 失效或被封 → 降级手动模式
    ERROR = "error"            # 4xx/5xx/超时 → 重试后降级


@dataclass
class FeedResult:
    state: FeedState
    payload: dict[str, Any] | None = None
    http: int | None = None
    detail: str = ""


class LolEsports:
    def __init__(self, api_key: str = PUBLIC_KEY, timeout: int = 20) -> None:
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.key = api_key
        self.timeout = timeout

    # ---------- persisted API（需要 x-api-key）----------
    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        r = self.s.get(f"{API}/{path}", params={"hl": "en-US", **params},
                       headers={"x-api-key": self.key}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def live(self) -> list[dict[str, Any]]:
        try:
            d = self._get("getLive")
        except requests.RequestException:
            return []
        return ((d.get("data") or {}).get("schedule") or {}).get("events") or []

    def event_details(self, match_id: str) -> dict[str, Any] | None:
        try:
            d = self._get("getEventDetails", id=match_id)
        except requests.RequestException:
            return None
        return (d.get("data") or {}).get("event")

    def teams(self) -> list[dict[str, Any]]:
        """全量队伍，给出干净的 code（如 MKOI / KC），用于剥离 summonerName 前缀。"""
        try:
            d = self._get("getTeams")
        except requests.RequestException:
            return []
        return ((d.get("data") or {}).get("teams")) or []

    # ---------- livestats feed（无需鉴权）----------
    def window(self, game_id: str) -> FeedResult:
        try:
            r = self.s.get(f"{FEED}/window/{game_id}", timeout=self.timeout)
        except requests.RequestException as e:
            return FeedResult(FeedState.ERROR, detail=str(e))
        if r.status_code == 204:
            return FeedResult(FeedState.NOT_STARTED, http=204)
        if r.status_code == 403:
            return FeedResult(FeedState.FORBIDDEN, http=403)
        if r.status_code != 200:
            return FeedResult(FeedState.ERROR, http=r.status_code, detail=r.text[:200])
        try:
            return FeedResult(FeedState.READY, payload=r.json(), http=200)
        except ValueError as e:
            return FeedResult(FeedState.ERROR, http=200, detail=f"非 JSON: {e}")

    def wait_for_start(
        self, game_id: str, *, interval: int = 30, max_204_minutes: int = 15
    ) -> FeedResult:
        """轮询到 204→200 跳变。连续 204 超过 max_204_minutes 判定为该赛事无 livestats
        覆盖，返回 NOT_STARTED 让上层切手动模式——否则前端会永远转圈（验收⑥）。"""
        deadline = time.time() + max_204_minutes * 60
        while True:
            res = self.window(game_id)
            if res.state is not FeedState.NOT_STARTED:
                return res
            if time.time() > deadline:
                res.detail = f"连续 204 超过 {max_204_minutes} 分钟，判定为无 livestats 覆盖"
                return res
            time.sleep(interval)


def parse_rosters(payload: dict[str, Any]) -> dict[str, Any]:
    """从 window 的 gameMetadata 抽出双方阵容。

    返回 champion 用的是 DDragon 的 **字符串 id**（如 LeeSin），
    不是数字 key —— feed 给的就是字符串。转换由调用方用 champion_map 完成。
    """
    gm = payload.get("gameMetadata") or {}
    out: dict[str, Any] = {"patch": gm.get("patchVersion"),
                           "esports_game_id": payload.get("esportsGameId")}
    for feed_side, side in (("blueTeamMetadata", "blue"), ("redTeamMetadata", "red")):
        t = gm.get(feed_side) or {}
        ps = t.get("participantMetadata") or []
        out[f"{side}_team_esports_id"] = t.get("esportsTeamId")
        out[f"{side}_player_ids"] = [str(p.get("esportsPlayerId")) for p in ps]
        out[f"{side}_champ_ids"] = [p.get("championId") for p in ps]
        out[f"{side}_roles"] = [p.get("role") for p in ps]
        out[f"{side}_names"] = [p.get("summonerName") for p in ps]
    return out
