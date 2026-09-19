"""lolesports 官方 API 接入层（PLAN.md §3.3）。

两个物理上完全独立的服务，认证与错误语义都不一样，所以分成两个 client：

1. ``EsportsApiClient``  -> ``esports-api.lolesports.com/persisted/gw``
   需要 ``x-api-key``（社区长期公开的那把 key），不带直接 403。
   用途：枚举联赛 / 赛程 / 赛事详情 / 战队名册。

2. ``LivestatsClient``   -> ``feed.lolesports.com/livestats/v1``
   **无需任何鉴权**。用途：``window`` 端点拿 ``gameMetadata``。
   §3.3.5 的核心技巧就在这里：对 completed 的 gameId **不带 startingTime**
   调 window，会返回开局首帧 + 完整 gameMetadata，里面有
   ``(esportsPlayerId, championId, role, esportsTeamId)`` 四元组。
   这是建 OE playerid ↔ esportsPlayerId 对照表的唯一自动化途径（§3.11.3 路径 1）。

错误分类（§8 阶段 6 验收⑤要求四种状态各有单测）：

    204  -> NoContentError    比赛未开始 / 该局无 livestats 覆盖。**不是错误，是信息**，
                              批量回填时必须计入「无覆盖」而不是「失败」。
    403  -> ForbiddenError    api key 失效。**不重试**，重试只会更快被封。
    404  -> NotFoundError     gameId 不存在。**不重试**。
                              [实测 2026-09-18] window 对不存在的 gameId 返回 404 而非 204，
                              手册 §3.3.3 只写了 204，这里补上区分。
    429  -> RateLimitedError  重试，尊重 Retry-After。
    5xx  -> ServerError       重试（指数退避 + 抖动）。
    其余 4xx -> ClientError   不重试。

限流策略：[实测 2026-09-18] 单机连续 40 次 + 30/60 并发 burst 打两个域名，
**全部 200，没有观察到任何 429**。但没观察到不等于没有，所以仍然内置一个
令牌桶默认 8 req/s —— 这个数在实测上限之下，且足以在 10 分钟内跑完全量回填。
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

# ── 常量 ───────────────────────────────────────────────────────────────
ESPORTS_API_BASE = "https://esports-api.lolesports.com/persisted/gw"
# [实测 2026-09-18] 镜像返回与主域完全一致的 49,466 字节，可作故障切换用
ESPORTS_API_MIRROR = "https://prod-relapi.ewp.gg/persisted/gw"
LIVESTATS_BASE = "https://feed.lolesports.com/livestats/v1"
# 社区公开 key，写死在 lolesports.com 前端 bundle 里；[实测 2026-09-18] 仍然有效
PUBLIC_API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
UA = "lolwp/1.0 (qxing@uchicago.edu)"

# feed 的 role -> OE 的 position（§3.11.2 表 2，常量 5 条）
ROLE_MAP: dict[str, str] = {
    "top": "top",
    "jungle": "jng",
    "mid": "mid",
    "bottom": "bot",
    "support": "sup",
}


# ── 异常层次 ────────────────────────────────────────────────────────────
class LolesportsError(Exception):
    """本模块所有异常的基类。"""


class NoContentError(LolesportsError):
    """HTTP 204：比赛未开始或该局没有 livestats 覆盖。不是故障。"""


class NotFoundError(LolesportsError):
    """HTTP 404：资源不存在。不重试。"""


class ForbiddenError(LolesportsError):
    """HTTP 403：api key 失效或被封。不重试，直接让调用方告警。"""


class RateLimitedError(LolesportsError):
    """HTTP 429。"""


class ServerError(LolesportsError):
    """HTTP 5xx。"""


class ClientError(LolesportsError):
    """其它 4xx。不重试。"""


# ── 限速器 ──────────────────────────────────────────────────────────────
class _TokenBucket:
    """线程安全的令牌桶。批量回填会开线程池，所以必须加锁。"""

    def __init__(self, rate_per_sec: float, burst: int | None = None) -> None:
        self.rate = float(rate_per_sec)
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_sec))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(wait)


# ── 基类 ────────────────────────────────────────────────────────────────
@dataclass
class _Attempt:
    status: int
    body: bytes


class _BaseClient:
    """带令牌桶 + 分类重试的 HTTP 外壳。"""

    def __init__(
        self,
        *,
        rate_per_sec: float = 8.0,
        max_retries: int = 4,
        timeout: float = 20.0,
        session: requests.Session | None = None,
    ) -> None:
        self.bucket = _TokenBucket(rate_per_sec)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": UA})
        # 计数器：供 /health 的 gauge 和批量任务的摘要用
        self.counts: dict[str, int] = {}

    def _bump(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def _raise_for_status(self, status: int, body: bytes) -> None:
        if status == 204:
            raise NoContentError("204 no content")
        if status == 403:
            raise ForbiddenError(f"403 forbidden: {body[:200]!r}")
        if status == 404:
            raise NotFoundError("404 not found")
        if status == 429:
            raise RateLimitedError("429 rate limited")
        if 500 <= status < 600:
            raise ServerError(f"{status} server error")
        if 400 <= status < 500:
            raise ClientError(f"{status}: {body[:200]!r}")

    def request_raw(self, url: str, params: dict[str, Any] | None = None) -> bytes:
        """返回响应体字节。204 抛 NoContentError（空体也是有意义的信号）。"""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.bucket.acquire()
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # 连接层错误，可重试
                last_exc = exc
                self._bump("net_error")
                self._sleep_backoff(attempt)
                continue
            self._bump(f"http_{r.status_code}")
            if r.status_code == 200:
                return r.content
            try:
                self._raise_for_status(r.status_code, r.content)
            except (RateLimitedError, ServerError) as exc:
                last_exc = exc
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(60.0, float(retry_after)))
                else:
                    self._sleep_backoff(attempt)
                continue
            # NoContent / NotFound / Forbidden / ClientError 由 _raise_for_status
            # 直接抛出并穿透本函数（不重试）。走到这里说明状态码既不是 200，
            # 也没被分类 —— 例如 2xx/3xx 里的怪值。显式炸掉，不要静默当成功。
            raise LolesportsError(
                f"未分类的 HTTP 状态 {r.status_code}: {r.content[:200]!r}"
            )
        assert last_exc is not None
        raise last_exc

    def _sleep_backoff(self, attempt: int) -> None:
        # 指数退避 + 抖动；抖动是必须的，否则线程池会同步重试形成尖峰
        time.sleep(min(30.0, (2.0**attempt) * 0.5) * (0.5 + random.random()))

    def request_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        import json

        return json.loads(self.request_raw(url, params))


# ── esports-api ─────────────────────────────────────────────────────────
class EsportsApiClient(_BaseClient):
    """``esports-api.lolesports.com/persisted/gw`` 封装。"""

    def __init__(
        self,
        api_key: str = PUBLIC_API_KEY,
        base: str = ESPORTS_API_BASE,
        *,
        hl: str = "en-US",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.base = base.rstrip("/")
        self.hl = hl
        self.session.headers.update({"x-api-key": api_key})

    def _get(self, path: str, **params: Any) -> Any:
        params.setdefault("hl", self.hl)
        params = {k: v for k, v in params.items() if v is not None}
        return self.request_json(f"{self.base}/{path}", params)

    # --- 端点 ---------------------------------------------------------
    def get_leagues(self) -> list[dict[str, Any]]:
        return self._get("getLeagues")["data"]["leagues"]

    def get_live(self) -> list[dict[str, Any]]:
        return self._get("getLive")["data"]["schedule"].get("events", [])

    def get_teams(self, team_slug: str | None = None) -> list[dict[str, Any]]:
        """不带 id 一次返回全量（[实测 2026-09-18] 1,548,877 字节）。"""
        return self._get("getTeams", id=team_slug)["data"]["teams"]

    def get_schedule(
        self, league_id: str | None = None, page_token: str | None = None
    ) -> dict[str, Any]:
        """返回 ``{'pages': {'older','newer'}, 'events': [...]}``，每页 80 条。"""
        return self._get("getSchedule", leagueId=league_id, pageToken=page_token)[
            "data"
        ]["schedule"]

    def iter_schedule_backwards(
        self,
        league_id: str | None = None,
        *,
        stop_before: str | None = None,
        max_pages: int = 200,
    ) -> Iterator[dict[str, Any]]:
        """沿 ``pages.older`` 向过去翻页，逐个 yield event。

        ``stop_before``：ISO 日期串（如 ``'2024-01-01'``）。一旦某页最早的
        ``startTime`` 早于它就停。**不是精确截断**——同一页里更晚的事件仍会 yield，
        由调用方自己过滤。这样写是因为分页 token 不支持按时间跳转。
        """
        token: str | None = None
        for _ in range(max_pages):
            sch = self.get_schedule(league_id, token)
            events = sch.get("events") or []
            if not events:
                return
            yield from events
            oldest = min(e["startTime"] for e in events)
            if stop_before is not None and oldest < stop_before:
                return
            token = (sch.get("pages") or {}).get("older")
            if not token:
                return

    def get_completed_events(
        self, league_id: str | None = None, tournament_id: str | None = None
    ) -> list[dict[str, Any]]:
        """最近约 30 天的已完赛事件，**内联 games[] 的 id**，省掉 getEventDetails。

        [实测 2026-09-18] 不带参数返回 300 个 event / 1,222 个 game，
        时间跨度 2026-08-17 → 2026-09-18。**没有分页**，带 tournamentId 只是过滤，
        不能用来拉全部历史 —— 历史必须走 ``iter_schedule_backwards`` + ``get_event_details``。
        """
        return self._get(
            "getCompletedEvents", leagueId=league_id, tournamentId=tournament_id
        )["data"]["schedule"].get("events", [])

    def get_tournaments_for_league(self, league_id: str) -> list[dict[str, Any]]:
        leagues = self._get("getTournamentsForLeague", leagueId=league_id)["data"][
            "leagues"
        ]
        return leagues[0]["tournaments"] if leagues else []

    def get_event_details(self, match_id: str) -> dict[str, Any]:
        return self._get("getEventDetails", id=match_id)["data"]["event"]

    def game_ids_of_match(self, match_id: str) -> list[dict[str, Any]]:
        """返回 ``[{'number','id','state'}]``。

        **不要用 ``gameId == matchId + number`` 这个猜测**：[实测 2026-09-18]
        对 getCompletedEvents 的 1,222 个 game 验证，只有 1,175 个成立（96.2%），
        47 个不成立（雪花 ID 跨块时会跳号）。必须老实调 getEventDetails。
        """
        ev = self.get_event_details(match_id)
        return (ev.get("match") or {}).get("games") or []


# ── livestats feed ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Participant:
    """window ``gameMetadata`` 里的一名选手。"""

    esports_game_id: str
    esports_match_id: str
    esports_team_id: str
    side: str  # 'blue' / 'red'
    participant_id: int
    esports_player_id: str
    summoner_name: str
    champion_id: str  # DDragon 的 id（'KSante'），不是数字 key
    role: str  # feed 口径 'top'/'jungle'/'mid'/'bottom'/'support'
    oe_position: str  # 已映射到 OE 口径 'top'/'jng'/'mid'/'bot'/'sup'
    patch_version: str


class LivestatsClient(_BaseClient):
    """``feed.lolesports.com/livestats/v1`` 封装。无鉴权。"""

    def __init__(self, base: str = LIVESTATS_BASE, **kw: Any) -> None:
        super().__init__(**kw)
        self.base = base.rstrip("/")

    def window(self, game_id: str, starting_time: str | None = None) -> dict[str, Any]:
        """``startingTime`` 省略时返回本局**最早**的帧（§3.3.3 约束 4）。

        对回填 crosswalk 而言这正是我们要的：最早帧 = 开局，gameMetadata 完整，
        且响应最小（[实测] 约 16.8 KB）。实时轮询才需要传 startingTime，
        且必须能被 10 秒整除（§3.3.3 约束 1）。
        """
        params = {"startingTime": starting_time} if starting_time else None
        return self.request_json(f"{self.base}/window/{game_id}", params)

    def window_metadata(self, game_id: str) -> list[Participant]:
        """调 window 并把 ``gameMetadata`` 摊平成 10 条 Participant。

        抛 NoContentError 表示该局无 livestats 覆盖（未开始 / 太老 / 非官方转播），
        抛 NotFoundError 表示 gameId 根本不存在。两者调用方都应记账而非崩。
        """
        payload = self.window(game_id)
        return parse_window_metadata(payload)


def parse_window_metadata(payload: dict[str, Any]) -> list[Participant]:
    """纯函数，便于单测。结构不符时抛 ValueError 而不是静默返回空表。

    静默返回空表是 §9.4 那条风险的典型形态：join 不上不报错，
    player rating 退化成先验值，指标只轻微下滑。
    """
    gm = payload.get("gameMetadata")
    if not gm:
        raise ValueError("window 响应缺少 gameMetadata")
    game_id = str(payload.get("esportsGameId", ""))
    match_id = str(payload.get("esportsMatchId", ""))
    patch = str(gm.get("patchVersion", ""))
    out: list[Participant] = []
    for side, key in (("blue", "blueTeamMetadata"), ("red", "redTeamMetadata")):
        team = gm.get(key) or {}
        team_id = str(team.get("esportsTeamId", ""))
        for p in team.get("participantMetadata") or []:
            role = str(p.get("role", ""))
            out.append(
                Participant(
                    esports_game_id=game_id,
                    esports_match_id=match_id,
                    esports_team_id=team_id,
                    side=side,
                    participant_id=int(p["participantId"]),
                    esports_player_id=str(p["esportsPlayerId"]),
                    summoner_name=str(p.get("summonerName", "")),
                    champion_id=str(p.get("championId", "")),
                    role=role,
                    oe_position=ROLE_MAP.get(role, role),
                    patch_version=patch,
                )
            )
    if len(out) != 10:
        raise ValueError(f"gameMetadata 应有 10 名选手，实得 {len(out)}")
    return out


# ── 名字规范化（§3.11.3 路径 3 的兜底）──────────────────────────────────
def strip_team_code(name: str, team_code: str) -> str:
    """剥离 summonerName 的队伍前缀。

    LPL 没有空格分隔符（``'WECube'``），LCS 有（``'SR Fudge'``）。
    用该队 code 同时覆盖两种格式 —— 这是 §3.11.3 唯一能在 LPL 上执行的写法。
    """
    import re

    if not team_code:
        return name
    return re.sub(rf"^{re.escape(team_code)}\s*", "", name, flags=re.I)


def norm_name(s: str) -> str:
    """去变音符 + 去非字母数字 + 小写。**只用于兜底，绝不做 fuzzy 近似匹配**。"""
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())
