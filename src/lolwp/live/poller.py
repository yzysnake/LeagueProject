"""实时轮询器：getLive → gameId → window 等 204→200 → 解析阵容 → 预测（§7.7.1）。

────────────────────────────────────────────────────────────────────────
唯一权威的开赛触发器是 window 的 204 → 200 跳变
────────────────────────────────────────────────────────────────────────
**绝不能写 `if game.state == "unstarted": continue`。** [实测 2026-09-19 10:0x]
LEC Movistar KOI vs Karmine Corp（matchId 115548681803406327）的 game 1 在
getEventDetails 里 state=inProgress，而 window 端点返回 204；手册 §3.3.4 记录的
另一场 LEC game4 的空窗长达 **20 分 50 秒**。schedule 的 inProgress 标的是
「局间休息 + BP 开始」，不是「已开打」。按 state 过滤会**跳过整个真正该密集
轮询的时段**，然后在真正开赛时前端还在转圈。

所以本轮询器的 watchlist 规则是「state != completed 就盯着」，
开没开打一律以 window 的 HTTP 状态码为准。

────────────────────────────────────────────────────────────────────────
为什么不带 startingTime
────────────────────────────────────────────────────────────────────────
我们只要 `gameMetadata`（阵容），**永远不要 frames**（赛中金币/击杀，进
pre-game 模型就是泄漏，§3.3.2）。不带 startingTime 时 feed 返回的是本局
**最早**的一帧 + 完整 gameMetadata（§3.3.3 约束 4）——对我们正好，
而且顺手绕开了另外两条约束：时间戳必须被 10 秒整除（否则 400）、
以及 50 秒延迟下限（否则 400）。这两个 400 在本模块里结构性地不可能发生。

────────────────────────────────────────────────────────────────────────
降级（§7.7.3，架构级要求）
────────────────────────────────────────────────────────────────────────
  x-api-key 被轮换/封禁 → client.live() 返回 [] → 本器标 api_degraded 并告警，
                          **what-if 完全不受影响**（它不经过这里）
  连续 204 > 15 分钟     → 该局标 NO_COVERAGE，前端提示手动摆阵容（验收⑥）
  window 403            → FORBIDDEN，停止轮询该局，不重试（重试只会更快被封）
  4xx/5xx/超时          → ERROR，指数退避后重试，连续失败到上限后标 NO_COVERAGE
  crosswalk 未命中      → resolve 侧退化到赛区中位数 + warning，不阻塞

**核心设计：手动摆阵容是默认模式，实时是增强。实时源整个消失时整站不受影响。**

────────────────────────────────────────────────────────────────────────
线程纪律
────────────────────────────────────────────────────────────────────────
`run_forever` 的每一次 tick 都包在 try/except Exception 里。轮询线程死掉的
后果是前端永远显示上一份快照且没有任何提示——比报错难查得多。异常只记进
`self.errors` 供 /health 暴露。

礼貌：getLive ≥ 60 s，window ≥ 30 s（§7.7.4）。两个节流都在本模块里强制，
不依赖调用方自觉；`_now`/`_sleep` 可注入，单测因此不需要真的睡 30 秒。

────────────────────────────────────────────────────────────────────────
时钟用墙钟，不用 time.monotonic
────────────────────────────────────────────────────────────────────────
macOS 上 `time.monotonic()` 是 `mach_absolute_time()`，**合盖睡眠时不走**。
这里的每个时限（getLive 60 秒节流、15 分钟无覆盖、5 分钟宽限、1 小时清理）
说的都是比赛的真实时间。用 monotonic 时，Mac 合盖后只在每 15 分钟一次、
每次 2–60 秒的 DarkWake 里醒来，节流器按「醒着的时间」算，常常认为离上次
getLive 还不到 60 秒而跳过刷新——新开打的局根本进不了 watchlist。
[实测 2026-10-23] Worlds 瑞士轮 8 局里 KC–AL、LOS–T1、LYON–TSW 三局整局
都在睡眠期间打完，一局也没抓到；同期 watchlist 里早就有的两局倒是在 DarkWake 里
被补抓到了（赛后补录）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from lolwp.live.lolesports import FeedResult, FeedState, LolEsports
from lolwp.live.resolve import Resolver, to_draft

log = logging.getLogger(__name__)


class GameStatus(str, Enum):
    """一局在轮询器眼里的状态。每一种前端都要有对应文案，不能只处理 READY。"""

    WAITING = "waiting"            # 已进 watchlist，window 还是 204（BP 中 / 读条前）
    READY = "ready"                # 已拿到阵容并出了预测
    NO_COVERAGE = "no_coverage"    # 连续 204 超时，或反复出错：该场无 livestats 覆盖
    FORBIDDEN = "forbidden"        # feed 403
    ERROR = "error"                # 其他错误，仍会重试
    FINISHED = "finished"          # getLive 里已经消失 / state=completed


@dataclass
class GameState:
    game_id: str
    match_id: str | None = None
    league_name: str | None = None
    block_name: str | None = None
    team_codes: list[str] = field(default_factory=list)
    status: GameStatus = GameStatus.WAITING
    first_seen: float = 0.0
    last_poll: float = 0.0
    # 该局所属 match 最后一次出现在 getLive 里的时刻。**判定「这局没了」只能靠它**，
    # 不能靠"本轮 alive 里没有它"——见 tick_live 里的三分支注释。
    last_seen_in_schedule: float = 0.0
    n_resurrected: int = 0
    n_204: int = 0
    n_error: int = 0
    first_204: float | None = None
    first_200: float | None = None
    predicted_at: float | None = None
    meta_hash: str | None = None
    message: str = "等待开局读条（通常 1-2 分钟）…"
    prediction: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds_waiting(self) -> float:
        if self.first_204 is None:
            return 0.0
        return max(0.0, (self.last_poll or self.first_204) - self.first_204)

    @property
    def latency_200_to_pred(self) -> float | None:
        """window 首次 200 → 预测就绪的秒数（验收⑤要求 ≤35 秒）。"""
        if self.first_200 is None or self.predicted_at is None:
            return None
        return self.predicted_at - self.first_200


@dataclass
class PollerConfig:
    live_interval_s: float = 60.0      # §7.7.4 礼貌下限
    window_interval_s: float = 30.0    # §7.7.4 礼貌下限
    no_coverage_after_s: float = 15 * 60.0   # 验收⑥
    # 标了「无覆盖」之后仍然低频回探。**这是对验收⑥的补充而不是违背**：
    # 手册 §3.3.4 自己记录的那场 LEC game4 空窗长达 20 分 50 秒，
    # 比 15 分钟阈值还长 6 分钟。也就是说按验收⑥的阈值，**手册自己那个实测样本
    # 会被误判成「无覆盖」**。前端必须在 15 分钟时停止转圈（这是验收⑥的目的），
    # 但后台没有任何理由真的放弃——所以 UI 切手动模式，轮询降到 2 分钟一次，
    # 真的开打了再自动恢复。
    no_coverage_poll_s: float = 120.0
    # getLive 返回空时用 getTeams 做一次鉴权探针。**没有这个探针就无法区分
    # 「今天没有比赛」和「key 被轮换了」**——两者都是空列表。getTeams 不带 id
    # 恒定返回 1,578 支队伍，空即鉴权失败。1.5 MB/次，所以 15 分钟才探一次。
    auth_probe_s: float = 900.0
    max_consecutive_errors: int = 8
    # 一个 match 从 getLive 里消失多久之后，才敢把它名下还没打的局判成「已结束」。
    # **[实测 2026-09-19 16:45:48Z] 没有这个宽限期就是线上事故**：那一刻 getLive
    # 返回非空但 getEventDetails 抖了一下，8 局正在进行的比赛（CBLOL + LIT）被整批
    # 判成 FINISHED，且此后 1 小时（finished_ttl_s）再也没恢复——两个多小时后另起
    # 进程才重新看到它们，其中 115565671526403084 当时压根还没开打。
    schedule_grace_s: float = 300.0
    # FINISHED 的局保留多久再从内存里清掉（长跑几天不能无限涨）
    finished_ttl_s: float = 3600.0
    backoff_base_s: float = 5.0
    backoff_max_s: float = 120.0
    tier_warn_only: bool = True

    def __post_init__(self) -> None:
        # 把礼貌写成硬约束，而不是注释。调小了会打爆唯一的免费实时数据源。
        if self.live_interval_s < 60.0:
            raise ValueError("getLive 轮询间隔不得低于 60 秒（§7.7.4）")
        if self.window_interval_s < 30.0:
            raise ValueError("window 轮询间隔不得低于 30 秒（§7.7.4）")


class Poller:
    """把实时比赛接到 app.predict 上。

    用法（单线程）::

        p = Poller(LolEsports(), Resolver.load(), art)
        p.start()            # 后台线程
        ...
        p.snapshot()         # 前端每秒读一次，纯内存

    也可以手动打 tick（单测走这条路，不需要线程也不需要 sleep）::

        p.tick_live(); p.tick_windows()
    """

    def __init__(
        self,
        client: LolEsports,
        resolver: Resolver,
        artifact: Any | None = None,
        cfg: PollerConfig | None = None,
        *,
        now: Callable[[], float] = time.time,   # 墙钟：见模块文档「时钟用墙钟」
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self.client = client
        self.resolver = resolver
        self.artifact = artifact
        self.cfg = cfg or PollerConfig()
        self._now = now
        self._sleep = sleep
        # 必须是 tz-aware UTC：draft_features_from_state 要拿它和状态快照的
        # as_of 相减算衰减，naive 与 aware 相减会直接抛 TypeError。
        self._wall = wall_clock or (lambda: pd.Timestamp.now(tz="UTC"))

        # 装载 artifact 的文件与当时的修改时间：文件被换过（每日重解实力快照 / 重训）就重新装载
        self.artifact_path: str | None = None
        self.artifact_mtime: float | None = None

        self.games: dict[str, GameState] = {}
        self.api_degraded = False
        self.errors: list[str] = []
        self.n_live_polls = 0
        self.n_window_polls = 0
        self._last_live = -1e18
        self._last_window = -1e18
        self._last_auth_probe = -1e18
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── 对外只读接口 ────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """前端唯一的数据入口：纯内存，不触发任何网络调用。"""
        with self._lock:
            return {
                "api_degraded": self.api_degraded,
                "n_live_polls": self.n_live_polls,
                "n_window_polls": self.n_window_polls,
                "errors": list(self.errors[-5:]),
                "games": [
                    {
                        "game_id": g.game_id,
                        "match_id": g.match_id,
                        "league": g.league_name,
                        "teams": list(g.team_codes),
                        "status": g.status.value,
                        "message": g.message,
                        "seconds_waiting": round(g.seconds_waiting, 1),
                        "latency_200_to_pred_s": (
                            None if g.latency_200_to_pred is None
                            else round(g.latency_200_to_pred, 3)
                        ),
                        "prediction": g.prediction,
                        "warnings": list(g.warnings),
                    }
                    for g in self.games.values()
                ],
            }

    def health(self) -> dict[str, Any]:
        """/health 用的 gauge（§3.11.4：覆盖率只做 gauge + 告警，绝不做启动 assert）。"""
        with self._lock:
            by = {s.value: 0 for s in GameStatus}
            rates: list[float] = []
            for g in self.games.values():
                by[g.status.value] += 1
                if g.prediction:
                    rates.append(float(g.prediction.get("player_resolve_rate", 0.0)))
            return {
                "api_degraded": self.api_degraded,
                "n_watched": len(self.games),
                "by_status": by,
                "live_resolve_rate": (sum(rates) / len(rates)) if rates else None,
                "n_errors": len(self.errors),
            }

    # ── tick 1：每 60 秒刷新 watchlist ──────────────────────────────────
    def tick_live(self, force: bool = False) -> int:
        """刷新 watchlist。返回本次 watchlist 里的局数。

        `client.live()` 内部把网络异常吞成 []（无效 x-api-key 实测就是这个行为），
        所以空列表有两种截然不同的成因：**今天没比赛** 和 **key 被轮换了**。
        两者不能混为一谈——前者什么都不该做，后者要给前端亮降级灯（验收⑦）。
        区分办法是拿 getTeams 做鉴权探针（它恒定返回 1,578 支队伍）。
        """
        t = self._now()
        if not force and t - self._last_live < self.cfg.live_interval_s:
            return len(self.games)
        self._last_live = t
        self.n_live_polls += 1

        events = self.client.live()
        alive: set[str] = set()
        # 本轮 getLive 里出现过的 match（不管 details 拿没拿到）
        seen_matches: set[str] = set()
        # 本轮 **成功读到 games[] 列表** 的 match。只有这些 match 名下的局，
        # 才可以凭「不在 alive 里」被判成已结束——这是权威信息。
        refreshed_matches: set[str] = set()
        for ev in events:
            match = ev.get("match") or {}
            match_id = str(match.get("id") or ev.get("id") or "")
            if not match_id:
                # getLive 实测会返回没有 match 的占位 event（2026-09-19 的 LCS 那条
                # 就是 match=None）。它不代表任何一局的死活，直接跳过。
                continue
            seen_matches.add(match_id)
            league = ((ev.get("league") or {}).get("name")) or None
            codes = [str(t_.get("code")) for t_ in (match.get("teams") or []) if t_.get("code")]
            detail = self.client.event_details(match_id)
            if not detail:
                # getEventDetails 单点失败 ≠ 比赛结束。什么都不做，下一轮再问。
                log.warning("getEventDetails(%s) 失败，本轮不据此判定其名下任何一局已结束",
                            match_id)
                continue
            refreshed_matches.add(match_id)
            league = ((detail.get("league") or {}).get("name")) or league
            for g in ((detail.get("match") or {}).get("games") or []):
                gid = str(g.get("id") or "")
                if not gid:
                    continue
                # ★ 绝不要写 if g["state"] == "unstarted": continue（§3.3.4）
                if g.get("state") == "completed":
                    continue
                alive.add(gid)
                with self._lock:
                    st = self.games.get(gid)
                    if st is None:
                        st = GameState(game_id=gid, first_seen=t)
                        self.games[gid] = st
                    st.match_id = match_id
                    st.league_name = league
                    st.block_name = ev.get("blockName") or st.block_name
                    st.last_seen_in_schedule = t
                    if codes:
                        st.team_codes = codes

        degraded: bool | None = None
        if events:
            degraded = False
        elif t - self._last_auth_probe >= self.cfg.auth_probe_s:
            # 空列表有两种成因，必须分开：今天没比赛 vs key 被轮换（验收⑦）
            self._last_auth_probe = t
            degraded = not self.client.teams()
            if degraded:
                log.error("getLive 与 getTeams 同时返回空，判定 x-api-key 失效，切手动模式")

        with self._lock:
            if degraded is not None:
                self.api_degraded = degraded
            for gid, st in list(self.games.items()):
                # ⚠️ 判「这局没了」必须分三种情形。合成一句 `gid not in alive`
                # 就是 2026-09-19 16:45:48Z 那次事故的成因（见 schedule_grace_s）。
                if events and gid in alive:
                    # (1) 还在赛程里。若之前被误判过，在这里自愈——没有这条，
                    #     一次误判会一直挂到 finished_ttl_s（1 小时）才有机会翻身。
                    if st.status is GameStatus.FINISHED:
                        st.n_resurrected += 1
                        st.status = GameStatus.WAITING
                        st.message = "等待开局读条（通常 1-2 分钟）…"
                        log.warning("gameId=%s 曾被误判为已结束，现已重新出现在赛程里，恢复轮询",
                                    gid)
                elif events and st.match_id in refreshed_matches:
                    # (2) 该 match 的 games[] 本轮**成功读到了**，而这一局不在里面
                    #     （或 state=completed）。这是权威信息，立即判结束。
                    if st.status is not GameStatus.FINISHED:
                        st.status = GameStatus.FINISHED
                        st.message = "该局已结束或已从赛程中移除"
                # (3) 整个 match 从 getLive 里消失了。真结束和 API 抖动长得一模一样，
                #     只能靠「连续消失够久」区分。宽限期内什么都不做。
                elif (events and st.match_id not in seen_matches
                        and st.status is not GameStatus.FINISHED
                        and t - st.last_seen_in_schedule > self.cfg.schedule_grace_s):
                    st.status = GameStatus.FINISHED
                    st.message = "该局已结束或已从赛程中移除"
                # 剩下一种：match 在 getLive 里但 details 没拿到 → 本轮不做任何判定。
                if (st.status is GameStatus.FINISHED
                        and t - max(st.last_poll, st.first_seen) > self.cfg.finished_ttl_s):
                    del self.games[gid]
            return len(self.games)

    # ── tick 2：每 30 秒轮询 window ─────────────────────────────────────
    def tick_windows(self, force: bool = False) -> int:
        """对 watchlist 里每个未决的 gameId 打一次 window。返回本次实际请求数。"""
        t = self._now()
        if not force and t - self._last_window < self.cfg.window_interval_s:
            return 0
        self._last_window = t

        with self._lock:
            pending = [
                g for g in self.games.values()
                if g.status in (GameStatus.WAITING, GameStatus.ERROR, GameStatus.NO_COVERAGE)
            ]
        n = 0
        for g in pending:
            if g.status is GameStatus.NO_COVERAGE:
                if t - g.last_poll < self.cfg.no_coverage_poll_s:
                    continue
            elif g.status is GameStatus.ERROR and g.n_error:
                # 指数退避：错误越多等越久，但不超过 backoff_max_s
                wait = min(self.cfg.backoff_base_s * 2 ** (g.n_error - 1), self.cfg.backoff_max_s)
                if t - g.last_poll < wait:
                    continue
            res = self.client.window(g.game_id)
            n += 1
            self.n_window_polls += 1
            self._handle_window(g, res, t)
        return n

    def _handle_window(self, g: GameState, res: FeedResult, t: float) -> None:
        with self._lock:
            g.last_poll = t
            if res.state is FeedState.NOT_STARTED:
                g.n_204 += 1
                g.n_error = 0
                if g.first_204 is None:
                    g.first_204 = t
                if t - g.first_204 > self.cfg.no_coverage_after_s:
                    if g.status is not GameStatus.NO_COVERAGE:
                        log.warning("gameId=%s 连续 204 超过 %.0f 分钟，判定无 livestats 覆盖",
                                    g.game_id, self.cfg.no_coverage_after_s / 60)
                    g.status = GameStatus.NO_COVERAGE
                    g.message = "该场次不支持实时读取，请到 what-if 页手动摆阵容"
                else:
                    g.status = GameStatus.WAITING
                    g.message = "等待开局读条（通常 1-2 分钟）…"
                return
            if res.state is FeedState.FORBIDDEN:
                g.status = GameStatus.FORBIDDEN
                g.message = "实时源拒绝访问（403），已切手动模式"
                log.error("window 403 gameId=%s，停止轮询该局", g.game_id)
                return
            if res.state is FeedState.ERROR:
                g.n_error += 1
                if g.n_error >= self.cfg.max_consecutive_errors:
                    g.status = GameStatus.NO_COVERAGE
                    g.message = "实时源连续失败，已切手动模式"
                else:
                    g.status = GameStatus.ERROR
                    g.message = f"实时源暂时不可用（{res.http or res.detail[:40]}），重试中…"
                self._note_error(f"window {g.game_id}: {res.http} {res.detail[:80]}")
                return

            # ---- 200：204 → 200 的跳变就是开赛触发器 ----
            if g.first_200 is None:
                g.first_200 = t
            g.n_error = 0
            payload = res.payload or {}
            h = _meta_hash(payload)
            if g.meta_hash == h and g.status is GameStatus.READY:
                return  # ★ 幂等：同一局同一份阵容只预测一次（§7.7.1）
            g.meta_hash = h
        # 解析与预测放在锁外：它们是纯 CPU，且绝不能让一次解析异常卡住整张表
        self._predict(g, payload)

    def predict_window(
        self, payload: dict[str, Any], league_name: str | None,
        now: pd.Timestamp | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """一份 window payload → (预测, warnings)。纯 CPU，不碰轮询状态。

        实时轮询和赛后补录（scripts/live_watch.py --backfill）**共用这一个函数**：
        两条路径各拼一遍字段，台账里同一种记录就会长出两种形状。
        now = 预测所站的时刻，只影响英雄强度与熟练度的衰减；补录传开局时间，
        这样补出来的概率和当时实时算的一样。
        """
        rg = self.resolver.resolve_window(payload, feed_league_name=league_name)
        out: dict[str, Any] = {
            "esports_game_id": rg.esports_game_id,
            "league_canon": rg.league_canon,
            "patch": list(rg.patch) if rg.patch else None,
            "blue_team": rg.blue.oe_team_id,
            "red_team": rg.red.oe_team_id,
            "blue_champ_keys": list(rg.blue.champ_keys),
            "red_champ_keys": list(rg.red.champ_keys),
            "blue_names": list(rg.blue.summoner_names),
            "red_names": list(rg.red.summoner_names),
            **rg.stats,
        }
        if self.artifact is not None:
            from lolwp.app.predict import predict as _predict_fn

            d = to_draft(rg, self.artifact.prior_state, now=self._wall() if now is None else now)
            pred = _predict_fn(self.artifact, d)
            out.update({
                "p_blue": pred.p_blue,
                # 加近期状态之前的概率：盯盘落盘后，近期状态的残差用它算（与训练时口径一致）
                "p_base": 1.0 / (1.0 + math.exp(-(pred.logit - pred.contributions.get("form_diff", 0.0)))),
                "logit": pred.logit,
                "layers": pred.layers,
                "contributions": pred.contributions,
            })
            rg.warnings.extend(pred.warnings)
        return out, list(rg.warnings)

    def _predict(self, g: GameState, payload: dict[str, Any]) -> None:
        try:
            out, warns = self.predict_window(payload, g.league_name)
            with self._lock:
                g.prediction = out
                g.warnings = warns
                g.status = GameStatus.READY
                g.predicted_at = self._now()
                g.message = "阵容已读取，预测就绪"
        except Exception as e:  # 解析失败绝不能让轮询线程死掉
            log.exception("解析/预测失败 gameId=%s", g.game_id)
            with self._lock:
                g.status = GameStatus.ERROR
                g.n_error += 1
                g.message = "阵容解析失败，请手动摆阵容"
                self._note_error(f"predict {g.game_id}: {e!r}")

    def reload_artifact_if_changed(self) -> bool:
        """artifact 文件换过（`artifact.refresh` 重解了实力快照、或重训过）就重新装载。返回是否换了。

        已出的预测不重算（同一份阵容只预测一次）；之后开打的局用新快照。
        """
        if not self.artifact_path:
            return False
        try:
            m = os.path.getmtime(self.artifact_path)
        except OSError:
            return False
        if m == self.artifact_mtime:
            return False
        art = load_serving_artifact(self.artifact_path)
        with self._lock:
            self.artifact, self.artifact_mtime = art, m
        return True

    def _note_error(self, msg: str) -> None:
        self.errors.append(msg)
        if len(self.errors) > 50:
            del self.errors[:-50]

    # ── 线程 ────────────────────────────────────────────────────────────
    def tick(self) -> None:
        """一次完整 tick。**任何异常都吞掉**，只记进 errors。"""
        for fn in (self.tick_live, self.tick_windows):
            try:
                fn()
            except Exception as e:  # 一次 tick 失败不得终止线程
                log.exception("%s 抛异常", fn.__name__)
                with self._lock:
                    self._note_error(f"{fn.__name__}: {e!r}")

    def seconds_until_due(self) -> float:
        """距离下一次该发请求还有多久。**这决定验收⑤的检出上界，不只是省几次空转。**

        原来写的是「每 5 秒 tick 一次，节流器自己判断该不该发」。那样实际的 window
        轮询间隔不是 30 秒，而是「30 秒向上取整到下一个 tick」**再加上每个 tick 的
        工作耗时**——tick_live 要串行打 1+N 个 HTTP，N 是当前比赛场数。
        实测（2026-09-19 18:40-18:45，2 场 LEC/LIT 并发）：313 秒里 11 轮，
        **平均 31.3 秒/轮**，而且 N 越大漂得越多。
        验收⑤给的预算是「首次 200 → 页面显示 ≤35 秒」，检出上界就已经吃掉 31 秒，
        留给前端刷新的只剩 4 秒——不能再让它随比赛场数漂。

        按 deadline 算等待时间后，间隔 = 30 秒 + 一轮请求的耗时，与 tick_s 无关。
        """
        t = self._now()
        return min(self._last_live + self.cfg.live_interval_s,
                   self._last_window + self.cfg.window_interval_s) - t

    def run_forever(self, tick_s: float = 5.0) -> None:
        while not self._stop.is_set():
            self.tick()
            # 上限仍是 tick_s：_stop 事件要能在 tick_s 内被看到，进程不能卡在长 sleep 里。
            self._sleep(max(0.0, min(tick_s, self.seconds_until_due())))

    def start(self, tick_s: float = 5.0) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.run_forever, args=(tick_s,), name="lolwp-poller", daemon=True
        )
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)


def _meta_hash(payload: dict[str, Any]) -> str:
    """幂等键：只哈希 gameMetadata。frames 每帧都变，把它算进来等于没有幂等。"""
    gm = payload.get("gameMetadata") or {}
    return hashlib.sha256(
        json.dumps(gm, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def load_serving_artifact(artifact_path: str = "models/prod.json"):
    """加载 artifact 并装好降级评分。

    `install_fallback_ratings` 喂的数据必须和 artifact **同一口径**，两边 as-of 才对齐：
    dev 口径的 artifact 配 load_dev()（serving 也不碰 holdout）；揭盲后全量重训的
    artifact（meta.data_scope == "all"，预注册修订 A4）配 load_all()。
    """
    from lolwp.features.loader import load_all, load_dev
    from lolwp.live.resolve import install_fallback_ratings
    from lolwp.models.artifact import Artifact

    art = Artifact.load(artifact_path)
    games = load_all() if art.meta.get("data_scope") == "all" else load_dev()
    install_fallback_ratings(art.prior_state, games)
    return art


def build_default_poller(artifact_path: str = "models/prod.json") -> Poller:
    """生产用的装配：加载 artifact、装好降级评分、建好解析器。"""
    m = os.path.getmtime(artifact_path)
    pol = Poller(LolEsports(), Resolver.load(), load_serving_artifact(artifact_path))
    pol.artifact_path, pol.artifact_mtime = artifact_path, m
    return pol


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = build_default_poller()
    p.start()
    try:
        while True:
            time.sleep(10)
            snap = p.snapshot()
            print(json.dumps(snap, ensure_ascii=False, default=str)[:1200])
    except KeyboardInterrupt:
        p.stop()
