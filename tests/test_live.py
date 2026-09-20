"""实时链路单测（PLAN.md 阶段 6 验收⑤⑥⑦）。

三组：

A. **四种 HTTP 状态各有单测**（验收⑤第一条）：200 / 204 / 403 / 其他错误。
   全部走 mock，**零真实网络**——单测不能依赖"此刻恰好有比赛在打"。
B. **train-serve skew golden test**：同一场真实比赛，走 live 解析路径算出的特征
   必须与走 silver 路径算出的特征**逐位一致**。夹具是真实 window payload
   （tests/fixtures/window_completed_{lec,lpl,lck}.json，本机实测 200 拉下来的）。
   这条专抓三类"必然会发生"的 bug：role 命名（bottom vs bot）、
   英雄名（DDragon 字符串 id vs 数字 key）、patch 解析（四段式 vs 两段式）。
   **LPL 那份是关键**：feed 的 summonerName 在 LPL 没有空格分隔符
   （`WBGZika` 而不是 `WBG Zika`），任何靠名字切前缀的实现都会在这里翻车。
C. 轮询状态机：15 分钟切无覆盖、幂等、异常不杀线程、无效 key 降级。

另有 tests/fixtures/window_live.json —— **真实进行中的比赛**的 payload
（LEC Movistar KOI vs Karmine Corp，gameId 115548681803406328，
本机 2026-09-19T15:15:27Z 抓到 204→200 跳变的那一刻）。
它比造假数据有价值的地方在于：feed 给的 blue 方是 KC，而同一时刻
getEventDetails 的 `games[0].teams[].side` 说 blue 是 MKOI —— 两边是反的。
手册 §3.3.4 说"Bo5 后续局的 side 在未开始时是占位值不可信"，这份夹具把它
钉成了回归测试（test_live_fixture_side_comes_from_window）。
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from lolwp.live.lolesports import FeedResult, FeedState, LolEsports, parse_rosters
from lolwp.live.poller import GameStatus, Poller, PollerConfig, _meta_hash
from lolwp.live.resolve import (
    FALLBACK_PLAYER_PREFIX,
    FALLBACK_TEAM_PREFIX,
    FEED_ROLE_TO_OE,
    ROLES,
    Resolver,
    build_team_crosswalk,
    install_fallback_ratings,
    norm_feed_patch,
    to_draft,
)

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures")
REPO = os.path.dirname(HERE)
SILVER = os.path.join(REPO, "data", "silver", "games.parquet")
ARTIFACT = os.path.join(REPO, "models", "prod.json")
PLAYER_XW = os.path.join(REPO, "data", "crosswalk", "player_crosswalk.parquet")

# (夹具名, esportsGameId, OE gameid, feed 的联赛名)
GOLDEN = [
    ("window_completed_lec.json", "115548668059589355", "LOLTMNT05_196187", "LEC"),
    ("window_completed_lpl.json", "115616219464541906", "13282-13282_game_5", "LPL"),
    ("window_completed_lck.json", "115548128962971893", "LOLTMNT02_411342", "LCK"),
]

needs_data = pytest.mark.skipif(
    not (os.path.exists(SILVER) and os.path.exists(PLAYER_XW)),
    reason="需要 silver 与 crosswalk",
)
needs_artifact = pytest.mark.skipif(not os.path.exists(ARTIFACT), reason="需要 models/prod.json")


# ════════════════════════════════════════════════════════════════════════
# 假的 HTTP 层：requests.Session.get 的最小替身
# ════════════════════════════════════════════════════════════════════════
class FakeResponse:
    def __init__(self, status: int, payload: object = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """按 URL 前缀路由的假 session。记录所有请求，供礼貌性断言。"""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, **kw: object) -> FakeResponse:
        self.calls.append(url)
        for prefix, resp in self.routes.items():
            if prefix in url:
                if callable(resp):
                    return resp(url)
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return FakeResponse(404, text="no route")


def make_client(routes: dict[str, object]) -> tuple[LolEsports, FakeSession]:
    c = LolEsports()
    s = FakeSession(routes)
    c.s = s  # type: ignore[assignment]
    return c, s


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIX, name)) as fh:
        return json.load(fh)


# ════════════════════════════════════════════════════════════════════════
# A. 四种 HTTP 状态各一条（验收⑤第一条）
# ════════════════════════════════════════════════════════════════════════
def test_window_200_returns_ready_with_payload():
    payload = load_fixture("window_live.json")
    c, _ = make_client({"/window/": FakeResponse(200, payload)})
    r = c.window("115548681803406328")
    assert r.state is FeedState.READY
    assert r.http == 200
    assert r.payload is not None
    assert r.payload["gameMetadata"]["patchVersion"] == "16.17.810.4348"


def test_window_204_is_not_started_not_an_error():
    """204 是**信息不是错误**：比赛还没发第一帧。不能当失败重试到爆。"""
    c, _ = make_client({"/window/": FakeResponse(204)})
    r = c.window("115548681803406329")
    assert r.state is FeedState.NOT_STARTED
    assert r.http == 204
    assert r.payload is None


def test_window_403_is_forbidden_and_must_not_retry():
    c, _ = make_client({"/window/": FakeResponse(403, text="forbidden")})
    r = c.window("x")
    assert r.state is FeedState.FORBIDDEN
    assert r.http == 403


def test_window_404_and_500_and_timeout_are_error():
    """[实测 2026-09-18] 不存在的 gameId 返回 404 而不是 204，必须与 204 分开处理。"""
    import requests

    c, _ = make_client({"/window/": FakeResponse(404, text="not found")})
    r = c.window("does-not-exist")
    assert r.state is FeedState.ERROR and r.http == 404

    c, _ = make_client({"/window/": FakeResponse(500, text="boom")})
    assert c.window("x").state is FeedState.ERROR

    c, _ = make_client({"/window/": requests.ConnectTimeout("timeout")})
    r = c.window("x")
    assert r.state is FeedState.ERROR and r.http is None


def test_window_200_with_non_json_body_is_error_not_crash():
    c, _ = make_client({"/window/": FakeResponse(200, None, text="<html>")})
    r = c.window("x")
    assert r.state is FeedState.ERROR
    assert "非 JSON" in r.detail


def test_invalid_api_key_yields_empty_live_not_exception():
    """验收⑦：key 失效时 /live 返回空 + warning，what-if 完全不受影响。"""
    c, _ = make_client({"getLive": FakeResponse(403, text="forbidden")})
    assert c.live() == []
    assert c.event_details("1") is None
    assert c.teams() == []


# ════════════════════════════════════════════════════════════════════════
# B. 解析层
# ════════════════════════════════════════════════════════════════════════
def test_role_map_is_the_five_constants():
    assert FEED_ROLE_TO_OE == {
        "top": "top", "jungle": "jng", "mid": "mid", "bottom": "bot", "support": "sup"
    }
    assert tuple(FEED_ROLE_TO_OE.values()) == ROLES


@pytest.mark.parametrize(
    ("raw", "expect"),
    [
        ("16.17.810.4348", (16, 17)),
        ("16.10.776.5552", (16, 10)),  # ← 零填充：解析成 (16, 1) 就会早 9 个版本
        ("14.18.617.4123", (14, 18)),
        ("16.17", (16, 17)),
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_norm_feed_patch(raw, expect):
    assert norm_feed_patch(raw) == expect


def test_norm_feed_patch_rejects_float():
    """float 入参必须显式炸掉：JSON number 会把 16.10 变成 16.1，再变成 patch 16.01。"""
    with pytest.raises(TypeError):
        norm_feed_patch(16.10)


@needs_data
def test_champion_map_is_a_bijection_on_feed_ids():
    r = Resolver.load()
    assert r.champion("KSante") == 897
    assert r.champion("LeeSin") == 64
    assert r.champion("MonkeyKing") == 62  # 悟空：dd_id 与显示名不一样
    assert r.champion("NotAChampion") is None
    assert len(set(r.champ_map.values())) == len(r.champ_map)


@needs_data
def test_resolver_parses_live_fixture_all_ten_players():
    """真实进行中的比赛：10 名选手 / 2 支队伍 / 10 个英雄全部解析成功。"""
    r = Resolver.load()
    rg = r.resolve_window(load_fixture("window_live.json"), feed_league_name="LEC")
    assert rg.stats["n_players_resolved"] == 10
    assert rg.stats["n_champs_resolved"] == 10
    assert rg.stats["n_teams_resolved"] == 2
    assert rg.league_canon == "LEC"
    assert rg.patch == (16, 17)
    assert rg.warnings == []
    assert rg.blue.roles == list(ROLES)


@needs_data
def test_live_fixture_side_comes_from_window_not_schedule():
    """§3.3.4：Bo5 的 side 以 window 的 blueTeamMetadata 为准，schedule 的不可信。

    实测同一时刻 getEventDetails 说 game1 blue=MKOI(103461966965149786)，
    而 window 说 blue=KC(111692118851466302)。两边是反的。
    """
    r = Resolver.load()
    rg = r.resolve_window(load_fixture("window_live.json"), feed_league_name="LEC")
    assert rg.blue.esports_team_id == "111692118851466302"  # KC
    assert rg.red.esports_team_id == "103461966965149786"   # MKOI


@needs_data
def test_lpl_summoner_names_have_no_space_separator():
    """§3.11.3 实测：LPL 的 summonerName 没有空格（`WBGZika`）。

    本模块**不靠名字解析任何东西**，所以这条只是把事实钉住：
    哪天有人往解析路径里加了"按空格切前缀"的逻辑，这条会提醒他 LPL 没有空格。
    """
    rg = Resolver.load().resolve_window(load_fixture("window_completed_lpl.json"),
                                        feed_league_name="LPL")
    names = [n for n in rg.blue.summoner_names if n]
    assert names and all(" " not in n for n in names), names
    assert rg.stats["n_players_resolved"] == 10


@needs_data
def test_unknown_entities_degrade_and_warn_but_never_raise():
    """§3.11.4：未命中只告警不阻塞；退化到赛区中位数而不是崩。"""
    r = Resolver.load()
    payload = {
        "esportsGameId": "0",
        "gameMetadata": {
            "patchVersion": "16.17.810.4348",
            "blueTeamMetadata": {
                "esportsTeamId": "999999999999999999",  # 不存在的队
                "participantMetadata": [
                    {"participantId": i + 1, "esportsPlayerId": f"88888888888888{i}",
                     "summonerName": f"NEW p{i}", "championId": "LeeSin", "role": role}
                    for i, role in enumerate(["top", "jungle", "mid", "bottom", "support"])
                ],
            },
            "redTeamMetadata": {
                "esportsTeamId": "111692118851466302",
                "participantMetadata": [
                    {"participantId": i + 6, "esportsPlayerId": "103495716771322725",
                     "summonerName": "KC Canna", "championId": "NoSuchChampion", "role": role}
                    for i, role in enumerate(["top", "jungle", "mid", "bottom", "support"])
                ],
            },
        },
    }
    rg = r.resolve_window(payload, feed_league_name="LEC")
    assert rg.blue.oe_team_id is None
    assert rg.stats["n_players_resolved"] == 5      # 蓝方 5 个新人全未命中
    assert rg.stats["n_champs_resolved"] == 5       # 红方 5 个假英雄全未命中
    assert rg.red.champ_keys == [None] * 5
    assert any("不在对照表中" in w for w in rg.warnings)
    assert any("不在 champion_map" in w for w in rg.warnings)


@needs_data
def test_unknown_league_falls_into_unknown_bucket():
    r = Resolver.load()
    rg = r.resolve_window(load_fixture("window_live.json"), feed_league_name="Totally New League")
    assert rg.league_canon == "UNKNOWN"
    assert any("UNKNOWN" in w for w in rg.warnings)


@needs_data
def test_team_crosswalk_is_a_function_and_covers_the_live_match():
    xw = build_team_crosswalk(out_path=None)
    assert xw.esports_team_id.is_unique, "esportsTeamId → oe_teamid 必须是函数"
    m = dict(zip(xw.esports_team_id, xw.oe_teamid))
    # MKOI 有两个候选（2024 的 MAD Lions KOI / 2025+ 的 Movistar KOI），
    # 规则必须选中当前在用的那个，否则会把战队打回两年前的 Elo。
    assert m["103461966965149786"] == "oe:team:4acc40654379b4da680be1bd3129dfa"
    assert m["111692118851466302"] == "oe:team:f3aa9bbd924ce3af42ac53f8326bc7c"


@needs_data
def test_team_crosswalk_rebuild_is_deterministic():
    a = build_team_crosswalk(out_path=None)
    b = build_team_crosswalk(out_path=None)
    pd.testing.assert_frame_equal(a, b)


@needs_data
def test_team_crosswalk_as_of_cut_changes_nothing_for_stable_teams():
    """as-of 重建：截到 2024 年底时，MKOI 只能解析成当时的 MAD Lions KOI。

    这条把"对照表是 as-of 的、不是永恒真理"这件事写进测试。
    serving 必须用最新的表，回放历史比赛则必须用当时的表。
    """
    old = build_team_crosswalk(out_path=None, until="2024-12-31")
    m = dict(zip(old.esports_team_id, old.oe_teamid))
    assert m["103461966965149786"] == "oe:team:6a0084dacefb228be863fbb64cefe11"


# ── B2. train-serve skew golden test ────────────────────────────────────
@needs_data
@needs_artifact
@pytest.mark.parametrize(("fixture", "esports_gid", "oe_gid", "feed_league"), GOLDEN)
def test_train_serve_skew_features_identical(fixture, esports_gid, oe_gid, feed_league):
    """**同一场比赛，live 路径与 silver 路径算出的特征必须逐位一致。**

    这是整个实时链路唯一能自动发现"训练/推理口径漂了"的测试。
    不一致的典型原因，按发生概率排：
      1. role 排序（feed 按 participantId 1..5，silver 按 OE position）
      2. 英雄 id 体系（feed 给 "LeeSin"，模型要 64）
      3. patch 解析（四段式 16.17.810.4348 → (16,17)）
      4. 队伍/选手 id 解析走错了对照表（weak/alias 混进 primary）
    """
    from lolwp.app.predict import Draft, build_features
    from lolwp.models.artifact import Artifact

    sv = pd.read_parquet(SILVER)
    rows = sv[sv.game_id == oe_gid]
    assert len(rows) == 1, f"silver 里找不到 {oe_gid}"
    row = rows.iloc[0]
    # 夹具全部取自 holdout 起始日之前，serving 路径不得触碰 holdout
    assert row.kickoff_utc < pd.Timestamp("2026-06-01", tz="UTC")

    art = Artifact.load(ARTIFACT)
    r = Resolver.load()
    rg = r.resolve_window(load_fixture(fixture), feed_league_name=feed_league)
    assert rg.esports_game_id == esports_gid

    # 逐字段对齐（出错时能一眼看出是哪一类 bug）
    assert rg.league_canon == row.league_canon
    assert rg.patch == (int(row.patch_major), int(row.patch_minor))
    assert rg.blue.oe_team_id == row.blue_team_id
    assert rg.red.oe_team_id == row.red_team_id
    assert rg.blue.player_ids == list(row.blue_player_ids)
    assert rg.red.player_ids == list(row.red_player_ids)
    assert rg.blue.champ_keys == list(row.blue_champ_keys)
    assert rg.red.champ_keys == list(row.red_champ_keys)

    d_live = to_draft(rg, art.prior_state, now=row.kickoff_utc)
    d_silver = Draft(
        league=row.league_canon,
        blue_team_id=row.blue_team_id,
        red_team_id=row.red_team_id,
        blue_player_ids=list(row.blue_player_ids),
        red_player_ids=list(row.red_player_ids),
        blue_champ_keys=list(row.blue_champ_keys),
        red_champ_keys=list(row.red_champ_keys),
        now=row.kickoff_utc,
    )
    f_live, _ = build_features(art, d_live)
    f_silver, _ = build_features(art, d_silver)
    assert set(f_live) == set(f_silver)
    for k in f_live:
        assert f_live[k] == pytest.approx(f_silver[k], abs=1e-12), k


@needs_data
@needs_artifact
def test_fallback_ratings_are_league_medians_not_global_init():
    """队伍未知时退化为**该赛区中位数**，而不是全局 1500。"""
    import numpy as np

    from lolwp.features.loader import load_dev
    from lolwp.models.artifact import Artifact

    art = Artifact.load(ARTIFACT)
    dev = load_dev()
    install_fallback_ratings(art.prior_state, dev)
    te = art.prior_state["team_elo"]
    pe = art.prior_state["player_elo"]
    init = float(art.prior_state["elo_cfg"]["init"])
    lec = te[f"{FALLBACK_TEAM_PREFIX}LEC"]
    assert lec > init, "一线赛区的队伍中位实力应当高于全局初始值 1500"
    for pos in ROLES:
        assert f"{FALLBACK_PLAYER_PREFIX}LEC:{pos}" in pe
    assert np.isfinite(lec)


@needs_data
@needs_artifact
def test_to_draft_uses_sentinels_for_unknown_entities():
    from lolwp.app.predict import predict
    from lolwp.features.loader import load_dev
    from lolwp.models.artifact import Artifact

    art = Artifact.load(ARTIFACT)
    install_fallback_ratings(art.prior_state, load_dev())
    r = Resolver.load()
    payload = load_fixture("window_live.json")
    payload["gameMetadata"]["blueTeamMetadata"]["esportsTeamId"] = "0"
    rg = r.resolve_window(payload, feed_league_name="LEC")
    d = to_draft(rg, art.prior_state)
    assert d.blue_team_id == f"{FALLBACK_TEAM_PREFIX}LEC"
    p = predict(art, d)          # 绝不能抛异常
    assert 0.0 < p.p_blue < 1.0


# ════════════════════════════════════════════════════════════════════════
# C. 轮询状态机
# ════════════════════════════════════════════════════════════════════════
class StubResolver:
    """poller 的单测不需要真对照表。"""

    def resolve_window(self, payload, *, feed_league_name=None):
        from lolwp.live.resolve import ResolvedGame, ResolvedSide

        def side(tag):
            return ResolvedSide(tag, f"oe:team:{tag}", "primary",
                                [f"p{tag}{i}" for i in range(5)], [1, 2, 3, 4, 5],
                                list(ROLES), [None] * 5, [None] * 5, [None] * 5)

        return ResolvedGame("g", "m", feed_league_name or "UNKNOWN", (16, 17), "16.17",
                            side("b"), side("r"), [], {"player_resolve_rate": 1.0})


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _live_event(match_id: str, game_ids: list[str], state: str = "inProgress"):
    return {
        "getLive": FakeResponse(200, {"data": {"schedule": {"events": [
            {"id": match_id, "state": "inProgress", "league": {"name": "LEC"},
             "match": {"id": match_id, "teams": [{"code": "MKOI"}, {"code": "KC"}]}},
        ]}}}),
        "getEventDetails": FakeResponse(200, {"data": {"event": {
            "league": {"name": "LEC"},
            "match": {"games": [{"id": g, "number": i + 1, "state": state}
                                for i, g in enumerate(game_ids)]},
        }}}),
    }


def make_poller(routes, cfg=None):
    c, s = make_client(routes)
    clk = Clock()
    p = Poller(c, StubResolver(), None, cfg or PollerConfig(), now=clk, sleep=lambda _: None)
    return p, s, clk


def test_poller_watches_unstarted_games_because_state_is_not_the_trigger():
    """★ 绝不能按 state 过滤：state=unstarted 的局也必须进 watchlist（§3.3.4）。"""
    routes = _live_event("m1", ["g1", "g2"], state="unstarted")
    routes["/window/"] = FakeResponse(204)
    p, _, _ = make_poller(routes)
    p.tick_live(force=True)
    assert set(p.games) == {"g1", "g2"}
    assert all(g.status is GameStatus.WAITING for g in p.games.values())


def test_poller_skips_completed_games():
    routes = _live_event("m1", ["g1"], state="completed")
    routes["/window/"] = FakeResponse(204)
    p, _, _ = make_poller(routes)
    p.tick_live(force=True)
    assert p.games == {}


def test_poller_204_to_200_transition_produces_prediction():
    """唯一权威触发器：204 → 200 跳变。"""
    payload = load_fixture("window_live.json")
    state = {"n": 0}

    def window(url):
        state["n"] += 1
        return FakeResponse(204) if state["n"] <= 3 else FakeResponse(200, payload)

    routes = _live_event("m1", ["g1"])
    routes["/window/"] = window
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    for _ in range(3):
        p.tick_windows(force=True)
        clk.advance(30)
    assert p.games["g1"].status is GameStatus.WAITING
    p.tick_windows(force=True)
    g = p.games["g1"]
    assert g.status is GameStatus.READY
    assert g.prediction and g.prediction["patch"] == [16, 17]
    assert g.latency_200_to_pred is not None and g.latency_200_to_pred < 35.0


def test_poller_marks_no_coverage_after_15_minutes_of_204():
    """验收⑥：连续 204 超过 15 分钟自动切「无覆盖」，不能让前端永远转圈。"""
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    for _ in range(29):          # 29 × 30 s = 14.5 分钟
        p.tick_windows(force=True)
        clk.advance(30)
    assert p.games["g1"].status is GameStatus.WAITING
    clk.advance(60)
    p.tick_windows(force=True)
    g = p.games["g1"]
    assert g.status is GameStatus.NO_COVERAGE
    assert "手动摆阵容" in g.message


def test_poller_is_idempotent_on_repeated_200():
    """同一局同一份阵容只预测一次（§7.7.1 的 predicted_cache）。"""
    payload = load_fixture("window_live.json")
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(200, payload)
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    p.tick_windows(force=True)
    first = p.games["g1"].predicted_at
    for _ in range(5):
        clk.advance(30)
        p.tick_windows(force=True)
    assert p.games["g1"].predicted_at == first


def test_meta_hash_ignores_frames():
    """幂等键只哈希 gameMetadata：frames 每帧都变，算进去等于没有幂等。"""
    payload = load_fixture("window_live.json")
    other = json.loads(json.dumps(payload))
    other["frames"] = [{"rfc460Timestamp": "different"}]
    assert _meta_hash(payload) == _meta_hash(other)


def test_poller_403_stops_polling_that_game():
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(403, text="forbidden")
    p, s, _ = make_poller(routes)
    p.tick_live(force=True)
    p.tick_windows(force=True)
    assert p.games["g1"].status is GameStatus.FORBIDDEN
    n = len([c for c in s.calls if "/window/" in c])
    p.tick_windows(force=True)
    assert len([c for c in s.calls if "/window/" in c]) == n, "403 之后不得继续重试"


def test_poller_errors_backoff_then_give_up_to_no_coverage():
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(500, text="boom")
    cfg = PollerConfig(max_consecutive_errors=3, backoff_base_s=1.0)
    p, _, clk = make_poller(routes, cfg)
    p.tick_live(force=True)
    for _ in range(3):
        p.tick_windows(force=True)
        clk.advance(300)
    assert p.games["g1"].status is GameStatus.NO_COVERAGE
    assert p.errors


def test_poller_survives_resolver_exception():
    """解析炸了也不能杀线程，只把该局标 error。"""
    class Boom:
        def resolve_window(self, payload, *, feed_league_name=None):
            raise RuntimeError("解析炸了")

    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(200, load_fixture("window_live.json"))
    c, _ = make_client(routes)
    p = Poller(c, Boom(), None, now=Clock(), sleep=lambda _: None)
    p.tick_live(force=True)
    p.tick_windows(force=True)          # 不抛
    assert p.games["g1"].status is GameStatus.ERROR
    assert any("解析炸了" in e for e in p.errors)


def test_poller_tick_never_raises_even_if_client_explodes():
    class Exploding:
        def live(self):
            raise RuntimeError("网络炸了")

        def window(self, gid):
            raise RuntimeError("网络炸了")

    p = Poller(Exploding(), StubResolver(), None, now=Clock(), sleep=lambda _: None)
    p.tick()                              # 不抛
    assert any("网络炸了" in e for e in p.errors)


def test_poller_marks_api_degraded_when_key_is_invalid():
    """验收⑦：key 无效 → /live 返回空 → 标降级 + 告警；what-if 不经过这里。"""
    routes = {"getLive": FakeResponse(403, text="forbidden")}
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    clk.advance(60)
    p.tick_live(force=True)
    assert p.api_degraded is True
    assert p.snapshot()["api_degraded"] is True
    assert p.health()["n_watched"] == 0


def test_poller_respects_polite_intervals():
    """§7.7.4：getLive ≥ 60 s、window ≥ 30 s，节流由本模块强制，不靠调用方自觉。"""
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, s, clk = make_poller(routes)
    p.tick_live()
    p.tick_windows()
    n_live = len([c for c in s.calls if "getLive" in c])
    n_win = len([c for c in s.calls if "/window/" in c])
    for _ in range(10):                    # 10 次 tick，时钟只走 5 秒
        clk.advance(0.5)
        p.tick_live()
        p.tick_windows()
    assert len([c for c in s.calls if "getLive" in c]) == n_live
    assert len([c for c in s.calls if "/window/" in c]) == n_win
    clk.advance(60)
    p.tick_live()
    p.tick_windows()
    assert len([c for c in s.calls if "getLive" in c]) == n_live + 1


def test_poller_config_rejects_impolite_intervals():
    with pytest.raises(ValueError):
        PollerConfig(live_interval_s=5)
    with pytest.raises(ValueError):
        PollerConfig(window_interval_s=1)


def test_parse_rosters_returns_feed_native_ids():
    """lolesports.parse_rosters 给的是 feed 原生 id，转换由 resolve 做，不能搞混。"""
    r = parse_rosters(load_fixture("window_live.json"))
    assert r["patch"] == "16.17.810.4348"
    assert r["blue_champ_ids"][0] == "Gragas"      # 字符串 id，不是数字 key
    assert r["blue_roles"] == ["top", "jungle", "mid", "bottom", "support"]


def test_feed_result_dataclass_defaults():
    r = FeedResult(FeedState.NOT_STARTED)
    assert r.payload is None and r.http is None and r.detail == ""


def test_no_coverage_still_recovers_if_the_game_finally_starts():
    """标了「无覆盖」之后仍然低频回探，真开打了要能自动恢复。

    理由：手册 §3.3.4 自己实测的那场 LEC game4 空窗 **20 分 50 秒**，
    比验收⑥的 15 分钟阈值还长。前端该在 15 分钟停转圈（验收⑥），
    但后台永久放弃就会漏掉手册自己那个样本。
    """
    payload = load_fixture("window_live.json")
    started = {"yes": False}

    def window(url):
        return FakeResponse(200, payload) if started["yes"] else FakeResponse(204)

    routes = _live_event("m1", ["g1"])
    routes["/window/"] = window
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    for _ in range(40):                    # 20 分钟连续 204
        p.tick_windows(force=True)
        clk.advance(30)
    assert p.games["g1"].status is GameStatus.NO_COVERAGE
    started["yes"] = True                  # 20 分 50 秒后真的开打了
    clk.advance(120)
    p.tick_windows(force=True)
    assert p.games["g1"].status is GameStatus.READY


def test_empty_live_with_healthy_key_is_not_degraded():
    """「今天没比赛」不等于「key 挂了」。两者都是空列表，必须用 getTeams 区分。"""
    routes = {
        "getLive": FakeResponse(200, {"data": {"schedule": {"events": []}}}),
        "getTeams": FakeResponse(200, {"data": {"teams": [{"id": "1"}]}}),
    }
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    clk.advance(1000)
    p.tick_live(force=True)
    assert p.api_degraded is False


def test_finished_games_are_evicted_after_ttl():
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes, PollerConfig(finished_ttl_s=10.0))
    p.tick_live(force=True)
    assert "g1" in p.games
    routes["getEventDetails"] = FakeResponse(
        200, {"data": {"event": {"league": {"name": "LEC"}, "match": {"games": []}}}}
    )
    clk.advance(60)
    p.tick_live(force=True)
    p.tick_live(force=True)
    assert p.games == {}


@needs_data
def test_every_tier1_and_international_feed_league_resolves():
    """一线 + 国际赛的 feed 联赛名一个都不许掉进 UNKNOWN 桶。

    EWC(7月) 与 KeSPA Cup(12月) 整个落在 holdout 段，从数据里数不出来，
    必须由显式常量表兜住——**这是显式映射，不是 fuzzy**。
    """
    from lolwp.live.resolve import FEED_LEAGUE_FALLBACK

    r = Resolver.load()
    for name, canon in FEED_LEAGUE_FALLBACK.items():
        got, _ = r.league(name)
        assert got == canon, f"{name} → {got}，应为 {canon}"
    assert r.league("完全不存在的联赛")[0] == "UNKNOWN"


def test_api_outage_does_not_drop_the_watchlist():
    """getLive 抖一下返回空时，绝不能把正在盯的比赛整批判成已结束。"""
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.WAITING
    routes["getLive"] = FakeResponse(500, text="boom")
    routes["getTeams"] = FakeResponse(200, {"data": {"teams": [{"id": "1"}]}})
    clk.advance(1000)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.WAITING, "API 抖动不得清空 watchlist"
    assert p.api_degraded is False


def test_event_details_outage_does_not_drop_the_watchlist():
    """★ 线上事故回归：getLive 成功但 getEventDetails 抖一下，绝不能判「比赛结束」。

    **[实测 2026-09-19 16:45:48Z]** 修复前的实现只看「本轮 alive 里有没有它」，
    结果那一刻 8 局（CBLOL RED-VKS 4 局 + LIT CG-HMB 4 局）被整批判成 FINISHED，
    而且因为 FINISHED 不会被重新激活，此后整整 1 小时（finished_ttl_s）
    不再轮询——其中 115565671526403084 当时压根还没开打，两小时后另起进程
    才重新看到它并出了预测。

    getEventDetails 失败是**单点故障**，它只说明"这一轮没问到"，
    不说明"这些局没了"。
    """
    routes = _live_event("m1", ["g1", "g2"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    assert {g.status for g in p.games.values()} == {GameStatus.WAITING}

    routes["getEventDetails"] = FakeResponse(500, text="boom")  # 只有 details 挂
    clk.advance(100)
    p.tick_live(force=True)
    assert [p.games[g].status for g in ("g1", "g2")] == [GameStatus.WAITING] * 2, (
        "getEventDetails 单点失败不得把 watchlist 判成已结束"
    )
    # 而且必须还在轮询（FINISHED 会被 tick_windows 跳过）
    clk.advance(100)
    assert p.tick_windows(force=True) == 2


def test_live_event_without_match_id_does_not_evict_anything():
    """getLive 实测会返回 match=None 的占位 event（2026-09-19 的 LCS 那条）。

    它是非空 events，但不携带任何一局的死活信息。旧实现会因为
    `events and gid not in alive` 成立而把整张表判成结束。
    """
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes)
    p.tick_live(force=True)
    routes["getLive"] = FakeResponse(200, {"data": {"schedule": {"events": [
        {"id": "x", "state": "inProgress", "league": {"name": "LCS"}, "match": None},
    ]}}})
    clk.advance(100)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.WAITING


def test_match_vanishing_from_schedule_finishes_only_after_grace():
    """match 整个从 getLive 消失：宽限期内不判死，超过宽限期才判死。

    真结束和 API 抖动在这一层长得一模一样，唯一能区分的信号是「连续消失多久」。
    """
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes, PollerConfig(schedule_grace_s=300.0))
    p.tick_live(force=True)
    # m1 消失，换成另一场毫不相干的比赛（events 非空）
    routes["getLive"] = FakeResponse(200, {"data": {"schedule": {"events": [
        {"id": "m9", "state": "inProgress", "league": {"name": "LCK"},
         "match": {"id": "m9", "teams": [{"code": "T1"}]}},
    ]}}})
    routes["getEventDetails"] = FakeResponse(200, {"data": {"event": {
        "league": {"name": "LCK"}, "match": {"games": []}}}})
    clk.advance(120)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.WAITING, "120 秒 < 宽限期 300 秒，不得判死"
    clk.advance(400)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.FINISHED


def test_completed_in_refreshed_details_finishes_immediately():
    """对照上一条：details **成功读到**且这局不在里面，是权威信息，立刻判死，不等宽限期。"""
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes, PollerConfig(schedule_grace_s=10_000.0))
    p.tick_live(force=True)
    routes["getEventDetails"] = _live_event("m1", ["g1"], state="completed")["getEventDetails"]
    clk.advance(100)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.FINISHED


def test_wrongly_finished_game_is_resurrected_when_it_reappears():
    """误判之后必须能自愈：重新出现在赛程里就恢复轮询，不用等 1 小时 TTL。"""
    routes = _live_event("m1", ["g1"])
    routes["/window/"] = FakeResponse(204)
    p, _, clk = make_poller(routes, PollerConfig(schedule_grace_s=60.0))
    p.tick_live(force=True)
    routes["getLive"] = FakeResponse(200, {"data": {"schedule": {"events": [
        {"id": "m9", "state": "inProgress", "league": {"name": "LCK"},
         "match": {"id": "m9", "teams": []}},
    ]}}})
    routes["getEventDetails"] = FakeResponse(200, {"data": {"event": {
        "league": {"name": "LCK"}, "match": {"games": []}}}})
    clk.advance(200)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.FINISHED
    # m1 又回来了
    back = _live_event("m1", ["g1"])
    routes["getLive"], routes["getEventDetails"] = back["getLive"], back["getEventDetails"]
    clk.advance(100)
    p.tick_live(force=True)
    assert p.games["g1"].status is GameStatus.WAITING
    assert p.games["g1"].n_resurrected == 1
