"""实时盯盘 + 赛后对账的守卫测试。

对账那部分踩过一个真实的坑：`getEventDetails` 的 `games[].teams[]`
**只有 id 和 side，没有逐局胜负**。第一版假设有，结果四条记录全是空值
而且不报错——典型的「不报错不等于对」。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "reports" / "live_watch.jsonl"
SCRIPT = ROOT / "scripts" / "live_watch.py"


def test_script_documents_the_api_limitation():
    """那个坑必须写在代码里，否则下一个人还会踩。"""
    src = SCRIPT.read_text()
    assert "没有逐局胜负" in src, "对账逻辑没写明 API 的这个限制"
    assert "终局帧" in src and "系列比分" in src, "没说明逐局胜负的两路来源"


def test_ledger_is_append_only_jsonl():
    if not LEDGER.exists():
        pytest.skip("尚无盯盘记录")
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    assert rows
    for r in rows:
        assert r.get("kind") in {"prediction", "result"}, f"未知记录类型 {r.get('kind')}"
        if r["kind"] == "prediction":
            assert 0.0 <= r["p_blue"] <= 1.0
            assert r.get("game_id") and r.get("match_id")


def test_predictions_are_not_duplicated():
    """脚本对已记录的 gameId 会跳过，launchd 崩溃重启时不能重复写。"""
    if not LEDGER.exists():
        pytest.skip("尚无盯盘记录")
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    ids = [r["game_id"] for r in rows if r["kind"] == "prediction"]
    assert len(ids) == len(set(ids)), f"预测被重复写入：{len(ids)} 条 / {len(set(ids))} 个唯一"


def test_results_only_recorded_when_unambiguous():
    """终局帧判不出、且系列比分也推不出时，blue_won 必须留空而不是猜一个；
    判出的必须注明来源。"""
    if not LEDGER.exists():
        pytest.skip("尚无盯盘记录")
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    for r in rows:
        if r["kind"] == "result" and r.get("blue_won") is not None:
            assert isinstance(r["blue_won"], bool)
            if "source" in r:
                assert r["source"] in {"feed", "series", "feed+series"}


@pytest.mark.skipif(not (ROOT / "deploy" / "com.lolwp.livewatch.plist").exists(),
                    reason="无 plist")
def test_livewatch_plist_is_restartable():
    """盯盘要跑十几个小时，崩了必须自动重启；脚本对已记录的局会跳过，天然幂等。"""
    import plistlib
    p = ROOT / "deploy" / "com.lolwp.livewatch.plist"
    d = plistlib.loads(p.read_text().replace("__REPO__", str(ROOT)).encode())
    assert d["Label"] == "com.lolwp.livewatch"
    assert d["RunAtLoad"] is True
    # 必须无条件重启：脚本每个窗口结束都 exit 0，SuccessfulExit=false 会让它一去不回
    assert d["KeepAlive"] is True, "正常退出后不会重新盯盘"


def _load_lw():
    import importlib.util
    spec = importlib.util.spec_from_file_location("live_watch", SCRIPT)
    lw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lw)
    return lw


def test_series_blue_won_only_when_determined():
    """系列比分只在「最后一局」或「零封」时能定逐局胜负，其余必须返回 None。"""
    lw = _load_lw()

    def g(n, state="completed"):
        return {"id": f"g{n}", "number": n, "state": state}
    teams = [{"id": "A", "result": {"gameWins": 3}}, {"id": "B", "result": {"gameWins": 1}}]
    games = [g(1), g(2), g(3), g(4), g(5, "unneeded")]
    assert lw._series_blue_won(games[0], games, teams, "A") is None    # 3-1 中间局：不可定
    assert lw._series_blue_won(games[3], games, teams, "B") is False   # 最后一局：A 胜，蓝方是 B
    assert lw._series_blue_won(games[3], games, teams, "A") is True
    sweep = [{"id": "A", "result": {"gameWins": 2}}, {"id": "B", "result": {"gameWins": 0}}]
    assert lw._series_blue_won(games[0], games[:2], sweep, "A") is True  # 零封：每局都是 A
    live = games[:3] + [g(4, "inProgress")]
    assert lw._series_blue_won(games[0], live, teams, "A") is None     # 系列没打完
    assert lw._series_blue_won(games[3], games, teams, "X") is None    # feed 的队伍不在本场：无从对齐
    assert lw._series_blue_won(games[3], games, teams, None) is None


# ---- 用假的 lolesports 接口把 reconcile 整个跑一遍（不打网络） ----

def _iso(minute):
    return f"2026-10-07T21:{minute:02d}:00.000Z"


def _frame(minute, state="in_game", b_inhib=0, r_inhib=0, b_gold=None, r_gold=None):
    return {"rfc460Timestamp": _iso(minute), "gameState": state,
            "blueTeam": {"inhibitors": b_inhib, "totalGold": b_gold},
            "redTeam": {"inhibitors": r_inhib, "totalGold": r_gold}}


def _window(blue_id, red_id, end=30, winner=None):
    """feed 的 window：首帧在第 0 分钟、终局在 end 分钟。winner=None 表示 feed 停在对局中途、
    没有 finished 帧；跳过终局后 feed 一直回最后的帧（实测如此）。"""
    def payload(starting):
        md = {"blueTeamMetadata": {"esportsTeamId": blue_id},
              "redTeamMetadata": {"esportsTeamId": red_id}}
        if starting is None:
            return {"gameMetadata": md, "frames": [_frame(0)]}
        t = min(int(starting[14:16]), end)
        if t < end or winner is None:
            return {"gameMetadata": md, "frames": [_frame(t)]}
        return {"gameMetadata": md, "frames": [_frame(end, "finished",
                                                       b_inhib=int(winner == "blue"),
                                                       r_inhib=int(winner == "red"),
                                                       b_gold=61000, r_gold=52000)]}
    return payload


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._p = status, payload
        self.content = b"{}" if payload is not None else b""

    def json(self):
        return self._p


class _FakeSession:
    def __init__(self, events, windows):
        self.headers, self.events, self.windows, self.calls = {}, events, windows, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("/getEventDetails"):
            return _Resp(200, {"data": {"event": self.events[params["id"]]}})
        w = self.windows.get(url.rsplit("/", 1)[-1])
        return _Resp(204) if w is None else _Resp(200, w((params or {}).get("startingTime")))


def test_feed_game_reads_sides_times_and_winner():
    lw = _load_lw()
    s = _FakeSession({}, {"g": _window("B", "A", end=30, winner="blue")})
    fg = lw._feed_game(s, "g")
    assert (fg["blue_id"], fg["red_id"], fg["blue_won"]) == ("B", "A", True)
    assert (fg["started_at"], fg["ended_at"]) == (_iso(0), _iso(30))


def test_feed_game_stops_when_feed_ends_without_finished_frame():
    """feed 停在对局中途时不能一路跳满 2 小时：帧不再前进就停，胜负留空。"""
    lw = _load_lw()
    s = _FakeSession({}, {"g": _window("B", "A", end=30, winner=None)})
    fg = lw._feed_game(s, "g")
    assert fg["blue_won"] is None and fg["ended_at"] is None
    assert len(s.calls) <= 10
    assert lw._feed_game(_FakeSession({}, {}), "nope") is None   # 204：拿不到 window


def _event(games, wins):
    return {"match": {"teams": [{"id": "A", "code": "AAA", "name": "Team A",
                                 "result": {"gameWins": wins[0]}},
                                {"id": "B", "code": "BBB", "name": "Team B",
                                 "result": {"gameWins": wins[1]}}],
                      "games": [{"id": gid, "number": i + 1, "state": "completed",
                                 # ⚠️ 故意和 feed 反着写：getEventDetails 的 side 不可信
                                 "teams": [{"id": "A", "side": "blue"}, {"id": "B", "side": "red"}]}
                                for i, gid in enumerate(games)]}}


def _pred(gid, mid, p_blue=0.4):
    return {"kind": "prediction", "game_id": gid, "match_id": mid, "recorded_at": "2026-10-07 21:05:00",
            "league": "TEST", "teams": ["AAA", "BBB"], "p_blue": p_blue,
            "blue_names": [f"BBB p{i}" for i in range(5)], "red_names": [f"AAA p{i}" for i in range(5)]}


def test_reconcile_uses_feed_sides_not_event_sides(tmp_path, monkeypatch):
    """B 2-0 零封，feed 里 B 在蓝方，getEventDetails 却说 A 在蓝方。
    g1 的 feed 没有 finished 帧，只能靠系列比分：必须按 feed 的蓝方算成「蓝胜」，
    按 event 的 side 会写成「红胜」——命中与否、log loss、Brier 全部翻过来。
    g2 两路都有答案且一致，不能因为 side 对不上被误判成 conflict。"""
    import requests

    lw = _load_lw()
    ledger = tmp_path / "live_watch.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [_pred("g1", "m1"), _pred("g2", "m1")]) + "\n")
    monkeypatch.setattr(lw, "LEDGER", str(ledger))
    fake = _FakeSession({"m1": _event(["g1", "g2"], (0, 2))},
                        {"g1": _window("B", "A", winner=None), "g2": _window("B", "A", winner="blue")})
    monkeypatch.setattr(requests, "Session", lambda: fake)
    monkeypatch.setattr(lw, "_resolver", lambda: None)
    lw.reconcile(pause_s=0)

    res = {r["game_id"]: r for r in map(json.loads, ledger.read_text().splitlines())
           if r["kind"] == "result"}
    assert res["g1"]["blue_won"] is True and res["g1"]["source"] == "series"
    assert res["g2"]["blue_won"] is True and res["g2"]["source"] == "feed+series"
    for r in res.values():
        assert (r["game_winner"], r["blue_team"], r["red_team"]) == ("Team B", "BBB", "AAA")
        assert r["started_at"] == _iso(0)


def test_reconcile_backfills_old_results_without_touching_the_verdict(tmp_path, monkeypatch):
    """早期判出胜负、但没带 feed 蓝红方与时间的结果：补一条带字段的记录，胜负照旧；
    终局帧和旧胜负矛盾时不补，留给人看。坏行不能让对账崩掉。"""
    import requests

    lw = _load_lw()
    ledger = tmp_path / "live_watch.jsonl"
    old = [_pred("g1", "m1"), _pred("g2", "m1"),
           {"kind": "result", "game_id": "g1", "blue_won": True, "p_blue": 0.4},
           {"kind": "result", "game_id": "g2", "blue_won": False, "p_blue": 0.4}]
    ledger.write_text("\n".join(json.dumps(r) for r in old) + "\n{\"kind\": \"resu\n")
    monkeypatch.setattr(lw, "LEDGER", str(ledger))
    fake = _FakeSession({"m1": _event(["g1", "g2"], (0, 2))},
                        {"g1": _window("B", "A", winner="blue"), "g2": _window("B", "A", winner="blue")})
    monkeypatch.setattr(requests, "Session", lambda: fake)
    monkeypatch.setattr(lw, "_resolver", lambda: None)
    lw.reconcile(pause_s=0)

    _, results = lw._load_ledger()
    assert results["g1"]["blue_won"] is True and results["g1"]["ended_at"] == _iso(30)
    assert "ended_at" not in results["g2"]          # 矛盾：没补
    assert lw._seen_games() == {"g1", "g2"}


# ---- 赛后补录：盯盘停摆 / 机器睡眠时整局错过的局，事后从赛程补出来 ----

_TRAINED_ON = "2026-05-31 21:55:09+00:00"


def _sched_event(mid, start, state="completed", league="Worlds"):
    return {"startTime": start, "state": state, "type": "match", "league": {"name": league},
            "match": {"id": mid, "teams": [{"code": "AAA"}, {"code": "BBB"}]}}


def _details(mid, games, league="Worlds"):
    return {"league": {"name": league},
            "match": {"id": mid, "teams": [{"id": "A", "code": "AAA"}, {"id": "B", "code": "BBB"}],
                      "games": [{"id": gid, "number": i + 1, "state": st}
                                for i, (gid, st) in enumerate(games)]}}


class _SchedSession(_FakeSession):
    """getSchedule 两页（新 → 旧），其余同 _FakeSession。"""
    def __init__(self, pages, events, windows=None):
        super().__init__(events, windows or {})
        self.pages = pages

    def get(self, url, params=None, timeout=None):
        if url.endswith("/getSchedule"):
            self.calls.append((url, params))
            i = 0 if not (params or {}).get("pageToken") else int(params["pageToken"])
            older = str(i + 1) if i + 1 < len(self.pages) else None
            return _Resp(200, {"data": {"schedule": {"events": self.pages[i],
                                                     "pages": {"older": older}}}})
        return super().get(url, params, timeout)


class _FakeFeed:
    """pol.client.window：gid → 首帧时间（200）；不在表里 → 204。"""
    def __init__(self, starts):
        self.starts, self.asked = starts, []

    def window(self, gid):
        from lolwp.live.lolesports import FeedResult, FeedState
        self.asked.append(gid)
        if gid not in self.starts:
            return FeedResult(FeedState.NOT_STARTED, http=204)
        return FeedResult(FeedState.READY, http=200, payload={
            "gameMetadata": {"blueTeamMetadata": {}, "redTeamMetadata": {}},
            "frames": [{"rfc460Timestamp": self.starts[gid]}]})


class _FakePol:
    def __init__(self, starts):
        from types import SimpleNamespace
        self.artifact = SimpleNamespace(meta={"trained_on": _TRAINED_ON})
        self.client = _FakeFeed(starts)
        self.nows = {}

    def predict_window(self, payload, league, now=None):
        gid = next(g for g, t in self.client.starts.items()
                   if t == payload["frames"][0]["rfc460Timestamp"])
        self.nows[gid] = now
        return ({"p_blue": 0.42, "layers": {"实力": -0.3}, "blue_champ_keys": [1] * 5,
                 "red_champ_keys": [2] * 5, "blue_names": ["BBB x"] * 5,
                 "red_names": ["AAA y"] * 5}, ["w"])


def _backfill_world(tmp_path, monkeypatch, ledger_rows=()):
    from datetime import UTC, datetime

    lw = _load_lw()
    ledger = tmp_path / "live_watch.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in ledger_rows))
    monkeypatch.setattr(lw, "LEDGER", str(ledger))
    lw._SKIP_GAMES.clear()
    lw._SETTLED_MATCHES.clear()
    pages = [
        [_sched_event("m1", "2026-10-15T18:00:00Z"),                     # 打完的 BO5：补
         _sched_event("m3", "2026-10-24T19:00:00Z", state="unstarted"),  # 没开打：不碰
         _sched_event("m4", "2026-10-16T10:00:00Z", league="LCK")],      # 别的赛事
        [_sched_event("m2", "2026-10-01T18:00:00Z"),                     # 早于 since：不碰
         _sched_event("m5", "2026-05-20T18:00:00Z")],
    ]
    events = {"m1": _details("m1", [("g1", "completed"), ("g2", "completed"),
                                    ("g3", "completed"), ("g4", "unneeded")]),
              "m4": _details("m4", [("g7", "completed")], league="LCK")}
    s = _SchedSession(pages, events)
    pol = _FakePol({"g1": "2026-10-15T18:16:34.661Z", "g2": "2026-10-15T19:14:55.707Z",
                    "g7": "2026-10-16T10:20:00.000Z"})
    since = datetime(2026, 10, 10, tzinfo=UTC)
    return lw, ledger, s, pol, since


def test_backfill_fills_only_completed_games_the_model_never_saw(tmp_path, monkeypatch):
    """g1 补；g2 台账里已有；g3 打完了 feed 还是 204（无覆盖）；g4 unneeded；
    m3 没开打；m2 早于 since。补出来的那条站在开局时刻预测，并标 source=backfill。"""
    lw, ledger, s, pol, since = _backfill_world(
        tmp_path, monkeypatch, [_pred("g2", "m1")])
    assert lw.backfill(pol, s, since, league="Worlds", pause_s=0) == 1
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    new = rows[-1]
    assert (new["game_id"], new["match_id"], new["source"]) == ("g1", "m1", "backfill")
    assert new["started_at"] == "2026-10-15T18:16:34.661Z" and new["league"] == "Worlds"
    assert new["teams"] == ["AAA", "BBB"] and new["p_blue"] == 0.42
    assert str(pol.nows["g1"])[:16] == "2026-10-15 18:16"
    assert pol.client.asked == ["g1", "g3"]                    # g2/g4/m3/m2/LCK 都没去问 feed
    assert not any(c[1] and c[1].get("id") in {"m2", "m3", "m4"} for c in s.calls)

    # 再跑一遍：g1 已在台账、g3 已知无覆盖、m1 已清完——不重复写，也不再去问
    assert lw.backfill(pol, s, since, league="Worlds", pause_s=0) == 0
    assert pol.client.asked == ["g1", "g3"]
    assert lw._seen_games() == {"g1", "g2"}


def test_backfill_skips_games_inside_the_training_window(tmp_path, monkeypatch):
    """开局早于 artifact 训练截止的局：模型训练时可能见过它，补出来的不是样本外预测。"""
    from datetime import UTC, datetime

    lw, ledger, s, pol, _ = _backfill_world(tmp_path, monkeypatch)
    pol.client.starts["g9"] = "2026-05-20T18:10:00.000Z"
    s.events.update(m2=_details("m2", []), m5=_details("m5", [("g9", "completed")]))
    assert lw.backfill(pol, s, datetime(2026, 5, 1, tzinfo=UTC), pause_s=0) == 3
    ids = {json.loads(x)["game_id"] for x in ledger.read_text().splitlines()}
    assert ids == {"g1", "g2", "g7"}                          # 不带 league 时别的赛事也补；g9 不补


def test_backfill_does_not_duplicate_a_game_another_process_just_wrote(tmp_path, monkeypatch):
    """make backfill 与盯盘并行：读赛程之后、落盘之前盯盘刚写下 g1，补录不能再写一遍。"""
    lw, ledger, s, pol, since = _backfill_world(tmp_path, monkeypatch)
    orig = pol.predict_window

    def racing(payload, league, now=None):
        out = orig(payload, league, now)
        if payload["frames"][0]["rfc460Timestamp"] == pol.client.starts["g1"]:
            with open(ledger, "a") as fh:
                fh.write(json.dumps(_pred("g1", "m1")) + "\n")
        return out
    pol.predict_window = racing
    lw.backfill(pol, s, since, league="Worlds", pause_s=0)
    ids = [json.loads(x)["game_id"] for x in ledger.read_text().splitlines()]
    assert ids.count("g1") == 1


def test_nap_wakes_promptly_after_the_machine_slept(monkeypatch):
    """合盖睡眠：time.sleep 冻结、墙钟照走。醒来后 _nap 要在 1 秒切片内返回，并报出真实间隔。"""
    lw = _load_lw()
    clock = {"t": 1000.0}
    naps = []

    def fake_sleep(dt):
        naps.append(dt)
        clock["t"] += dt + (900 if len(naps) == 3 else 0)   # 第 3 片里睡了 15 分钟
    monkeypatch.setattr(lw.time, "time", lambda: clock["t"])
    monkeypatch.setattr(lw.time, "sleep", fake_sleep)
    gap = lw._nap(30)
    assert gap > lw.WAKE_GAP_S and len(naps) == 3 and max(naps) <= 1.0


def test_poller_measures_time_with_the_wall_clock():
    """monotonic（mach_absolute_time）在 Mac 睡眠时不走，节流会在 DarkWake 里挡住 getLive。"""
    import inspect
    import time

    from lolwp.live.poller import Poller
    assert inspect.signature(Poller).parameters["now"].default is time.time


# ---- 实力快照随赛果重解：对账记阵容、补录守住快照截止、重解失败不拖垮盯盘 ----

class _FakeResolver:
    """按 feed 的 esportsTeamId 拼出 OE id；A 的对照是 contested。"""
    def resolve_window(self, payload, feed_league_name=None):
        from types import SimpleNamespace as NS
        md = payload["gameMetadata"]

        def side(key):
            tid = md[key]["esportsTeamId"]
            return NS(oe_team_id=f"oe:team:{tid}", team_confidence="contested" if tid == "A" else "primary",
                      player_ids=[f"oe:player:{tid}{i}" for i in range(5)])
        return NS(league_canon="WLDs", blue=side("blueTeamMetadata"), red=side("redTeamMetadata"))


def test_reconcile_records_lineup_and_final_gold(tmp_path, monkeypatch):
    """对账把这局的 BP 后阵容解析成 OE id 写进 result（给实力快照重解），顺带记终局双方总经济。
    已判出、缺阵容的老结果只补字段；补齐之后再对账一条都不追加。"""
    import requests

    lw = _load_lw()
    ledger = tmp_path / "live_watch.jsonl"
    old = [_pred("g1", "m1"), _pred("g2", "m1"),
           {"kind": "result", "game_id": "g1", "blue_won": True, "p_blue": 0.4,
            "started_at": _iso(0), "ended_at": _iso(30)}]          # 老结果：有时间、没阵容
    ledger.write_text("".join(json.dumps(r) + "\n" for r in old))
    monkeypatch.setattr(lw, "LEDGER", str(ledger))
    fake = _FakeSession({"m1": _event(["g1", "g2"], (0, 2))},
                        {"g1": _window("B", "A", winner="blue"), "g2": _window("B", "A", winner="blue")})
    monkeypatch.setattr(requests, "Session", lambda: fake)
    assert lw.reconcile(_FakeResolver(), quiet=True, pause_s=0) == 2   # g2 新判出 + g1 补阵容

    _, res = lw._load_ledger()
    for gid in ("g1", "g2"):
        r = res[gid]
        assert r["blue_won"] is True and r["league_canon"] == "WLDs"
        assert (r["blue_oe_team"], r["blue_team_conf"]) == ("oe:team:B", "primary")
        assert (r["red_oe_team"], r["red_team_conf"]) == ("oe:team:A", "contested")
        assert r["blue_oe_players"] == [f"oe:player:B{i}" for i in range(5)]
        assert (r["blue_gold"], r["red_gold"]) == (61000, 52000)
    n_lines = len(ledger.read_text().splitlines())
    assert lw.reconcile(_FakeResolver(), quiet=True, pause_s=0) == 0
    assert len(ledger.read_text().splitlines()) == n_lines


def test_backfill_skips_games_older_than_the_rating_snapshot(tmp_path, monkeypatch):
    """快照重解之后，开局早于快照里最晚一局终局的局不补：补出来的预测会用上它开打之后的赛果。"""
    lw, ledger, s, pol, since = _backfill_world(tmp_path, monkeypatch)
    pol.artifact.meta["ratings_cutoff"] = "2026-10-15 19:00:00+00:00"   # g1（18:16）之后、g2 之前
    assert lw.backfill(pol, s, since, league="Worlds", pause_s=0) == 1
    ids = [json.loads(x)["game_id"] for x in ledger.read_text().splitlines()]
    assert ids == ["g2"]


def test_record_carries_the_ratings_snapshot():
    lw = _load_lw()
    r = lw._record({"game_id": "g"}, {"p_blue": 0.5}, "BT4_LR@2026-09-18", "2026-10-24 21:30")
    assert (r["model"], r["ratings"]) == ("BT4_LR@2026-09-18", "2026-10-24 21:30")


def test_refresh_ratings_swaps_the_artifact_and_never_raises(capsys):
    """有新快照就换上并报一句；没新比赛不出声；重解出错只打日志——盯盘不能因此停。"""
    from types import SimpleNamespace as NS

    lw = _load_lw()

    class Pol:
        def __init__(self):
            self.artifact, self.swaps = NS(meta={"ratings_as_of": "2026-09-18 23:20:30+00:00"}), 0

        def reload_artifact_if_changed(self):
            self.swaps += 1
            self.artifact = NS(meta={"ratings_as_of": "2026-10-24 21:00:01+00:00", "n_live_games": 57})
            return True

    pol = Pol()
    msg = lw._refresh_ratings(pol, refresh=lambda: object())
    assert "2026-10-24 21:00" in msg and "57" in msg and pol.swaps == 1
    assert lw._refresh_ratings(pol, refresh=lambda: None) is None and pol.swaps == 1

    def boom():
        raise ValueError("bad silver")
    assert lw._refresh_ratings(pol, refresh=boom) is None
    assert "bad silver" in capsys.readouterr().out
