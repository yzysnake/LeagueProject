"""「近期战绩」页背后的台账整理逻辑。"""
import json
import math

import pytest

from lolwp.live.ledger import load_review, summarize


def _write(tmp_path, rows):
    p = tmp_path / "live_watch.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return str(p)


def _pred(gid, p_blue, teams, blue, red, at="2026-10-07 12:00:00", league="WSCI"):
    return {"kind": "prediction", "game_id": gid, "match_id": "m", "recorded_at": at,
            "league": league, "teams": teams, "p_blue": p_blue,
            "blue_names": [f"{blue} p{i}" for i in range(5)],
            "red_names": [f"{red} p{i}" for i in range(5)]}


def _res(gid, blue_won):
    return {"kind": "result", "game_id": gid, "blue_won": blue_won}


def test_sides_come_from_player_names_not_teams_order(tmp_path):
    """换边后 teams 顺序不变但蓝红方反了——必须以选手前缀为准。"""
    df = load_review(_write(tmp_path, [_pred("g4", 0.69, ["MU", "DSG"], "DSG", "MU")]))
    assert (df.loc[0, "blue"], df.loc[0, "red"]) == ("DSG", "MU")


def test_latest_result_wins_and_pending(tmp_path):
    path = _write(tmp_path, [
        _pred("a", 0.7, ["X", "Y"], "X", "Y"), _res("a", None), _res("a", True),
        _pred("b", 0.3, ["X", "Y"], "X", "Y"), _res("b", None),
        _pred("c", 0.6, ["X", "Y"], "X", "Y"),
    ])
    st = load_review(path).set_index("game_id")["status"].to_dict()
    assert st == {"a": "命中", "b": "待定", "c": "待定"}


def test_summarize_ignores_pending(tmp_path):
    path = _write(tmp_path, [
        _pred("a", 0.8, ["X", "Y"], "X", "Y"), _res("a", True),
        _pred("b", 0.8, ["X", "Y"], "X", "Y"), _res("b", False),
        _pred("c", 0.5, ["X", "Y"], "X", "Y"),
    ])
    sm = summarize(load_review(path))
    assert (sm["n"], sm["n_done"], sm["n_pending"]) == (3, 2, 1)
    assert sm["hit_rate"] == 0.5
    assert sm["brier"] == pytest.approx((0.2 ** 2 + 0.8 ** 2) / 2)
    assert sm["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.2)) / 2)


def test_empty_or_missing_ledger(tmp_path):
    df = load_review(str(tmp_path / "nope.jsonl"))
    assert df.empty and summarize(df)["hit_rate"] is None


def _pred_names(gid, teams, blue_names, red_names, at="2026-10-07 12:00:00"):
    return {"kind": "prediction", "game_id": gid, "match_id": "m", "recorded_at": at,
            "league": "X", "teams": teams, "p_blue": 0.6,
            "blue_names": blue_names, "red_names": red_names}


def _sides(tmp_path, *rows):
    df = load_review(_write(tmp_path, list(rows)))
    return dict(zip(df["game_id"], zip(df["blue"], df["red"])))


def test_missing_names_do_not_crash(tmp_path):
    """resolver 对缺位 / 空 summonerName 给 None，watch() 原样写成 null（2024 春 LPL 整场 10 个空名）。"""
    got = _sides(tmp_path,
                 _pred_names("a", ["DK", "T1"], ["DK x", None, "DK y", "DK z", None], [None] * 5),
                 _pred_names("b", ["DK", "T1"], [None] * 5, [None] * 5))
    assert got == {"a": ("DK", "T1"), "b": ("?", "?")}


def test_names_glued_to_the_tag(tmp_path):
    """LPL / DCGI 淘汰赛的 feed 名字不带空格："VITNaakNako"。这局 VIT 在蓝方，赛程里排第二。"""
    got = _sides(tmp_path, _pred_names("g", ["KT", "VIT"],
                                       ["VITNaakNako", "VITLyncas", "VITFIESTA", "VITCarzzy", "VITFleshy"],
                                       ["KTPerfecT", "KTCuzz", "KTBdd", "KTDeft", "KTBeryL"]))
    assert got == {"g": ("VIT", "KT")}


def test_substitute_with_another_tag_is_outvoted(tmp_path):
    """MKF 的替补挂主队 tag "MKOI"：多数票仍是 MKF；且绝不能把同一队同时写成蓝方和红方。"""
    got = _sides(tmp_path, _pred_names("g", ["UCAM", "MKF"],
                                       ["MKOI NightSlayer", "MKF XnS", "MKF Fresskowy", "MKF a", "MKF b"],
                                       [f"UCAM p{i}" for i in range(5)]))
    assert got == {"g": ("MKF", "UCAM")}


def test_one_unreadable_side_is_paired_never_duplicated(tmp_path):
    """蓝方名字认不出、红方是赛程里排第一的队：蓝方只能是另一队，不能按赛程顺序猜成同一队。"""
    got = _sides(tmp_path, _pred_names("g", ["GL", "MKF"], ["a", "b", "c", "d", "e"],
                                       [f"GL p{i}" for i in range(5)]))
    assert got == {"g": ("MKF", "GL")}


def test_reconciled_sides_take_precedence(tmp_path):
    """对账写进 result 的 blue_team / red_team 来自 feed 的 esportsTeamId，比名字前缀可靠。"""
    rows = [_pred_names("g", ["A", "B"], [f"A p{i}" for i in range(5)], [f"B p{i}" for i in range(5)]),
            {"kind": "result", "game_id": "g", "blue_won": True, "blue_team": "B", "red_team": "A"}]
    assert _sides(tmp_path, *rows) == {"g": ("B", "A")}


def test_bad_lines_are_skipped_and_counted(tmp_path):
    p = tmp_path / "live_watch.jsonl"
    good = json.dumps(_pred("a", 0.7, ["X", "Y"], "X", "Y"))
    p.write_text(good + '\n{"kind": "resu\n[1, 2]\n\n' + json.dumps(_res("a", True)) + "\n")
    df = load_review(str(p))
    assert df.attrs["n_bad_lines"] == 2 and df.loc[0, "status"] == "命中"


def test_game_time_and_late_capture(tmp_path):
    """有开局时间就按开局时间排、显示；落盘晚于终局的标 late（赛后补录）。"""
    rows = [_pred("live", 0.7, ["X", "Y"], "X", "Y", at="2026-10-07 21:05:00"),
            {"kind": "result", "game_id": "live", "blue_won": True,
             "started_at": "2026-10-07T21:00:00.000Z", "ended_at": "2026-10-07T21:30:00.000Z"},
            _pred("late", 0.7, ["X", "Y"], "X", "Y", at="2026-10-07 23:59:00"),
            {"kind": "result", "game_id": "late", "blue_won": False,
             "started_at": "2026-10-07T20:00:00.000Z", "ended_at": "2026-10-07T20:30:00.000Z"},
            _pred("pending", 0.7, ["X", "Y"], "X", "Y", at="2026-10-07 22:00:00")]
    df = load_review(_write(tmp_path, rows)).set_index("game_id")
    assert df.loc["live", "late"] is False and df.loc["late", "late"] is True
    assert df.loc["pending", "late"] is None
    assert list(df.index) == ["pending", "live", "late"]   # 按比赛时间，不是落盘时间


def test_same_capture_time_orders_by_game_id(tmp_path):
    """盯盘补抓时几局同一秒落盘：同一系列里 game_id 随局数递增，新的一局排前面。"""
    rows = [_pred(g, 0.5, ["X", "Y"], "X", "Y", at="2026-10-07 23:32:14")
            for g in ("117322971012796390", "117322971012796392", "117322971012796391")]
    assert list(load_review(_write(tmp_path, rows))["game_id"]) == [
        "117322971012796392", "117322971012796391", "117322971012796390"]


def test_backfilled_games_count_as_late_and_carry_their_kickoff(tmp_path):
    """补录的局（盯盘停摆 / 睡眠时整局错过）：对账前就有开局时间，且一律算赛后补录。"""
    bf = dict(_pred("bf", 0.6, ["X", "Y"], "X", "Y", at="2026-10-24 18:00:00"),
              source="backfill", started_at="2026-10-15T18:16:34.661Z")
    df = load_review(_write(tmp_path, [bf])).set_index("game_id")
    assert bool(df.loc["bf", "late"]) and bool(df.loc["bf", "backfill"])
    assert str(df.loc["bf", "game_time"])[:16] == "2026-10-15 18:16"
