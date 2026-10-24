"""站点的选择目录与阵容板事件：纯函数，不起 Streamlit。

名单这一层要守住的事：一个位置可以有多名选手（替补、轮换），默认是最近一场的首发；
阵容板发回来的是整份阵容，Python 侧负责校验（未知英雄、同一英雄出现两次、过期事件）。
"""
import json
import os

import pandas as pd
import pytest

from lolwp.app import catalog
from lolwp.app.board import apply_event, board_data
from lolwp.app.catalog import ROLES

T = pd.Timestamp


def _games(rows):
    """rows: (kickoff, league, blue_id, blue_name, red_id, red_name, blue_keys, red_keys)"""
    return pd.DataFrame([{"kickoff_utc": T(k, tz="UTC"), "league_canon": lg, "blue_team_id": b, "blue_team_name": bn,
                          "red_team_id": r, "red_team_name": rn, "blue_champ_keys": bk, "red_champ_keys": rk}
                         for k, lg, b, bn, r, rn, bk, rk in rows])


def _players(rows):
    """rows: (gameid, date, teamid, position, playerid, playername)"""
    return pd.DataFrame(rows, columns=["gameid", "date", "teamid", "position", "playerid", "playername"])


def test_rosters_keep_every_player_per_role_and_default_to_the_last_lineup():
    """替补 Sub 只打过一场、而且是最近那场：默认首发就是他，常规首发 Main 也在名单里。"""
    teams = pd.DataFrame([{"team_id": "A", "last_game": T("2026-09-10", tz="UTC")}])
    rows = []
    for i, day in enumerate(["2026-08-01", "2026-08-20", "2026-09-10"]):
        top = ("pS", "Sub") if day == "2026-09-10" else ("pM", "Main")
        rows += [(f"g{i}", f"{day} 10:00:00", "A", "top", *top)]
        rows += [(f"g{i}", f"{day} 10:00:00", "A", r, f"p_{r}", f"N_{r}") for r in ROLES[1:]]
    rows += [("old", "2024-01-01 10:00:00", "A", "top", "pOld", "Retired")]   # 一年以前：不进名单
    ro = catalog.build_rosters(_players(rows), teams)["A"]
    assert [p["name"] for p in ro["roles"]["top"]] == ["Sub", "Main"]        # 按最近出场排
    assert [p["games"] for p in ro["roles"]["top"]] == [1, 2]
    assert ro["default"] == ["pS", "p_jng", "p_mid", "p_bot", "p_sup"]


def test_rosters_of_a_retired_team_use_its_own_last_year():
    """退役队伍：窗口按它自己的最后一场往前一年，而不是按今天，否则名单是空的。"""
    teams = pd.DataFrame([{"team_id": "OLD", "last_game": T("2019-11-10", tz="UTC")}])
    rows = [("g", "2019-11-10 10:00:00", "OLD", r, f"p{r}", f"n{r}") for r in ROLES]
    ro = catalog.build_rosters(_players(rows), teams)["OLD"]
    assert all(len(ro["roles"][r]) == 1 for r in ROLES) and None not in ro["default"]


def test_home_league_ignores_international_events():
    g = _games([("2026-03-01", "LCK", "T1", "T1", "X", "X", [1] * 5, [2] * 5),
                ("2026-07-01", "MSI", "T1", "T1", "Y", "Y", [1] * 5, [2] * 5),
                ("2026-07-02", "MSI", "T1", "T1", "Y", "Y", [1] * 5, [2] * 5),
                ("2015-01-01", "LCK", "Z", "Zeta Gaming", "X", "X", [1] * 5, [2] * 5)])
    t = catalog.build_teams(g, {"T1": {"code": "T1", "esports_id": "e1"}}).set_index("team_id")
    assert t.loc["T1", "league"] == "LCK"                 # MSI 打得更多，但所属赛区是 LCK
    assert t.loc["Y", "league"] == "MSI"                  # 只打过国际赛：取最常见的
    assert bool(t.loc["T1", "active"]) and not bool(t.loc["Z", "active"])
    assert t.loc["Z", "code"] == "ZG"                     # 没有官方代码：首字母兜底


def test_team_codes_prefer_the_mapping_with_more_games():
    xw = pd.DataFrame({"esports_team_id": ["e1", "e2"], "oe_teamid": ["T", "T"], "n_games": [300, 5]})
    tj = [{"id": "e1", "code": "T1"}, {"id": "e2", "code": "T1A"}]
    assert catalog.team_codes(xw, tj)["T"] == {"code": "T1", "esports_id": "e1"}


def test_role_pools_need_volume_and_share():
    """位置池：该位置 ≥3 场、且占该英雄出场 ≥5%；按该位置出场数降序。"""
    cmap = {"version": "v", "champions": [{"dd_id": n, "dd_key": k, "name": n, "title": ""}
                                          for n, k in [("Ahri", 1), ("Galio", 2), ("Rare", 3)]]}
    rows = [("2026-09-01", "LCK", "A", "A", "B", "B", [None, None, 1, None, None], [None, None, 2, None, None])] * 5
    rows += [("2026-09-02", "LCK", "A", "A", "B", "B", [None, None, 2, None, 3], [None] * 5)] * 2
    ch = catalog.build_champions(cmap, None, _games(rows))
    assert ch["pool"]["mid"] == [2, 1]                    # Galio 7 场 > Ahri 5 场
    assert ch["pool"]["sup"] == []                        # Rare 只有 2 场：不进池
    assert ch["by_key"][3]["n"]["sup"] == 2 and ch["by_key"][1]["zh"] == "Ahri"   # 没有中文表退回英文


def test_chinese_names_only_used_when_versions_match():
    cmap = {"version": "16.18.1", "champions": [{"dd_id": "Akali", "dd_key": 84, "name": "Akali", "title": ""}]}
    zh = {"version": "16.18.1", "champions": {"Akali": {"name": "阿卡丽", "epithet": "离群之刺"}}}
    g = _games([])
    assert catalog.build_champions(cmap, zh, g)["by_key"][84]["zh"] == "阿卡丽"
    assert catalog.build_champions(cmap, {**zh, "version": "16.20.1"}, g)["by_key"][84]["zh"] == "Akali"


def test_strength_percentiles_rank_players():
    pct = catalog.strength_percentiles({"a": -1.0, "b": 0.0, "c": 2.0, "d": 0.5})
    assert pct["c"] == 100 and pct["a"] == 25 and catalog.strength_percentiles({}) == {}


# ---- 阵容板事件 ----


def _sim():
    return {"rev": 3, "seq": 0,
            "blue": {"team": "A", "players": ["a1", "a2", "a3", "a4", "a5"], "champs": [None] * 5},
            "red": {"team": "B", "players": ["b1", "b2", "b3", "b4", "b5"], "champs": [None] * 5}}


def _ev(blue_champs, red_champs, blue_players=None, rev=3, seq=1):
    return {"blue": {"champs": blue_champs, "players": blue_players or ["a1", "a2", "a3", "a4", "a5"]},
            "red": {"champs": red_champs, "players": ["b1", "b2", "b3", "b4", "b5"]}, "rev": rev, "seq": seq}


def test_board_event_applies_the_full_draft():
    sim = _sim()
    changed = apply_event(sim, _ev([1, 2, None, None, None], [None] * 5, ["sub", "a2", "a3", "a4", "a5"]),
                          {1, 2, 3})
    assert changed and sim["blue"]["champs"] == [1, 2, None, None, None]
    assert sim["blue"]["players"][0] == "sub" and sim["seq"] == 1


def test_board_event_rejects_unknown_and_duplicate_champions():
    """一局里同一英雄只能出现一次：后出现的（红方那个）作废；目录里没有的 key 一律丢掉。"""
    sim = _sim()
    apply_event(sim, _ev([1, 999, None, None, None], [1, 2, None, None, None]), {1, 2})
    assert sim["blue"]["champs"] == [1, None, None, None, None]
    assert sim["red"]["champs"] == [None, 2, None, None, None]


def test_stale_board_event_keeps_python_side_players():
    """换队那一刻浏览器里还在点：带旧 rev 的事件只采纳英雄，选手保持新队的首发。"""
    sim = _sim()
    apply_event(sim, _ev([1, None, None, None, None], [None] * 5, ["old"] * 5, rev=2, seq=7), {1})
    assert sim["blue"]["champs"][0] == 1 and sim["blue"]["players"] == ["a1", "a2", "a3", "a4", "a5"]
    assert sim["seq"] == 7


def test_board_event_garbage_is_ignored():
    sim = _sim()
    assert apply_event(sim, None, {1}) is False and apply_event(sim, "x", {1}) is False
    assert apply_event(sim, {"blue": {"champs": "nope"}, "rev": 3}, {1}) is False
    assert sim == _sim()


def test_board_data_carries_rosters_and_draft():
    champs = {"list": [{"key": 1, "id": "Ahri", "en": "Ahri", "zh": "阿狸", "epithet": "九尾妖狐",
                        "icon": None, "n": {r: 0 for r in ROLES}}], "pool": {r: [] for r in ROLES}}
    ro = {"A": {"roles": {"top": [{"id": "a1", "name": "Zeus", "games": 3, "last": "2026-09-01", "pct": 90}]}}}
    d = board_data(_sim(), champs, ro, lambda t: {"name": t, "code": t, "league": "LCK", "logo": None})
    assert d["sides"]["blue"]["players"][0][0]["name"] == "Zeus" and d["sides"]["red"]["players"][0] == []
    assert d["draft"]["blue"]["players"][0] == "a1" and d["rev"] == 3
    json.dumps(d)                                          # 必须能序列化给前端


# ---- 设计系统里带逻辑的那几个 ----


def test_matchup_escapes_names_and_names_the_favourite():
    from lolwp.app import ui
    h = ui.matchup({"name": "<b>A</b>", "code": "A"}, {"name": "B", "code": "B"}, 0.3)
    assert "<b>A</b>" not in h and "&lt;b&gt;A&lt;/b&gt;" in h
    assert "模型看好 <b>B</b>" in h and "30.0%" in h and "70.0%" in h


def test_waterfall_ends_at_the_predicted_probability():
    import math

    from lolwp.app import ui
    steps = [("赛况层", "", 0.01), ("实力层", "", -0.9), ("阵容层", "", 0.2)]
    z = 0.16 + sum(c for _, _, c in steps)
    h = ui.waterfall(0.16, steps)
    assert f"{1 / (1 + math.exp(-z)):.1%}" in h
    ui.waterfall(0.16, [("实力层", "", 9.0)])             # 极端值也不能崩（坐标轴裁到 [0, 1]）


def test_calibration_note_only_in_the_overconfident_band():
    from lolwp.app import ui
    assert ui.calibration_note(0.6) is None
    assert f"{0.8880:.1%}" in ui.calibration_note(0.12) and f"{0.9255:.1%}" in ui.calibration_note(0.97)


@pytest.mark.skipif(not os.path.exists("data/ddragon/champion_map.json"), reason="需先跑 ddragon.sync()")
def test_real_champion_catalog_covers_every_champion():
    with open("data/ddragon/champion_map.json") as fh:
        cmap = json.load(fh)
    g = pd.read_parquet("data/silver/games.parquet") if os.path.exists("data/silver/games.parquet") else _games([])
    ch = catalog.build_champions(cmap, None, g[g.league_tier >= 1] if "league_tier" in g else g)
    assert len(ch["list"]) == cmap["n_champions"]
    assert all(ch["pool"][r] for r in ROLES), "每个位置都应该有常用英雄"


def test_layer_cards_match_the_waterfall_and_add_up():
    """层卡片与瀑布同一口径：截距 + 三层逐段加起来正好等于 最终胜率 − 50%。"""
    import math

    from lolwp.app import ui
    lay = {"实力": 0.62, "阵容": -0.21, "赛况": 0.004}
    steps = ui.layer_steps(0.16, lay)
    p = 1 / (1 + math.exp(-(0.16 + sum(lay.values()))))
    base = (1 / (1 + math.exp(-0.16)) - 0.5) * 100
    assert sum(steps.values()) + base == pytest.approx((p - 0.5) * 100)
    h = ui.waterfall(0.16, [(f"{n}层", "", lay[n]) for n in ui.LAYER_ORDER])
    for n in ui.LAYER_ORDER:
        assert f"{steps[n]:+.1f} pp" in h
