"""阵容模拟页与 What-if 页：用真实 artifact + 真实名单跑真实页面。

阵容板本身是浏览器里的组件（AppTest 跑不了 JS），所以这里分两层：
组件发事件之后 Python 侧怎么落（test_app_catalog.py 测 apply_event），
以及 st.session_state["sim"] 变了之后页面算出的概率、提示、what-if 是否跟着变（本文件）。
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "src" / "lolwp" / "app" / "main.py")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(ROOT / "models" / "prod.json") and os.path.exists(ROOT / "data" / "bronze" / "all.parquet")),
    reason="需先跑 artifact.train() 与 make bronze")


def _boom(*a, **k):
    raise RuntimeError("离线")


@pytest.fixture
def at():
    with patch("lolwp.live.poller.build_default_poller", _boom):
        a = AppTest.from_file(APP, default_timeout=300)
        a.run()
        yield a


def _board(at) -> dict:
    return json.loads(at.get("bidi_component")[0].proto.json)


def _html(at) -> str:
    """不含全局样式与开场脚本（那两段里有所有 class 名）。"""
    return "\n".join(e.proto.body for e in at.get("html")
                     if not e.proto.body.lstrip().startswith(("<style", "<script")))


def _p_blue(at) -> str:
    h = _html(at)
    i = h.index('class="p blue')
    return h[h.index(">", i) + 1:h.index("%", i) + 1]


def test_board_gets_every_champion_role_pools_and_both_rosters(at):
    assert len(at.exception) == 0
    d = _board(at)
    assert len(d["champs"]) >= 170 and all(d["pool"][r] for r in ("top", "jng", "mid", "bot", "sup"))
    for side in ("blue", "red"):
        roles = d["sides"][side]["players"]
        assert all(roles), f"{side} 有位置没有任何候选选手"
        sel = d["draft"][side]["players"]
        assert all(any(p["id"] == pid for p in lst) for pid, lst in zip(sel, roles)), "默认首发不在名单里"
    # 没选英雄时阵容层只剩「全按位置基准」的那一点偏移，一个也没选应提示 10 个空位
    assert "10 个位置待选" in _html(at)


def test_substitute_changes_the_probability(at):
    """同一个位置换成替补，实力层要跟着变——这就是「选手 ID 可选」的意义。"""
    d = _board(at)
    lst = next(lst for lst in d["sides"]["blue"]["players"] if len(lst) > 1)
    slot = d["sides"]["blue"]["players"].index(lst)
    before = _p_blue(at)
    sim = at.session_state["sim"]
    sub = next(p["id"] for p in lst if p["id"] != sim["blue"]["players"][slot])
    sim["blue"]["players"][slot] = sub
    at.run()
    assert len(at.exception) == 0 and _p_blue(at) != before


def test_picked_champions_feed_the_prediction_and_what_if(at):
    sim = at.session_state["sim"]
    d = _board(at)
    sim["blue"]["champs"] = [d["pool"][r][0] for r in ("top", "jng", "mid", "bot", "sup")]
    sim["red"]["champs"] = [d["pool"][r][1] for r in ("top", "jng", "mid", "bot", "sup")]
    at.run()
    assert len(at.exception) == 0
    assert "个位置待选" not in _html(at)
    draft = at.session_state["draft"]
    assert list(draft.blue_champ_keys) == sim["blue"]["champs"]

    at.switch_page("views/whatif.py").run()
    assert len(at.exception) == 0
    labels = {m.label for m in at.metric}
    assert {"当前蓝色方胜率", "最多提升", "最多下降"} <= labels
    # 候选里不能有本局已被选走的英雄（选项是展示文案「中文 · English（Δ）」）
    with open(ROOT / "data" / "ddragon" / "champion_map.json") as fh:
        en = {c["dd_key"]: c["name"] for c in json.load(fh)["champions"]}
    picked = {en[k] for k in sim["blue"]["champs"] + sim["red"]["champs"]}
    opts = at.selectbox(key="wi_pick").options
    assert opts and not any(o.split(" · ", 1)[1].rsplit("（", 1)[0] in picked for o in opts)


def test_what_if_apply_writes_back_to_the_simulator(at):
    sim = at.session_state["sim"]
    at.switch_page("views/whatif.py").run()
    pick = at.selectbox(key="wi_pick").value
    rev = sim["rev"]
    next(b for b in at.button if b.label == "应用到阵容模拟").click().run()
    sim = at.session_state["sim"]
    assert sim["blue"]["champs"][2] == int(pick) and sim["rev"] == rev + 1   # 默认换蓝方中单


def test_changing_team_resets_players_but_keeps_champions(at):
    sim = at.session_state["sim"]
    sim["blue"]["champs"][0] = 266
    old_team, rev = sim["blue"]["team"], sim["rev"]
    from lolwp.app import data
    t = data.teams()                                       # 选项是展示文案，set_value 要的是 team_id
    new_team = next(x for x in t.loc[t["active"], "team_id"] if x not in (old_team, sim["red"]["team"]))
    at.selectbox(key="sim_blue_team").set_value(new_team).run()
    sim = at.session_state["sim"]
    assert sim["blue"]["team"] == new_team and sim["rev"] == rev + 1
    assert sim["blue"]["champs"][0] == 266
    assert sim["blue"]["players"] == data.rosters()[new_team]["default"]   # 新队的最近一场首发
    assert _board(at)["draft"]["blue"]["players"] == sim["blue"]["players"]


def test_swap_moves_teams_players_and_champions(at):
    sim = at.session_state["sim"]
    sim["blue"]["champs"][1] = 64
    blue, red = dict(sim["blue"]), dict(sim["red"])
    at.button(key="sim_swap").click().run()
    sim = at.session_state["sim"]
    assert sim["blue"]["team"] == red["team"] and sim["red"]["team"] == blue["team"]
    assert sim["red"]["champs"][1] == 64
    assert at.selectbox(key="sim_blue_team").value == red["team"]


def test_same_team_on_both_sides_is_refused_politely(at):
    sim = at.session_state["sim"]
    at.selectbox(key="sim_red_team").set_value(sim["blue"]["team"]).run()
    assert len(at.exception) == 0
    assert "同一支队伍" in _html(at)


def test_team_choice_survives_a_page_round_trip(at):
    """切到 What-if 再回来：多页应用里不渲染的控件值默认会被丢掉，下拉框回到第一项，
    而阵容板还是原来的队——两边对不上。控件用 persist_state="session" 守住。"""
    from lolwp.app import data
    t = data.teams()
    sim = at.session_state["sim"]
    new_team = next(x for x in t.loc[t["active"], "team_id"] if x not in (sim["blue"]["team"], sim["red"]["team"]))
    at.selectbox(key="sim_blue_team").set_value(new_team).run()
    at.switch_page("views/whatif.py").run()
    at.switch_page("views/simulate.py").run()
    assert len(at.exception) == 0
    assert at.selectbox(key="sim_blue_team").value == new_team == at.session_state["sim"]["blue"]["team"]
    assert _board(at)["sides"]["blue"]["team"]["name"] == data.team_row(new_team)["name"]
