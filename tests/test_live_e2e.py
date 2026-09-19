"""实时链路的端到端测试，夹具是真实抓到的比赛。

`tests/fixtures/window_live_lec_g1.json` 是 2026-09-19 10:15:26 从
feed.lolesports.com 抓到的真实 payload（LEC，Karmine Corp vs Movistar KOI，
16,803 字节）。用真数据而不是造的假数据，因为这条链路的 bug 全都藏在
"两边命名不一样"这类细节里，假数据造不出来。
"""
import json
import os

import pandas as pd
import pytest

from lolwp.live.lolesports import parse_rosters
from lolwp.live.resolve_team import build_player_team_index, resolve_team

FIX = "tests/fixtures/window_live_lec_g1.json"
SILVER = "data/silver/games.parquet"
CW = "data/crosswalk/player_crosswalk.parquet"
ART = "models/prod.json"

FEED_TO_OE_ROLE = {"top": "top", "jungle": "jng", "mid": "mid",
                   "bottom": "bot", "support": "sup"}
OE_ORDER = ("top", "jng", "mid", "bot", "sup")

pytestmark = pytest.mark.skipif(not os.path.exists(FIX), reason="需要实时夹具")


@pytest.fixture(scope="module")
def roster():
    with open(FIX) as fh:
        return parse_rosters(json.load(fh))


def test_fixture_is_a_real_payload(roster):
    assert roster["esports_game_id"] == "115548681803406328"
    assert roster["patch"] == "16.17.810.4348"
    for side in ("blue", "red"):
        assert len(roster[f"{side}_player_ids"]) == 5
        assert len(roster[f"{side}_champ_ids"]) == 5


def test_role_names_differ_between_feed_and_oe(roster):
    """**这是本文件存在的首要理由。**

    feed 用 top/jungle/mid/bottom/support，OE 用 top/jng/mid/bot/sup。
    照搬会让三个位置错位，而且不报错——只是特征悄悄算到别人头上。
    """
    feed_roles = set(roster["blue_roles"]) | set(roster["red_roles"])
    assert feed_roles == {"top", "jungle", "mid", "bottom", "support"}
    assert feed_roles != set(OE_ORDER)
    assert all(FEED_TO_OE_ROLE[r] in OE_ORDER for r in feed_roles)


def test_patch_format_differs_from_oe(roster):
    """feed 给四段式 16.17.810.4348，OE 给两段式零填充 16.17。"""
    parts = roster["patch"].split(".")
    assert len(parts) == 4
    assert (int(parts[0]), int(parts[1])) == (16, 17)


@pytest.mark.skipif(not os.path.exists(CW), reason="需要 crosswalk")
def test_all_ten_players_resolve(roster):
    cw = pd.read_parquet(CW)
    prim = cw[cw.confidence == "primary"]
    e2o = dict(zip(prim.esports_player_id.astype(str), prim.oe_playerid))
    ids = roster["blue_player_ids"] + roster["red_player_ids"]
    hit = [e2o.get(str(p)) for p in ids]
    assert all(hit), f"未解析: {[i for i, h in zip(ids, hit) if not h]}"


def test_all_ten_champions_resolve(roster):
    with open("data/ddragon/champion_map.json") as fh:
        cm = json.load(fh)
    by_id = {c["dd_id"]: c["dd_key"] for c in cm["champions"]}
    ids = roster["blue_champ_ids"] + roster["red_champ_ids"]
    assert all(by_id.get(c) for c in ids), f"未解析: {[c for c in ids if not by_id.get(c)]}"


@pytest.mark.skipif(not (os.path.exists(SILVER) and os.path.exists(CW)),
                    reason="需要 silver 与 crosswalk")
def test_team_resolution_is_deterministic_and_correct(roster):
    """队伍靠已解析的选手反推，不允许按队名模糊匹配。"""
    g = pd.read_parquet(SILVER)
    idx = build_player_team_index(g)
    cw = pd.read_parquet(CW)
    prim = cw[cw.confidence == "primary"]
    e2o = dict(zip(prim.esports_player_id.astype(str), prim.oe_playerid))
    for side, expect in (("blue", "Karmine Corp"), ("red", "Movistar KOI")):
        oe = [e2o.get(str(p)) for p in roster[f"{side}_player_ids"]]
        res = resolve_team(oe, idx)
        assert res.oe_team_name == expect, f"{side} 解析成 {res.oe_team_name}"
        assert res.votes == 5 and res.confidence == 1.0


def test_team_resolution_refuses_to_guess():
    """票数不过半时必须返回 None 而不是猜一个。"""
    idx = {"p1": ("t1", "Team One", pd.Timestamp("2026-01-01", tz="UTC")),
           "p2": ("t2", "Team Two", pd.Timestamp("2026-01-01", tz="UTC"))}
    res = resolve_team(["p1", "p2", None, None, None], idx, min_votes=3)
    assert res.oe_team_id is None
    assert "拒绝猜测" in res.detail


def test_team_resolution_handles_all_unknown():
    res = resolve_team([None] * 5, {})
    assert res.oe_team_id is None and res.total == 0


@pytest.mark.skipif(not (os.path.exists(ART) and os.path.exists(SILVER) and os.path.exists(CW)),
                    reason="需要完整环境")
def test_end_to_end_prediction_from_live_payload(roster):
    """验收⑤：真实进行中的比赛，从 window JSON 一路到胜率，零告警。"""
    from lolwp.app.predict import Draft, predict
    from lolwp.models.artifact import Artifact

    g = pd.read_parquet(SILVER)
    idx = build_player_team_index(g)
    cw = pd.read_parquet(CW)
    prim = cw[cw.confidence == "primary"]
    e2o = dict(zip(prim.esports_player_id.astype(str), prim.oe_playerid))
    with open("data/ddragon/champion_map.json") as fh:
        by_id = {c["dd_id"]: c["dd_key"] for c in json.load(fh)["champions"]}

    def side(pfx):
        slot = {}
        for pid, ch, ro in zip(roster[f"{pfx}_player_ids"], roster[f"{pfx}_champ_ids"],
                               roster[f"{pfx}_roles"]):
            slot[FEED_TO_OE_ROLE[ro]] = (e2o.get(str(pid)), by_id.get(ch))
        return [slot[k] for k in OE_ORDER]

    b, r = side("blue"), side("red")
    tb = resolve_team([x[0] for x in b], idx)
    tr = resolve_team([x[0] for x in r], idx)
    art = Artifact.load(ART)
    p = predict(art, Draft("LEC", tb.oe_team_id, tr.oe_team_id,
                           [x[0] for x in b], [x[0] for x in r],
                           [x[1] for x in b], [x[1] for x in r]))
    assert 0.0 < p.p_blue < 1.0
    assert p.warnings == [], f"完整解析不应有告警：{p.warnings}"
    assert abs(sum(p.contributions.values()) + art.intercept - p.logit) < 1e-9
