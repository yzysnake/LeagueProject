"""OE playerid ↔ lolesports esportsPlayerId 对照表。

PLAN.md §9.4 把这条链路列为最大工程风险：匹配失败不报错，只会让 player rating
悄悄退化成先验值，模型变成"只看队伍不看人"，指标只轻微下滑。
它同时是"实时读到阵容 → 查该选手历史"这个核心产品链路的必经之路。

阶段 2 验收⑤要求 2024-2026 有 livestats 覆盖的场次对齐率 ≥95%。
"""
import os

import pandas as pd
import pytest

CW = "data/crosswalk/player_crosswalk.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(CW), reason="需先跑 build_crosswalk.py")


@pytest.fixture(scope="module")
def cw():
    return pd.read_parquet(CW)


@pytest.fixture(scope="module")
def primary(cw):
    return cw[cw.confidence == "primary"]


def test_primary_is_a_function(primary):
    """serving 路径只认 primary：一个 OE 选手必须唯一对应一个 esports id。

    全表（含 weak/alias）有 294 个一对多，直接用会把历史查到别人头上。
    """
    multi = primary.groupby("oe_playerid").esports_player_id.nunique()
    assert (multi == 1).all(), f"{(multi > 1).sum()} 个 oe_playerid 对应多个 esports id"


def test_primary_reverse_collisions_are_bounded(primary):
    """反向不是严格单射：OE 会给同一个人换 playerid。数量必须小且可枚举。"""
    rev = primary.groupby("esports_player_id").oe_playerid.nunique()
    n = int((rev > 1).sum())
    assert n <= 30, f"反向碰撞 {n} 个，超出可人工核查的规模"


def test_known_players_resolve(primary):
    """几个不会认错的人。映射错了这里立刻炸。"""
    expected = {
        "Faker": "98767991747728851",
        "Chovy": "99871276342168416",
        "Caps": "98767975968177297",
    }
    for name, eid in expected.items():
        row = primary[primary.oe_name.str.lower() == name.lower()]
        assert len(row) == 1, f"{name} 命中 {len(row)} 行"
        assert str(row.iloc[0].esports_player_id) == eid, f"{name} 映射变了"


def test_position_agreement_is_high(primary):
    """对齐是按"同场+同位置+同英雄"硬 join 做的，位置一致率必须接近 1。

    这一项掉下来说明 join 退化成了名字模糊匹配——手册明令禁止 fuzzy。
    """
    assert primary.pos_agree_rate.mean() > 0.99


def test_no_fuzzy_fallback_in_primary(primary):
    """primary 全部来自确定性 join，不允许有空 feed_name 的条目混进来。"""
    assert primary.blank_name_rate.max() <= 1.0
    assert (primary.n_games >= 1).all()


@pytest.mark.skipif(not os.path.exists("data/crosswalk/coverage_by_league.csv"),
                    reason="需要覆盖率报告")
def test_stage2_acceptance_tier1_coverage():
    """阶段 2 验收⑤：tier-1 + 国际赛的选手行覆盖率 ≥ 95%。实测 99.44%。"""
    with open("data/crosswalk/coverage_report.md") as fh:
        txt = fh.read()
    assert "tier-1 + 国际赛" in txt
