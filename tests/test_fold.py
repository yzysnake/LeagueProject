"""12 行 → 1 行折叠的正确性。

按行切分 = 标签泄漏（PLAN.md §2.2 列为"三条泄漏通道里最蠢也最常见的一条"），
所以折叠的唯一性断言必须是硬测试，不是注释。
"""
import os

import pandas as pd
import pytest

from lolwp.store.fold import ROLES, fold_games, load_champ_key_map

BRONZE = "data/bronze/all.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(BRONZE), reason="需先跑 bronze.build()")


@pytest.fixture(scope="module")
def games():
    # 只取两年，测试要快；正确性与年份无关
    b = pd.read_parquet(BRONZE)
    b = b[b.gameid.isin(b[b.date.str[:4].isin(["2025", "2026"])].gameid.unique())]
    return fold_games(b)


def test_one_row_per_game(games):
    assert games.game_id.nunique() == len(games)


def test_label_is_binary_and_complementary(games):
    assert set(games.y_blue_win.unique()) <= {0, 1}
    assert games.y_blue_win.notna().all()


def test_blue_winrate_in_plausible_range(games):
    """实测全量 0.5327。偏离太远说明蓝红方向搞反了或过滤出了偏倚样本。"""
    wr = games.y_blue_win.mean()
    assert 0.50 < wr < 0.57, f"蓝方胜率 {wr:.4f} 不合理"


def test_five_slots_per_side(games):
    for col in ["blue_player_ids", "red_player_ids", "blue_champ_keys", "red_champ_keys"]:
        assert games[col].map(len).eq(5).all(), f"{col} 不是定长 5"


def test_champion_keys_resolve(games):
    """英雄 key 必须全部落在 Data Dragon 的合法 key 集合里。"""
    valid = set(load_champ_key_map().values())
    for col in ["blue_champ_keys", "red_champ_keys"]:
        for keys in games[col]:
            for k in keys:
                assert pd.isna(k) or int(k) in valid, f"非法英雄 key {k}"


def test_mirror_picks_are_rare_and_flagged(games):
    """镜像英雄是真实存在的特殊赛制，不是数据错误，但必须被标记出来。

    实测 2025-2026 共 10 场，全部是 2025-12 的德玛西亚杯（DCup）。
    比例突然上升说明 OE 数据出了问题，或者赛制变了需要重新评估。
    """
    n = int(games.has_mirror_pick.sum())
    assert n / len(games) < 0.001, f"镜像场次比例异常：{n}/{len(games)}"
    if n:
        # 全部落在非 tier-1 赛区，不会污染评测集
        assert not games[games.has_mirror_pick].league_raw.isin(
            ["LPL", "LCK", "LEC", "LCS", "LCP", "CBLOL", "PCS", "VCS"]
        ).any(), "tier-1 赛区出现镜像英雄，需人工核查"


def test_mirror_flag_matches_recomputation(games):
    """标记列必须与重算一致，不能只是个摆设。"""
    for flag, b, r in zip(games.has_mirror_pick, games.blue_champ_keys, games.red_champ_keys):
        ks = [int(x) for x in list(b) + list(r) if not pd.isna(x)]
        expect = len(ks) == 10 and len(set(ks)) != 10
        assert bool(flag) == expect


def test_teams_differ(games):
    assert (games.blue_team_id != games.red_team_id).all()


def test_sorted_by_kickoff(games):
    assert games.kickoff_utc.is_monotonic_increasing


def test_roles_are_canonical():
    assert ROLES == ("top", "jng", "mid", "bot", "sup")


def test_no_forbidden_columns_leaked(games):
    """折叠不得凭空引入赛后列。"""
    from lolwp.features.whitelist import classify
    meta_ok = {
        "game_id", "kickoff_utc", "season_year", "y_blue_win", "data_snapshot_sha",
        "league_raw", "patch_raw", "blue_team_id", "red_team_id",
        "blue_team_name", "red_team_name", "game_in_match_raw",
        "blue_player_ids", "red_player_ids", "blue_champ_keys", "red_champ_keys",
        "blue_player_missing", "red_player_missing",
        "blue_champ_missing", "red_champ_missing",
        "has_ban_data", "has_first_pick", "blue_first_pick", "has_pick_order",
        "has_mirror_pick",
    }
    for c in games.columns:
        if c in meta_ok:
            continue
        assert classify(c) in {"allowed", "record_only"}, f"{c} 不该出现在折叠结果里"
