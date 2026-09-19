"""silver 层的建模宽表。

这一层是训练与推理的共同输入，任何静默错误都会同时污染两边。
"""
import os

import pandas as pd
import pytest
import yaml

SILVER = "data/silver/games.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(SILVER), reason="需先跑 silver.build()")


@pytest.fixture(scope="module")
def g():
    return pd.read_parquet(SILVER)


def test_one_row_per_game(g):
    assert g.game_id.nunique() == len(g)


def test_series_id_complete(g):
    assert g.series_id.notna().all()
    assert g.game_in_series.max() <= 5
    assert g.game_in_series.min() >= 1


def test_no_series_straddles_holdout_cutoff(g):
    """最关键的一条：任何 series 都不能一半在训练集、一半在 holdout。

    跨边界的 series 会让同一对队伍同一天的状态泄漏进 holdout，
    PLAN.md §4.8.2 把它列为比标签泄漏更严重的一类。
    """
    with open("eval/HOLDOUT_START") as fh:
        start = fh.read().strip()
    in_ho = g.kickoff_utc.dt.strftime("%Y-%m-%d") >= start
    per = g.assign(in_ho=in_ho).groupby("series_id").in_ho.nunique()
    straddling = per[per > 1]
    assert len(straddling) == 0, f"{len(straddling)} 个 series 横跨 holdout 边界"


def test_league_tier_respects_frozen_scope(g):
    """评测口径必须以冻结的 eval/tier1_leagues.yaml 为准，含 A2 的分段降级。"""
    with open("eval/tier1_leagues.yaml") as fh:
        cfg = yaml.safe_load(fh)
    # PCS/VCS：2024-12-31 之前 tier 1，之后 tier 0
    for lg in ["PCS", "VCS"]:
        assert lg in cfg["tier1_until"]
        sub = g[g.league_raw == lg]
        if len(sub) == 0:
            continue
        day = sub.kickoff_utc.dt.strftime("%Y-%m-%d")
        before = sub[day <= "2024-12-31"]
        after = sub[day > "2024-12-31"]
        if len(before):
            assert (before.league_tier == 1).all(), f"{lg} 2024 前应为 tier 1"
        if len(after):
            assert (after.league_tier == 0).all(), f"{lg} 2025 起应为 tier 0"


def test_worlds_is_tier2_and_uses_lowercase_code(g):
    """WLDs 小写 s。写成 WLDS 会静默漏掉世界赛（修订 A1 的起因）。"""
    assert "WLDs" in set(g.league_raw)
    assert "WLDS" not in set(g.league_raw)
    assert (g[g.league_raw == "WLDs"].league_tier == 2).all()


def test_holdout_membership_matches_date_rule(g):
    """冻结名单必须能由「日期截止线 + 赛区口径」重算出来，不能是手工名单。"""
    with open("eval/holdout_gameids_20260919_v3.txt") as fh:
        frozen = set(fh.read().split())
    with open("eval/HOLDOUT_START") as fh:
        start = fh.read().strip()
    day = g.kickoff_utc.dt.strftime("%Y-%m-%d")
    recomputed = set(g[(day >= start) & (g.league_tier >= 1)].game_id)
    assert recomputed == frozen, (
        f"重算与冻结名单不一致：多 {len(recomputed - frozen)}，少 {len(frozen - recomputed)}"
    )


def test_patch_features_present_and_monotone(g):
    for c in ["patch_major", "patch_minor", "patch_idx", "patch_age_days"]:
        assert c in g.columns, f"缺列 {c}"
    ok = g.patch_age_days.notna()
    assert (g.loc[ok, "patch_age_days"] >= 0).all(), "patch_age_days 出现负值"


def test_crosswalk_attached(g):
    for c in ["blue_esports_ids", "red_esports_ids"]:
        assert c in g.columns
        assert g[c].map(len).eq(5).all()


def test_asof_never_after_kickoff(g):
    """特征冻结时刻必须 ≤ 开赛时刻，这是 point-in-time 正确性的底线。"""
    assert (g.asof_ts <= g.kickoff_utc).all()


def test_no_label_leakage_columns(g):
    from lolwp.features.whitelist import FORBIDDEN_RE
    bad = [c for c in g.columns if FORBIDDEN_RE.search(c)]
    assert not bad, f"silver 出现禁用列：{bad}"
