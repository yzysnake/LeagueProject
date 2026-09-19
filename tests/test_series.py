"""series_id 会话切分测试。

为什么这套测试比"非空率 100% + game_in_series ≤5"重要
------------------------------------------------------
PLAN.md §2.2 说得很清楚：v1.0 的两条验收对**被 UTC 零点劈成两段的 Bo5**（game 号
分别是 1-3 和 4-5）**两条都通过**，完全抓不到。所以这里每一条正向断言都配了一条
**负控制**——先证明被证伪的日历日写法确实会在同一个真实案例上失败，再证明 6h 会话
切分把它修好。只有正向断言的测试是没有牙齿的。

真实案例全部来自本机快照 data/raw/*.csv（13 年 102,084 场），不是编出来的。
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from lolwp.store.series import (
    GAP_DEFAULT,
    MAX_GAMES_PER_SERIES,
    SeriesConfig,
    attach_series,
    audit,
    build_game_in_series,
    build_series_id,
    calendar_day_series_id,
    count_split_series,
    pair_key,
    series_span_hours,
    series_stats,
)

RAW_GLOB = os.path.join(_ROOT, "data", "raw", "*_LoL_esports_match_data_from_OraclesElixir.csv")

#: 本机快照列名走 league_raw（configs/league_canonical.yaml 属于 silver 的活，这里不依赖它），
#: 并开启队名兜底——全量有 1,343/1,323 场缺 blue/red teamid。
CFG = SeriesConfig(
    league_col="league_raw",
    blue_name_col="blue_team_name",
    red_name_col="red_team_name",
)


# --------------------------------------------------------------------------- #
# 小工具：用字面量搭一张一场一行的表                                             #
# --------------------------------------------------------------------------- #
def mk(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(game_id, league, 'YYYY-mm-dd HH:MM:SS', game_no, blue, red), ...]"""
    df = pd.DataFrame(
        rows,
        columns=["game_id", "league_raw", "kickoff_utc", "game_no", "blue_team_id", "red_team_id"],
    )
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    return df


def n_series(df: pd.DataFrame, **kw) -> int:
    return int(build_series_id(df, CFG, **kw).nunique())


# =========================================================================== #
# 一、跨 UTC 零点：真实案例不得被劈开（配负控制）                                 #
# =========================================================================== #

# LCS 2026-01-24/25 Disguised vs Sentinels，手册 §2.2 点名的第一个案例
LCS_MIDNIGHT_BO3 = mk(
    [
        ("LOLTMNT01_325749", "LCS", "2026-01-24 23:14:34", 1,
         "oe:team:289284911fb6f41c0ccaecc3a0c2891", "oe:team:f90422f1cbcb24bc2a855202582ec29"),
        ("LOLTMNT01_325761", "LCS", "2026-01-25 00:22:15", 2,
         "oe:team:f90422f1cbcb24bc2a855202582ec29", "oe:team:289284911fb6f41c0ccaecc3a0c2891"),
        ("LOLTMNT01_324691", "LCS", "2026-01-25 01:17:06", 3,
         "oe:team:289284911fb6f41c0ccaecc3a0c2891", "oe:team:f90422f1cbcb24bc2a855202582ec29"),
    ]
)

# LCS 2026-02-14/15 Sentinels vs Disguised 的完整 Bo5，g4/g5 跨零点
LCS_MIDNIGHT_BO5 = mk(
    [
        ("LOLTMNT02_354540", "LCS", "2026-02-14 21:11:50", 1, "A", "B"),
        ("LOLTMNT02_353493", "LCS", "2026-02-14 22:05:04", 2, "A", "B"),
        ("LOLTMNT02_354548", "LCS", "2026-02-14 22:58:02", 3, "A", "B"),
        ("LOLTMNT02_354552", "LCS", "2026-02-14 23:52:02", 4, "A", "B"),
        ("LOLTMNT02_353505", "LCS", "2026-02-15 00:41:15", 5, "B", "A"),
    ]
)

# CBLOL 2026-08-07/08 LOUD vs FURIA
CBLOL_MIDNIGHT = mk(
    [
        ("LOLTMNT01_439279", "CBLOL", "2026-08-07 23:48:44", 1,
         "oe:team:658a021a796aa5501503220adb38cf4", "oe:team:87e3e1e75839f75a10bbc66e894d806"),
        ("LOLTMNT01_440253", "CBLOL", "2026-08-08 00:46:12", 2,
         "oe:team:87e3e1e75839f75a10bbc66e894d806", "oe:team:658a021a796aa5501503220adb38cf4"),
    ]
)


@pytest.mark.parametrize(
    "df,name", [(LCS_MIDNIGHT_BO3, "LCS Bo3"), (LCS_MIDNIGHT_BO5, "LCS Bo5"), (CBLOL_MIDNIGHT, "CBLOL")]
)
def test_cross_midnight_series_stays_whole(df: pd.DataFrame, name: str) -> None:
    """手册点名的三个跨零点真实案例，6h 会话切分必须给出**唯一**一个 series。"""
    assert n_series(df) == 1, name
    gis = build_game_in_series(df, cfg=CFG)
    assert gis.tolist() == list(range(1, len(df) + 1)), name


@pytest.mark.parametrize(
    "df,name", [(LCS_MIDNIGHT_BO3, "LCS Bo3"), (LCS_MIDNIGHT_BO5, "LCS Bo5"), (CBLOL_MIDNIGHT, "CBLOL")]
)
def test_negative_control_calendar_day_does_split_them(df: pd.DataFrame, name: str) -> None:
    """负控制：被证伪的日历日写法在**同一批数据**上确实劈成两段。

    没有这条，上面那条正向断言等于没测——一个"永远返回同一个常量"的实现也能过。
    """
    assert calendar_day_series_id(df, CFG).nunique() == 2, name
    # 而且 v1.0 的两条老验收对劈开完全无感：非空率仍是 100%，段内局号仍 ≤5
    cal = calendar_day_series_id(df, CFG)
    assert cal.notna().all()
    assert build_game_in_series(df, cal, CFG).max() <= MAX_GAMES_PER_SERIES


# =========================================================================== #
# 二、同日两个独立系列不得被合并                                                 #
# =========================================================================== #

# NLC 2026-05-27 Bulldog vs DMG：两局都是 game 1，间隔 2h07m —— 两个独立 Bo1
NLC_TWO_SERIES = mk(
    [
        ("LOLTMNT05_196376", "NLC", "2026-05-27 18:06:05", 1,
         "oe:team:4ccccd4b7c3dba7d799d52c4b78634d", "oe:team:147e28c0910250269fbc430f9f42010"),
        ("LOLTMNT05_196385", "NLC", "2026-05-27 20:13:06", 1,
         "oe:team:4ccccd4b7c3dba7d799d52c4b78634d", "oe:team:147e28c0910250269fbc430f9f42010"),
    ]
)

# EBL 2026-08-13 RLX vs Hooligans：[1,2] 之后又来一个 game 1
EBL_TWO_SERIES = mk(
    [
        ("LOLTMNT04_181942", "EBL", "2026-08-13 18:38:46", 1,
         "oe:team:722d77867e22e3b96da8b96258451b5", "oe:team:05cfed851ffe67f56c8f3635cceaaeb"),
        ("LOLTMNT04_181945", "EBL", "2026-08-13 19:27:54", 2,
         "oe:team:722d77867e22e3b96da8b96258451b5", "oe:team:05cfed851ffe67f56c8f3635cceaaeb"),
        ("LOLTMNT04_181949", "EBL", "2026-08-13 21:20:40", 1,
         "oe:team:722d77867e22e3b96da8b96258451b5", "oe:team:05cfed851ffe67f56c8f3635cceaaeb"),
    ]
)


def test_same_day_independent_series_not_merged() -> None:
    """game 号重复 = 两个独立系列的铁证，纯 6h 时间规则修不了，必须靠二次切分。"""
    assert n_series(NLC_TWO_SERIES) == 2
    sid = build_series_id(EBL_TWO_SERIES, CFG)
    assert sid.nunique() == 2
    assert sid.tolist() == [sid.iloc[0], sid.iloc[0], sid.iloc[2]]
    assert build_game_in_series(EBL_TWO_SERIES, sid, CFG).tolist() == [1, 2, 1]


def test_negative_control_pure_6h_rule_would_merge_them() -> None:
    """负控制：关掉 game 号去重（= 手册 §2.2 的原版实现）确实会把它们粘成一个。

    这条同时是"与手册偏离"的存档——偏离点、代价、开关全在这里。
    """
    assert n_series(NLC_TWO_SERIES, game_col=None) == 1
    assert n_series(EBL_TWO_SERIES, game_col=None) == 1


def test_missing_team_id_falls_back_to_name() -> None:
    """LPLOL 2026-02-12 Elegance vs Rage Bait：两队 teamid 全缺，只能靠队名。

    缺 id 时绝不能 fillna 成同一个常量——那会把所有缺 id 的场次粘成一个巨型 series。
    """
    df = pd.DataFrame(
        {
            "game_id": ["LOLTMNT02_351645", "LOLTMNT02_351685"],
            "league_raw": ["LPLOL", "LPLOL"],
            "kickoff_utc": pd.to_datetime(["2026-02-12 20:35:58", "2026-02-12 23:27:36"]),
            "game_no": [1, 1],
            "blue_team_id": [None, None],
            "red_team_id": [None, None],
            "blue_team_name": ["Rage Bait e-Sports", "Rage Bait e-Sports"],
            "red_team_name": ["Elegance", "Elegance"],
        }
    )
    assert build_series_id(df, CFG).nunique() == 2      # 队名兜底 + game 号去重
    # 不给兜底列时退化为逐行哨兵：两场互不相识，仍是 2 个 series，绝不粘连
    bare = SeriesConfig(league_col="league_raw")
    assert build_series_id(df, bare).nunique() == 2


def test_missing_team_id_never_merges_unrelated_games() -> None:
    """四场互不相干、teamid 全缺的比赛，必须是 4 个 series 而不是 1 个。"""
    df = mk(
        [
            ("g1", "X", "2026-01-01 10:00:00", 1, None, None),
            ("g2", "X", "2026-01-01 11:00:00", 1, None, None),
            ("g3", "X", "2026-01-01 12:00:00", 1, None, None),
            ("g4", "X", "2026-01-01 13:00:00", 1, None, None),
        ]
    )
    assert build_series_id(df, SeriesConfig(league_col="league_raw")).nunique() == 4


# =========================================================================== #
# 三、切分规则本身的性质                                                        #
# =========================================================================== #

def test_side_swap_keeps_same_series() -> None:
    """系列内每局换边，队对键必须是**无序对**。"""
    df = mk(
        [
            ("g1", "LCK", "2026-03-01 08:00:00", 1, "T1", "GEN"),
            ("g2", "LCK", "2026-03-01 09:00:00", 2, "GEN", "T1"),
        ]
    )
    assert n_series(df) == 1


def test_gap_just_over_threshold_splits() -> None:
    """6h 是闭区间上界：恰好 6h 归同一 series，6h+1s 开新 series。"""
    same = mk([("g1", "LEC", "2026-03-01 08:00:00", 1, "A", "B"),
               ("g2", "LEC", "2026-03-01 14:00:00", 2, "A", "B")])
    diff = mk([("g1", "LEC", "2026-03-01 08:00:00", 1, "A", "B"),
               ("g2", "LEC", "2026-03-01 14:00:01", 2, "A", "B")])
    assert n_series(same) == 1
    assert n_series(diff) == 2
    assert GAP_DEFAULT == pd.Timedelta(hours=6)


def test_same_teams_different_league_not_merged() -> None:
    """同一对队伍同一天在两个赛事各打一局（如联赛 + 杯赛），不能并成一个 series。"""
    df = mk([("g1", "LEC", "2026-03-01 08:00:00", 1, "A", "B"),
             ("g2", "EWC", "2026-03-01 09:00:00", 1, "A", "B")])
    assert n_series(df) == 2


def test_long_pause_inside_bo5_is_kept_together() -> None:
    """Bo5 中间的长暂停（实测最长的一例首局后停了 4.27h，CLS 2016-08-07）不能被切开。

    这正是阈值不能取 3h/4h 的原因：那会把真系列切成两段，落进不同 fold。
    """
    df = mk(
        [
            ("g1", "CLS", "2016-08-07 16:24:48", 1, "A", "B"),
            ("g2", "CLS", "2016-08-07 20:41:00", 2, "A", "B"),
            ("g3", "CLS", "2016-08-07 22:16:00", 3, "A", "B"),
            ("g4", "CLS", "2016-08-07 23:03:00", 4, "A", "B"),
            ("g5", "CLS", "2016-08-08 00:26:57", 5, "A", "B"),
        ]
    )
    assert n_series(df) == 1
    assert float(series_span_hours(df, cfg=CFG).iloc[0]) == pytest.approx(8.04, abs=0.01)
    assert n_series(df, gap=pd.Timedelta(hours=4)) == 2      # 负控制：4h 阈值就切坏了


def test_out_of_order_game_numbers_stay_merged() -> None:
    """LAS 2026-09-13 Gen.G Scholars vs Nongshim：局号 [2,1,3] 只是乱序、没有重复。

    乱序更可能是 OE 时间戳噪声。两类错误代价不对称（劈开真系列 = §4.8.2 第 4 条，
    比标签泄漏更严重；粘连两个系列只损失一点有效样本量），所以这里**保守地不切**。
    """
    df = mk(
        [
            ("LOLTMNT01_460268", "LAS", "2026-09-13 07:04:02", 2, "GENS", "NSA"),
            ("LOLTMNT01_460292", "LAS", "2026-09-13 07:46:53", 1, "NSA", "GENS"),
            ("LOLTMNT01_461256", "LAS", "2026-09-13 08:37:52", 3, "GENS", "NSA"),
        ]
    )
    assert n_series(df) == 1


def test_identical_timestamps_get_distinct_ids() -> None:
    """同分同秒开赛的两个独立系列（全量 24 例，如 BIG 2021-02-02）不能共用一个 key。"""
    df = mk(
        [
            ("a", "BIG", "2021-02-02 00:56:05", 1, "T1", "T2"),
            ("b", "BIG", "2021-02-02 00:56:05", 1, "T1", "T2"),
            ("c", "BIG", "2021-02-02 01:54:22", 2, "T1", "T2"),
        ]
    )
    sid = build_series_id(df, CFG)
    assert sid.nunique() == 2
    assert sum("#" in s for s in sid.unique()) == 1     # 第二个挂 #2 后缀


def test_deterministic_under_row_permutation() -> None:
    """输入行序变了，series_id 不能变——否则复现性直接报废。

    全量有 44 对相邻局 kickoff 完全同秒，只按时间排序时先后由输入行序决定。
    """
    df = pd.concat([LCS_MIDNIGHT_BO5, NLC_TWO_SERIES, EBL_TWO_SERIES], ignore_index=True)
    a = build_series_id(df, CFG)
    shuffled = df.sample(frac=1.0, random_state=7)
    b = build_series_id(shuffled, CFG).reindex(df.index)
    assert a.tolist() == b.tolist()
    assert list(build_series_id(df, CFG).index) == list(df.index)   # 索引对齐


def test_attach_series_and_empty_frame() -> None:
    out = attach_series(LCS_MIDNIGHT_BO5, CFG)
    assert {"series_id", "game_in_series"} <= set(out.columns)
    assert len(out) == len(LCS_MIDNIGHT_BO5)
    empty = LCS_MIDNIGHT_BO5.iloc[:0]
    assert len(build_series_id(empty, CFG)) == 0
    assert len(build_game_in_series(empty, cfg=CFG)) == 0


def test_loud_failures_on_bad_input() -> None:
    """列缺失 / 时间列类型不对 / game_id 重复，必须**报错**而不是静默给个错 key。"""
    with pytest.raises(KeyError, match="缺列"):
        build_series_id(LCS_MIDNIGHT_BO5.drop(columns=["league_raw"]), CFG)
    bad_ts = LCS_MIDNIGHT_BO5.assign(kickoff_utc=LCS_MIDNIGHT_BO5.kickoff_utc.astype(str))
    with pytest.raises(TypeError, match="datetime64"):
        build_series_id(bad_ts, CFG)
    dupe = pd.concat([LCS_MIDNIGHT_BO3, LCS_MIDNIGHT_BO3], ignore_index=True)
    with pytest.raises(ValueError, match="重复"):
        build_series_id(dupe, CFG)


# =========================================================================== #
# 四、全量真实数据（13 年 102,084 场）                                          #
# =========================================================================== #

@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    """把 13 个原始 CSV 的 team 行（participantid 100/200）折成一场一行。

    只是测试用的最小折叠，生产口径在 store/silver.py；这里刻意不依赖它，
    免得 silver 的 bug 把 series 的测试一起带绿。
    """
    files = sorted(glob.glob(RAW_GLOB))
    if len(files) < 13:
        pytest.skip(f"需要 data/raw 下 13 个 OE CSV，实际 {len(files)} 个")
    cols = ["gameid", "league", "date", "game", "participantid", "side", "teamid", "teamname"]
    frames = [pd.read_csv(f, usecols=cols, low_memory=False) for f in files]
    t = pd.concat(frames, ignore_index=True)
    t = t[t.participantid.isin([100, 200])]
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    blue = t[t.side == "Blue"].set_index("gameid")
    red = t[t.side == "Red"].set_index("gameid").reindex(blue.index)
    return pd.DataFrame(
        {
            "game_id": blue.index,
            "league_raw": blue["league"].values,
            "kickoff_utc": blue["date"].values,
            "game_no": blue["game"].values,
            "blue_team_id": blue["teamid"].values,
            "red_team_id": red["teamid"].values,
            "blue_team_name": blue["teamname"].values,
            "red_team_name": red["teamname"].values,
        }
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def sid_all(games: pd.DataFrame) -> pd.Series:
    return build_series_id(games, CFG)


@pytest.fixture(scope="module")
def audit_all(games: pd.DataFrame, sid_all: pd.Series) -> dict:
    return audit(games, sid_all, CFG)


def test_snapshot_shape(games: pd.DataFrame) -> None:
    """快照护栏：数字变了说明数据换了，下面所有实测断言都要重新标定。"""
    assert len(games) == 102_084
    assert games.game_id.is_unique
    assert int((games.kickoff_utc.dt.year == 2026).sum()) == 8_851


def test_series_id_nonnull_and_stable(games: pd.DataFrame, sid_all: pd.Series) -> None:
    assert sid_all.notna().all()
    assert sid_all.nunique() == 59_670          # 实测 2026-09-18
    # 每个 series 只能属于一个 (league, 队对)——后缀消解之后不允许再有碰撞
    pk = pair_key(games, CFG)
    key = games.league_raw.astype(str) + "||" + pk
    assert int(key.groupby(sid_all).nunique().max()) == 1


def test_max_game_in_series_le_5(games: pd.DataFrame, sid_all: pd.Series) -> None:
    """Bo5 是物理上限。>5 说明切分过宽。"""
    assert int(build_game_in_series(games, sid_all, CFG).max()) <= MAX_GAMES_PER_SERIES


def test_no_series_longer_than_bo5(audit_all: dict) -> None:
    assert audit_all["too_long"]["count"] == 0
    assert audit_all["max_len"] == MAX_GAMES_PER_SERIES


def test_series_span_within_12h(games: pd.DataFrame, sid_all: pd.Series) -> None:
    """手册断言 (b)：同一 series 跨度 ≤12h。实测最大 8.04h（CLS 2016-08-07 的 Bo5）。"""
    span = series_span_hours(games, sid_all, CFG)
    assert float(span.max()) <= 12.0
    assert float(span.max()) == pytest.approx(8.04, abs=0.02)


def test_no_split_series_left(games: pd.DataFrame, sid_all: pd.Series) -> None:
    """手册断言 (c)：不存在"同 league 同队对、相邻两局间隔 <6h 却 series_id 不同"。

    全量残留 1 例（OTBLX 2021-06-13，同队对局号按时间排成 [2,1,2]，怎么切都得罪一边），
    2026 为 0。把上界写死成 1 是为了让任何新增的劈开立刻让 CI 变红。
    """
    assert count_split_series(games, sid_all, CFG) <= 1
    g26 = games[games.kickoff_utc.dt.year == 2026]
    assert count_split_series(g26, cfg=CFG) == 0


def test_no_duplicate_game_numbers_within_series(audit_all: dict) -> None:
    """二次切分的目标：全库不再有同一 series 里出现两次 game 1 这种事。"""
    assert audit_all["game_no_duplicated"]["count"] == 0
    assert audit_all["game_no_gt_5"]["count"] == 0


def test_game_numbers_are_1_to_n_for_the_vast_majority(audit_all: dict) -> None:
    """手册断言 (a) 的**可执行版本**。

    字面照抄"sorted(game) == 1..n"会永远红：全量 201 个 series 的局号不连续，其中
    105 个是单局 series——OE 压根没收录同系列的其它局（pattern 大头 `(2,)` 79 次、
    `(3,)` 25 次）。那是数据覆盖缺口，不是切分错误。所以断言改成比例门槛。
    """
    assert audit_all["game_no_not_1_to_n"]["rate"] < 0.005
    assert audit_all["game_no_not_1_to_n"]["count"] == 201


def test_m_bar_matches_measurement(games: pd.DataFrame, sid_all: pd.Series) -> None:
    """m̄ 直接进 §6.3 的 deff = 1 + (m̄−1)·ICC，测错会让 MDE 整体偏。

    实测 2026-09-18：全量 1.711 / 2026 全赛区 2.187（手册写的 2.14 偏低）/
    tier1+国际赛 2026 高达 2.547——**评估集用的是后者**，别拿全量的 1.71 去算 deff。
    """
    full = series_stats(games, sid_all, CFG)
    assert 1.65 < full["m_bar"] < 1.80
    assert full["m_bar"] == pytest.approx(1.7108, abs=0.001)

    g26 = games[games.kickoff_utc.dt.year == 2026]
    s26 = series_stats(g26, cfg=CFG)
    assert 2.10 < s26["m_bar"] < 2.30
    assert s26["m_bar"] == pytest.approx(2.1871, abs=0.001)
    assert s26["length_dist"] == {1: 1162, 2: 1468, 3: 1069, 4: 194, 5: 154}

    tier1 = {"LPL", "LCK", "LEC", "LCS", "LCP", "CBLOL", "PCS", "VCS",
             "LTA", "LTA N", "LTA S", "MSI", "WLDS", "FST", "EWC", "IC"}
    ev = g26[g26.league_raw.str.upper().isin(tier1)]
    s_ev = series_stats(ev, cfg=CFG)
    assert 2.40 < s_ev["m_bar"] < 2.70
    assert s_ev["m_bar"] == pytest.approx(2.5472, abs=0.001)


def test_calendar_day_baseline_is_broken_on_real_data(games: pd.DataFrame) -> None:
    """复现手册 §2.2 的 89——负控制的总闸门。

    这条一旦不再是 89，要么数据换了，要么谁把切分逻辑改坏了，两种都必须有人看一眼。
    """
    g26 = games[games.kickoff_utc.dt.year == 2026]
    cal26 = calendar_day_series_id(g26, CFG)
    assert count_split_series(g26, cal26, CFG, respect_game_no=False) == 89
    cal_all = calendar_day_series_id(games, CFG)
    assert count_split_series(games, cal_all, CFG, respect_game_no=False) == 1138
    # 而 6h 会话切分把 89 全修好了
    assert count_split_series(g26, cfg=CFG) == 0
