"""Glicko-2 双轨的验收测试（PLAN.md §4.2 G2 / §4.6.3 / §4.7.1 T2 档 A）。

分四组：
  1. 算法正确性 —— 对 Glickman 论文第 3 页的算例，以及对 PyPI `glicko2` 参考实现；
  2. 时序正确性 —— T2 档 A「未来删除不变性」必须 **bit-identical**，
     以及 rating period 的批量语义（period 内互相看不见）；
  3. RD 的行为 —— 休赛期膨胀、冷启动、上限截断；
  4. training/serving 一致性 —— 状态快照路径与状态机路径逐位相同。
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import pandas as pd
import pytest

from lolwp.models.glicko2 import (
    ANCHOR,
    FEATURE_COLS,
    SCALE,
    Glicko2Config,
    Rating,
    _period_start_scalar,
    drift,
    expected_score,
    features_from_glicko_state,
    g_factor,
    rd_report,
    run_glicko_state_machine,
    solve_volatility,
    update_rating,
    win_prob_band,
)

SILVER = "data/silver/games.parquet"
needs_data = pytest.mark.skipif(not os.path.exists(SILVER), reason="需先跑 silver.build()")


def _elo(r: float) -> float:
    return (r - ANCHOR) / SCALE


def _rd(x: float) -> float:
    return x / SCALE


# ==========================================================================
# 1. 算法正确性
# ==========================================================================


def test_glickman_paper_worked_example():
    """Glickman《Example calculation》：1500/200/0.06 打三场 → 1464.06 / 151.52 / 0.05999。

    这是整个模块唯一的外部真值。改了任何一步公式，这条会先红。
    论文给的是四舍五入到小数点后两位的值，所以容差取 0.01。
    """
    r = Rating(_elo(1500), _rd(200), 0.06)
    opps = [
        (_elo(1400), _rd(30), 1.0, 1.0),
        (_elo(1550), _rd(100), 0.0, 1.0),
        (_elo(1700), _rd(300), 0.0, 1.0),
    ]
    out = update_rating(r, opps, elapsed=1.0, cfg=Glicko2Config(tau=0.5))
    assert abs(out.rating - 1464.06) < 0.01
    assert abs(out.rd - 151.52) < 0.01
    assert abs(out.sigma - 0.05999) < 1e-5


def test_paper_intermediate_quantities():
    """逐步核对论文的中间量：g(φ)、E、v、Δ。整体对不代表每一步都对。"""
    assert abs(g_factor(_rd(30)) - 0.9955) < 1e-4
    assert abs(g_factor(_rd(100)) - 0.9531) < 1e-4
    assert abs(g_factor(_rd(300)) - 0.7242) < 1e-4
    mu = _elo(1500)
    assert abs(expected_score(mu, _elo(1400), _rd(30)) - 0.639) < 1e-3
    assert abs(expected_score(mu, _elo(1550), _rd(100)) - 0.432) < 1e-3
    assert abs(expected_score(mu, _elo(1700), _rd(300)) - 0.303) < 1e-3


def test_volatility_solver_matches_paper():
    """论文算例里 v=1.7785、Δ=-0.4834、φ=200/173.7178 时 σ'=0.05999。"""
    s = solve_volatility(0.06, -0.4834, _rd(200), 1.7785, tau=0.5)
    assert abs(s - 0.05999) < 1e-5


def test_volatility_solver_is_a_root():
    """求根器返回的必须真是 f(x)=0 的根，而不是"迭代到上限就返回"。"""
    sigma, delta, phi, v, tau = 0.06, 0.9, _rd(180), 1.2, 0.5
    s = solve_volatility(sigma, delta, phi, v, tau=tau)
    x = 2.0 * math.log(s)
    a = math.log(sigma**2)
    ex = math.exp(x)
    f = ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2) - (x - a) / tau**2
    # 收敛判据是 log 空间的 |B−A| < eps=1e-6，对应的 f 残差量级在 1e-8
    assert abs(f) < 1e-6


def test_cross_check_against_reference_package():
    """与 PyPI `glicko2` 2.1.0 交叉验证 200 组随机输入。

    主实现必须自己写（要能导出状态快照供推理复用），但"自己写"不等于"可以不对数"。
    注意参考实现的 `_f()` 里把论文的 φ² 写成了 μ²（内部尺度），在 μ≠0 时两者会有
    微小分歧；本测试因此对 σ' 放宽到 2e-3，对 rating / RD 仍按 1e-2 卡死。
    """
    pkg = pytest.importorskip("glicko2", reason="pip install glicko2（可选的交叉验证）")
    rng = random.Random(0)
    worst_r = worst_rd = worst_v = 0.0
    for _ in range(200):
        r0 = rng.uniform(1100, 1900)
        rd0 = rng.uniform(40, 350)
        k = rng.randint(1, 8)
        opp_r = [rng.uniform(1100, 1900) for _ in range(k)]
        opp_rd = [rng.uniform(30, 350) for _ in range(k)]
        out = [float(rng.randint(0, 1)) for _ in range(k)]

        mine = update_rating(
            Rating(_elo(r0), _rd(rd0), 0.06),
            [(_elo(a), _rd(b), c, 1.0) for a, b, c in zip(opp_r, opp_rd, out)],
            elapsed=1.0,
            cfg=Glicko2Config(tau=0.5),
        )
        ref = pkg.Player(rating=r0, rd=rd0, vol=0.06)
        ref.update_player(opp_r, opp_rd, out)
        worst_r = max(worst_r, abs(mine.rating - ref.rating))
        worst_rd = max(worst_rd, abs(mine.rd - ref.rd))
        worst_v = max(worst_v, abs(mine.sigma - ref.vol))
    assert worst_r < 1e-2, worst_r
    assert worst_rd < 1e-2, worst_rd
    assert worst_v < 2e-3, worst_v


def test_no_opponents_only_inflates_rd():
    """论文 step 6：没打比赛的 period 只膨胀 RD，μ 与 σ 一动不动。"""
    r = Rating(_elo(1600), _rd(80), 0.06)
    out = update_rating(r, [], elapsed=1.0, cfg=Glicko2Config())
    assert out.mu == r.mu
    assert out.sigma == r.sigma
    assert out.rd > r.rd


def test_pairwise_weight_controls_information_content():
    """5v5 拆成 5 个完全相关的伪对局，按 1 计权会假装有 5 份信息。

    这条锁死 `pairwise_weight` 的语义：w=1/5 时 RD 的收缩应当与"打一场"同量级，
    w=1 时会明显过度收缩。
    """
    r = Rating(0.0, _rd(120), 0.06)
    opp = (_elo(1500), _rd(120), 1.0)
    one = update_rating(r, [(*opp, 1.0)], cfg=Glicko2Config())
    five_w1 = update_rating(r, [(*opp, 1.0)] * 5, cfg=Glicko2Config())
    five_w02 = update_rating(r, [(*opp, 0.2)] * 5, cfg=Glicko2Config())
    assert five_w1.rd < one.rd - 10          # 按 1 计权：RD 塌得多得多
    assert abs(five_w02.rd - one.rd) < 1e-9  # 按 1/5 计权：与一场等价


# ==========================================================================
# 2. 时序正确性
# ==========================================================================


def _synth(n_games: int = 900, n_teams: int = 24, seed: int = 3) -> pd.DataFrame:
    """合成赛程：真实实力固定，比赛按时间推进，中间插一段 60 天休赛期。"""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1.0, n_teams)
    players = [[f"p{t}_{j}" for j in range(5)] for t in range(n_teams)]
    t0 = pd.Timestamp("2021-01-04", tz="UTC")
    rows = []
    for i in range(n_games):
        a, b = rng.choice(n_teams, 2, replace=False)
        p = 1.0 / (1.0 + math.exp(-(skill[a] - skill[b])))
        day = i // 6 * 2 + (0 if i < n_games // 2 else 60)  # 后半段整体推迟 60 天
        ts = t0 + pd.Timedelta(days=int(day), hours=int(i % 6))
        rows.append(
            {
                "game_id": f"g{i}",
                "series_id": f"s{i // 3}",
                "kickoff_utc": ts,
                "league_canon": "LCK" if i % 3 else "LPL",
                "league_tier": 1 + (i % 17 == 0),
                "blue_team_id": f"t{a}",
                "red_team_id": f"t{b}",
                "blue_player_ids": players[a],
                "red_player_ids": players[b],
                "y_blue_win": int(rng.random() < p),
            }
        )
    return pd.DataFrame(rows).sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)


def test_requires_sorted_input():
    df = _synth(60)
    with pytest.raises(ValueError, match="升序"):
        run_glicko_state_machine(df.iloc[::-1].reset_index(drop=True))


def test_first_appearance_emits_prior_not_posterior():
    """红线：先读状态写特征，再用赛果更新。首秀那一场必须是 1500 / RD₀。"""
    df = _synth(40)
    f = run_glicko_state_machine(df, cfg=Glicko2Config(period="game"))
    assert f.blue_team_glicko.iloc[0] == pytest.approx(1500.0)
    assert f.red_team_glicko.iloc[0] == pytest.approx(1500.0)
    assert f.team_rd_blue.iloc[0] == pytest.approx(350.0)
    assert f.team_glicko_diff.iloc[0] == 0.0


@pytest.mark.parametrize("period", ["game", "day", "week", "month"])
def test_future_deletion_invariance_bit_identical(period):
    """T2 档 A（§4.7.1）：删掉未来的比赛，历史那一场的特征必须**逐位相同**。

    在线递归类特征没有任何借口通不过这一档 —— 一旦不是 bit-identical，
    就说明状态机里有一条路径读到了 `>= t` 的信息。
    """
    df = _synth(700)
    cfg = Glicko2Config(period=period)
    full = run_glicko_state_machine(df, cfg=cfg)
    rng = random.Random(11)
    for i in sorted(rng.sample(range(200, len(df)), 12)):
        cut = run_glicko_state_machine(df.iloc[: i + 1].reset_index(drop=True), cfg=cfg)
        np.testing.assert_array_equal(
            full.iloc[i].to_numpy(dtype=float), cut.iloc[i].to_numpy(dtype=float)
        )


def test_rating_period_is_batched_not_online():
    """Glicko-2 的定义性质：同一 rating period 内的比赛读同一份快照。

    构造两场同一天、同一对阵的比赛（period="week"）：
    第二场的特征必须与第一场**完全相同**（看不见第一场的结果）；
    换成 period="game" 后第二场必须不同（看得见）。
    """
    ts = pd.Timestamp("2022-03-01", tz="UTC")
    rows = []
    for i in range(4):
        rows.append(
            {
                "game_id": f"g{i}", "series_id": "s0",
                "kickoff_utc": ts + pd.Timedelta(hours=i),
                "league_canon": "LCK", "league_tier": 1,
                "blue_team_id": "A", "red_team_id": "B",
                "blue_player_ids": [f"a{j}" for j in range(5)],
                "red_player_ids": [f"b{j}" for j in range(5)],
                "y_blue_win": 1,
            }
        )
    df = pd.DataFrame(rows)
    wk = run_glicko_state_machine(df, cfg=Glicko2Config(period="week"))
    assert wk.team_glicko_diff.nunique() == 1  # 一周内四场，读到的状态完全一样
    assert wk.team_glicko_diff.iloc[0] == 0.0

    gm = run_glicko_state_machine(df, cfg=Glicko2Config(period="game"))
    assert gm.team_glicko_diff.iloc[0] == 0.0
    assert gm.team_glicko_diff.iloc[1] > 0.0          # 第二场已经看见第一场赢了
    assert gm.team_glicko_diff.is_monotonic_increasing  # 连赢四场，差距单调扩大


def test_run_is_deterministic():
    df = _synth(400)
    a = run_glicko_state_machine(df)
    b = run_glicko_state_machine(df)
    pd.testing.assert_frame_equal(a, b)


def test_all_feature_columns_finite():
    df = _synth(500)
    f = run_glicko_state_machine(df)
    assert list(f.columns) == list(FEATURE_COLS)
    assert np.isfinite(f.to_numpy(dtype=float)).all()


def test_n_games_counts_games_not_opponent_rows():
    """`*_n_games_min` 数的是**场次**，不是对局记录条数。

    pairwise 模式下一场 5v5 会给每名选手登记 5 条对局记录（5 个对手）。
    如果计数跟着记录条数走，选手的"出场数"会虚高 5 倍，
    于是 rd_report 的冷启动分箱、以及任何拿这一列做门槛的下游逻辑全部错位
    ——这正是本模块修过的一个真 bug，所以钉死在这里。
    """
    ts = pd.Timestamp("2022-05-02", tz="UTC")
    rows = [
        {
            "game_id": f"g{i}", "series_id": f"s{i}",
            "kickoff_utc": ts + pd.Timedelta(days=7 * i),
            "league_canon": "LCK", "league_tier": 1,
            "blue_team_id": "A", "red_team_id": "B",
            "blue_player_ids": [f"a{j}" for j in range(5)],
            "red_player_ids": [f"b{j}" for j in range(5)],
            "y_blue_win": i % 2,
        }
        for i in range(6)
    ]
    df = pd.DataFrame(rows)
    f = run_glicko_state_machine(df, cfg=Glicko2Config(period="week"))
    # 第 i 场（0-based）开赛前，两队/十人各打过 i 场
    np.testing.assert_array_equal(f.team_n_games_min.to_numpy(), np.arange(6.0))
    np.testing.assert_array_equal(f.player_n_games_min.to_numpy(), np.arange(6.0))
    # 快照路径必须给出同一个计数（否则线上的冷启动判定和离线不一致）
    _, state = run_glicko_state_machine(df, cfg=Glicko2Config(period="week"), return_state=True)
    assert state["player"]["a0"][4] == 6.0
    assert state["team"]["A"][4] == 6.0


def test_missing_player_ids_fall_back_to_max_rd():
    """选手名单缺失时不能假装知道：diff 给 0，RD 顶到上限。"""
    df = _synth(30)
    df = df.assign(blue_player_ids=[[] for _ in range(len(df))])
    f = run_glicko_state_machine(df)
    assert (f.player_glicko_diff == 0.0).all()
    assert (f.player_rd_max_blue == 350.0).all()


# ==========================================================================
# 3. RD 的行为：休赛期膨胀、冷启动、上限
# ==========================================================================


def test_drift_monotone_and_capped():
    r = Rating(0.0, _rd(60), 0.06)
    prev = r.rd
    for weeks in (1, 4, 10, 100, 10_000):
        cur = drift(r, weeks, 350.0 / SCALE).rd
        assert cur >= prev
        prev = cur
    assert drift(r, 10_000, 350.0 / SCALE).rd == pytest.approx(350.0)


def test_offseason_gap_inflates_rd_by_measurable_amount():
    """实测数据里最大无比赛空档 34 天。这条把"34 天到底涨多少"钉成一个数。

    σ=0.06、膨胀周期 7 天：34 天 = 4.857 个周期，
    RD 60 → √(60² + 4.857·(0.06·173.7178)²) = **64.2**。
    **这个量级本身就是一条结论**：Glickman 的默认 σ₀=0.06 让休赛期膨胀相当弱
    （90 天也只到 70.7，365 天才到 96.2）。想让它更凶只能调
    `inflation_period_days`（调小 = 同样天数注入更多方差），
    而那是一个必须预注册的超参，不能在 OOS 上挑。
    """
    r = Rating(0.0, _rd(60), 0.06)
    out = drift(r, 34.0 / 7.0, 350.0 / SCALE)
    expected = math.sqrt(60.0**2 + (34.0 / 7.0) * (0.06 * SCALE) ** 2)
    assert out.rd == pytest.approx(expected, rel=1e-12)
    assert 64.0 < out.rd < 64.5
    assert drift(r, 90.0 / 7.0, 350.0 / SCALE).rd == pytest.approx(70.7, abs=0.1)
    assert drift(r, 365.0 / 7.0, 350.0 / SCALE).rd == pytest.approx(96.2, abs=0.1)
    # 把膨胀周期调到 1 天，同样 34 天的空档就凶得多
    assert drift(r, 34.0 / 1.0, 350.0 / SCALE).rd == pytest.approx(85.4, abs=0.1)


def test_state_machine_rd_rises_after_long_layoff():
    """状态机层面的同一件事：休赛 60 天后，同一支队伍的 RD 必须比赛季末更大。"""
    df = _synth(900)
    f = run_glicko_state_machine(df, cfg=Glicko2Config(period="week"))
    d = df.assign(rd=f.team_rd_blue, gap=df.kickoff_utc.diff().dt.days.fillna(0))
    resume = int(d.gap.idxmax())
    before = d.rd.iloc[resume - 30 : resume].mean()
    after = d.rd.iloc[resume : resume + 30].mean()
    assert after > before, (before, after)


def test_cold_start_rd_is_far_above_veteran_rd():
    """RD 才是本模块的主产出：新面孔与老面孔必须被明确区分开。"""
    df = _synth(1200)
    f = run_glicko_state_machine(df, cfg=Glicko2Config(period="week"))
    cold = f.team_rd_blue[f.team_n_games_min == 0]
    warm = f.team_rd_blue[f.team_n_games_min > 100]
    assert len(cold) and len(warm)
    assert cold.mean() > 2.5 * warm.mean()


def test_rd_report_shape():
    df = _synth(900)
    f = run_glicko_state_machine(df)
    rep = rd_report(df, f)
    assert set(rep) == {
        "by_team_experience", "by_player_experience",
        "by_season_segment", "by_international",
    }
    assert rep["by_team_experience"].n.sum() == len(df)


def test_win_prob_band_widens_with_uncertainty():
    """§4.6.4：不确定性大的对局，UI 区间必须明显更宽，且点估计更靠近 0.5。"""
    strong = Rating(_elo(1700), _rd(50), 0.06)
    weak = Rating(_elo(1500), _rd(50), 0.06)
    p_sure, lo_sure, hi_sure = win_prob_band(strong, weak)
    fuzzy = Rating(_elo(1500), _rd(300), 0.06)
    p_fuzzy, lo_fuzzy, hi_fuzzy = win_prob_band(strong, fuzzy)
    assert (hi_fuzzy - lo_fuzzy) > 2.5 * (hi_sure - lo_sure)
    assert abs(p_fuzzy - 0.5) < abs(p_sure - 0.5)
    assert lo_sure < p_sure < hi_sure


# ==========================================================================
# 4. training / serving 一致性
# ==========================================================================


@pytest.mark.parametrize("period", ["game", "day", "week", "biweek", "month"])
def test_features_from_state_matches_state_machine(period):
    """golden test：快照路径与状态机路径必须逐位相同（§7.4 training/serving skew）。

    做法：把最后一场所属 period 整段切掉，用更早的历史导出快照，再用快照复算最后
    一场的特征，与状态机在同一场上写出的那一行比。两条路径只要有一个公式抄错就会红。
    **五种 period 都要过**：推理路径自己要把 `now` floor 到同一张 period 网格上，
    不 floor 的话 elapsed 会多出最多一个 period，RD 就会比训练时宽一点点——
    这种偏差小到不会报错，只会让线上比离线差，是最难查的一类 skew。
    """
    df = _synth(600)
    cfg = Glicko2Config(period=period)
    full = run_glicko_state_machine(df, cfg=cfg)
    last = df.iloc[-1]
    start = pd.Timestamp(
        round(_period_start_scalar(last.kickoff_utc, period) * 86_400e9), tz="UTC"
    )
    hist = df[df.kickoff_utc < start].reset_index(drop=True)
    _, state = run_glicko_state_machine(hist, cfg=cfg, return_state=True)
    # now 传**真实开赛时刻**（线上就是这么调的）
    got = features_from_glicko_state(
        state,
        blue_team_id=last.blue_team_id,
        red_team_id=last.red_team_id,
        blue_player_ids=last.blue_player_ids,
        red_player_ids=last.red_player_ids,
        now=last.kickoff_utc,
    )
    want = full.iloc[-1].to_dict()
    for k in FEATURE_COLS:
        assert got[k] == pytest.approx(want[k], abs=1e-9), k


def test_snapshot_flushes_partial_period_so_serving_is_fresher_than_training():
    """记录一条会被忽略的语义：**导出快照时那半个没走完的 period 已经被 flush 了**。

    线上流程是"拿全部历史导出快照 → 预测下一场"。若 period="week"，快照里已经含了
    本周早些时候的比赛，而训练时同一周的特征读的是周初状态——线上比离线更新鲜。
    实测这个 skew 对 log loss 是好事（0.62403 → 0.62091）、对 ECE 是坏事
    （0.0073 → 0.0110），详见模块 docstring ⑧。

    本测试不主张哪种对，只把行为钉死：
      * period="week"：用"含同周比赛"的快照复算，会与状态机那一行**不同**；
      * period="game"：逐场即 period，怎么切快照都逐位一致，**不存在这个 skew**。
    """
    ts = pd.Timestamp("2022-03-01", tz="UTC")  # 周二，同周内连打 4 场
    df = pd.DataFrame(
        [
            {
                "game_id": f"g{i}",
                "series_id": f"s{i}",
                "kickoff_utc": ts + pd.Timedelta(days=i),
                "league_canon": "LCK",
                "league_tier": 1,
                "blue_team_id": "A",
                "red_team_id": "B",
                "blue_player_ids": [f"a{j}" for j in range(5)],
                "red_player_ids": [f"b{j}" for j in range(5)],
                "y_blue_win": 1,
            }
            for i in range(4)
        ]
    )
    last = df.iloc[-1]
    kw = {
        "blue_team_id": last.blue_team_id,
        "red_team_id": last.red_team_id,
        "blue_player_ids": last.blue_player_ids,
        "red_player_ids": last.red_player_ids,
        "now": last.kickoff_utc,
    }
    for period, expect_same in (("week", False), ("game", True)):
        cfg = Glicko2Config(period=period)
        online = run_glicko_state_machine(df, cfg=cfg).iloc[-1]
        _, state = run_glicko_state_machine(
            df.iloc[:-1].reset_index(drop=True), cfg=cfg, return_state=True
        )
        got = features_from_glicko_state(state, **kw)
        diff = max(abs(got[c] - float(online[c])) for c in FEATURE_COLS)
        if expect_same:
            assert diff < 1e-9, f"period=game 必须无 skew，实测 {diff}"
        else:
            # 快照已经把同周前三场（A 三连胜）吃进去了，状态机那一行还停在周初
            assert diff > 1.0, f"period=week 的残余应当看得见，实测 {diff}"
            assert got["team_glicko_diff"] > float(online["team_glicko_diff"])


def test_features_from_state_inflates_rd_for_stale_entities():
    """线上推理最容易漏的一步：半年没打的队不能顶着半年前那个窄 RD 出现。"""
    df = _synth(600)
    cfg = Glicko2Config(period="week")
    _, state = run_glicko_state_machine(df, cfg=cfg, return_state=True)
    kw = {
        "blue_team_id": "t0", "red_team_id": "t1",
        "blue_player_ids": [f"p0_{j}" for j in range(5)],
        "red_player_ids": [f"p1_{j}" for j in range(5)],
    }
    now = df.kickoff_utc.max()
    fresh = features_from_glicko_state(state, now=now, **kw)
    stale = features_from_glicko_state(state, now=now + pd.Timedelta(days=180), **kw)
    assert stale["team_rd_blue"] > fresh["team_rd_blue"] + 10
    assert stale["player_rd_blue"] > fresh["player_rd_blue"]


def test_state_snapshot_is_json_serializable():
    import json

    df = _synth(200)
    _, state = run_glicko_state_machine(df, return_state=True)
    assert json.loads(json.dumps(state))["n_games"] == 200


def test_unknown_ids_get_prior_at_serving_time():
    """未见过的 teamid（§9 冷启动表）：rating=1500、RD 顶到上限，而不是报错。"""
    df = _synth(200)
    _, state = run_glicko_state_machine(df, return_state=True)
    f = features_from_glicko_state(
        state,
        blue_team_id="brand-new-org", red_team_id="another-new-org",
        blue_player_ids=["x1", "x2", "x3", "x4", "x5"],
        red_player_ids=["y1", "y2", "y3", "y4", "y5"],
    )
    assert f["team_glicko_diff"] == 0.0
    assert f["team_rd_blue"] == pytest.approx(350.0)
    assert f["player_rd_max_blue"] == pytest.approx(350.0)


# ==========================================================================
# 5. 真实数据上的冒烟测试
# ==========================================================================


@needs_data
def test_real_data_smoke_and_no_holdout():
    """真实数据上跑一遍：形状、有限性，以及绝不碰 holdout。"""
    from lolwp.features.loader import holdout_start, load_dev

    df = load_dev().iloc[-6000:].reset_index(drop=True)
    assert df.kickoff_utc.max() < holdout_start()
    f = run_glicko_state_machine(df, cfg=Glicko2Config(period="week"))
    assert len(f) == len(df)
    assert np.isfinite(f.to_numpy(dtype=float)).all()
    assert f.team_rd_blue.between(20.0, 350.0).all()
    assert f.player_rd_blue.between(20.0, 350.0).all()
