"""分层 BT 的单测。

重点不在"跑得通"，而在三件会静默出错的事：
  1. **未来信息泄漏**——rating 是回归拟合出来的，最容易在窗口边界上多吃一天。
     用 §4.7.1 T2 的未来删除不变性钉死，且要求 **bit-identical**（档 A）。
  2. **梯度写错**——解析梯度和目标函数不一致时 L-BFGS 会收敛到一个错误的点，
     不报错，只是指标差一点点。用有限差分逐项核对。
  3. **收缩方向反了**——冷启动的兜底链写反（新人给 0 而不是给队伍层）同样静默。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lolwp.models.hier_bt import (
    FEATURE_COLUMNS,
    HierBTConfig,
    _build_design,
    _neg_log_post,
    bridge_census,
    fit_snapshot,
    infer_team_region,
    refit_grid,
    rolling_ratings,
    variance_components,
)

T0 = pd.Timestamp("2020-01-01", tz="UTC")


def make_games(
    n: int = 1200,
    *,
    n_teams_per_region: int = 8,
    regions: tuple[str, ...] = ("RA", "RB"),
    seed: int = 0,
    region_offset: dict[str, float] | None = None,
    p_sub: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """按模型自己的生成过程造数据：每队 5 名固定选手，选手潜力 ~ N(赛区偏移, 0.25)。

    两个刻意加进来的细节，少了哪个这份合成数据都会骗人：
      * **蓝红方随机**——否则强队永远在蓝方，全部信号被截距吃掉，
        模型会"什么都学不到"却仍然收敛（这个坑在写这份测试时真的踩到了）。
      * **`p_sub` 的替补**——固定阵容下同队 5 人完全共线，选手层根本不可识别；
        真实职业赛里替补/转会正是把选手从队伍里分离出来的唯一信息源。

    返回 (games, true_player_rating)。
    """
    rng = np.random.default_rng(seed)
    region_offset = region_offset or dict.fromkeys(regions, 0.0)
    team_region: dict[str, str] = {}
    roster: dict[str, list[str]] = {}
    pool: dict[str, list[str]] = {g: [] for g in regions}
    truth: dict[str, float] = {}
    for g in regions:
        for i in range(n_teams_per_region):
            t = f"{g}_T{i}"
            team_region[t] = g
            roster[t] = [f"{t}_P{j}" for j in range(5)]
            for p in roster[t]:
                truth[p] = region_offset[g] + rng.normal(0.0, 0.25)
                pool[g].append(p)

    def lineup(team: str) -> list[str]:
        out = list(roster[team])
        if rng.random() < p_sub:
            slot = int(rng.integers(5))
            cand = pool[team_region[team]][int(rng.integers(len(pool[team_region[team]])))]
            out[slot] = cand
        return out

    rows = []
    for k in range(n):
        if k % 12 == 0 and len(regions) > 1:          # ~8% 跨赛区，与实测量级相当
            ga, gb = rng.choice(len(regions), size=2, replace=False)
            a = f"{regions[ga]}_T{rng.integers(n_teams_per_region)}"
            b = f"{regions[gb]}_T{rng.integers(n_teams_per_region)}"
            league, tier = "INTL", 2
        else:
            g = regions[int(rng.integers(len(regions)))]
            ia, ib = rng.choice(n_teams_per_region, size=2, replace=False)
            a, b = f"{g}_T{ia}", f"{g}_T{ib}"
            league, tier = g, 1
        if rng.random() < 0.5:                        # 蓝红方随机
            a, b = b, a
        la, lb = lineup(a), lineup(b)
        eta = sum(truth[p] for p in la) - sum(truth[p] for p in lb)
        y = int(rng.random() < 1.0 / (1.0 + np.exp(-eta)))
        rows.append(
            {
                "game_id": f"g{k}",
                "kickoff_utc": T0 + pd.Timedelta(hours=6 * k),
                "league_canon": league,
                "league_tier": tier,
                "blue_team_id": a,
                "red_team_id": b,
                "blue_player_ids": la,
                "red_player_ids": lb,
                "y_blue_win": y,
            }
        )
    return pd.DataFrame(rows), truth


# ---------------------------------------------------------------------------
# 1. 目标函数与梯度
# ---------------------------------------------------------------------------
def test_gradient_matches_finite_difference() -> None:
    df, _ = make_games(200, n_teams_per_region=4, seed=1)
    cfg = HierBTConfig(half_life_days=1e6, window_years=10.0)
    as_of = df.kickoff_utc.max() + pd.Timedelta(days=1)
    d = _build_design(df, as_of, cfg)
    n = len(d.pid) + 2 * len(d.tid) + len(d.lid) + 1
    rng = np.random.default_rng(7)
    theta = rng.normal(0.0, 0.3, n)
    _, g = _neg_log_post(theta, d, cfg)
    eps = 1e-6
    for j in rng.choice(n, size=25, replace=False):
        up = theta.copy()
        dn = theta.copy()
        up[j] += eps
        dn[j] -= eps
        num = (_neg_log_post(up, d, cfg)[0] - _neg_log_post(dn, d, cfg)[0]) / (2 * eps)
        assert abs(num - g[j]) < 1e-4 * max(1.0, abs(g[j])), (j, num, g[j])


# ---------------------------------------------------------------------------
# 2. 参数恢复
# ---------------------------------------------------------------------------
def test_recovers_player_ordering() -> None:
    df, truth = make_games(3000, seed=2)
    cfg = HierBTConfig(half_life_days=1e6, window_years=10.0, sigma_player=0.25)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1), cfg)
    keys = sorted(snap.r_player)
    est = np.array([snap.r_player[k] for k in keys])
    tru = np.array([truth[k] for k in keys])
    r = float(np.corrcoef(est, tru)[0, 1])
    # 同队 5 人高度共线（只有替补局把他们分开），单个选手不可能被完美分离；
    # 能到 0.5 就说明选手层信号确实被抓住了，而不只是抄了队伍层。
    assert r > 0.5, r


def test_identifies_region_offset() -> None:
    """跨赛区对战是 L_region 的唯一信息源。造一个 RA 比 RB 强的世界，看能不能量出来。"""
    df, _ = make_games(3000, seed=3, region_offset={"RA": 0.20, "RB": -0.20})
    cfg = HierBTConfig(half_life_days=1e6, window_years=10.0)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1), cfg)
    assert snap.l_region["RA"] > snap.l_region["RB"]
    # 真值：每队 5 人 × 0.40 的赛区差 = 队伍层差 2.0，ℓ 是"每个选手位"的尺度 → 0.40
    gap = snap.l_region["RA"] - snap.l_region["RB"]
    assert 0.10 < gap < 0.60, gap


def test_region_unidentifiable_without_bridges() -> None:
    """没有跨赛区对战时 L_region 必须塌回先验（=0），而不是编一个数出来。"""
    df, _ = make_games(1200, seed=4, region_offset={"RA": 0.3, "RB": -0.3})
    df = df[df.league_tier < 2].reset_index(drop=True)      # 删掉全部跨区赛
    cfg = HierBTConfig(half_life_days=1e6, window_years=10.0)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1), cfg)
    assert abs(snap.l_region["RA"] - snap.l_region["RB"]) < 0.02


# ---------------------------------------------------------------------------
# 3. 时间正确性（§4.7.1 T2 档 A）
# ---------------------------------------------------------------------------
def test_future_deletion_invariance_snapshot() -> None:
    df, _ = make_games(2000, seed=5)
    cut = df.kickoff_utc.iloc[1200]
    cfg = HierBTConfig(window_years=10.0)
    a = fit_snapshot(df, cut, cfg)
    b = fit_snapshot(df[df.kickoff_utc < cut].reset_index(drop=True), cut, cfg)
    assert a.r_player.keys() == b.r_player.keys()
    diff = max(abs(a.r_player[k] - b.r_player[k]) for k in a.r_player)
    assert diff == 0.0, diff          # bit-identical，档 A
    assert a.intercept == b.intercept
    assert a.l_region == b.l_region


def test_future_deletion_invariance_rolling() -> None:
    """整条滚动链路的未来删除不变性：删掉 cut 之后的比赛，cut 之前每一行特征逐位相等。"""
    df, _ = make_games(2000, seed=6)
    cfg = HierBTConfig(window_years=10.0, refit_every_days=30)
    cut = df.kickoff_utc.iloc[1400]
    full_feats = rolling_ratings(df, cfg, start=T0 + pd.Timedelta(days=90))
    trunc = df[df.kickoff_utc < cut].reset_index(drop=True)
    trunc_feats = rolling_ratings(trunc, cfg, start=T0 + pd.Timedelta(days=90))
    m = (df.kickoff_utc < cut).to_numpy()
    for c in FEATURE_COLUMNS:
        a = full_feats[c].to_numpy()[m]
        b = trunc_feats[c].to_numpy()
        assert np.array_equal(a, b), (c, np.abs(a - b).max())


def test_snapshot_ignores_games_at_or_after_as_of() -> None:
    """边界是 `<`，不是 `<=`。多吃开赛当刻那一场就是把赛果喂进了自己的特征。"""
    df, _ = make_games(800, seed=7)
    cfg = HierBTConfig(window_years=10.0)
    as_of = df.kickoff_utc.iloc[500]
    snap = fit_snapshot(df, as_of, cfg)
    assert snap.diagnostics["n_games"] == 500


def test_perturbed_init_reaches_same_optimum() -> None:
    """MAP 是凸问题的点估计：换起点应当落到同一个解（档 B 的跨 seed 判据，这里更严）。"""
    df, _ = make_games(1500, seed=8)
    cfg = HierBTConfig(window_years=10.0)
    as_of = df.kickoff_utc.max() + pd.Timedelta(days=1)
    a = fit_snapshot(df, as_of, cfg)
    rng = np.random.default_rng(99)
    warm = fit_snapshot(df, as_of, cfg)
    warm.r_player = {k: v + rng.normal(0, 0.2) for k, v in warm.r_player.items()}
    warm.m_team = {k: v + rng.normal(0, 0.2) for k, v in warm.m_team.items()}
    b = fit_snapshot(df, as_of, cfg, warm=warm)
    dmax = max(abs(a.r_player[k] - b.r_player[k]) for k in a.r_player)
    assert dmax < 1e-3, dmax


# ---------------------------------------------------------------------------
# 4. 赛区归属与桥接普查
# ---------------------------------------------------------------------------
def test_infer_team_region_ignores_international() -> None:
    df, _ = make_games(600, seed=9)
    tr = infer_team_region(df)
    assert set(tr.values()) == {"RA", "RB"}      # 绝不能出现 "INTL"
    assert tr["RA_T0"] == "RA"


def test_bridge_census_counts_pairs() -> None:
    df, _ = make_games(600, seed=10)
    bc = bridge_census(df)
    assert bc["n_pairs"] == 1                     # 只有 RA-RB 一对
    assert bc["n_cross"] == int(bc["pairs"][("RA", "RB")])
    assert bc["n_cross"] > 0


# ---------------------------------------------------------------------------
# 5. 冷启动兜底链（§5.5）
# ---------------------------------------------------------------------------
def test_cold_start_fallback_chain() -> None:
    df, _ = make_games(1500, seed=11, region_offset={"RA": 0.25, "RB": -0.25})
    cfg = HierBTConfig(window_years=10.0)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1), cfg)
    # 没见过的选手 → 其队伍的 m_T，而不是 0
    assert snap.player_rating("brand_new_guy", "RA_T0") == snap.m_team["RA_T0"]
    # 没见过的队 → 该赛区的 ℓ_g（此处 team_region 也查不到 → UNKNOWN → 0）
    assert snap.team_level("never_seen_team") == 0.0
    f = snap.features(
        blue_team_id="RA_T0", red_team_id="RB_T0",
        blue_player_ids=["brand_new_guy"] * 5,
        red_player_ids=[f"RB_T0_P{j}" for j in range(5)],
    )
    assert f["bt_known_players"] == 5.0
    assert f["bt_league_diff"] == pytest.approx(
        snap.l_region["RA"] - snap.l_region["RB"]
    )


def test_features_are_antisymmetric() -> None:
    """蓝红互换必须让所有 diff 变号。§2.4：镜像对称是硬约束，写反了指标只差一点点。"""
    df, _ = make_games(1200, seed=12)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1),
                        HierBTConfig(window_years=10.0))
    bl = [f"RA_T0_P{j}" for j in range(5)]
    rd = [f"RB_T1_P{j}" for j in range(5)]
    a = snap.features(blue_team_id="RA_T0", red_team_id="RB_T1",
                      blue_player_ids=bl, red_player_ids=rd)
    b = snap.features(blue_team_id="RB_T1", red_team_id="RA_T0",
                      blue_player_ids=rd, red_player_ids=bl)
    for c in ("bt_player_diff", "bt_league_diff", "bt_chem_diff", "bt_team_level_diff"):
        assert a[c] == pytest.approx(-b[c], abs=1e-12)
    assert a["bt_known_players"] == b["bt_known_players"]


# ---------------------------------------------------------------------------
# 6. 转会继承 —— 这是"建在选手层"相对"建在队伍层"的全部意义
# ---------------------------------------------------------------------------
def test_transfer_carries_rating() -> None:
    """一名强选手从强队转到弱队，新队的预测强度必须立刻上抬。

    队伍层 rating 做不到这件事（teamid 没变，历史还是弱队的），
    这条测试就是把"为什么非得建在选手层"钉成可执行的断言。
    """
    df, truth = make_games(4000, n_teams_per_region=6, regions=("RA",), seed=13,
                           p_sub=0.35)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1),
                        HierBTConfig(window_years=10.0, half_life_days=1e6))
    order = sorted(truth, key=lambda p: truth[p])
    worst, best = order[:5], order[-5:]
    weak_team = worst[0].rsplit("_P", 1)[0]
    opp = "RA_T5" if weak_team != "RA_T5" else "RA_T4"
    opp_line = [f"{opp}_P{j}" for j in range(5)]
    before = snap.features(blue_team_id=weak_team, red_team_id=opp,
                           blue_player_ids=worst, red_player_ids=opp_line)["bt_player_diff"]
    after = snap.features(blue_team_id=weak_team, red_team_id=opp,
                          blue_player_ids=best, red_player_ids=opp_line)["bt_player_diff"]
    assert after > before + 0.3, (before, after)


# ---------------------------------------------------------------------------
# 7. 杂项
# ---------------------------------------------------------------------------
def test_refit_grid_is_anchored_to_data_start() -> None:
    """网格锚在数据首日：换评测起点不能让同一场比赛拿到不同快照。"""
    df, _ = make_games(400, seed=14)
    cfg = HierBTConfig(refit_every_days=14)
    g1 = refit_grid(df, cfg, start=T0 + pd.Timedelta(days=28))
    g2 = refit_grid(df, cfg, start=T0 + pd.Timedelta(days=56))
    assert set(g2).issubset(set(g1))


def test_rolling_ratings_shape_and_index() -> None:
    df, _ = make_games(600, seed=15)
    df.index = pd.RangeIndex(100, 700)
    feats = rolling_ratings(df, HierBTConfig(refit_every_days=30, window_years=10.0))
    assert list(feats.columns) == list(FEATURE_COLUMNS)
    assert feats.index.equals(df.index)
    assert np.isfinite(feats.to_numpy()).all()


def test_empty_window_returns_zero_snapshot() -> None:
    df, _ = make_games(300, seed=16)
    snap = fit_snapshot(df, T0 - pd.Timedelta(days=1), HierBTConfig())
    assert snap.diagnostics["fitted"] is False
    assert snap.features(
        blue_team_id="RA_T0", red_team_id="RA_T1",
        blue_player_ids=["a"] * 5, red_player_ids=["b"] * 5,
    )["bt_player_diff"] == 0.0


def test_variance_components_reports_all_layers() -> None:
    df, _ = make_games(1500, seed=17)
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1),
                        HierBTConfig(window_years=10.0))
    vc = variance_components(snap)
    assert vc["n_players"] == 80  # 2 赛区 × 8 队 × 5 人
    assert vc["sd_player_total"] > 0
    assert vc["sd_player_within_team"] >= 0


# ---------------------------------------------------------------------------
# 8. 补充的时间正确性：标签扰动不变性
# ---------------------------------------------------------------------------
def test_future_label_perturbation_invariance() -> None:
    """把 T 之后所有比赛的**胜负翻转**，T 之前的特征必须逐位不变。

    这条比未来删除不变性更尖锐：删除能被"窗口下界也跟着动"之类的 bug 掩盖过去
    （两边都少了同一批数据，差值可能仍然相等），而翻转只改 `y`，
    行数、权重、设计矩阵全都不动。只要目标函数里漏进了一个 `t >= as_of` 的 y，
    这条就会立刻红，而未来删除不变性未必。
    """
    df, _ = make_games(2000, seed=21)
    cfg = HierBTConfig(window_years=10.0, refit_every_days=30)
    cut = df.kickoff_utc.iloc[1300]
    flipped = df.copy()
    m_future = (flipped.kickoff_utc >= cut).to_numpy()
    flipped.loc[m_future, "y_blue_win"] = 1 - flipped.loc[m_future, "y_blue_win"]
    a = rolling_ratings(df, cfg, start=T0 + pd.Timedelta(days=90))
    b = rolling_ratings(flipped, cfg, start=T0 + pd.Timedelta(days=90))
    past = ~m_future
    for c in FEATURE_COLUMNS:
        x, z = a[c].to_numpy()[past], b[c].to_numpy()[past]
        assert np.array_equal(x, z), (c, float(np.abs(x - z).max()))
    # 反向哨兵：翻转之后的段必须真的变了，否则上面那条是"什么都没测"
    assert not np.array_equal(
        a["bt_player_diff"].to_numpy()[m_future], b["bt_player_diff"].to_numpy()[m_future]
    )


def test_refit_is_bit_identical_across_runs() -> None:
    """同一输入连跑两次必须逐位相同（§4.7.1 T2 **档 A**，比 BT 被要求的档 B 严一档）。

    MAP + L-BFGS 本身是确定性的，但 `_build_design` 里的 id 分配、
    "并列时取谁"的打破平局规则一旦依赖 dict/groupby 顺序就会漂。
    """
    df, _ = make_games(1500, seed=22)
    cfg = HierBTConfig(window_years=10.0)
    as_of = df.kickoff_utc.max() + pd.Timedelta(days=1)
    a = fit_snapshot(df, as_of, cfg)
    b = fit_snapshot(df.copy(), as_of, cfg)
    assert a.r_player == b.r_player
    assert a.m_team == b.m_team
    assert a.c_team == b.c_team
    assert a.l_region == b.l_region
    assert a.intercept == b.intercept


def test_unknown_team_in_known_region_falls_back_to_l_region() -> None:
    """兜底链的中间一环：队伍没见过、但赛区见过 → 用 ℓ_g，不是 0。

    `test_cold_start_fallback_chain` 只覆盖了"赛区也不认识"那一端（→0），
    中间这一环写错（直接 return 0.0）不会被它抓到。
    """
    df, _ = make_games(1500, seed=23, region_offset={"RA": 0.25, "RB": -0.25})
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1),
                        HierBTConfig(window_years=10.0))
    snap.team_region["expansion_team"] = "RA"
    assert "expansion_team" not in snap.m_team
    assert snap.team_level("expansion_team") == snap.l_region["RA"]
    # 新队里的新人 → 该赛区水平，而不是全球平均
    assert snap.player_rating("rookie", "expansion_team") == snap.l_region["RA"]


# ---------------------------------------------------------------------------
# 9. 名单不全（silver 里 3.6% 的比赛真的只有 3-4 个 playerid）
# ---------------------------------------------------------------------------
def _drop_one(df: pd.DataFrame, rows: np.ndarray, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = [
        list(v)[:4] if i in set(rows.tolist()) else list(v)
        for i, v in enumerate(out[col])
    ]
    return out


def test_missing_roster_slot_uses_team_level_not_zero() -> None:
    """只报了 4 个人时，第 5 个位子必须按"这支队的一名普通选手"算，而不是按 0 算。

    按 0 算会让强赛区的队伍**因为缺数据**而被削弱（rating 的零点是全球平均，
    不是该赛区平均），而且只对那几场稳定地给错方向 —— 总指标上看不出来。
    """
    df, _ = make_games(1500, seed=24, region_offset={"RA": 0.35, "RB": -0.35})
    snap = fit_snapshot(df, df.kickoff_utc.max() + pd.Timedelta(days=1),
                        HierBTConfig(window_years=10.0))
    full5 = [f"RA_T0_P{j}" for j in range(5)]
    red = [f"RB_T0_P{j}" for j in range(5)]
    a = snap.features(blue_team_id="RA_T0", red_team_id="RB_T0",
                      blue_player_ids=full5, red_player_ids=red)
    b = snap.features(blue_team_id="RA_T0", red_team_id="RB_T0",
                      blue_player_ids=full5[:4], red_player_ids=red)
    # 补位取 m_T，所以缺一个人时 blue 的和恰好少掉"那个人 − m_T"这一截
    assert b["bt_player_diff"] == pytest.approx(
        a["bt_player_diff"] - (snap.r_player[full5[4]] - snap.m_team["RA_T0"]), abs=1e-12
    )
    # 关键断言：RA 是强区，缺人也绝不能把 blue 打到弱于"按 0 补位"那么惨
    zero_pad = a["bt_player_diff"] - snap.r_player[full5[4]]
    assert b["bt_player_diff"] > zero_pad
    assert b["bt_known_players"] == 9.0


def test_design_charges_missing_slots_to_m_team() -> None:
    """名单不全的那几场，账要记到 m_T 上，不能硬塞给在场的 4 个人。

    这里直接查设计矩阵而不是查拟合结果：统计量的对照组很难挑干净
    （少报的那个位子通常是某个固定选手，他本来就会因为少了 25% 的出场而被多收缩），
    而"A 的第 i 行在 m_blue 列上是不是恰好 +1"是个确定性事实，写成断言不会漂。
    """
    df, _ = make_games(400, n_teams_per_region=4, seed=25)
    rng = np.random.default_rng(3)
    rows = rng.choice(len(df), size=40, replace=False)
    cut = _drop_one(df, rows, "blue_player_ids")
    cfg = HierBTConfig(window_years=10.0, half_life_days=1e6)
    as_of = df.kickoff_utc.max() + pd.Timedelta(days=1)

    d_full = _build_design(df, as_of, cfg)
    d_cut = _build_design(cut, as_of, cfg)
    n_p, n_t = len(d_cut.pid), len(d_cut.tid)
    m_off = n_p + n_t + 1
    A = d_cut.A.toarray()
    Af = d_full.A.toarray()
    # 完整名单：m 块整列为 0 —— 模块顶部"m 只进先验"的论证对 96.4% 的比赛原样成立
    assert np.all(Af[:, m_off:] == 0.0)
    hit = set(rows.tolist())
    for i in range(len(cut)):
        t = d_cut.tid[str(cut.blue_team_id.iloc[i])]
        want = 1.0 if i in hit else 0.0
        assert A[i, m_off + t] == want, (i, A[i, m_off + t], want)
        # 在场选手仍然是 ±1，没有被放大去顶替缺位那个人
        assert A[i, :n_p].sum() == pytest.approx(
            len(list(cut.blue_player_ids.iloc[i])) - len(list(cut.red_player_ids.iloc[i]))
        )
    # 红方一侧没动过 → 红队的 m 列必须全 0
    for i in range(len(cut)):
        t = d_cut.tid[str(cut.red_team_id.iloc[i])]
        if str(cut.red_team_id.iloc[i]) != str(cut.blue_team_id.iloc[i]):
            contrib = A[i, m_off + t]
            assert contrib == 0.0, (i, contrib)


def test_gradient_matches_finite_difference_with_missing_slots() -> None:
    """m_T 进了似然之后梯度多了一块，必须重新逐项核对。"""
    df, _ = make_games(300, n_teams_per_region=4, seed=26)
    rng = np.random.default_rng(11)
    df = _drop_one(df, rng.choice(len(df), size=60, replace=False), "red_player_ids")
    cfg = HierBTConfig(half_life_days=1e6, window_years=10.0)
    d = _build_design(df, df.kickoff_utc.max() + pd.Timedelta(days=1), cfg)
    n = len(d.pid) + 2 * len(d.tid) + len(d.lid) + 1
    theta = rng.normal(0.0, 0.3, n)
    _, g = _neg_log_post(theta, d, cfg)
    eps = 1e-6
    for j in rng.choice(n, size=40, replace=False):
        up, dn = theta.copy(), theta.copy()
        up[j] += eps
        dn[j] -= eps
        num = (_neg_log_post(up, d, cfg)[0] - _neg_log_post(dn, d, cfg)[0]) / (2 * eps)
        assert abs(num - g[j]) < 1e-4 * max(1.0, abs(g[j])), (j, num, g[j])


# ---------------------------------------------------------------------------
# 软标签（赛后经济差，修订 A6）：似然里的 y 换成 [0,1] 里的值，时间正确性一条不能松
# ---------------------------------------------------------------------------
def _with_soft(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["y_margin"] = np.clip(out.y_blue_win * 0.7 + rng.uniform(0, 0.3, len(out)), 0.01, 0.99)
    return out


def test_soft_target_uses_the_named_column() -> None:
    df = _with_soft(make_games(400, seed=31)[0], 31)
    as_of = df.kickoff_utc.max() + pd.Timedelta(days=1)
    cfg = HierBTConfig(window_years=10.0)
    assert np.array_equal(_build_design(df, as_of, cfg, "y_margin").y, df.y_margin.to_numpy())
    assert np.array_equal(_build_design(df, as_of, cfg).y, df.y_blue_win.to_numpy(dtype=float))
    soft = fit_snapshot(df, as_of, cfg, target="y_margin")
    assert soft.diagnostics["target"] == "y_margin"
    assert soft.r_player != fit_snapshot(df, as_of, cfg).r_player


def test_future_soft_label_perturbation_invariance() -> None:
    """只改 T 之后各局的软标签（赛后经济差），T 之前的特征必须逐位不变——
    任何一局的赛后统计都进不了开赛不晚于它的那些局的特征。"""
    df = _with_soft(make_games(2000, seed=32)[0], 32)
    cfg = HierBTConfig(window_years=10.0, refit_every_days=30)
    cut = df.kickoff_utc.iloc[1300]
    moved = df.copy()
    m_future = (moved.kickoff_utc >= cut).to_numpy()
    moved.loc[m_future, "y_margin"] = 1 - moved.loc[m_future, "y_margin"]
    a = rolling_ratings(df, cfg, start=T0 + pd.Timedelta(days=90), target="y_margin")
    b = rolling_ratings(moved, cfg, start=T0 + pd.Timedelta(days=90), target="y_margin")
    past = ~m_future
    for c in FEATURE_COLUMNS:
        x, z = a[c].to_numpy()[past], b[c].to_numpy()[past]
        assert np.array_equal(x, z), (c, float(np.abs(x - z).max()))
    assert not np.array_equal(
        a["bt_player_diff"].to_numpy()[m_future], b["bt_player_diff"].to_numpy()[m_future]
    )
