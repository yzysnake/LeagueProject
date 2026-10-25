"""近期状态（form）第二层：EWMA 本身、时间正确性、β 的拟合、训练与推理两条路逐位一致。"""
import numpy as np
import pandas as pd
import pytest

from lolwp.models.form import (
    DEFAULT,
    FormConfig,
    fit_beta,
    form_at,
    form_diff,
    form_entries,
    team_form,
)

T0 = pd.Timestamp("2026-01-01", tz="UTC")


def _games(n=400, seed=0, teams=8, series_len=3):
    """合成赛程：每个系列连打 series_len 局（同系列间隔 1 小时），系列之间隔 1 天。"""
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    while len(rows) < n:
        a, b = rng.choice(teams, 2, replace=False)
        for g in range(series_len):
            blue, red = (a, b) if g % 2 == 0 else (b, a)
            rows.append({"game_id": f"g{len(rows)}", "series_id": f"s{k}",
                         "kickoff_utc": T0 + pd.Timedelta(days=k, hours=g),
                         "blue_team_id": f"T{blue}", "red_team_id": f"T{red}",
                         "y_blue_win": int(rng.random() < 0.5)})
        k += 1
    df = pd.DataFrame(rows[:n])
    return df, np.clip(rng.uniform(0.3, 0.7, len(df)), 0.01, 0.99)


def test_team_form_is_a_shrunk_ewma():
    cfg = FormConfig(half_life_games=1.0, prior_games=1.0)          # a = 0.5
    assert team_form([], cfg) == 0.0
    # 最近一局权重 1、上一局 0.5：(0.4 + 0.5·(−0.2)) / (1 + 0.5 + 1)
    assert team_form([-0.2, 0.4], cfg) == pytest.approx(0.3 / 2.5)
    assert team_form([9.0] + [0.1] * 60, FormConfig(window=60)) == team_form([0.1] * 60, FormConfig(window=60))


def test_form_never_sees_its_own_or_later_results():
    """一局的 form 只用开赛更早的局：翻转 T 之后各局的赛果，T 及之前的 form 逐位不变；
    同一局自己的赛果也不影响自己。"""
    g, p = _games()
    a = form_diff(g, p)
    cut = 250
    flipped = g.copy()
    flipped.loc[cut:, "y_blue_win"] = 1 - flipped.loc[cut:, "y_blue_win"]
    b = form_diff(flipped, p)
    assert np.array_equal(a[:cut + 1], b[:cut + 1])
    assert not np.array_equal(a[cut + 1:], b[cut + 1:])


def test_simultaneous_games_do_not_see_each_other():
    g = pd.DataFrame({"series_id": ["s1", "s2"], "kickoff_utc": [T0, T0],
                      "blue_team_id": ["A", "A"], "red_team_id": ["B", "C"], "y_blue_win": [1, 0]})
    assert list(form_diff(g, np.array([0.5, 0.5]))) == [0.0, 0.0]


def test_series_games_count_only_when_configured():
    """第 2 局开打时第 1 局的赛果已知：默认算进去；include_series=False（对照）时只看之前的系列。"""
    g = pd.DataFrame({"series_id": ["s1", "s1"], "kickoff_utc": [T0, T0 + pd.Timedelta(hours=1)],
                      "blue_team_id": ["A", "B"], "red_team_id": ["B", "A"], "y_blue_win": [1, 1]})
    p = np.array([0.5, 0.5])
    with_s = form_diff(g, p, DEFAULT)
    assert with_s[0] == 0.0 and with_s[1] < 0                       # 第 1 局 A 超预期 → 第 2 局蓝方 B 吃亏
    assert form_diff(g, p, FormConfig(include_series=False))[1] == 0.0


def test_form_diff_is_antisymmetric():
    g, p = _games(200, seed=3)
    swapped = g.rename(columns={"blue_team_id": "red_team_id", "red_team_id": "blue_team_id"})
    swapped["y_blue_win"] = 1 - g.y_blue_win
    assert np.allclose(form_diff(swapped, 1 - p), -form_diff(g, p))


def test_fit_beta_recovers_the_slope():
    rng = np.random.default_rng(5)
    z, f = rng.normal(0, 1, 20000), rng.normal(0, 0.3, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(z + 0.8 * f)))).astype(float)
    assert fit_beta(z, f, y) == pytest.approx(0.8, abs=0.1)
    assert fit_beta(z, np.zeros_like(z), y) == 0.0


def test_serving_state_matches_training_exactly():
    """推理侧（form_entries + form_at）与训练侧（form_diff）对同一局给出同一个数：train-serve 不许 skew。"""
    g, p = _games(300, seed=7)
    f = form_diff(g, p)
    for k in (40, 150, 299):
        state = {"cfg": {"half_life_games": 5.0, "prior_games": 3.0, "window": 60, "include_series": True},
                 "silver": form_entries(g.iloc[:k], p[:k])}
        now = g.kickoff_utc.iloc[k]
        got = form_at(state, g.blue_team_id.iloc[k], now) - form_at(state, g.red_team_id.iloc[k], now)
        assert got == pytest.approx(f[k], abs=1e-12)


def test_serving_state_is_as_of_and_merges_live_entries():
    """OE 与台账两部分合并、按时间排；只用开赛早于 now 的条目（补录站在开局时刻也不会用上之后的局）。"""
    cfg = {"half_life_games": 1.0, "prior_games": 1.0, "window": 60, "include_series": True}
    state = {"cfg": cfg, "silver": {"A": [[T0.isoformat(), 0.4]]},
             "live": {"A": [[(T0 + pd.Timedelta(days=2)).isoformat(), -0.2]]}}
    c = FormConfig(**cfg)
    assert form_at(state, "A") == pytest.approx(team_form([0.4, -0.2], c))
    assert form_at(state, "A", T0 + pd.Timedelta(days=1)) == pytest.approx(team_form([0.4], c))
    assert form_at(state, "A", T0) == 0.0 and form_at(state, "nobody") == 0.0


def test_entries_keep_only_recent_teams_and_the_window():
    g, p = _games(300, seed=9)
    e = form_entries(g, p, FormConfig(window=10))
    assert all(len(h) <= 10 for h in e.values())
    assert all(h == sorted(h, key=lambda x: x[0]) for h in e.values())
    assert form_entries(g, p, since=g.kickoff_utc.max() + pd.Timedelta(days=1)) == {}
