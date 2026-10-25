"""三组位置搭配的英雄协同（features/role_pairs.py，登记在 backlog 待重测）：时间正确性与衰减的量纲。"""
import numpy as np
import pandas as pd
import pytest

from lolwp.features.role_pairs import role_pair_synergy

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def _games(kicks, blue, red, y):
    # silver 的开赛时间是微秒精度：测试数据也用同样的精度（量纲 bug 就是在这里藏住的）
    return pd.DataFrame({"kickoff_utc": pd.DatetimeIndex(kicks).as_unit("us"),
                         "blue_champ_keys": blue, "red_champ_keys": red, "y_blue_win": y})


def test_decay_uses_the_stated_half_life():
    """同一对 ADC+辅助隔 730 天再出现：上一局的残差权重正好减半。"""
    duo = [1, 2, 3, 10, 20]                     # top, jng, mid, bot, sup
    # 红方第二局换一套全新的英雄：特征里只剩蓝方那一对的历史
    g = _games([T0, T0 + pd.Timedelta(days=730)], [duo, duo], [[4, 5, 6, 7, 8], [40, 50, 60, 70, 80]], [1, 0])
    s = role_pair_synergy(g, np.array([0.5, 0.5]), k=1.0, hl_days=730.0, pairs={"bot+sup": (3, 4)})
    assert s[0] == 0.0
    assert s[1] == pytest.approx(0.5 * 0.5 / (0.5 + 1.0))  # 残差 +0.5，权重 0.5，收缩 k=1


def test_never_sees_its_own_or_later_results():
    rng = np.random.default_rng(0)
    n = 300
    blue = [list(rng.integers(0, 6, 5)) for _ in range(n)]
    red = [list(rng.integers(0, 6, 5)) for _ in range(n)]
    y = rng.integers(0, 2, n)
    kicks = [T0 + pd.Timedelta(hours=6 * i) for i in range(n)]
    p = rng.uniform(0.3, 0.7, n)
    a = role_pair_synergy(_games(kicks, blue, red, y), p)
    y2 = y.copy()
    y2[200:] = 1 - y2[200:]
    b = role_pair_synergy(_games(kicks, blue, red, y2), p)
    assert np.array_equal(a[:201], b[:201]) and not np.array_equal(a[201:], b[201:])


def test_only_the_three_role_pairs_count():
    """上单与辅助都没换、但三组位置对全换了：不相干的「上单+辅助」组合不带任何历史，特征为 0；
    对照：只把 ADC+辅助那一对原样再用一次，就有历史了。"""
    p = np.array([0.5, 0.5])
    g = _games([T0, T0 + pd.Timedelta(days=1)],
               [[1, 2, 3, 4, 5], [1, 99, 98, 97, 5]],
               [[6, 7, 8, 9, 10], [60, 70, 80, 90, 100]], [1, 1])
    assert role_pair_synergy(g, p)[1] == 0.0
    g2 = _games([T0, T0 + pd.Timedelta(days=1)],
                [[1, 2, 3, 4, 5], [11, 12, 13, 4, 5]],
                [[6, 7, 8, 9, 10], [60, 70, 80, 90, 100]], [1, 1])
    assert role_pair_synergy(g2, p)[1] > 0
