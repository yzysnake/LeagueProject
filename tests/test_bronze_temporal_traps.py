"""bronze 层的时间语义陷阱。

**`year` 列是赛季年，不是 `date` 的日历年。** 实测 2,709 场（2.65%）两者不一致，
而且**两个方向都有**：

- 赛季年领先 2,612 场：欧洲二级联赛（LIT/PRM/LFL/EBL）的 2027 Winter 赛段
  在 2026-08-28 就开打，所以 2026 年的文件里有 69 场 year=2027 的比赛。
- 赛季年落后 97 场：年末杯赛跨年打完，如德玛西亚杯 year=2017 却在
  2018-01-04 进行；NASG 2020 在 2021 年 1 月。

后果：任何按 `year` 做时间切分 / 训练窗口 / 衰减权重的代码，都会让这 2,709 场
落进错误的时间桶——**这是一条不报错的时间泄漏通道**。
全项目一律以 `date` 为时间坐标，`year` 只能当作赛季标签使用。
"""
import os

import pandas as pd
import pytest

BRONZE = "data/bronze/all.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(BRONZE), reason="需先跑 bronze.build()")


@pytest.fixture(scope="module")
def games():
    df = pd.read_parquet(BRONZE, columns=["gameid", "year", "date", "position", "league"])
    return df[df.position == "team"].drop_duplicates("gameid")


def test_year_column_is_season_not_calendar(games):
    """记录这个事实，防止将来有人"顺手"用 year 做时间切分。"""
    mismatch = games[games.year != games.date.str[:4]]
    assert len(mismatch) > 0, "若此断言失败，说明 OE 改了语义，需重新评估本测试"
    ahead = (mismatch.year.astype(int) > mismatch.date.str[:4].astype(int)).sum()
    behind = (mismatch.year.astype(int) < mismatch.date.str[:4].astype(int)).sum()
    # 两个方向都存在，且领先方向占绝大多数（赛段提前开打比杯赛跨年常见得多）
    assert ahead > 0 and behind > 0, f"ahead={ahead} behind={behind}"
    assert ahead > behind * 5, f"方向比例异常 ahead={ahead} behind={behind}"
    # 错配幅度不超过一年；超过就是真的数据损坏
    delta = (mismatch.year.astype(int) - mismatch.date.str[:4].astype(int)).abs()
    assert delta.max() == 1, f"出现跨度大于一年的错配：{delta.max()}"


def test_season_year_ahead_of_calendar_is_bounded(games):
    """错配比例应在 1%-5% 量级。突然飙升说明 OE 数据出了问题。"""
    rate = (games.year != games.date.str[:4]).mean()
    assert 0.005 < rate < 0.06, f"year/date 错配率异常：{rate:.2%}"


def test_dates_are_iso_sortable(games):
    """date 是字符串，全项目依赖字符串比较做时间过滤，格式必须严格一致。"""
    assert games.date.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()


def test_holdout_cutoff_uses_date_not_year(games):
    """用 date 与用 year 切 holdout 会得到不同的集合，差异必须是已知的。"""
    by_date = set(games[games.date >= "2026-06-01"].gameid)
    by_year = set(games[games.year >= "2026"].gameid)
    assert by_date != by_year
    assert len(by_year - by_date) > 0, "用 year 切会多收进早于截止线的比赛"
