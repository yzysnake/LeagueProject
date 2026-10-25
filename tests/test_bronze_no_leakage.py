"""bronze 层必须物理上不含任何赛后统计列。

这是列级泄漏防线落在存储层的体现：即使后面有人写错代码想用 gamelength，
也拿不到——它根本不在文件里。
"""
import os

import pandas as pd
import pytest

from lolwp.features.whitelist import classify

BRONZE = "data/bronze/all.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(BRONZE), reason="需先跑 bronze.build()")


@pytest.fixture(scope="module")
def cols():
    return pd.read_parquet(BRONZE, columns=None).columns.tolist()


def test_no_forbidden_columns(cols):
    """rating_label（终局经济、时长）同样不许进 bronze：它们只在 data/silver/margins.parquet 里，
    当历史局的 BT 标签用（修订 A6）。"""
    bad = [c for c in cols if c != "data_snapshot_sha" and classify(c) in {"forbidden", "rating_label"}]
    assert not bad, f"bronze 里出现了禁用列：{bad}"


def test_known_leaky_columns_absent(cols):
    """点名几个最容易被误用的赛后列。"""
    for c in ["gamelength", "kills", "deaths", "totalgold", "golddiff", "dragons",
              "damagetochampions", "visionscore", "firstblood", "towers"]:
        assert c not in cols, f"{c} 不该出现在 bronze"


def test_result_present_but_marked_label_only(cols):
    assert "result" in cols
    assert classify("result") == "label_only"


def test_snapshot_sha_present(cols):
    assert "data_snapshot_sha" in cols
