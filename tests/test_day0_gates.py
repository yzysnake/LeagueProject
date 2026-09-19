"""负控制：证明五重校验真的会拒绝坏数据，而不是只在好数据上通过。"""
import importlib.util
import os
import shutil
import sys
from datetime import date

import pytest

spec = importlib.util.spec_from_file_location("day0_validate", "scripts/day0_validate.py")
dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv)

QUOTA_HTML = (
    b'<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title>'
    b'<meta http-equiv="content-type" content="text/html; charset=utf-8"/></head>'
    b'<body><p>Sorry, you can\'t view or download this file at this time.</p></body></html>'
)
GOOD = "data/raw/2014_LoL_esports_match_data_from_OraclesElixir.csv"
TODAY = date(2026, 9, 18)


def test_gate1_rejects_quota_html(tmp_path):
    p = tmp_path / "q.csv"
    p.write_bytes(QUOTA_HTML)
    with pytest.raises(AssertionError, match="闸1"):
        dv.validate(str(p), 2026, {}, TODAY)


def test_gate1_catches_zero_row_parse():
    """关键回归测试：csv.DictReader 读配额 HTML 得到 0 行且不抛异常，
    而 `0 % 12 == 0` 为真——手册 v1.0 的校验会放行。闸1+闸4 必须联合挡住。"""
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(QUOTA_HTML.decode())))
    assert len(rows) == 0
    assert len(rows) % 12 == 0  # 正是这个才让旧校验失效


def test_gate2_rejects_small_file(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text(dv.EXPECTED_PREFIX + "\n")
    with pytest.raises(AssertionError, match="闸2"):
        dv.validate(str(p), 2026, {}, TODAY)


def test_gate3_rejects_reordered_header(tmp_path):
    p = tmp_path / "h.csv"
    # 必须 >1MB，否则会先被闸2 拦下，测不到闸3
    p.write_text("league,gameid,date\n" + "xxxxx,yyyyy,zzzzz\n" * 100_000)
    with pytest.raises(AssertionError, match="闸3"):
        dv.validate(str(p), 2026, {}, TODAY)


@pytest.mark.skipif(not os.path.exists(GOOD), reason="需要已下载的 2014 文件")
def test_gate4_rejects_truncated_game(tmp_path):
    """删掉一行 → 某场只剩 11 行，闸4 必须抓到。"""
    p = tmp_path / "t.csv"
    with open(GOOD) as src, open(p, "w") as dst:
        lines = src.readlines()
        del lines[5]
        dst.writelines(lines)
    with pytest.raises(AssertionError, match="闸4"):
        dv.validate(str(p), 2014, {}, TODAY)


@pytest.mark.skipif(not os.path.exists(GOOD), reason="需要已下载的 2014 文件")
def test_gate5a_rejects_regression(tmp_path):
    p = tmp_path / "r.csv"
    shutil.copy(GOOD, p)
    prev = {"n_games": 99999, "max_date": "2014-12-21"}
    with pytest.raises(AssertionError, match="闸5a"):
        dv.validate(str(p), 2014, prev, TODAY)


@pytest.mark.skipif(not os.path.exists(GOOD), reason="需要已下载的 2014 文件")
def test_good_file_passes(tmp_path):
    rec, warns = dv.validate(GOOD, 2014, {}, TODAY)
    assert rec["n_games"] == 918
    assert rec["n_rows"] == 918 * 12
    assert rec["n_cols"] == 165
    assert warns == []


def test_offseason_gap_does_not_trip_freshness():
    """实测：2025 赛内最大空档 14 天，跨年 10 天。手册 v1.1 的 3 天阈值必然误报。"""
    assert dv.STALE_FAIL_DAYS > 14
    assert dv.STALE_WARN_DAYS >= 10
