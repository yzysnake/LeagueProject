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
    """实测全量 2014-2026 的最大无比赛空档是 34 天（2024-12-08 → 2025-01-11）。

    手册 v1.0 的 3 天、我自己第一版的 21 天、v1.2 建议的 35 天都不够安全。
    阈值必须严格大于历史最大空档，且留出余量。
    """
    OBSERVED_MAX_GAP = 34
    assert dv.STALE_FAIL_DAYS > OBSERVED_MAX_GAP + 5, (
        f"阈值 {dv.STALE_FAIL_DAYS} 天对历史最大空档 {OBSERVED_MAX_GAP} 天余量不足"
    )
    assert dv.STALE_WARN_DAYS >= 10


def test_winter_break_exempts_hard_fail(tmp_path, monkeypatch):
    """1 月初跑时，当年文件可能一场比赛都还没有或只有零星几场，不得硬失败。"""
    from datetime import date as _date
    p = tmp_path / "w.csv"
    shutil.copy(GOOD, p)
    # 假装今天是 1 月 20 日，而文件最后一场是去年 12 月 21 日（相隔 30 天）
    rec, warns = dv.validate(str(p), 2014, {}, _date(2015, 1, 20))
    assert rec["n_games"] == 918
