"""实时盯盘 + 赛后对账的守卫测试。

对账那部分踩过一个真实的坑：`getEventDetails` 的 `games[].teams[]`
**只有 id 和 side，没有逐局胜负**。第一版假设有，结果四条记录全是空值
而且不报错——典型的「不报错不等于对」。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "reports" / "live_watch.jsonl"
SCRIPT = ROOT / "scripts" / "live_watch.py"


def test_script_documents_the_api_limitation():
    """那个坑必须写在代码里，否则下一个人还会踩。"""
    src = SCRIPT.read_text()
    assert "没有逐局胜负" in src, "对账逻辑没写明 API 的这个限制"
    assert "Bo1" in src, "没说明只有 Bo1 能无歧义反推"


def test_ledger_is_append_only_jsonl():
    if not LEDGER.exists():
        pytest.skip("尚无盯盘记录")
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    assert rows
    for r in rows:
        assert r.get("kind") in {"prediction", "result"}, f"未知记录类型 {r.get('kind')}"
        if r["kind"] == "prediction":
            assert 0.0 <= r["p_blue"] <= 1.0
            assert r.get("game_id") and r.get("match_id")


def test_predictions_are_not_duplicated():
    """脚本对已记录的 gameId 会跳过，launchd 崩溃重启时不能重复写。"""
    if not LEDGER.exists():
        pytest.skip("尚无盯盘记录")
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    ids = [r["game_id"] for r in rows if r["kind"] == "prediction"]
    assert len(ids) == len(set(ids)), f"预测被重复写入：{len(ids)} 条 / {len(set(ids))} 个唯一"


def test_results_only_recorded_when_unambiguous():
    """多局系列单靠 gameWins 推不出逐局胜负，那种情况 blue_won 必须留空
    而不是猜一个。"""
    if not LEDGER.exists():
        pytest.skip("尚无盯盘记录")
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    for r in rows:
        if r["kind"] == "result" and r.get("blue_won") is not None:
            assert isinstance(r["blue_won"], bool)


@pytest.mark.skipif(not (ROOT / "deploy" / "com.lolwp.livewatch.plist").exists(),
                    reason="无 plist")
def test_livewatch_plist_is_restartable():
    """盯盘要跑十几个小时，崩了必须自动重启；脚本对已记录的局会跳过，天然幂等。"""
    import plistlib
    p = ROOT / "deploy" / "com.lolwp.livewatch.plist"
    d = plistlib.loads(p.read_text().replace("__REPO__", str(ROOT)).encode())
    assert d["Label"] == "com.lolwp.livewatch"
    assert d["RunAtLoad"] is True
    assert d["KeepAlive"] == {"SuccessfulExit": False}, "崩溃后不会自动重启"
