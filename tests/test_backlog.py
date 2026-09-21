"""重测台账的守卫。

台账存在的理由：本项目有一批假设是「现在测不出来」而非「已被证伪」，
它们应当随数据积累自动复查。没有机制就只能靠人凭感觉想起来——而人想不起来。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "reports" / "retest_ledger.jsonl"


def test_every_parked_hypothesis_is_registered():
    """项目里写过「暂不采纳、待重测」的四个假设，必须都在台账里。

    漏登记一个，它就永远不会被复查。
    """
    from lolwp.ops.backlog import REGISTRY
    keys = {h.key for h in REGISTRY}
    for k in ["counter_reliability", "synergy_reliability",
              "series_score_diff", "mastery_diff"]:
        assert k in keys, f"{k} 未登记，将永远不会被复查"


def test_registry_entries_are_self_documenting():
    """每条都要写清门禁与当初搁置的理由，否则几年后没人看得懂该怎么判。"""
    from lolwp.ops.backlog import REGISTRY
    for h in REGISTRY:
        assert h.title and h.gate and h.why_parked
        assert len(h.why_parked) > 20, f"{h.key} 的搁置理由太简略"
        assert callable(h.run), f"{h.key} 没有可执行的重测实现"


@pytest.mark.skipif(not LEDGER.exists(), reason="台账尚未产生")
def test_ledger_is_append_only_and_parsable():
    """趋势比单次结论重要，所以台账必须是 append-only 的逐行 JSON。"""
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    assert rows, "台账为空"
    need = {"key", "run_date", "n_games", "statistic", "threshold", "passed", "margin"}
    for r in rows:
        assert need <= set(r), f"台账缺字段：{need - set(r)}"
    # 日期必须单调不减——append-only 的直接体现
    dates = [r["run_date"] for r in rows]
    assert dates == sorted(dates), "台账出现了乱序，说明有人改写而不是追加"


@pytest.mark.skipif(not LEDGER.exists(), reason="台账尚未产生")
def test_nothing_that_passed_the_gate_is_still_parked():
    """若某个假设已越过门禁却还留在 REGISTRY 且未进生产模型，
    说明有人跑过重测但没跟进。这条测试就是那个提醒。"""
    from lolwp.models.artifact import PROD_FEATURES
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r["key"]] = r          # 后出现的覆盖先出现的
    feature_of = {"series_score_diff": "series_score_diff",
                  "mastery_diff": "mastery_diff"}
    for key, r in latest.items():
        if not r["passed"]:
            continue
        col = feature_of.get(key)
        if col is None:
            pytest.fail(f"{key} 已越过门禁（余量 {r['margin']:.2f}×），"
                        "请人工评估并更新 REGISTRY")
        assert col in PROD_FEATURES, (
            f"{key} 已越过门禁（余量 {r['margin']:.2f}×）但 {col} 仍不在生产模型里。"
            "要么加进去，要么在 REGISTRY 里写明为何仍不采纳。")


def test_makefile_exposes_retest():
    mk = (ROOT / "Makefile").read_text()
    for t in ["retest:", "retest-history:", "install-cron:"]:
        assert t in mk, f"Makefile 缺 {t}"
    assert "scripts/retest.py" in mk
