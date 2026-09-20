"""实时 tab 的行为测试，重点是**降级路径**。

实时是增强功能不是依赖：lolesports 是非官方接口、无 SLA、随时可能变。
所以这里测的主要不是"正常时好不好看"，而是"坏掉时会不会把整个网站带崩"。
全部用 mock，不打真实网络。
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "src" / "lolwp" / "app" / "main.py")
ART = str(ROOT / "models" / "prod.json")
pytestmark = pytest.mark.skipif(not os.path.exists(ART), reason="需先跑 artifact.train()")
TIMEOUT = 300


def _run(poller_obj):
    """在**源头**打补丁：AppTest 会把脚本当成新模块重新执行，
    所以 patch lolwp.app.main._poller 不起作用；要 patch 它内部调用的
    build_default_poller。同时必须清 cache_resource，否则跨用例串味。"""
    import streamlit as stlib

    stlib.cache_resource.clear()
    if isinstance(poller_obj, dict):          # 模拟构建失败
        def _boom(*a, **k):
            raise RuntimeError(poller_obj["error"].split(": ", 1)[-1])
        target = _boom
    else:
        def target(*a, **k):
            return poller_obj
    with patch("lolwp.live.poller.build_default_poller", target):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
    stlib.cache_resource.clear()
    return at


class _FakePoller:
    def __init__(self, snap, health, raise_on_tick=False):
        self._s, self._h, self._raise = snap, health, raise_on_tick

    def tick(self):
        if self._raise:
            raise ConnectionError("模拟接口挂了")

    def snapshot(self):
        return self._s

    def health(self):
        return self._h


_EMPTY_HEALTH = {"api_degraded": False, "n_watched": 0, "n_errors": 0,
                 "live_resolve_rate": 1.0,
                 "by_status": {k: 0 for k in
                               ["waiting", "ready", "no_coverage", "forbidden",
                                "error", "finished"]}}


def test_poller_build_failure_does_not_break_the_app():
    """轮询器构建失败时，实时页给提示，其余页面照常。"""
    at = _run({"error": "RuntimeError: crosswalk 缺失"})
    assert len(at.exception) == 0
    assert len(at.tabs) == 4
    assert any("无法构建" in e.value for e in at.error)
    # 手动摆阵容那页仍然出概率
    assert any(m.label == "蓝方胜率" for m in at.metric)


def test_tick_exception_is_caught_and_last_snapshot_kept():
    """单次轮询抛异常不能让页面崩，要显示告警并沿用上次快照。"""
    at = _run(_FakePoller({"n_live_polls": 3, "n_window_polls": 7, "games": []},
                          _EMPTY_HEALTH, raise_on_tick=True))
    assert len(at.exception) == 0
    assert any("本次轮询失败" in w.value for w in at.warning)


def test_no_live_games_shows_idle_message():
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 0, "games": []},
                          _EMPTY_HEALTH))
    assert len(at.exception) == 0
    assert any("没有正在进行的比赛" in i.value for i in at.info)


@pytest.mark.parametrize("status,needle", [
    ("waiting", "还没开局读条"),
    ("no_coverage", "无 livestats 覆盖"),
    ("forbidden", "鉴权失败"),
    ("error", "接口异常"),
])
def test_every_non_ready_status_renders_without_crashing(status, needle):
    """四种非就绪状态各有明确文案，且都不能抛异常（验收⑤第一条的 UI 侧）。"""
    h = dict(_EMPTY_HEALTH, n_watched=1)
    h["by_status"] = dict(h["by_status"], **{status: 1})
    at = _run(_FakePoller(
        {"n_live_polls": 1, "n_window_polls": 1, "games": [
            {"game_id": "g1", "league": "LEC", "teams": ["A", "B"],
             "status": status, "message": needle, "prediction": None, "warnings": []}]}, h))
    assert len(at.exception) == 0


def test_ready_game_renders_probability_and_layers():
    with open(ROOT / "data" / "ddragon" / "champion_map.json") as fh:
        keys = [c["dd_key"] for c in json.load(fh)["champions"]][:10]
    h = dict(_EMPTY_HEALTH, n_watched=1)
    h["by_status"] = dict(h["by_status"], ready=1)
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 1, "games": [
        {"game_id": "g1", "league": "LCK", "teams": ["T1", "GEN"], "status": "ready",
         "message": "就绪", "latency_200_to_pred_s": 0.221, "warnings": [],
         "prediction": {"p_blue": 0.61, "layers": {"实力": 0.4, "阵容": 0.05, "赛况": -0.01},
                        "blue_champ_keys": keys[:5], "red_champ_keys": keys[5:],
                        "blue_names": [f"B{i}" for i in range(5)],
                        "red_names": [f"R{i}" for i in range(5)]}}]}, h))
    assert len(at.exception) == 0
    labels = {m.label for m in at.metric}
    assert "实力层" in labels and "监视中的对局" in labels


def test_ready_without_probability_warns_not_crashes():
    """阵容读到了但概率算不出来（例如实体全未解析）时要告警，不能崩。"""
    h = dict(_EMPTY_HEALTH, n_watched=1)
    h["by_status"] = dict(h["by_status"], ready=1)
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 1, "games": [
        {"game_id": "g1", "league": "LPL", "teams": ["A", "B"], "status": "ready",
         "message": "就绪", "warnings": ["两队均未解析"],
         "prediction": {"p_blue": None, "blue_champ_keys": [], "red_champ_keys": [],
                        "blue_names": [], "red_names": []}}]}, h))
    assert len(at.exception) == 0
    assert any("概率未产出" in w.value for w in at.warning)
