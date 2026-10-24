"""站点各页的行为测试，重点是**降级路径**。

实时是增强功能不是依赖：lolesports 是非官方接口、无 SLA、随时可能变。
所以这里测的主要不是"正常时好不好看"，而是"坏掉时会不会把整个网站带崩"。
全部用 mock，不打真实网络。

站点是 st.navigation 的多页应用：AppTest 先跑一次首页（阵容模拟），再 switch_page 到要测的页。
每次交互只重跑当前页——旧版五个 tab 在同一个脚本里，这里的很多断言是从「tab 下标」改成「页」的。
"""
import json
import os
from contextlib import contextmanager
from datetime import date
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
LIVE, REVIEW, HISTORY = "views/live.py", "views/review.py", "views/history.py"


def _clear_poller():
    """只清轮询器的缓存：它在源头打了补丁，跨用例会串味；artifact / 名单等缓存留着省时间。"""
    from lolwp.app import data

    data.poller.clear()


def _run(poller_obj, page=None):
    """在**源头**打补丁：AppTest 会把页面脚本当成新模块重新执行，所以要 patch
    data.poller() 内部调用的 build_default_poller。所有 run() 都要在 patch 里做。"""
    _clear_poller()
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
        if page:
            at.switch_page(page).run()
    _clear_poller()
    return at


def _html(at) -> str:
    """页面上所有 st.html 的内容（对阵卡、提示条、页脚都是 HTML）。
    不含全局样式与开场脚本：那两段里有所有 class 名，留着的话「页面上有 lw-matchup」恒为真。"""
    return "\n".join(e.proto.body for e in at.get("html")
                     if not e.proto.body.lstrip().startswith(("<style", "<script")))


def _footer(at) -> bool:
    return "Riot Games" in _html(at)


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


# 和真实 Poller.health() 一致：没有监视中的对局时解析率无从计算，是 None 而不是 1.0。
# 这里曾经写成 1.0，结果 UI 把 None 直接 :.0% 格式化的崩溃一直没被测出来。
_EMPTY_HEALTH = {"api_degraded": False, "n_watched": 0, "n_errors": 0,
                 "live_resolve_rate": None,
                 "by_status": {k: 0 for k in
                               ["waiting", "ready", "no_coverage", "forbidden",
                                "error", "finished"]}}


def test_poller_build_failure_does_not_break_the_app():
    """轮询器构建失败时，实时页给提示；阵容模拟（首页）照常出概率。"""
    at = _run({"error": "RuntimeError: crosswalk 缺失"})
    assert len(at.exception) == 0
    assert "lw-matchup" in _html(at) and "%" in _html(at)       # 首页的胜率主视觉
    assert {"实力层", "阵容层", "赛况层"} <= {m.label for m in at.metric}
    at = _run({"error": "RuntimeError: crosswalk 缺失"}, LIVE)
    assert len(at.exception) == 0
    assert any("无法构建" in e.value for e in at.error)


def test_tick_exception_is_caught_and_last_snapshot_kept():
    """单次轮询抛异常不能让页面崩，要显示告警并沿用上次快照。"""
    at = _run(_FakePoller({"n_live_polls": 3, "n_window_polls": 7, "games": []},
                          _EMPTY_HEALTH, raise_on_tick=True), LIVE)
    assert len(at.exception) == 0
    assert any("本次轮询失败" in w.value for w in at.warning)


def test_no_live_games_shows_idle_message():
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 0, "games": []},
                          _EMPTY_HEALTH), LIVE)
    assert len(at.exception) == 0
    assert "没有正在进行的比赛" in _html(at)


@pytest.mark.parametrize("status,needle", [
    ("waiting", "BP 中"),
    ("no_coverage", "无实时覆盖"),
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
             "status": status, "message": "消息", "prediction": None, "warnings": []}]}, h), LIVE)
    assert len(at.exception) == 0
    assert needle in _html(at)


def _ready_game(**pred):
    with open(ROOT / "data" / "ddragon" / "champion_map.json") as fh:
        keys = [c["dd_key"] for c in json.load(fh)["champions"]][:10]
    base = {"p_blue": 0.61, "logit": 0.447, "layers": {"实力": 0.4, "阵容": 0.05, "赛况": -0.01},
            "blue_champ_keys": keys[:5], "red_champ_keys": keys[5:],
            "blue_names": [f"T1 B{i}" for i in range(5)], "red_names": [f"GEN R{i}" for i in range(5)]}
    # 赛程顺序故意和 feed 的蓝红方反着写（换边后就是这样）
    return {"game_id": "g1", "league": "LCK", "teams": ["GEN", "T1"], "status": "ready",
            "message": "就绪", "latency_200_to_pred_s": 0.221, "warnings": [],
            "prediction": {**base, **pred}}


def test_ready_game_renders_probability_layers_and_lineups():
    h = dict(_EMPTY_HEALTH, n_watched=1)
    h["by_status"] = dict(h["by_status"], ready=1)
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 1, "games": [_ready_game()]}, h), LIVE)
    assert len(at.exception) == 0
    labels = {m.label for m in at.metric}
    assert "实力层" in labels and "监视中的对局" in labels
    html = _html(at)
    assert "61.0%" in html and "lw-lineup" in html
    # 蓝红方认的是 feed（选手名 tag），不是赛程顺序：对阵卡里蓝方 T1 在前
    mu = html[html.index('class="lw-matchup"'):]
    assert mu.index(">T1<") < mu.index(">GEN<")


def test_ready_without_probability_warns_not_crashes():
    """阵容读到了但概率算不出来（例如实体全未解析）时要告警，不能崩。"""
    h = dict(_EMPTY_HEALTH, n_watched=1)
    h["by_status"] = dict(h["by_status"], ready=1)
    g = _ready_game(p_blue=None, blue_champ_keys=[], red_champ_keys=[], blue_names=[], red_names=[])
    g["warnings"] = ["两队均未解析"]
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 1, "games": [g]}, h), LIVE)
    assert len(at.exception) == 0
    assert any("概率未产出" in w.value for w in at.warning)
    assert "两队均未解析" in _html(at)


def test_review_page_renders_real_ledger():
    """「近期战绩」页读真实台账，必须出指标、不抛异常。"""
    if not (ROOT / "reports" / "live_watch.jsonl").exists():
        pytest.skip("尚无盯盘记录")
    at = _run({"error": "RuntimeError: 离线"}, REVIEW)
    assert len(at.exception) == 0
    labels = {m.label for m in at.metric}
    assert {"命中率", "log loss", "Brier", "待定"} <= labels


def test_idle_poller_renders_resolve_rate_dash_and_footer():
    """没比赛时（解析率 None）实时页显示「—」，页脚照常渲染。"""
    at = _run(_FakePoller({"n_live_polls": 1, "n_window_polls": 0, "games": []},
                          _EMPTY_HEALTH), LIVE)
    assert len(at.exception) == 0
    assert next(m for m in at.metric if m.label == "实体解析率").value == "—"
    assert _footer(at), "脚本没跑完：页脚空白"


class _HealthBoom(_FakePoller):
    def health(self):
        raise KeyError("by_status")


def test_live_panel_crash_stays_inside_the_live_page():
    """实时面板内部任何异常都只能留在实时页里：就地显示，页脚照常，其它页不受影响。"""
    at = _run(_HealthBoom({"n_live_polls": 1, "n_window_polls": 0, "games": []},
                          _EMPTY_HEALTH), LIVE)
    assert len(at.exception) == 1          # 就地显示在实时页
    assert _footer(at)
    at.switch_page(HISTORY).run()
    assert len(at.exception) == 0 and {"命中率", "样本"} <= {m.label for m in at.metric}


# ---- 「近期战绩」页：用合成台账驱动真实页面 ----


def _game(gid, league, day, p_blue, blue_won, late=False, teams=("AAA", "BBB")):
    """一局 = 一条预测 + （可选）一条带开局 / 终局时间的结果。late=True 表示终局后才落盘。"""
    rec_at = f"{day} 23:00:00" if late else f"{day} 10:05:00"
    rows = [{"kind": "prediction", "game_id": gid, "match_id": "m" + gid, "recorded_at": rec_at,
             "league": league, "teams": list(teams), "p_blue": p_blue,
             "blue_names": [f"{teams[0]} p{i}" for i in range(5)],
             "red_names": [f"{teams[1]} p{i}" for i in range(5)]}]
    if blue_won is not None:
        rows.append({"kind": "result", "game_id": gid, "blue_won": blue_won,
                     "blue_team": teams[0], "red_team": teams[1],
                     "started_at": f"{day}T10:00:00.000Z", "ended_at": f"{day}T10:35:00.000Z"})
    return rows


_BASE = (_game("1", "WSCI", "2026-09-23", 0.8, True) + _game("2", "WSCI", "2026-09-23", 0.3, False)
         + _game("3", "LPL", "2026-10-01", 0.7, False) + _game("4", "LPL", "2026-10-02", 0.6, True, late=True)
         + _game("5", "LCK", "2026-10-03", 0.55, None))


def _write_ledger(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


@contextmanager
def _review_app(tmp_path, rows=_BASE):
    """真实页面 + 合成台账。所有 run() 都要在 patch 里做：出了 with 再 run 会去建真的轮询器。"""
    ledger = tmp_path / "live_watch.jsonl"
    _write_ledger(ledger, rows)

    def _boom(*a, **k):
        raise RuntimeError("离线")
    _clear_poller()
    with patch("lolwp.live.poller.build_default_poller", _boom), \
            patch("lolwp.live.ledger.LEDGER", str(ledger)):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        at.switch_page(REVIEW).run()
        yield at, ledger
    _clear_poller()


def _review_metrics(at):
    return {m.label: m.value for m in at.metric}


def _games_table(at):
    return at.dataframe[-1].value


def test_review_page_defaults_cover_everything(tmp_path):
    with _review_app(tmp_path) as (at, _):
        assert len(at.exception) == 0
        m = _review_metrics(at)
        assert (m["对局"], m["命中率"], m["待定"]) == ("5", "75.0%", "1")
        t = _games_table(at)
        assert len(t) == 5 and list(t["备注"]).count("赛后补录") == 1
        by = at.dataframe[0].value.set_index("赛事")
        assert by.loc["LCK", "命中率"] == "—"          # 全是待定的赛事显示 —，不是 None
        assert "实时 4" in _html(at) and "赛后补录 1" in _html(at)


def test_result_filter_only_narrows_the_table(tmp_path):
    """只看命中时，命中率还得是全体的 75%，不能变成恒为 100% 的自证。"""
    with _review_app(tmp_path) as (at, _):
        at.multiselect(key="review_status").set_value(["命中"]).run()
        assert _review_metrics(at)["命中率"] == "75.0%"
        assert set(_games_table(at)["结果"]) == {"✅ 命中"}


def test_new_games_show_up_in_an_open_session(tmp_path):
    """会话开着时台账多出新的一天、新的赛事：默认视图和自定义日期都必须自动包含，
    不能停在首次渲染时的日期上限和赛事列表上。"""
    with _review_app(tmp_path) as (at, ledger):
        at.radio(key="review_span").set_value("自定义").run()
        assert _review_metrics(at)["对局"] == "5"
        _write_ledger(ledger, _BASE + _game("6", "Worlds", "2026-10-15", 0.6, True))
        at.run()
        assert _review_metrics(at)["对局"] == "6"
        assert "Worlds" in set(_games_table(at)["赛事"])
        at.radio(key="review_span").set_value("全部").run()
        assert _review_metrics(at)["对局"] == "6"


def test_half_picked_range_means_from_that_day_on(tmp_path):
    """范围控件只点了起始日时显示「起始日 – 空」，统计也得是「从那天到最新」，不是只看那一天。"""
    with _review_app(tmp_path) as (at, _):
        at.radio(key="review_span").set_value("自定义").run()
        next(d for d in at.date_input if d.label == "日期范围").set_value((date(2026, 10, 1),)).run()
        assert len(at.exception) == 0
        assert _review_metrics(at)["对局"] == "3"


def test_strict_toggle_drops_late_captures(tmp_path):
    with _review_app(tmp_path) as (at, _):
        at.toggle(key="review_strict").set_value(True).run()
        m = _review_metrics(at)
        assert (m["对局"], m["命中率"]) == ("4", "66.7%")


def test_backfilled_games_are_labelled_and_strict_drops_them(tmp_path):
    """整局错过、事后补录的局：备注单独写明，「排除赛后补录」也要把它们排掉。"""
    bf = _game("7", "Worlds", "2026-10-15", 0.7, True)
    bf[0].update(source="backfill", recorded_at="2026-10-24 18:00:00",
                 started_at="2026-10-15T10:00:00.000Z")
    with _review_app(tmp_path, _BASE + bf) as (at, _):
        assert "赛后补录（整局错过）" in set(_games_table(at)["备注"])
        assert "其中整局错过、事后补录 1" in _html(at)
        at.toggle(key="review_strict").set_value(True).run()
        assert _review_metrics(at)["对局"] == "4"


def test_review_page_crash_stays_on_that_page(tmp_path):
    """台账是外部进程写的；这一页无论怎么坏，页脚都得照常渲染，别的页也照常。"""
    def _bad(*a, **k):
        raise ValueError("模拟台账损坏")
    with patch("lolwp.live.ledger.load_review", _bad), _review_app(tmp_path) as (at, _):
        assert len(at.exception) == 1
        assert _footer(at)
        at.switch_page(HISTORY).run()
        assert len(at.exception) == 0
