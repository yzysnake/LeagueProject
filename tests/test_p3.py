"""P3 前瞻评估的闸门（scripts/p3_prospective.py）：一次性、没到预注册的触发日不动 OE、OE 拉不到就停、
拿到数据先重建再评估。评估本身的数字靠彩排（make p3-rehearse）验证，这里不跑。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _p3():
    spec = importlib.util.spec_from_file_location("p3", ROOT / "scripts" / "p3_prospective.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_refuses_to_run_twice(tmp_path, monkeypatch):
    p3 = _p3()
    rec = tmp_path / "p3.json"
    rec.write_text("{}")
    monkeypatch.delenv("LOLWP_P3_RERUN", raising=False)
    assert p3.run_real(str(rec)) == 2


def _world(p3, tmp_path, monkeypatch):
    monkeypatch.setattr(p3, "RECORD", str(tmp_path / "p3.json"))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return NS(returncode=0)
    return calls, run


def test_waits_for_the_preregistered_trigger_date(tmp_path, monkeypatch):
    p3 = _p3()
    calls, run = _world(p3, tmp_path, monkeypatch)
    assert p3.when_ready(p3.TRIGGER - pd.Timedelta(days=1), run=run) == 0
    assert calls == []                                       # 没到触发日：连 OE 都不碰


def test_stops_while_oe_is_still_blocked(tmp_path, monkeypatch):
    p3 = _p3()
    calls = []
    monkeypatch.setattr(p3, "RECORD", str(tmp_path / "p3.json"))

    def blocked(cmd, **kw):
        calls.append(cmd)
        return NS(returncode=1)                              # day0_fetch.sh 拿到配额错误页
    assert p3.when_ready(p3.TRIGGER, run=blocked) == 0
    assert calls == [["bash", "scripts/day0_fetch.sh"]]      # 不重建 silver、不评估


def test_does_nothing_once_done(tmp_path, monkeypatch):
    p3 = _p3()
    calls, run = _world(p3, tmp_path, monkeypatch)
    (tmp_path / "p3.json").write_text("{}")
    assert p3.when_ready(p3.TRIGGER + pd.Timedelta(days=30), run=run) == 0 and calls == []


def test_oe_counts_as_ready_only_with_games_after_the_cutoff(tmp_path):
    """Drive 配额解除后拿到的若还是 9/18 那份文件（内容没变），不算数据到了。"""
    p3 = _p3()
    raw = tmp_path / "2026.csv"
    pd.DataFrame({"date": ["2026-09-17 09:00:00", "2026-09-18 10:00:00"]}).to_csv(raw, index=False)
    assert not p3.raw_has_games_after(p3.CUTOFF, str(raw))
    pd.DataFrame({"date": ["2026-09-18 10:00:00", "2026-11-10 09:00:00"]}).to_csv(raw, index=False)
    assert p3.raw_has_games_after(p3.CUTOFF, str(raw))
    assert not p3.raw_has_games_after(p3.CUTOFF, str(tmp_path / "missing.csv"))


def test_rebuilds_the_data_then_evaluates_once(tmp_path, monkeypatch):
    p3 = _p3()
    calls, run = _world(p3, tmp_path, monkeypatch)
    monkeypatch.setattr(p3, "raw_has_games_after", lambda *a, **k: True)
    notes = []
    monkeypatch.setattr(p3, "_notify", notes.append)

    def fake_real(record=None):
        Path(p3.RECORD).write_text(json.dumps({"n_games": 120, "specs": {"S2": {"log_loss": 0.61}}}))
        return 0
    monkeypatch.setattr(p3, "run_real", fake_real)
    assert p3.when_ready(p3.TRIGGER, run=run) == 0
    mods = [c[-1] for c in calls if c[1:2] == ["-m"]]
    assert calls[0] == ["bash", "scripts/day0_fetch.sh"]
    assert mods == ["lolwp.store.bronze", "lolwp.store.silver", "lolwp.store.margins"]
    assert notes and "120" in notes[0]
