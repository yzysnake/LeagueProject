"""逐折选设定（eval/gate.adaptive_select，修订 A8 的默认协议）：不偷看被打分的那一折、只看近期、历史不足时退回首选。"""
import numpy as np
import pandas as pd
import pytest

from lolwp.eval.gate import adaptive_select, select_and_gate

T0 = pd.Timestamp("2020-01-01", tz="UTC")


def _world(n_folds=10, per=300, seed=0):
    """前一半的折里 A 是真概率、B 是常数 0.5；后一半反过来。"""
    rng = np.random.default_rng(seed)
    fold = np.repeat(np.arange(n_folds), per)
    kick = np.array([T0 + pd.Timedelta(days=28 * k, hours=h % 600) for k, h in zip(fold, range(len(fold)))])
    truth = rng.uniform(0.15, 0.85, len(fold))
    y = (rng.random(len(fold)) < truth).astype(float)
    early = fold < n_folds // 2
    preds = {"A": np.where(early, truth, 0.5), "B": np.where(early, 0.5, truth)}
    return preds, y, fold, kick


def test_short_window_follows_the_recent_winner():
    preds, y, fold, kick = _world()
    _, short = adaptive_select(preds, y, fold, kick, window=pd.Timedelta(days=60), min_rows=100)
    assert short[7] == "B"                       # 近两折都是 B 好
    _, long = adaptive_select(preds, y, fold, kick, window=pd.Timedelta(days=10_000), min_rows=100)
    assert long[7] == "A"                        # 窗口够长时，前五折 A 好的历史压过近两折


def test_choice_for_a_fold_never_sees_that_fold_or_later():
    preds, y, fold, kick = _world()
    _, a = adaptive_select(preds, y, fold, kick, min_rows=100)
    y2 = y.copy()
    y2[fold >= 6] = 1 - y2[fold >= 6]            # 只改第 6 折及以后的赛果
    _, b = adaptive_select(preds, y2, fold, kick, min_rows=100)
    assert a[:7] == b[:7]                        # 第 0–6 折的选择逐个不变（第 6 折只看第 0–5 折）


def test_falls_back_to_the_first_setting_without_enough_history():
    preds, y, fold, kick = _world(per=50)
    out, picks = adaptive_select(preds, y, fold, kick, min_rows=10_000)
    assert set(picks) == {"A"} and np.array_equal(out, preds["A"])


def test_select_and_gate_reports_the_picks():
    preds, y, fold, kick = _world()
    r = select_and_gate(preds, np.full(len(y), 0.5), y, fold, kick, np.arange(len(y)), min_rows=100)
    assert r["protocol"] == "per-window" and sum(r["picks"].values()) == 10
    assert r["delta"] > 0                        # 比常数 0.5 好
    with pytest.raises(ValueError, match="至少"):
        adaptive_select({}, y, fold, kick)
