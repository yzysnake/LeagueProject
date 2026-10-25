"""分层 BT 的赛后经济差软标签（预注册修订 A6）：标签本身、经济差表、白名单、缓存指纹。

赛后统计只许当**历史局**的 BT 标签：本局的照旧不许进本局特征，也不进 bronze。
快照的未来不变性对软标签同样成立，那一条钉在 tests/test_hier_bt.py。
"""
import math

import numpy as np
import pandas as pd
import pytest

from lolwp.features.whitelist import check_feature_columns, classify
from lolwp.models.margin import MARGIN_KAPPA, margin_target, with_margin_target


def test_margin_target_values_and_symmetry():
    """单位：千金 / 10 分钟。胜方中位数 3.34 → σ(0.5·3.34) ≈ 0.84；蓝红互换 → 1 − y'。"""
    y = margin_target([1.0], [63_400.0], [50_000.0], [2_400.0])     # 13.4k / 40 分钟 = 3.35 / 10 分钟
    assert y[0] == pytest.approx(1 / (1 + math.exp(-MARGIN_KAPPA * 13.4 / 4)))
    assert 0.83 < y[0] < 0.85
    flip = margin_target([0.0], [50_000.0], [63_400.0], [2_400.0])
    assert flip[0] == pytest.approx(1 - y[0])
    # 翻盘局：蓝方赢了但经济落后 → 软标签记经济领先的红方占优
    assert margin_target([1.0], [48_000.0], [52_000.0], [1_800.0])[0] < 0.5


def test_margin_target_falls_back_to_the_result():
    nan = float("nan")
    y = margin_target([1.0, 0.0, 1.0, 0.0], [nan, 60_000.0, 60_000.0, 60_000.0],
                      [50_000.0, nan, 50_000.0, 50_000.0], [1_800.0, 1_800.0, nan, 0.0])
    assert list(y) == [1.0, 0.0, 1.0, 0.0]


def test_with_margin_target_aligns_by_game_id():
    games = pd.DataFrame({"game_id": ["a", "b", "c"], "y_blue_win": [1, 0, 1]})
    games.attrs["holdout_start"] = "2026-06-01"
    margins = pd.DataFrame({"blue_gold": [40_000.0, 70_000.0], "red_gold": [60_000.0, 60_000.0],
                            "gamelength_s": [1_500.0, 2_000.0]},
                           index=pd.Index(["b", "a"], name="game_id"))   # 顺序打乱、c 缺
    out = with_margin_target(games, margins)
    assert out.y_margin.iloc[0] == pytest.approx(margin_target([1], [70_000], [60_000], [2_000])[0])
    assert out.y_margin.iloc[1] == pytest.approx(margin_target([0], [40_000], [60_000], [1_500])[0])
    assert out.y_margin.iloc[2] == 1.0                                # 缺经济 → 退回胜负
    assert out.attrs["holdout_start"] == "2026-06-01" and "y_margin" not in games


def test_gold_and_length_are_rating_labels_never_features():
    for c in ("totalgold", "gamelength"):
        assert classify(c) == "rating_label"
        with pytest.raises(AssertionError):
            check_feature_columns(["side", c])


def test_margins_table_from_raw_csv(tmp_path):
    """原始 CSV 的队伍行 → 每场一行；只收 silver 里有的局；蓝红方按 side。"""
    from lolwp.store import margins as M

    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for gid, bg, rg, length in (("g1", 61_000, 52_000, 1_900), ("g2", 48_000, 55_000, 2_300),
                                ("g3", 1, 2, 3)):
        for side, gold in (("Blue", bg), ("Red", rg)):
            rows.append({"gameid": gid, "side": side, "position": "team", "totalgold": gold,
                         "gamelength": length, "kills": 99})
            rows.append({"gameid": gid, "side": side, "position": "mid", "totalgold": 9,
                         "gamelength": length, "kills": 9})
    pd.DataFrame(rows).to_csv(raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv", index=False)
    silver = tmp_path / "games.parquet"
    pd.DataFrame({"game_id": ["g2", "g1"]}).to_parquet(silver)       # g3 不在 silver
    out = tmp_path / "margins.parquet"
    m = M.build(str(raw), str(silver), str(out))
    assert list(m.index) == ["g1", "g2"] and list(m.columns) == M.COLUMNS
    assert m.loc["g1"].tolist() == [61_000, 52_000, 1_900]
    assert m.loc["g2"].tolist() == [48_000, 55_000, 2_300]
    assert M.load(str(out)).equals(m)
    with pytest.raises(FileNotFoundError, match="make silver"):
        M.load(str(tmp_path / "nope.parquet"))


def test_cache_key_keeps_old_entries_and_tracks_soft_labels():
    """胜负标签的指纹与加 target 之前逐位相同（老缓存照样命中）；软标签的指纹跟着取值变。"""
    import hashlib
    import json
    from dataclasses import asdict

    from lolwp.models.bt_cache import _fingerprint
    from lolwp.models.hier_bt import HierBTConfig

    df = pd.DataFrame({"game_id": ["a", "b"], "y_blue_win": [1, 0], "y_margin": [0.8, 0.3],
                       "kickoff_utc": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)})
    cfg = HierBTConfig()
    h = hashlib.sha256()
    for part in (str(len(df)), str(df.kickoff_utc.min()), str(df.kickoff_utc.max())):
        h.update(part.encode())
    h.update(hashlib.sha256("|".join(df.game_id.astype(str)).encode()).digest())
    h.update(json.dumps(asdict(cfg), sort_keys=True, default=str).encode())
    assert _fingerprint(df, cfg) == h.hexdigest()[:16]
    soft = _fingerprint(df, cfg, "y_margin")
    assert soft != _fingerprint(df, cfg)
    df2 = df.assign(y_margin=[0.8, 0.31])
    assert _fingerprint(df2, cfg, "y_margin") != soft
    assert np.isfinite(df2.y_margin).all()
