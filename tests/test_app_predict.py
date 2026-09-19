"""阶段 5 验收：推理、分层分解、what-if、降级告警。"""
import json
import os
import time

import pandas as pd
import pytest

from lolwp.app.predict import Draft, predict, whatif_swap
from lolwp.models.artifact import Artifact

ART = "models/prod.json"
pytestmark = pytest.mark.skipif(not os.path.exists(ART), reason="需先跑 artifact.train()")


@pytest.fixture(scope="module")
def art():
    return Artifact.load(ART)


@pytest.fixture(scope="module")
def champs():
    with open("data/ddragon/champion_map.json") as fh:
        return json.load(fh)["champions"]


@pytest.fixture(scope="module")
def real_draft():
    g = pd.read_parquet("data/silver/games.parquet")
    r = g[(g.league_canon == "LCK") & (g.kickoff_utc < "2026-06-01")].iloc[-1]
    return Draft(r.league_canon, r.blue_team_id, r.red_team_id,
                 list(r.blue_player_ids), list(r.red_player_ids),
                 [int(x) for x in r.blue_champ_keys], [int(x) for x in r.red_champ_keys],
                 now=r.kickoff_utc)


def test_decomposition_closes_exactly(art, real_draft):
    """验收③：Σ贡献 + 截距 == logit(p)，误差 < 1e-9。"""
    p = predict(art, real_draft)
    assert abs(sum(p.contributions.values()) + art.intercept - p.logit) < 1e-9


def test_probability_in_range(art, real_draft):
    p = predict(art, real_draft)
    assert 0.0 < p.p_blue < 1.0
    import math
    assert abs(1 / (1 + math.exp(-p.logit)) - p.p_blue) < 1e-12


def test_layers_sum_to_total_contribution(art, real_draft):
    p = predict(art, real_draft)
    assert abs(sum(p.layers.values()) - sum(p.contributions.values())) < 1e-12
    assert set(p.layers) <= {"实力", "阵容", "赛况"}


def test_whatif_is_fast(art, real_draft, champs):
    """验收②：遍历全部英雄的 p99 端到端 < 500 ms。"""
    keys = [c["dd_key"] for c in champs]
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        whatif_swap(art, real_draft, "blue", 2, keys)
        times.append((time.perf_counter() - t0) * 1000)
    assert max(times) < 500, f"what-if 最慢 {max(times):.0f} ms"


def test_whatif_covers_all_champions(art, real_draft, champs):
    keys = [c["dd_key"] for c in champs]
    wi = whatif_swap(art, real_draft, "blue", 2, keys)
    assert len(wi) == len(keys)
    assert set(wi.champ_key) == set(keys)


def test_whatif_current_pick_has_zero_delta(art, real_draft, champs):
    """把某位置换成它当前已选的英雄，概率变化必须是 0。"""
    cur = real_draft.blue_champ_keys[2]
    wi = whatif_swap(art, real_draft, "blue", 2, [cur])
    assert abs(float(wi.delta_pp.iloc[0])) < 1e-9


def test_whatif_magnitude_is_small(art, real_draft, champs):
    """换一个英雄的影响应在几个百分点以内。若出现两位数 pp 变化说明模型或特征有问题。"""
    keys = [c["dd_key"] for c in champs]
    wi = whatif_swap(art, real_draft, "blue", 2, keys)
    assert wi.delta_pp.abs().max() < 10.0, f"单英雄影响 {wi.delta_pp.abs().max():.1f}pp 过大"


# ---------- 验收④：三类降级必须告警且不抛异常 ----------

def test_unknown_team_warns_not_raises(art, real_draft, champs):
    d = Draft(real_draft.league, "oe:team:NOT_A_REAL_TEAM", real_draft.red_team_id,
              real_draft.blue_player_ids, real_draft.red_player_ids,
              real_draft.blue_champ_keys, real_draft.red_champ_keys)
    p = predict(art, d)
    assert 0 < p.p_blue < 1
    assert any("战队" in w for w in p.warnings)


def test_unknown_player_warns_not_raises(art, real_draft):
    d = Draft(real_draft.league, real_draft.blue_team_id, real_draft.red_team_id,
              ["oe:player:GHOST"] * 5, real_draft.red_player_ids,
              real_draft.blue_champ_keys, real_draft.red_champ_keys)
    p = predict(art, d)
    assert 0 < p.p_blue < 1
    assert any("选手" in w for w in p.warnings)


def test_unknown_league_warns_not_raises(art, real_draft):
    d = Draft("NOT_A_LEAGUE", real_draft.blue_team_id, real_draft.red_team_id,
              real_draft.blue_player_ids, real_draft.red_player_ids,
              real_draft.blue_champ_keys, real_draft.red_champ_keys)
    p = predict(art, d)
    assert 0 < p.p_blue < 1
    assert any("赛区" in w for w in p.warnings)


def test_all_three_degradations_at_once(art):
    """三种降级同时发生也不能崩。"""
    d = Draft("NOPE", "no-team-a", "no-team-b", ["x"] * 5, ["y"] * 5,
              [1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    p = predict(art, d)
    assert 0 < p.p_blue < 1
    assert len(p.warnings) >= 3


# ---------- 验收①：离线可用 ----------

def test_icons_present_offline(champs):
    with open("data/ddragon/champion_map.json") as fh:
        ver = json.load(fh)["version"]
    missing = [c["dd_id"] for c in champs
               if not os.path.exists(f"frontend/public/dd/{ver}/{c['dd_id']}.png")]
    assert not missing, f"{len(missing)} 个英雄图标缺失，离线模式会破图"


def test_artifact_is_self_describing(art):
    """artifact 必须记得自己是怎么来的（§4.7.2）。"""
    for k in ["model", "trained_on", "n_train_games", "holdout_start", "oos_log_loss"]:
        assert k in art.meta


def test_artifact_trained_before_holdout(art):
    assert pd.Timestamp(art.meta["trained_on"]) < pd.Timestamp(art.meta["holdout_start"])
