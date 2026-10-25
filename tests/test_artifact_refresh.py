"""实力快照随赛果重解（artifact.refresh）与各处跟着换 artifact。

快照冻结一个月 log loss 就差 0.012（reports/exp_model_improvements.md §1.2），而外层系数是稳定的：
refresh 只重解快照，系数、其余状态与 trained_on 一律不动。这里用合成数据跑，不依赖 silver。
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from lolwp.models import artifact as A
from lolwp.models.hier_bt import HierBTConfig, HierBTSnapshot

T0 = pd.Timestamp("2026-01-01", tz="UTC")
TEAMS = [f"oe:team:{i}" for i in range(8)]


def _roster(team: str) -> list[str]:
    return [f"oe:player:{team[-1]}{k}" for k in range(5)]


def _silver(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 0.6, len(TEAMS))))
    rows = []
    for i in range(n):
        b, r = rng.choice(TEAMS, 2, replace=False)
        p = 1 / (1 + np.exp(-(strength[b] - strength[r])))
        rows.append({"game_id": f"oe{i}", "kickoff_utc": T0 + pd.Timedelta(hours=20 * i),
                     "league_canon": "LCK" if TEAMS.index(b) < 4 else "LPL", "league_tier": 1,
                     "blue_team_id": b, "red_team_id": r,
                     "blue_player_ids": np.array(_roster(b), dtype=object),
                     "red_player_ids": np.array(_roster(r), dtype=object),
                     "y_blue_win": int(rng.random() < p)})
    return pd.DataFrame(rows)


def _artifact(path, silver: pd.DataFrame, scope: str = "all", target: str = "y_blue_win") -> A.Artifact:
    games, live = A.rating_games(silver, None, target)
    snap, fresh = A._rating_snapshot(games, live, HierBTConfig(refit_every_days=3), target=target)
    art = A.Artifact(coef=dict.fromkeys(A.PROD_FEATURES, 0.5), intercept=0.1,
                     mu=dict.fromkeys(A.PROD_FEATURES, 0.0), sigma=dict.fromkeys(A.PROD_FEATURES, 1.0),
                     prior_state={"team_elo": {}}, draft_state={"champ_t1": {}}, bt_state=snap.to_dict(),
                     meta={"model": "BT4_LR", "trained_on": str(silver.kickoff_utc.max()),
                           "data_scope": scope, "rating_target": target, **fresh,
                           "ratings_n_games": len(games)})
    art.save(str(path))
    return art


def _ledger(path, games):
    """台账：每局一条带阵容、终局总经济的 result（对账解析出的 OE id）；终局时间留空。"""
    rows = []
    for gid, start, blue, red, blue_won, blue_players, league in games:
        rows.append({"kind": "result", "game_id": gid, "blue_won": blue_won, "started_at": start,
                     "ended_at": None, "league_canon": league,
                     "blue_gold": 60_000 if blue_won else 50_000, "red_gold": 55_000,
                     "blue_oe_team": blue, "blue_team_conf": "primary",
                     "red_oe_team": red, "red_team_conf": "primary",
                     "blue_oe_players": blue_players, "red_oe_players": _roster(red)})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


@pytest.fixture
def world(tmp_path, monkeypatch):
    silver = _silver()
    monkeypatch.setattr("lolwp.features.loader.load_all", lambda *a, **k: silver)
    last = silver.kickoff_utc.max()
    new = [f"oe:player:0{k}" for k in range(4)] + ["oe:player:ROOKIE"]   # 0 号队换了个新人
    ledger = _ledger(tmp_path / "live_watch.jsonl", [
        (f"g{i}", (last + pd.Timedelta(days=1, hours=i)).isoformat(), TEAMS[0], TEAMS[i + 1], True, new, "WLDs")
        for i in range(3)
    ] + [("old", (last - pd.Timedelta(days=3)).isoformat(), TEAMS[0], TEAMS[1], False, new, "LCK")])  # OE 已收录的那段
    path = tmp_path / "prod.json"
    return silver, path, ledger, _artifact(path, silver)


def test_refresh_adds_live_results_and_leaves_the_outer_model_alone(world):
    silver, path, ledger, art = world
    new = A.refresh(str(path), ledger=ledger)
    assert new is not None and new.meta["n_live_games"] == 3          # "old" 早于 OE 的 LCK 截止：不收
    assert pd.Timestamp(new.meta["ratings_as_of"]) > pd.Timestamp(art.meta["ratings_as_of"])
    snap = HierBTSnapshot.from_dict(new.bt_state)
    assert "oe:player:ROOKIE" in snap.r_player                         # 盯盘才见过的新人进了快照
    assert snap.as_of > silver.kickoff_utc.max()
    # 外层模型、其余状态、训练截止一律不动
    assert (new.coef, new.intercept, new.mu, new.sigma) == (art.coef, art.intercept, art.mu, art.sigma)
    assert (new.prior_state, new.draft_state) == (art.prior_state, art.draft_state)
    assert new.meta["trained_on"] == art.meta["trained_on"]
    on_disk = A.Artifact.load(str(path))
    assert on_disk.meta["ratings_as_of"] == new.meta["ratings_as_of"]
    # 截止 = 最晚一局按开赛 + 75 分钟（台账这几局没有终局时间）
    last_live = pd.Timestamp(silver.kickoff_utc.max()) + pd.Timedelta(days=1, hours=2)
    assert pd.Timestamp(new.meta["ratings_cutoff"]) == last_live + pd.Timedelta(minutes=75)


def test_refresh_is_a_no_op_without_new_games(world):
    _, path, ledger, _ = world
    assert A.refresh(str(path), ledger=ledger) is not None
    m = os.path.getmtime(path)
    assert A.refresh(str(path), ledger=ledger) is None
    assert os.path.getmtime(path) == m


def test_refresh_matches_a_fresh_fit_on_the_same_games(world):
    """暖启动只改 L-BFGS 的起点：结果要和从零解一遍一致（MAP 唯一）。"""
    silver, path, ledger, _ = world
    new = A.refresh(str(path), ledger=ledger)
    games, live = A.rating_games(silver, ledger)
    cold, _ = A._rating_snapshot(games, live, HierBTConfig(refit_every_days=3))
    warm = HierBTSnapshot.from_dict(new.bt_state)
    assert warm.as_of == cold.as_of
    diff = max(abs(warm.r_player[p] - cold.r_player[p]) for p in cold.r_player)
    assert diff < 1e-4


def test_refresh_refuses_dev_scope(tmp_path, monkeypatch):
    """dev 口径的 artifact 是复现揭盲用的，冻结才对；台账的局全在 holdout 期之后。"""
    silver = _silver()
    monkeypatch.setattr("lolwp.features.loader.load_all", lambda *a, **k: silver)
    path = tmp_path / "prod.json"
    _artifact(path, silver, scope="dev")
    with pytest.raises(RuntimeError, match="全量口径"):
        A.refresh(str(path), ledger=None)


def test_refresh_does_not_overwrite_a_model_retrained_meanwhile(world, monkeypatch):
    """读进来之后文件被 make artifact 换过：放弃这一次，绝不把新训的模型盖回旧系数。"""
    _, path, ledger, _ = world
    orig = A.rating_games

    def racing(full, led, target="y_blue_win"):
        out = orig(full, led, target)
        d = json.loads(path.read_text())
        d["coef"] = dict.fromkeys(A.PROD_FEATURES, 9.0)               # 另一个进程刚训完写下
        path.write_text(json.dumps(d))
        st = os.stat(path)
        os.utime(path, (st.st_atime, st.st_mtime + 5))
        return out
    monkeypatch.setattr(A, "rating_games", racing)
    assert A.refresh(str(path), ledger=ledger) is None
    assert set(A.Artifact.load(str(path)).coef.values()) == {9.0}


def test_poller_swaps_in_a_changed_artifact(tmp_path, monkeypatch):
    """盯盘 / 站点的轮询器：文件换过才重新装载，没换就不动。"""
    from lolwp.live import poller as P

    f = tmp_path / "prod.json"
    f.write_text("{}")
    loads = []
    monkeypatch.setattr(P, "load_serving_artifact", lambda p: loads.append(p) or {"v": len(loads)})
    pol = P.Poller(client=None, resolver=None, artifact={"v": 0})
    assert pol.reload_artifact_if_changed() is False                  # 不知道文件在哪：不动
    pol.artifact_path, pol.artifact_mtime = str(f), os.path.getmtime(f)
    assert pol.reload_artifact_if_changed() is False and pol.artifact == {"v": 0}
    st = os.stat(f)
    os.utime(f, (st.st_atime, st.st_mtime + 5))
    assert pol.reload_artifact_if_changed() is True and pol.artifact == {"v": 1}
    assert pol.reload_artifact_if_changed() is False and loads == [str(f)]


def test_api_reloads_a_changed_artifact_and_keeps_the_old_one_on_mismatch(monkeypatch):
    pytest.importorskip("fastapi")
    from lolwp.api import main as M

    mtime = {"v": 1.0}
    monkeypatch.setattr(M, "_mtime", lambda p: mtime["v"])
    monkeypatch.setattr(M, "S", {"art": "old", "art_mtime": 1.0})
    monkeypatch.setattr(M, "_load_artifact", lambda: "new")
    assert M._art() == "old"
    mtime["v"] = 2.0
    assert M._art() == "new"

    def mismatch():
        raise RuntimeError("artifact 与代码版本不匹配，拒绝启动。")
    monkeypatch.setattr(M, "_load_artifact", mismatch)
    mtime["v"] = 3.0
    assert M._art() == "new"                                          # 换了代码重训的：不换，接着用旧的
    assert M.S["art_mtime"] == 3.0                                    # 同一个坏文件只试一次


def _margins(silver: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """合成的赛后经济差：胜方领先 1–10 千金，30 分钟一局。"""
    rng = np.random.default_rng(seed)
    lead = rng.uniform(1_000, 10_000, len(silver)) * np.where(silver.y_blue_win == 1, 1, -1)
    return pd.DataFrame({"blue_gold": 55_000 + lead, "red_gold": np.full(len(silver), 55_000.0),
                         "gamelength_s": np.full(len(silver), 1_800.0)},
                        index=pd.Index(silver.game_id, name="game_id"))


def test_refresh_keeps_the_margin_target(tmp_path, monkeypatch):
    """软标签训出来的 artifact，重解时也必须用软标签：外层系数是在这个标签的快照特征上训的。
    台账的局没有终局时间（拿不到时长）→ 退回胜负，和 silver 缺经济的局一样。"""
    from lolwp.models.margin import with_margin_target

    silver = _silver()
    margins = _margins(silver)
    monkeypatch.setattr("lolwp.features.loader.load_all", lambda *a, **k: silver)
    monkeypatch.setattr("lolwp.store.margins.load", lambda *a, **k: margins)
    last = silver.kickoff_utc.max()
    ledger = _ledger(tmp_path / "live_watch.jsonl", [
        (f"g{i}", (last + pd.Timedelta(days=1, hours=i)).isoformat(), TEAMS[0], TEAMS[i + 1], True,
         _roster(TEAMS[0]), "WLDs") for i in range(3)])
    path = tmp_path / "prod.json"
    _artifact(path, with_margin_target(silver, margins), target="y_margin")
    new = A.refresh(str(path), ledger=ledger)
    assert new is not None and new.meta["rating_target"] == "y_margin"
    snap = HierBTSnapshot.from_dict(new.bt_state)
    assert snap.diagnostics["target"] == "y_margin"
    games, live = A.rating_games(with_margin_target(silver, margins), ledger, "y_margin")
    assert list(live.y_margin) == [1.0, 1.0, 1.0]
    cold, _ = A._rating_snapshot(games, live, HierBTConfig(refit_every_days=3), target="y_margin")
    win, _ = A._rating_snapshot(games, live, HierBTConfig(refit_every_days=3))
    gap = lambda a, b: max(abs(a.r_player[p] - b.r_player[p]) for p in b.r_player)
    assert gap(snap, cold) < 1e-4 and gap(snap, win) > 1e-3


def test_refresh_rebuilds_the_live_part_of_the_form_layer(world, tmp_path):
    """近期状态：β 与 OE 那部分训练时定死，台账那部分每次重解时按开赛前落盘的预测重建（只收 tier>=1 的局）。"""
    silver, path, _, _ = world
    d = json.loads(path.read_text())
    d["prior_state"]["form"] = {"cfg": {"half_life_games": 5.0, "prior_games": 3.0, "window": 60,
                                        "include_series": True},
                                "beta": 0.7, "silver": {"keep": [["2026-01-01T00:00:00+00:00", 0.1]]}, "live": {}}
    path.write_text(json.dumps(d))
    last = silver.kickoff_utc.max()
    ledger = _ledger(tmp_path / "led.jsonl", [
        (f"g{i}", (last + pd.Timedelta(days=1, hours=i)).isoformat(), TEAMS[0], TEAMS[i + 1], True,
         _roster(TEAMS[0]), "LCK") for i in range(2)])
    with open(ledger, "a") as fh:                                     # 只有 g0 有开赛前的预测
        fh.write(json.dumps({"kind": "prediction", "game_id": "g0", "p_blue": 0.66, "p_base": 0.6}) + "\n")
    new = A.refresh(str(path), ledger=ledger)
    form = new.prior_state["form"]
    assert form["beta"] == 0.7 and form["silver"] == {"keep": [["2026-01-01T00:00:00+00:00", 0.1]]}
    assert set(form["live"]) == {TEAMS[0], TEAMS[1]}
    assert form["live"][TEAMS[0]][0][1] == pytest.approx(1 - 0.6)      # 蓝方赢了，残差 = 1 − p_base
