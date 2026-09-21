"""FastAPI 契约测试（PLAN.md §7.5）。

重点不是"正常时返回什么"，而是**契约不能被悄悄改掉**：
端点集合、请求校验、artifact 版本断言。
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "prod.json").exists(),
                                reason="需先跑 artifact.train()")
ROLES = ["top", "jng", "mid", "bot", "sup"]


@pytest.fixture(scope="module")
def client():
    from lolwp.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def body():
    import pandas as pd
    g = pd.read_parquet(ROOT / "data" / "silver" / "games.parquet")
    r = g[(g.league_canon == "LCK") & (g.kickoff_utc < "2026-06-01")].iloc[-1]
    return {
        "league": "LCK",
        "blue": [{"champion": int(k), "role": ro, "player_id": p}
                 for k, ro, p in zip(r.blue_champ_keys, ROLES, r.blue_player_ids)],
        "red": [{"champion": int(k), "role": ro, "player_id": p}
                for k, ro, p in zip(r.red_champ_keys, ROLES, r.red_player_ids)],
        "blue_team_id": r.blue_team_id, "red_team_id": r.red_team_id,
    }


def test_all_seven_endpoints_exist():
    """契约锁死：少一个端点就说明有人改了接口而没改文档。"""
    from lolwp.api.main import app
    paths = {r.path for r in app.routes if hasattr(r, "methods")}
    for p in ["/health", "/champions", "/players", "/predict",
              "/predict/batch", "/explain", "/live"]:
        assert p in paths, f"缺端点 {p}"


def test_health_reports_model_and_data_scale(client):
    h = client.get("/health").json()
    assert h["ok"] and h["model_version"] == "BT4_LR"
    assert h["n_champions"] == 173
    assert h["n_known_players"] > 5000 and h["n_known_teams"] > 1000
    assert h["run_id"]                      # 回滚时用它确认换了实例


def test_champions_are_complete_with_icons(client):
    ch = client.get("/champions").json()
    assert len(ch) == 173
    assert all(c["icon"] and c["icon"].endswith(".png") for c in ch)


def test_predict_is_deterministic_and_closed(client, body):
    a = client.post("/predict", json=body).json()
    b = client.post("/predict", json=body).json()
    assert a["p_blue"] == b["p_blue"], "同一请求两次结果不同"
    assert 0 < a["p_blue"] < 1
    s = sum(c["delta_logit"] for c in a["contributions"])
    assert abs(s + (a["logit"] - s) - a["logit"]) < 1e-12
    assert set(a["layers"]) <= {"实力", "阵容", "赛况"}


def test_predict_matches_in_process_path(client, body):
    """HTTP 路径与进程内路径必须给出**完全相同**的概率。

    两条路径若产生差异，就是 train-serve skew 的一种变体——
    用户在网页上看到的和离线评估的不是同一个模型。
    """
    from lolwp.app.predict import Draft, predict
    from lolwp.models.artifact import Artifact
    art = Artifact.load(str(ROOT / "models" / "prod.json"))
    bp = [s["player_id"] for s in body["blue"]]
    rp = [s["player_id"] for s in body["red"]]
    bc = [s["champion"] for s in body["blue"]]
    rc = [s["champion"] for s in body["red"]]
    local = predict(art, Draft(body["league"], body["blue_team_id"],
                               body["red_team_id"], bp, rp, bc, rc))
    http = client.post("/predict", json=body).json()
    assert abs(local.p_blue - http["p_blue"]) < 1e-12


@pytest.mark.parametrize("mutate,why", [
    (lambda b: b.__setitem__("blue", b["blue"][:4]), "只给 4 个位置"),
    (lambda b: b.__setitem__("blue", [dict(b["blue"][0])] * 5), "位置重复"),
])
def test_request_validation_rejects_bad_drafts(client, body, mutate, why):
    bad = json.loads(json.dumps(body))
    mutate(bad)
    assert client.post("/predict", json=bad).status_code == 422, why


def test_unknown_champion_is_rejected_not_silently_defaulted(client, body):
    """未知英雄必须 422。静默给个默认值是最坏的处理方式。"""
    bad = json.loads(json.dumps(body))
    bad["blue"][0]["champion"] = 99999
    r = client.post("/predict", json=bad)
    assert r.status_code == 422 and "99999" in r.text


def test_unknown_entities_warn_but_still_predict(client, body):
    bad = json.loads(json.dumps(body))
    bad["blue_team_id"] = "oe:team:NOPE"
    bad["league"] = "NOT_A_LEAGUE"
    r = client.post("/predict", json=bad).json()
    assert 0 < r["p_blue"] < 1
    assert len(r["warnings"]) >= 2


def test_batch_is_fast_and_complete(client, body):
    ch = client.get("/champions").json()
    keys = [c["dd_key"] for c in ch]
    r = client.post("/predict/batch",
                    json={"base": body, "side": "blue", "role": "mid",
                          "candidates": keys}).json()
    assert len(r["items"]) == len(keys)
    assert r["elapsed_ms"] < 500          # 验收②同一条线
    assert max(abs(i["delta_pp"]) for i in r["items"]) < 10


def test_explain_states_the_honest_caveat(client, body):
    """叙述里必须带上「BP 贡献接近 0」这句——这是产品诚实性的一部分，
    不是可选的文案。"""
    e = client.post("/explain", json=body).json()
    assert "BP" in e["narrative"] and "接近 0" in e["narrative"]


def test_live_never_500s(client):
    """实时接口失败只降级，不能把 500 抛给前端。"""
    r = client.get("/live")
    assert r.status_code == 200
    assert "available" in r.json()


def test_artifact_version_assertion_exists():
    """启动时必须断言 artifact 与代码的特征集一致。

    不一致时继续跑会用错误的特征、且不报错——这是最难查的一类故障。
    """
    src = (ROOT / "src" / "lolwp" / "api" / "main.py").read_text()
    assert "PROD_FEATURES" in src and "拒绝启动" in src


@pytest.mark.skipif(not (ROOT / "docker-compose.yml").exists(), reason="无 compose")
def test_compose_commands_are_single_line():
    """YAML 折叠标量陷阱：`>` 里**缩进更深的行会保留换行**，把 shell 命令切断。

    [实测 2026-09-21] 我把 `--host 0.0.0.0` 换行并多缩进后，
    YAML 在那里插了换行，uvicorn 拿不到 --host，**静默退回默认的 127.0.0.1**，
    容器内 healthcheck 照样过（它走 localhost），但宿主机访问不到。
    这类 bug 只表现为"端口不通"，很难联想到 YAML 缩进。
    """
    import yaml
    with open(ROOT / "docker-compose.yml") as fh:
        d = yaml.safe_load(fh)
    for svc, spec in d["services"].items():
        cmd = spec.get("command", "")
        assert "\n" not in cmd.rstrip("\n"), f"{svc} 的 command 被 YAML 切成了多行"


@pytest.mark.skipif(not (ROOT / "docker-compose.yml").exists(), reason="无 compose")
def test_compose_uv_run_also_skips_dev():
    """`uv run` 默认会重新同步且**带上 dev 组**，会抵消前一步的 `uv sync --no-dev`。

    [实测] 不给 `uv run` 也加 --no-dev 时，容器里会多装 mypy/ruff/pytest 13 个包。
    """
    y = (ROOT / "docker-compose.yml").read_text()
    for line in y.splitlines():
        if "uv run" in line:
            assert "--no-dev" in line, f"这行的 uv run 少了 --no-dev：{line.strip()}"


@pytest.mark.skipif(not (ROOT / "docker-compose.yml").exists(), reason="无 compose")
def test_compose_api_binds_all_interfaces_inside_container():
    """容器内必须绑 0.0.0.0（否则宿主机映射不进去），
    对外暴露则由 ports 的 127.0.0.1 前缀限制。两者不矛盾。"""
    y = (ROOT / "docker-compose.yml").read_text()
    assert "--host 0.0.0.0" in y
    assert "--server.address 0.0.0.0" in y


@pytest.mark.skipif(not (ROOT / "docker-compose.yml").exists(), reason="无 compose")
def test_compose_enforces_single_worker_and_local_bind():
    """两条硬约束写进 compose：单 worker（内存）与只绑回环（安全）。"""
    y = (ROOT / "docker-compose.yml").read_text()
    assert "--workers 1" in y
    assert '"127.0.0.1:8000:8000"' in y and '"127.0.0.1:8501:8501"' in y
    assert 'profiles: ["online"]' in y      # 离线是默认状态
    assert "start_period: 300s" in y        # 冷启动要 5 分钟
