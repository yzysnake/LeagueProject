"""FastAPI 服务（PLAN.md §7.5）。

设计要点：
1. **artifact 只加载一次**，放在 lifespan 里。它含状态快照（2,070 队 / 10,863 选手 /
   113,058 条选手英雄计数），每个 worker 一份，所以 docker compose 里必须
   `--workers 1`——多 worker 内存线性翻倍。横向扩要加容器实例不是加 worker。
2. **推理与训练共用同一套特征函数**（app.predict → features_from_state /
   HierBTSnapshot.features），服务端不重写任何公式（§7.4 防 train-serve skew）。
3. **启动时断言 artifact 与代码版本一致**，不一致直接拒绝启动而不是打个警告
   然后用错误的特征跑（§4.7.2）。
4. 实时功能失败只让 /live 降级，不影响 /predict。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from lolwp.api.schemas import (
    ROLES,
    BatchItem,
    BatchPredictRequest,
    BatchPredictResponse,
    Champion,
    Contribution,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

ART_PATH = os.environ.get("LOLWP_ARTIFACT", "models/prod.json")
CMAP_PATH = "data/ddragon/champion_map.json"
SILVER = "data/silver/games.parquet"
RUN_ID = uuid.uuid4().hex[:12]

LABELS = {
    "side_prior": "蓝方基准（按赛区收缩）",
    "bt_player_diff": "选手实力差（分层 BT）",
    "bt_chem_diff": "队伍配合残差",
    "bt_league_diff": "赛区强度差",
    "champ_meta_diff": "英雄版本强度差",
    "team_elo_diff": "队伍 Elo 差",
    "player_elo_diff": "选手 Elo 差",
}
# 实测的高自信段校准（reports/stage8_comparison.md）
_EXTREME = [(0.95, 0.9255), (0.90, 0.9286), (0.85, 0.8880)]

S: dict[str, Any] = {}


def _load_state() -> dict[str, Any]:
    """同步加载。冷启动要读 10 MB 的 artifact 与状态快照，
    放在 lifespan 里用线程跑，避免阻塞事件循环。"""
    from lolwp.models.artifact import PROD_FEATURES, Artifact

    art = Artifact.load(ART_PATH)
    # 启动断言：artifact 的特征集必须与当前代码的 PROD_FEATURES 完全一致。
    # 不一致说明 artifact 与代码版本不匹配，继续跑会用错误的特征，且不报错。
    if list(art.coef) != list(PROD_FEATURES):
        raise RuntimeError(
            f"artifact 与代码版本不匹配，拒绝启动。\n"
            f"  artifact 特征: {list(art.coef)}\n"
            f"  代码 PROD_FEATURES: {list(PROD_FEATURES)}\n"
            f"  请重新 `make artifact`。"
        )
    with open(CMAP_PATH) as fh:
        cm = json.load(fh)
    return {"art": art, "champs": cm["champions"],
            "by_key": {c["dd_key"]: c for c in cm["champions"]},
            "dd_version": cm["version"], "poller": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import anyio

    S.update(await anyio.to_thread.run_sync(_load_state))
    yield
    S.clear()


app = FastAPI(title="LoL 职业赛胜率预测", version="1.0", lifespan=lifespan)


def _to_draft(req: PredictRequest):
    import pandas as pd

    from lolwp.app.predict import Draft

    def side(slots):
        by = {s.role: s for s in slots}
        return ([by[r].player_id for r in ROLES], [by[r].champion for r in ROLES])

    bp, bc = side(req.blue)
    rp, rc = side(req.red)
    unknown = [c for c in bc + rc if c not in S["by_key"]]
    if unknown:
        raise HTTPException(422, f"未知的英雄 key：{unknown}")
    now = pd.Timestamp(req.kickoff_utc, tz="UTC") if req.kickoff_utc else None
    return Draft(req.league, req.blue_team_id or "", req.red_team_id or "",
                 bp, rp, bc, rc, now=now)


def _confidence_note(p: float) -> str | None:
    hi = max(p, 1 - p)
    for lo, actual in _EXTREME:
        if hi >= lo:
            return (f"这个概率落在模型的高自信段。历史上模型给出 ≥{lo:.0%} 的比赛，"
                    f"实际胜率只有 {actual:.1%}——高自信段系统性过度自信。")
    return None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    art = S["art"]
    m = art.meta
    return HealthResponse(
        ok=True, model_version=m.get("model", "?"),
        trained_on=str(m.get("trained_on")), holdout_start=str(m.get("holdout_start")),
        n_train_games=int(m.get("n_train_games", 0)),
        oos_log_loss=float(m.get("oos_log_loss", 0)),
        holdout_log_loss=m.get("holdout_log_loss"),
        n_champions=len(S["champs"]),
        n_known_players=len(art.prior_state.get("player_elo", {})),
        n_known_teams=len(art.prior_state.get("team_elo", {})),
        live_available=S.get("poller") is not None,
        run_id=RUN_ID,
    )


@app.get("/champions", response_model=list[Champion])
def champions() -> list[Champion]:
    v = S["dd_version"]
    return [Champion(dd_key=c["dd_key"], dd_id=c["dd_id"], name=c["name"],
                     icon=f"/dd/{v}/{c['dd_id']}.png") for c in S["champs"]]


@app.get("/players")
def players(league: str = Query(..., description="canonical 赛区 code"),
            limit: int = 400) -> dict[str, Any]:
    """选手选择器数据：该赛区各队最近一场的首发。"""
    import pandas as pd

    if not os.path.exists(SILVER):
        raise HTTPException(503, "silver 未构建，请先 make silver")
    g = pd.read_parquet(SILVER, columns=["league_canon", "kickoff_utc", "blue_team_id",
                                         "red_team_id", "blue_team_name", "red_team_name",
                                         "blue_player_ids", "red_player_ids"])
    g = g[g.league_canon == league].sort_values("kickoff_utc", ascending=False)
    if g.empty:
        raise HTTPException(404, f"赛区 {league} 无数据")
    teams: dict[str, dict[str, Any]] = {}
    for r in g.head(limit).itertuples(index=False):
        for pfx in ("blue", "red"):
            tid = str(getattr(r, f"{pfx}_team_id"))
            if tid in teams or tid == "None":
                continue
            teams[tid] = {"team_id": tid, "team_name": getattr(r, f"{pfx}_team_name"),
                          "roster": [str(p) if p else None
                                     for p in getattr(r, f"{pfx}_player_ids")],
                          "roles": list(ROLES)}
    return {"league": league, "n_teams": len(teams), "teams": list(teams.values())}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    from lolwp.app.predict import predict as _p
    from lolwp.models.artifact import LAYER_OF

    art = S["art"]
    r = _p(art, _to_draft(req))
    return PredictResponse(
        p_blue=r.p_blue, logit=r.logit, layers=r.layers, warnings=r.warnings,
        model_version=art.meta.get("model", "?"),
        confidence_note=_confidence_note(r.p_blue),
        contributions=[
            Contribution(name=k, label=LABELS.get(k, k),
                         layer=LAYER_OF.get(k, "其它"),
                         delta_logit=v, value=r.features.get(k, float("nan")))
            for k, v in sorted(r.contributions.items(), key=lambda x: -abs(x[1]))
        ],
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """what-if 敏感度。向量化：实力层与赛况层只算一次。"""
    from lolwp.app.predict import whatif_swap

    t0 = time.perf_counter()
    d = _to_draft(req.base)
    slot = list(ROLES).index(req.role)
    bad = [c for c in req.candidates if c not in S["by_key"]]
    if bad:
        raise HTTPException(422, f"未知的英雄 key：{bad[:5]}")
    wi = whatif_swap(S["art"], d, req.side, slot, req.candidates)
    return BatchPredictResponse(
        base_p_blue=float(wi.attrs["base_p"]),
        items=[BatchItem(champion=int(r.champ_key), p_blue=float(r.p_blue),
                         delta_pp=float(r.delta_pp)) for r in wi.itertuples(index=False)],
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/explain")
def explain(req: PredictRequest) -> dict[str, Any]:
    """逐项贡献 + 一段自然语言。分层是为了让用户看到 draft 的真实量级。"""
    r = predict(req)
    lay = r.layers
    strongest = max(r.contributions, key=lambda c: abs(c.delta_logit)) if r.contributions else None
    text = (
        f"蓝方胜率 {r.p_blue:.1%}。"
        f"其中实力层贡献 {lay.get('实力', 0):+.3f}、阵容层 {lay.get('阵容', 0):+.3f}、"
        f"赛况层 {lay.get('赛况', 0):+.3f}（对 logit）。"
    )
    if strongest:
        text += f"影响最大的单项是「{strongest.label}」（{strongest.delta_logit:+.3f}）。"
    text += ("注意：本项目实测整个 BP 阶段的可测量贡献接近 0，"
             "换一个英雄通常只动 1-2 个百分点；决定胜负的主要是队伍与选手实力。")
    return {"p_blue": r.p_blue, "layers": lay, "narrative": text,
            "contributions": [c.model_dump() for c in r.contributions],
            "warnings": r.warnings, "confidence_note": r.confidence_note}


@app.get("/live")
def live() -> dict[str, Any]:
    """当前进行中的比赛。**失败只降级本端点**，不影响 /predict。"""
    if S.get("poller") is None:
        try:
            from lolwp.live.poller import build_default_poller
            S["poller"] = build_default_poller()
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"{type(e).__name__}: {e}", "games": []}
    try:
        S["poller"].tick()
        snap = S["poller"].snapshot()
        return {"available": True, "health": S["poller"].health(), **snap}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}", "games": []}
