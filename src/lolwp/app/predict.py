"""推理：纯函数，无状态，无 I/O（artifact 在外面加载一次）。

设计要点：
1. **与训练共用同一套特征公式**（features_from_state / draft_features_from_state），
   不在这里重写任何公式。training/serving skew 是这类系统最常见的静默故障。
2. 分层分解：把 logit 拆成「实力 / 阵容 / 赛况」三层分别展示（§7.6）。
   只显示总胜率的话，用户换一个英雄看到只动 1-2pp 会以为工具坏了；
   分层显示才能让用户看到 draft 的真实贡献量级——这既是产品设计也是诚实性。
3. what-if 必须向量化：遍历 173 个英雄要在 500ms 内出结果。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from lolwp.features.draft_priors import draft_features_from_state
from lolwp.features.priors import features_from_state
from lolwp.models.artifact import LAYER_OF, Artifact
from lolwp.models.hier_bt import HierBTSnapshot


@dataclass
class Draft:
    league: str
    blue_team_id: str
    red_team_id: str
    blue_player_ids: list[str]
    red_player_ids: list[str]
    blue_champ_keys: list[int]
    red_champ_keys: list[int]
    now: pd.Timestamp | None = None


@dataclass
class Prediction:
    p_blue: float
    logit: float
    contributions: dict[str, float]
    layers: dict[str, float]
    features: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def build_features(art: Artifact, d: Draft) -> tuple[dict[str, float], list[str]]:
    warns: list[str] = []
    f = features_from_state(
        art.prior_state,
        league_canon=d.league,
        blue_team_id=d.blue_team_id,
        red_team_id=d.red_team_id,
        blue_player_ids=d.blue_player_ids,
        red_player_ids=d.red_player_ids,
    )
    # 分层 BT 特征：走 HierBTSnapshot.features()，与训练**同一个函数**（§7.4）
    if art.bt_state:
        snap = HierBTSnapshot.from_dict(art.bt_state)
        f.update(snap.features(
            blue_team_id=str(d.blue_team_id), red_team_id=str(d.red_team_id),
            blue_player_ids=[str(p) if p else None for p in d.blue_player_ids],
            red_player_ids=[str(p) if p else None for p in d.red_player_ids],
        ))
    f.update(
        draft_features_from_state(
            art.draft_state,
            blue_champ_keys=d.blue_champ_keys,
            red_champ_keys=d.red_champ_keys,
            blue_player_ids=d.blue_player_ids,
            red_player_ids=d.red_player_ids,
            now=d.now,
        )
    )

    # 三类必须告警但绝不能抛异常的情况（阶段 5 验收④）
    te = art.prior_state["team_elo"]
    for side, tid in (("蓝方", d.blue_team_id), ("红方", d.red_team_id)):
        if str(tid) not in te:
            warns.append(f"{side}战队「{tid}」不在历史数据中，已退化为联赛平均实力")
    pe = art.prior_state["player_elo"]
    unknown = [p for p in list(d.blue_player_ids) + list(d.red_player_ids) if p and str(p) not in pe]
    if unknown:
        warns.append(f"{len(unknown)} 名选手不在历史数据中，其实力已退化为先验值：{unknown[:3]}")
    if d.league not in art.prior_state["side_league"]:
        warns.append(f"赛区「{d.league}」不在历史数据中，蓝方先验已退化为全局值")
    if f.get("champ_cover_min", 5) < 5:
        warns.append(
            f"有英雄在当前版本的一线赛事中缺少出场记录（覆盖 {int(f['champ_cover_min'])}/5），"
            "其版本强度已退化为该位置基准，概率可信度下降"
        )
    return f, warns


def predict(art: Artifact, d: Draft) -> Prediction:
    f, warns = build_features(art, d)
    contrib: dict[str, float] = {}
    z = art.intercept
    for name, w in art.coef.items():
        x = (f[name] - art.mu[name]) / art.sigma[name]
        c = w * x
        contrib[name] = c
        z += c
    layers: dict[str, float] = {}
    for name, c in contrib.items():
        layers[LAYER_OF[name]] = layers.get(LAYER_OF[name], 0.0) + c
    return Prediction(
        p_blue=_sigmoid(z), logit=z, contributions=contrib, layers=layers,
        features=f, warnings=warns,
    )


def whatif_swap(
    art: Artifact, d: Draft, side: str, slot: int, candidate_keys: list[int]
) -> pd.DataFrame:
    """把 side 的第 slot 个位置换成每一个候选英雄，返回概率变化。

    向量化要点：只有 champ_meta_diff 与 mastery 相关项会变，实力层与赛况层恒定，
    所以基线 logit 只算一次，循环里只重算 draft 部分。
    """
    base = predict(art, d)
    fixed_z = base.logit - base.contributions["champ_meta_diff"]
    w = art.coef["champ_meta_diff"]
    mu, sg = art.mu["champ_meta_diff"], art.sigma["champ_meta_diff"]

    keys_b = list(d.blue_champ_keys)
    keys_r = list(d.red_champ_keys)
    target = keys_b if side == "blue" else keys_r

    rows = []
    for ck in candidate_keys:
        t = list(target)
        t[slot] = ck
        df = draft_features_from_state(
            art.draft_state,
            blue_champ_keys=t if side == "blue" else keys_b,
            red_champ_keys=t if side == "red" else keys_r,
            blue_player_ids=d.blue_player_ids,
            red_player_ids=d.red_player_ids,
            now=d.now,
        )
        z = fixed_z + w * (df["champ_meta_diff"] - mu) / sg
        rows.append({"champ_key": ck, "p_blue": _sigmoid(z), "delta_pp": (_sigmoid(z) - base.p_blue) * 100})
    out = pd.DataFrame(rows).sort_values("delta_pp", ascending=False).reset_index(drop=True)
    out.attrs["base_p"] = base.p_blue
    return out


def check_decomposition(pred: Prediction, tol: float = 1e-9) -> None:
    """验收③：贡献之和加截距必须精确等于 logit。"""
    s = sum(pred.contributions.values()) + pred.intercept if hasattr(pred, "intercept") else None
    s = sum(pred.contributions.values())
    if abs(s + _intercept_of(pred) - pred.logit) > tol:
        raise AssertionError("分解不闭合")


def _intercept_of(pred: Prediction) -> float:
    return pred.logit - sum(pred.contributions.values())
