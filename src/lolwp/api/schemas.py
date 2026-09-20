"""API 契约（PLAN.md §7.5）。

`champion` 一律用 **Data Dragon 数字 key**，那是项目内部的唯一英雄 ID。
`player_id` / `team_id` 可以为 null，缺失时走 §5.5 的 fallback 并在
响应的 `warnings` 里说明——**绝不静默降级**。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Role = Literal["top", "jng", "mid", "bot", "sup"]
ROLES: tuple[Role, ...] = ("top", "jng", "mid", "bot", "sup")


class Slot(BaseModel):
    champion: int = Field(..., description="Data Dragon 数字 key")
    role: Role
    player_id: str | None = None


class PredictRequest(BaseModel):
    league: str = Field("LEC", description="canonical 赛区 code，如 LPL / LCK / WLDs")
    blue: list[Slot]
    red: list[Slot]
    blue_team_id: str | None = None
    red_team_id: str | None = None
    patch: str | None = None
    kickoff_utc: str | None = Field(None, description="ISO8601；不给则视作现在")

    @field_validator("blue", "red")
    @classmethod
    def _five_distinct_roles(cls, v: list[Slot]) -> list[Slot]:
        if len(v) != 5:
            raise ValueError(f"每边必须恰好 5 个位置，收到 {len(v)}")
        roles = [s.role for s in v]
        if len(set(roles)) != 5:
            raise ValueError(f"位置必须互不相同，收到 {roles}")
        return v


class Contribution(BaseModel):
    name: str
    label: str
    layer: str
    delta_logit: float
    value: float


class PredictResponse(BaseModel):
    p_blue: float
    logit: float
    contributions: list[Contribution]
    layers: dict[str, float]
    warnings: list[str] = []
    model_version: str
    # 高自信段的实测校准，前端据此提示（§5.6.4：截断区间必须诚实表达）
    confidence_note: str | None = None


class BatchPredictRequest(BaseModel):
    base: PredictRequest
    # 只改一个槽位：what-if 敏感度
    side: Literal["blue", "red"] = "blue"
    role: Role = "mid"
    candidates: list[int] = Field(..., description="候选英雄的 dd_key 列表")


class BatchItem(BaseModel):
    champion: int
    p_blue: float
    delta_pp: float


class BatchPredictResponse(BaseModel):
    base_p_blue: float
    items: list[BatchItem]
    elapsed_ms: float


class Champion(BaseModel):
    dd_key: int
    dd_id: str
    name: str
    icon: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    model_version: str
    trained_on: str
    holdout_start: str
    n_train_games: int
    oos_log_loss: float
    holdout_log_loss: float | None = None
    n_champions: int
    n_known_players: int
    n_known_teams: int
    live_available: bool
    run_id: str
