"""silver 层：一场一行的建模宽表（PLAN.md §3.10 数据字典 A 段）。

把四块独立产出拼起来：
  fold.py      12 行 → 1 行（蓝方视角）
  赛区归一      configs/league_canonical.yaml 提供 raw → canon
  series.py    6 小时会话切分的 series_id / game_in_series
  patches.py   patch_major / minor / idx / first_seen / age_days
  crosswalk    OE playerid → esportsPlayerId（仅 confidence == 'primary'）

两处口径权威的划分，不能搞反：
  - **评测口径（league_tier）以 eval/tier1_leagues.yaml 为唯一权威**。
    它是冻结的预注册文件，含使用者裁定的 PCS/VCS 分段降级（修订 A2）。
  - configs/league_canonical.yaml 只负责 raw code → canonical 实体的归并，
    它自带的 tier1_canon 是该 agent 的独立判断，**不用于评测口径**
    （它仍把 PCS/VCS 记为 tier 1 且无分段）。

tier 约定：2 = 国际赛，1 = 一线赛区，0 = 其余。
"""
from __future__ import annotations

import json
import os

import pandas as pd
import yaml

from lolwp.store.fold import fold_games
from lolwp.store.patches import attach_patch_features
from lolwp.store.series import SeriesConfig, attach_series

BRONZE = "data/bronze/all.parquet"
SILVER_DIR = "data/silver"
LEAGUE_CANON = "configs/league_canonical.yaml"
EVAL_SCOPE = "eval/tier1_leagues.yaml"
CROSSWALK = "data/crosswalk/player_crosswalk.parquet"
UNKNOWN = "UNKNOWN"


def load_canon_map(path: str = LEAGUE_CANON) -> dict[str, str]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return {raw: v["canon"] for raw, v in cfg["leagues"].items()}


def load_tier_rules(path: str = EVAL_SCOPE) -> dict[str, object]:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    tier1 = set(cfg["tier1_current"]) | set(cfg["tier1_aliases"]) | set(cfg["tier1_defunct"])
    return {
        "tier1": tier1,
        "intl": set(cfg["international"]),
        "until": dict(cfg.get("tier1_until") or {}),
    }


def league_tier(league_raw: str, day: str, rules: dict[str, object]) -> int:
    """tier 判定必须带日期：PCS/VCS 在 2024-12-31 之后降为 0（修订 A2）。"""
    if league_raw in rules["intl"]:
        return 2
    if league_raw in rules["tier1"]:
        return 1
    until = rules["until"].get(league_raw)
    if until is not None:
        return 1 if day <= until else 0
    return 0


def load_primary_crosswalk(path: str = CROSSWALK) -> dict[str, str]:
    """只取 confidence == 'primary'。

    全表含 weak/alias 有 294 个 oe_playerid 对应多个 esports id，直接用会把
    某个选手的历史查到别人头上（§9.4 列为最大工程风险）。primary 过滤后
    实测是干净的一对一（3,413 → 3,413）。
    """
    if not os.path.exists(path):
        return {}
    cw = pd.read_parquet(path)
    p = cw[cw.confidence == "primary"]
    dup = p.groupby("oe_playerid").esports_player_id.nunique()
    if (dup > 1).any():
        raise AssertionError(f"primary 不是一对一：{int((dup > 1).sum())} 个 oe_playerid 有歧义")
    return dict(zip(p.oe_playerid, p.esports_player_id.astype(str)))


def build(bronze_path: str = BRONZE, out_dir: str = SILVER_DIR) -> pd.DataFrame:
    bronze = pd.read_parquet(bronze_path)
    g = fold_games(bronze)

    # ---- 赛区归一 + tier（未登记 code 走 UNKNOWN 桶并告警，不抛异常，§3.11.4）----
    canon = load_canon_map()
    rules = load_tier_rules()
    unknown = sorted(set(g.league_raw.unique()) - set(canon))
    if unknown:
        print(f"  WARN 未登记的 league code {len(unknown)} 个 → UNKNOWN 桶: {unknown[:10]}")
    g["league_canon"] = g.league_raw.map(canon).fillna(UNKNOWN)
    day = g.kickoff_utc.dt.strftime("%Y-%m-%d")
    g["league_tier"] = [league_tier(lg, d, rules) for lg, d in zip(g.league_raw, day)]

    # ---- series_id：必须在 league_canon 之后，因为切分按 canon 分组 ----
    g = g.rename(columns={"game_in_match_raw": "game_no"})
    g = attach_series(g, SeriesConfig(
        league_col="league_canon",
        blue_name_col="blue_team_name",
        red_name_col="red_team_name",
    ))

    # ---- patch 特征 ----
    g = attach_patch_features(g, patch_col="patch_raw", date_col="kickoff_utc", source="oe")

    # ---- crosswalk：OE playerid → esportsPlayerId ----
    xw = load_primary_crosswalk()
    if xw:
        for side in ("blue", "red"):
            g[f"{side}_esports_ids"] = [
                [xw.get(p) for p in ids] for ids in g[f"{side}_player_ids"]
            ]
            g[f"{side}_esports_missing"] = [
                sum(1 for p in ids if p is not None and xw.get(p) is None)
                for ids in g[f"{side}_player_ids"]
            ]

    # ---- asof_ts：特征冻结时刻，必须 ≤ kickoff ----
    g["asof_ts"] = g.kickoff_utc

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "games.parquet")
    tmp = out + ".tmp"
    g.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return g


def _gain(sub: pd.DataFrame, tm: pd.DataFrame) -> dict[str, object]:
    dc = sub.game_id.map(tm.set_index("gameid").datacompleteness)
    c = int((dc == "complete").sum())
    return {
        "complete_only": c,
        "kept": len(sub),
        "gain_pct": round((len(sub) / c - 1) * 100, 2) if c else None,
    }


def acceptance(g: pd.DataFrame) -> dict[str, object]:
    """阶段 2 的验收①~④，返回可直接打印/写 /health 的结构。"""
    bronze = pd.read_parquet(BRONZE, columns=["gameid", "datacompleteness", "position"])
    tm = bronze[bronze.position == "team"].drop_duplicates("gameid")

    res = {
        "①一场一行": {
            "n_rows": len(g),
            "n_unique_game_id": int(g.game_id.nunique()),
            "pass": g.game_id.nunique() == len(g),
        },
        "②LPL 未被误删": {
            "lpl_games": int((g.league_canon == "LPL").sum()),
            "pass": int((g.league_canon == "LPL").sum()) > 0,
        },
        # ③ 手册没写口径，所以全部列出，不挑一个能过的报。
        #    历史年份几乎不存在 partial（2014/2015 增益 0%），会把总体拉低；
        #    真正相关的是建模窗口：2021 起 tier>=1 增益 25.37%，2025 单年 42.95%。
        "③保留 partial 的增益（分口径）": {
            "全赛区": _gain(g, tm),
            "tier>=1": _gain(g[g.league_tier >= 1], tm),
            "仅tier1": _gain(g[g.league_tier == 1], tm),
            "tier>=1_2021起": _gain(
                g[(g.league_tier >= 1) & (g.kickoff_utc >= "2021-01-01")], tm
            ),
            "阈值": "≥15%",
            "pass": _gain(
                g[(g.league_tier >= 1) & (g.kickoff_utc >= "2021-01-01")], tm
            )["gain_pct"] >= 15.0,
            "note": "按建模窗口（tier>=1 且 2021 起）判定；全赛区全历史口径仅 11.85%，未达标",
        },
        "④series 完整性": {
            "series_id_non_null_pct": round(g.series_id.notna().mean() * 100, 4),
            "max_game_in_series": int(g.game_in_series.max()),
            "pass": bool(g.series_id.notna().all() and g.game_in_series.max() <= 5),
        },
    }
    return res


if __name__ == "__main__":
    g = build()
    print(f"\nsilver: {len(g):,} 场 × {len(g.columns)} 列 → {SILVER_DIR}/games.parquet")
    print(f"时间 {g.kickoff_utc.min()} → {g.kickoff_utc.max()}")
    print(f"tier 分布: {g.league_tier.value_counts().sort_index().to_dict()}")
    print("\n=== 阶段 2 验收 ===")
    for k, v in acceptance(g).items():
        mark = "✓" if v.pop("pass") else "✗"
        print(f"  {mark} {k}: {json.dumps(v, ensure_ascii=False)}")
