"""把 bronze 的 12 行/场折叠成 1 行/场（蓝方视角）。

PLAN.md §2.2：样本单位是一张地图，不是一行 OE 记录。按行随机切分 = 标签直接泄漏。
折叠后必须满足 `df.game_id.nunique() == len(df)`。

本模块只做**与赛区归一 / series_id / patch_idx 无关**的那部分，那三项由各自模块
注入，避免这里变成一个什么都管的上帝函数。

实测依据（全量 102,084 场）：
  - 每场恰好 2 个 team 行（1 蓝 1 红）+ 10 个选手行，每个 (side, position) 唯一。
  - position 取值恰好 {top, jng, mid, bot, sup, team}，无变体（没有 bottom/utility）。
  - 42 场蓝红 result 同为 0（无胜者，40 场是 LDL 2018 的 partial），必须丢弃，
    否则会静默产出两条都标"负"的样本。
  - 10 场出现**镜像英雄**（同一英雄同时在双方阵容里），全部是 2025-12 的
    德玛西亚杯（DCup），属于特殊赛制而非数据错误。不在 tier-1 口径内，
    但必须打 `has_mirror_pick` 标记：任何"英雄出场"类编码在这些场次上
    蓝减红会抵消为 0，需要下游显式决定怎么处理。
"""
from __future__ import annotations

import json

import pandas as pd

ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
CHAMP_MAP = "data/ddragon/champion_map.json"


def load_champ_key_map(path: str = CHAMP_MAP) -> dict[str, int]:
    """OE 的 champion 展示名 → Data Dragon 数字 key。实测 173/173 精确匹配。"""
    with open(path) as fh:
        payload = json.load(fh)
    return {c["name"]: c["dd_key"] for c in payload["champions"]}


def fold_games(bronze: pd.DataFrame, champ_key: dict[str, int] | None = None) -> pd.DataFrame:
    """12 行 → 1 行（蓝方视角）。返回的每一列都必须是 BP 时点可见的。"""
    if champ_key is None:
        champ_key = load_champ_key_map()

    teams = bronze[bronze.position == "team"]
    players = bronze[bronze.position != "team"].copy()

    # ---- 标签：蓝方 team 行的 result；蓝红必须互补 ----
    res = teams.pivot_table(index="gameid", columns="side", values="result", aggfunc="first")
    valid = (res["Blue"].astype("float") + res["Red"].astype("float")) == 1
    dropped_label = int((~valid).sum())
    keep_ids = res.index[valid]

    teams = teams[teams.gameid.isin(keep_ids)]
    players = players[players.gameid.isin(keep_ids)]

    # ---- 场级元数据（取蓝方 team 行，这些列两边同值）----
    blue_t = teams[teams.side == "Blue"].set_index("gameid")
    red_t = teams[teams.side == "Red"].set_index("gameid")

    out = pd.DataFrame(index=blue_t.index)
    out["kickoff_utc"] = pd.to_datetime(blue_t["date"], utc=True)
    out["league_raw"] = blue_t["league"]
    out["season_year"] = blue_t["year"]          # 赛季年，≠ 日历年，见 test_bronze_temporal_traps
    out["split"] = blue_t["split"]
    out["playoffs"] = pd.to_numeric(blue_t["playoffs"], errors="coerce").astype("Int8")
    out["game_in_match_raw"] = pd.to_numeric(blue_t["game"], errors="coerce").astype("Int16")
    out["patch_raw"] = blue_t["patch"]
    out["datacompleteness"] = blue_t["datacompleteness"]      # 仅记录，禁止进特征
    out["data_snapshot_sha"] = blue_t["data_snapshot_sha"]

    out["blue_team_id"] = blue_t["teamid"]
    out["red_team_id"] = red_t["teamid"].reindex(out.index)
    out["blue_team_name"] = blue_t["teamname"]
    out["red_team_name"] = red_t["teamname"].reindex(out.index)

    out["y_blue_win"] = res.loc[out.index, "Blue"].astype("int8")

    # ---- 选手与英雄：按 (side, role) 展开成定长 5 元组 ----
    players["champ_key"] = players["champion"].map(champ_key).astype("Int32")
    for side, pfx in (("Blue", "blue"), ("Red", "red")):
        sub = players[players.side == side]
        pid = sub.pivot_table(index="gameid", columns="position", values="playerid", aggfunc="first")
        ck = sub.pivot_table(index="gameid", columns="position", values="champ_key", aggfunc="first")
        pid = pid.reindex(index=out.index, columns=list(ROLES))
        ck = ck.reindex(index=out.index, columns=list(ROLES))
        out[f"{pfx}_player_ids"] = pid.values.tolist()
        out[f"{pfx}_champ_keys"] = ck.astype("Int32").values.tolist()
        out[f"{pfx}_player_missing"] = pid.isna().sum(axis=1).astype("Int8")
        out[f"{pfx}_champ_missing"] = ck.isna().sum(axis=1).astype("Int8")

    # ---- 缺失指示（不是特征本身，是"这条特征能不能算"的开关）----
    bans = teams.pivot_table(index="gameid", columns="side",
                             values="ban1", aggfunc="first").reindex(out.index)
    out["has_ban_data"] = bans.notna().all(axis=1)
    fp = blue_t["firstPick"].reindex(out.index)
    out["has_first_pick"] = fp.notna()
    out["blue_first_pick"] = pd.to_numeric(fp, errors="coerce").astype("Int8")
    picks = teams.pivot_table(index="gameid", columns="side",
                              values="pick1", aggfunc="first").reindex(out.index)
    out["has_pick_order"] = picks.notna().all(axis=1)

    # 镜像英雄标记：正常职业赛制下 10 个英雄互不相同
    def _has_mirror(row) -> bool:
        ks = [int(x) for x in list(row["blue_champ_keys"]) + list(row["red_champ_keys"])
              if not pd.isna(x)]
        return len(ks) == 10 and len(set(ks)) != 10

    out["has_mirror_pick"] = out.apply(_has_mirror, axis=1)

    out = out.reset_index().rename(columns={"gameid": "game_id"})
    out = out.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)

    assert out.game_id.nunique() == len(out), "折叠后 game_id 不唯一"
    out.attrs["dropped_invalid_label"] = dropped_label
    return out


if __name__ == "__main__":
    bronze = pd.read_parquet("data/bronze/all.parquet")
    g = fold_games(bronze)
    print(f"折叠：{len(bronze):,} 行 → {len(g):,} 场（丢弃无胜者 {g.attrs['dropped_invalid_label']} 场）")
    print(f"时间范围 {g.kickoff_utc.min()} → {g.kickoff_utc.max()}")
    print(f"蓝方胜率 {g.y_blue_win.mean():.4f}")
    print(f"选手 ID 有缺失的场次 {(g.blue_player_missing.fillna(0) + g.red_player_missing.fillna(0) > 0).sum():,}")
    print(f"has_ban_data {g.has_ban_data.mean():.1%} | has_first_pick {g.has_first_pick.mean():.1%} | has_pick_order {g.has_pick_order.mean():.1%}")
