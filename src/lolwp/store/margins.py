"""赛后经济差表：每场一行（终局双方总经济、时长），**只给分层 BT 当历史局的训练标签用**（预注册修订 A6）。

bronze 在最早的环节就把赛后统计列物理丢掉了（泄漏防线落在存储层），这里不碰那条：
本表从原始 CSV 只读 `whitelist.RATING_LABEL` 里那两列，单独存成 data/silver/margins.parquet，
从不并进特征表。用它的只有 `models.margin.with_margin_target`：快照只吃开赛早于快照时刻的比赛，
所以任何一局的经济差都进不了它自己的特征。

蓝红方按 OE 的 side 列，与 silver 的 blue_* / red_* 同一口径（silver 也是按 side 折叠的）。
"""
from __future__ import annotations

import os

import pandas as pd

from lolwp.features.whitelist import RATING_LABEL

RAW_DIR = "data/raw"
SILVER = "data/silver/games.parquet"
MARGINS = "data/silver/margins.parquet"
COLUMNS = ["blue_gold", "red_gold", "gamelength_s"]


def build(raw_dir: str = RAW_DIR, silver_path: str = SILVER, out: str = MARGINS) -> pd.DataFrame:
    """原始 CSV 的队伍行 → silver 里每场一行。返回以 game_id 为索引、列为 COLUMNS 的表。"""
    ids = set(pd.read_parquet(silver_path, columns=["game_id"]).game_id)
    usecols = ["gameid", "side", "position", *sorted(RATING_LABEL)]
    parts = []
    for fn in sorted(f for f in os.listdir(raw_dir) if f.endswith("OraclesElixir.csv")):
        d = pd.read_csv(os.path.join(raw_dir, fn), usecols=usecols, low_memory=False,
                        dtype={"gameid": str, "side": str, "position": str})
        parts.append(d[(d.position == "team") & d.gameid.isin(ids)])
    t = pd.concat(parts, ignore_index=True).drop_duplicates(["gameid", "side"], keep="last")
    blue = t[t.side == "Blue"].set_index("gameid")
    red = t[t.side == "Red"].set_index("gameid")
    idx = blue.index.intersection(red.index)
    num = lambda s: pd.to_numeric(s.loc[idx], errors="coerce")
    m = pd.DataFrame({"blue_gold": num(blue.totalgold), "red_gold": num(red.totalgold),
                      "gamelength_s": num(blue.gamelength).fillna(num(red.gamelength))}, index=idx)
    m.index.name = "game_id"
    m = m.sort_index()
    tmp = out + ".tmp"
    m.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return m


def load(path: str = MARGINS) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在：先跑 `make silver`（会顺带建赛后经济差表）")
    return pd.read_parquet(path).set_index("game_id")


if __name__ == "__main__":
    m = build()
    n_silver = len(pd.read_parquet(SILVER, columns=["game_id"]))
    ok = m[COLUMNS].notna().all(axis=1)
    print(f"赛后经济差表 → {MARGINS}：{len(m):,} 场（silver {n_silver:,} 场），"
          f"三列齐全 {ok.mean():.2%}；缺的局 BT 退回胜负标签")
