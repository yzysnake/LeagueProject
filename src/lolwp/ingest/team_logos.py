"""战队队标：build 时落到 frontend/public/teams/{esports_team_id}.png，运行时只引本地路径。

和英雄图标同一条纪律（PLAN.md §7.6.3）：**运行时打 CDN 就不叫离线可用**。
队伍表用 crosswalk 阶段已经缓存的 getTeams 结果（data/crosswalk/_cache/teams.json），
只下载 tier≥1 里最近一年打过比赛、且 crosswalk 对得上的队（约 80 支），统一缩到
128 px 的 PNG：原图从 145 px 到 8334 px 不等，最大的一张 1 MB 多，页面上只显示 36 px。
队标是纯展示用的可选资产：下载失败只跳过那一支，站点用队伍代码徽章兜底。
"""
from __future__ import annotations

import io
import json
import os

import pandas as pd
import requests
from PIL import Image

from lolwp.app import catalog

UA = "lolwp/1.0 (qxing@uchicago.edu)"
OUT_DIR = "frontend/public/teams"
SIZE = 128


def sync(out_dir: str = OUT_DIR, *, silver: str = "data/silver/games.parquet",
         crosswalk: str = "data/crosswalk/team_crosswalk.parquet",
         teams_json: str = "data/crosswalk/_cache/teams.json") -> tuple[int, int]:
    """返回 (新下载数, 现役队伍里有队标的数)。已有的不重下。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(teams_json) as fh:
        tj = json.load(fh)
    g = pd.read_parquet(silver)
    teams = catalog.build_teams(g[g.league_tier >= 1],
                                catalog.team_codes(pd.read_parquet(crosswalk), tj))
    by_id = {str(t.get("id")): t for t in tj}
    n_new = n_have = 0
    for eid in teams.loc[teams["active"], "esports_id"].dropna():
        url = (by_id.get(eid) or {}).get("image")
        if not url:
            continue
        path = os.path.join(out_dir, f"{eid}.png")
        if os.path.exists(path):
            n_have += 1
            continue
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        except (requests.RequestException, OSError) as e:
            print(f"  跳过 {eid}：{e}")
            continue
        im.thumbnail((SIZE, SIZE), Image.LANCZOS)
        im.save(path + ".tmp", format="PNG", optimize=True)
        os.replace(path + ".tmp", path)
        n_new += 1
        n_have += 1
    return n_new, n_have


if __name__ == "__main__":
    new, have = sync()
    print(f"队标：新下载 {new} 张，现役队伍共 {have} 张 → {OUT_DIR}/")
