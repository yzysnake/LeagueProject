"""Data Dragon 接入：英雄元数据 + 三体系 ID 映射。

三套英雄标识体系，必须能互相往返：
  1. OE CSV 的 `champion` 列      —— 展示名，如 "Lee Sin" / "K'Sante" / "Nunu & Willump"
  2. Data Dragon 的 `id`          —— 驼峰无空格，如 "LeeSin" / "KSante" / "Nunu"
  3. Data Dragon 的 `key`（数字） —— livestats feed 的 championId 用的就是它

livestats 返回数字 key，OE 存展示名，前端图标按 DDragon id 取。三者对不上，
选手历史英雄数据就会静默丢失——这是 PLAN.md §9.4 列为最大工程风险的那条。
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{ver}/data/en_US/champion.json"
UA = "lolwp/1.0 (qxing@uchicago.edu)"
OUT_DIR = "data/ddragon"


def latest_version(timeout: int = 30) -> str:
    r = requests.get(VERSIONS_URL, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    versions = r.json()
    if not versions:
        raise RuntimeError("Data Dragon 版本列表为空")
    return versions[0]


def fetch_champions(version: str, timeout: int = 30) -> dict[str, Any]:
    r = requests.get(
        CHAMPION_URL.format(ver=version), headers={"User-Agent": UA}, timeout=timeout
    )
    r.raise_for_status()
    return r.json()["data"]


def build_map(champions: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dd_id, c in champions.items():
        rows.append(
            {
                "dd_id": dd_id,            # "LeeSin"
                "dd_key": int(c["key"]),   # 64
                "name": c["name"],         # "Lee Sin"  ← OE champion 列用这个
                "title": c["title"],
            }
        )
    rows.sort(key=lambda r: r["dd_key"])
    return rows


def sync(out_dir: str = OUT_DIR) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ver = latest_version()
    champs = fetch_champions(ver)
    rows = build_map(champs)

    names = {r["name"] for r in rows}
    ids = {r["dd_id"] for r in rows}
    keys = {r["dd_key"] for r in rows}
    if not (len(names) == len(ids) == len(keys) == len(rows)):
        raise RuntimeError(
            f"三体系存在重复，无法建立双射：name={len(names)} id={len(ids)} "
            f"key={len(keys)} rows={len(rows)}"
        )

    payload = {"version": ver, "n_champions": len(rows), "champions": rows}
    path = os.path.join(out_dir, "champion_map.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return payload


def load(out_dir: str = OUT_DIR) -> dict[str, Any]:
    with open(os.path.join(out_dir, "champion_map.json")) as fh:
        return json.load(fh)


if __name__ == "__main__":
    p = sync()
    print(f"Data Dragon {p['version']}：{p['n_champions']} 个英雄 → {OUT_DIR}/champion_map.json")
