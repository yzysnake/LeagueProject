"""英雄三体系 ID 往返映射测试。

PLAN.md §9.4 把「OE 与 lolesports 的 ID 对不上」列为最大工程风险：匹配失败不报错，
只会让特征静默退化。英雄这一层必须是严格双射。
"""
import csv
import glob
import json
import os

import pytest

MAP = "data/ddragon/champion_map.json"
pytestmark = pytest.mark.skipif(not os.path.exists(MAP), reason="需先跑 ddragon.sync()")


@pytest.fixture(scope="module")
def cmap():
    return json.load(open(MAP))


def test_bijection(cmap):
    rows = cmap["champions"]
    assert len({r["name"] for r in rows}) == len(rows)
    assert len({r["dd_id"] for r in rows}) == len(rows)
    assert len({r["dd_key"] for r in rows}) == len(rows)


def test_round_trip_all_three(cmap):
    """name → key → id → name 必须回到原点。"""
    rows = cmap["champions"]
    by_name = {r["name"]: r for r in rows}
    by_key = {r["dd_key"]: r for r in rows}
    by_id = {r["dd_id"]: r for r in rows}
    for r in rows:
        assert by_key[by_name[r["name"]]["dd_key"]]["dd_id"] == r["dd_id"]
        assert by_name[by_id[r["dd_id"]]["name"]]["dd_key"] == r["dd_key"]


def test_known_tricky_names(cmap):
    """撇号、空格、连字符、&：最容易在规范化时被改坏的几个。"""
    by_name = {r["name"]: r for r in cmap["champions"]}
    for name, dd_id in [
        ("Lee Sin", "LeeSin"),
        ("K'Sante", "KSante"),
        ("Kai'Sa", "Kaisa"),
        ("Cho'Gath", "Chogath"),
        ("Nunu & Willump", "Nunu"),
        ("Dr. Mundo", "DrMundo"),
        ("Renata Glasc", "Renata"),
        ("Wukong", "MonkeyKing"),
    ]:
        assert name in by_name, f"{name} 不在映射表里"
        assert by_name[name]["dd_id"] == dd_id, f"{name} → {by_name[name]['dd_id']}，期望 {dd_id}"


@pytest.mark.skipif(not glob.glob("data/raw/*.csv"), reason="需先下载 OE 数据")
def test_every_oe_champion_resolves(cmap):
    """全量 OE 2014-2026 里出现过的每个英雄名都必须能映射，一个都不能漏。"""
    by_name = {r["name"] for r in cmap["champions"]}
    seen = set()
    for p in sorted(glob.glob("data/raw/*_LoL_esports_match_data_from_OraclesElixir.csv")):
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                c = (row.get("champion") or "").strip()
                if c and row.get("position") != "team":
                    seen.add(c)
    unresolved = seen - by_name
    assert not unresolved, f"{len(unresolved)} 个英雄名无法映射: {sorted(unresolved)}"
