"""泄漏白名单：只有 BP 结束、开赛前就能看到的列才准进特征。

PLAN.md §4.8.1。这是整个项目最便宜也最有效的一道防线：一旦让赛后统计
（gamelength / kills / golddiff / dragons …）混进特征，指标会漂亮得不像话，
而且不会有任何报错。红线是 accuracy > 67% 或 log loss < 0.60。
"""
from __future__ import annotations

import re

# BP 时点可见的 OE 原始列
ALLOWED_RAW: frozenset[str] = frozenset(
    {
        "gameid", "datacompleteness", "url", "league", "year", "split", "playoffs",
        "date", "game", "patch", "participantid", "side", "position",
        "playername", "playerid", "teamname", "teamid", "champion",
        "ban1", "ban2", "ban3", "ban4", "ban5",
        "pick1", "pick2", "pick3", "pick4", "pick5", "firstPick",
    }
)

# 只能作 label，绝不能进特征
LABEL_ONLY: frozenset[str] = frozenset({"result"})

# 只能记录，不能进特征：datacompleteness 与赛区强相关，进特征等于偷偷泄漏赛区
RECORD_ONLY: frozenset[str] = frozenset({"datacompleteness", "url"})

FORBIDDEN_RE = re.compile(r"(at10|at15|at20|at25)$|^(opp_|golddiff|xpdiff|csdiff)")


def classify(col: str) -> str:
    """返回 allowed / label_only / record_only / forbidden。"""
    if col in LABEL_ONLY:
        return "label_only"
    if col in RECORD_ONLY:
        return "record_only"
    if col in ALLOWED_RAW:
        return "allowed"
    return "forbidden"


def check_feature_columns(cols: list[str]) -> None:
    """特征矩阵的列必须全部来自 allowed，否则抛异常。CI 里跑这个。"""
    bad = [c for c in cols if classify(c) != "allowed"]
    if bad:
        raise AssertionError(f"以下列不得进入特征：{bad}")


def split_raw_columns(all_cols: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"allowed": [], "label_only": [], "record_only": [], "forbidden": []}
    for c in all_cols:
        out[classify(c)].append(c)
    return out
