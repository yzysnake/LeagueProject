"""唯一的数据入口。holdout 隔离在这里硬编码，不给任何人绕过的机会。

PLAN.md §6.1.3：**排除机制是日期截止线，不是那份 gameid 名单。**
holdout 窗口在开发期内还在持续产生新比赛（2026 世界赛现在才打到第 8 场），
用枚举名单排除的话，新比赛既不在名单里、日期又落在 holdout 期内，
会直接流进训练集，时间隔离当场破掉。
"""
from __future__ import annotations

import os

import pandas as pd

SILVER = "data/silver/games.parquet"
HOLDOUT_START_FILE = "eval/HOLDOUT_START"
# final holdout 的揭盲记录（2026-09-20，git 437d1cb）。它在，才允许生产重训用全量数据。
REVEAL_RECORD = "reports/final_holdout_result.json"


def holdout_start() -> pd.Timestamp:
    with open(HOLDOUT_START_FILE) as fh:
        return pd.Timestamp(fh.read().strip(), tz="UTC")


def load_dev(
    path: str = SILVER,
    *,
    min_tier: int = 0,
    since: str | None = None,
) -> pd.DataFrame:
    """开发用数据：**物理上不可能包含 holdout**。

    训宽评窄：默认 min_tier=0（全赛区训练）。评测时再用 min_tier=1 收窄。
    """
    df = pd.read_parquet(path)
    cutoff = holdout_start()
    n_before = len(df)
    df = df[df.kickoff_utc < cutoff]
    if since:
        df = df[df.kickoff_utc >= pd.Timestamp(since, tz="UTC")]
    if min_tier:
        df = df[df.league_tier >= min_tier]
    df = df.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    df.attrs["holdout_start"] = cutoff
    df.attrs["excluded_by_holdout"] = n_before - int((pd.read_parquet(
        path, columns=["kickoff_utc"]).kickoff_utc < cutoff).sum())
    return df


def load_holdout(path: str = SILVER, *, min_tier: int = 1) -> pd.DataFrame:
    """final holdout。**开发期一次都不要调用它。**

    调用处必须在 experiments/ 下留记录（见 preregistration.md）。
    这个函数故意不给默认的便利用法，且会打印一条刺眼的警告。
    """
    if os.environ.get("LOLWP_ALLOW_HOLDOUT") != "yes":
        raise RuntimeError(
            "拒绝加载 holdout。这是 final holdout，开发期只允许揭盲一次。\n"
            "确实要揭盲请设 LOLWP_ALLOW_HOLDOUT=yes，并在 experiments/ 下记录本次调用。"
        )
    print("!!! 正在加载 FINAL HOLDOUT —— 这应当在整个项目里只发生一次 !!!")
    df = pd.read_parquet(path)
    df = df[(df.kickoff_utc >= holdout_start()) & (df.league_tier >= min_tier)]
    return df.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)


def holdout_revealed() -> bool:
    return os.path.exists(REVEAL_RECORD)


def load_all(path: str = SILVER) -> pd.DataFrame:
    """**只给生产重训和它的推理端用**：holdout 揭盲之后，生产模型改用全部数据（预注册修订 A4）。

    评测一律仍走 load_dev()：walk-forward、门禁、被搁置假设的重测都只看 holdout 之前，
    数字与揭盲前可比。用全量数据训出的模型在 holdout 期是样本内的，**不能再拿 holdout 评它**；
    它之后的样本外成绩看盯盘台账（预测在赛果出来之前就已落盘）。
    揭盲记录不在时拒绝——揭盲之前这样用就是把 holdout 漏进了训练。
    """
    if not holdout_revealed():
        raise RuntimeError(f"holdout 尚未揭盲（{REVEAL_RECORD} 不存在），拒绝加载全量数据：开发期只能用 load_dev()")
    df = pd.read_parquet(path).sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True)
    df.attrs["holdout_start"] = holdout_start()
    return df
