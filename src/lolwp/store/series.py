"""series_id 会话切分：把 OE 的"一场一行"切成 Bo1/Bo3/Bo5 系列赛分组键。

为什么这个文件值得单独存在
--------------------------
`series_id` 是全项目最 load-bearing 的键：分组 CV 切分、embargo、block bootstrap、
series 级 ICC、MDE 全挂在它上面（PLAN.md §2.2 / §4.8.2 第 4 条 / §6.3）。劈错了会让
同一个 Bo5 的 g1-g3 与 g4-g5 落进不同 fold——这比标签泄漏更隐蔽，因为所有列级检查都
是绿的。

三条实测结论（2026-09-18，13 年全量 102,084 场，代码见 tests/test_series.py）
----------------------------------------------------------------------------
1. **按 UTC 日历日构造是错的**（PLAN.md §2.2 已证伪）。本机快照复现：2026 全年
   8,851 场里恰好 **89 个真系列被零点劈开**（判据：同 league 同队对、相邻两局间隔
   < 6 小时、却落在两个日历日）。手册给的 89 完全对上。赛区分布 LRN 32 / NACL 22 /
   LCS 13 / LRS 11 / CBLOL 5 / CD 4 / AC 1 / LPLOL 1——清一色美洲（北美晚间开赛
   = UTC 次日凌晨）。全量 13 年是 1,138 对、1,121 个真系列被切碎。
2. **6 小时阈值经得起标定，不是拍脑袋。** 用 OE 自带的 `game` 列做弱标签
   （game 号 +1 = 同系列，n=42,356；game 号回到 1 = 新系列，n=42,299）扫阈值：

       T        同系列被切断(FN)      异系列被粘连(FP)
       3h       114 (0.269%)          172 (0.407%)
       4h        41 (0.097%)          204 (0.482%)
       5h        31 (0.073%)          221 (0.523%)
       6h        30 (0.071%)          235 (0.556%)   ← 采用
       12h       30 (0.071%)          243 (0.575%)
       24h       27 (0.064%)          485 (1.147%)

   FN 在 5.5h 处触底（30 例是不可约的：OE 里跨天续赛/赛会编号），6h→12h 整段平坦
   （102,084 场里只有 8 对相邻局的间隔落在 6–12h 区间，直方图上是一段真空）。
   而两类错误的代价**不对称**：FN = 真系列跨 fold（§4.8.2 说比标签泄漏更严重），
   FP = 两个系列被并成一个（只损失一点有效样本量）。所以阈值要取在 FN 平台上、
   宁可多粘不可多切 → **6h 是对的，保留手册取值**。间隔分布也支持：
   p50=0.93h、p99=2.20h、p99.9=3.77h，`[4.5h,6h)` 只有 20 对，`[6h,14h)` 只有 8 对。
3. **6h 单靠时间修不了反方向（两个独立系列被粘连）。** 2026 有 4 例同队对在 6 小时
   内打了两个独立系列（game 号出现 [1,1] / [1,2,1] 这种重复），全量 193 例。纯时间
   规则对它们无能为力（其中 44 对的 kickoff 时间戳**完全相同**，任何阈值都分不开）。
   因此本实现在 6h 分块之上加了一条**最小干预**的二次切分：**同一块内 game 号重复
   时，在重复处开新 series**。它只对 game 号出现重复的块生效（全量 193 个块），
   不碰 [2,1,3] 这类仅乱序、无重复的块（全量 42 例；乱序更可能是 OE 时间戳噪声，
   而"宁可多粘不可多切"要求保守）。加上这条后全量重复 game 号的 series 从 193 降到
   **0**，局号非 1..n 的 series 从 384 降到 201，代价是多出 203 个 series
   （59,467 → 59,670，+0.34%），m̄ 从 1.7166 变成 1.7108。

与手册的偏离
------------
- 手册 §2.2 的参考实现只有 6h 规则，没有 game 号去重子切分。上面第 3 条是实测依据。
  想要严格复刻手册行为：`SeriesConfig(game_col=None)`。
- 手册断言 (a)「同 series 内 game 号是从 1 开始、无重复的连续序列」**在全量上不可能
  成立**：全量 201 个 series（0.34%）的 game 号不是 1..n，其中 105 个是单局 series
  ——OE 根本没收录同系列的其它局，pattern 大头是 `(2,)` 79 次、`(3,)` 25 次；另外
  96 个是多局但缺号（`(2,3)` 48 次、`(1,3)` 29 次）。这是**数据覆盖缺口，不是切分
  错误**——照抄成 assert 会让 CI 永远红。可执行的等价断言是「同 series 内局号无重复
  + max(局号) ≤ 5 + 长度 ≤ 5 + 非 1..n 的比例 < 0.5%」，见 `audit()` 与
  tests/test_series.py::test_game_numbers_are_1_to_n_for_the_vast_majority。
- 局号只要求无重复、不要求严格递增：全量 42 个 series 的局号按开赛时间排是乱序的
  （如 LAS 2026-09-13 的 [2,1,3]），那是 OE 时间戳噪声，切开的代价比留着大。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

#: 会话切分阈值。见模块 docstring 第 2 条的标定表——别随手改，改了要重跑标定。
GAP_DEFAULT: pd.Timedelta = pd.Timedelta(hours=6)

#: 一个 series 的物理上限（Bo5）。超过它一定是切分过宽或数据异常。
MAX_GAMES_PER_SERIES: int = 5

#: series_id 里的时间戳格式（手册 §3.10 的字段示例用的就是分钟精度）
_TS_FMT = "%Y%m%dT%H%M"


@dataclass(frozen=True)
class SeriesConfig:
    """列名与阈值的集中配置，免得下游到处传一堆 kwargs。

    默认列名对齐 §3.10 的 silver 数据字典（`kickoff_utc` / `league_canon` /
    `blue_team_id` / `red_team_id` / `game_id`），`game_no` 是 OE `game` 列折叠后的名字。
    """

    gap: pd.Timedelta = GAP_DEFAULT
    league_col: str = "league_canon"
    time_col: str = "kickoff_utc"
    blue_col: str = "blue_team_id"
    red_col: str = "red_team_id"
    game_id_col: str = "game_id"
    #: OE 的 `game` 列（系列内第几局）。设成 None 关闭 game 号去重子切分（= 手册原版行为）
    game_col: str | None = "game_no"
    #: team_id 缺失时的兜底列（队名）。silver 会把缺 id 的场次整场丢掉，这里只是防御
    blue_name_col: str | None = None
    red_name_col: str | None = None


def _resolve(cfg: SeriesConfig | None, kwargs: dict[str, Any]) -> SeriesConfig:
    base = cfg if cfg is not None else SeriesConfig()
    return replace(base, **kwargs) if kwargs else base


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"build_series_id 缺列 {missing}；现有列 {list(df.columns)[:20]}…。"
            " 列名不一致就传 SeriesConfig(league_col=..., time_col=...) 显式指定。"
        )


def _team_key(df: pd.DataFrame, cfg: SeriesConfig) -> tuple[pd.Series, pd.Series]:
    """取两队的身份键。缺失值绝不能被 fillna 成同一个常量——那会把所有缺 id 的场次
    粘成一个巨型 series。这里退化为逐行唯一的哨兵值，保证"不可合并"。"""
    gid = df[cfg.game_id_col].astype("string")

    def one(col: str, fallback: str | None, tag: str) -> pd.Series:
        s = df[col].astype("string")
        if fallback is not None and fallback in df.columns:
            s = s.fillna(df[fallback].astype("string"))
        sentinel = "__NA_" + tag + "__" + gid
        return s.fillna(sentinel).astype(object)

    return (
        one(cfg.blue_col, cfg.blue_name_col, "B"),
        one(cfg.red_col, cfg.red_name_col, "R"),
    )


def pair_key(df: pd.DataFrame, cfg: SeriesConfig | None = None, **kwargs: Any) -> pd.Series:
    """无序队对键 `min|max`。无序是必须的：同一系列里蓝红方每局都换边。"""
    cfg = _resolve(cfg, kwargs)
    _require(df, [cfg.blue_col, cfg.red_col, cfg.game_id_col])
    b, r = _team_key(df, cfg)
    return pd.Series(
        ["|".join(sorted((str(x), str(y)))) for x, y in zip(b, r)],
        index=df.index,
        dtype=object,
    )


def _sorted_view(df: pd.DataFrame, cfg: SeriesConfig) -> pd.DataFrame:
    """按 (league, pair, kickoff, game_no, game_id) 排序的视图。

    为什么排序键要带 game_no 和 game_id：全量里有 **44 对相邻局的 kickoff 完全同秒**
    （OE 的 date 偶尔是排期时间不是开局时间）。只按时间排序时这些行的先后由 pandas
    的稳定排序 + 输入行序决定——同样的数据换个读取顺序就得到不同的 series_id。
    带上 game_no（系列内序）和 game_id（全局唯一）后排序完全确定。
    """
    cols = [cfg.league_col, cfg.time_col, cfg.blue_col, cfg.red_col, cfg.game_id_col]
    _require(df, cols)
    if not pd.api.types.is_datetime64_any_dtype(df[cfg.time_col]):
        raise TypeError(
            f"{cfg.time_col} 必须是 datetime64（OE 的 date 是 naive UTC），"
            f"实际 {df[cfg.time_col].dtype}"
        )
    if df[cfg.game_id_col].duplicated().any():
        n = int(df[cfg.game_id_col].duplicated().sum())
        raise ValueError(f"{cfg.game_id_col} 有 {n} 个重复——silver 必须先折叠成一场一行")

    work = pd.DataFrame(
        {
            "_league": df[cfg.league_col].astype("string").fillna("__NA_LEAGUE__").astype(object),
            "_pair": pair_key(df, cfg),
            "_t": df[cfg.time_col],
            "_gid": df[cfg.game_id_col].astype("string").astype(object),
        },
        index=df.index,
    )
    if cfg.game_col is not None and cfg.game_col in df.columns:
        work["_g"] = pd.to_numeric(df[cfg.game_col], errors="coerce")
    else:
        work["_g"] = np.nan
    return work.sort_values(
        ["_league", "_pair", "_t", "_g", "_gid"], kind="stable", na_position="last"
    )


def _blocks(work: pd.DataFrame, cfg: SeriesConfig) -> pd.Series:
    """在排序视图上算块号（连续整数），索引 = 排序后的 df.index。"""
    new_group = (work["_league"] != work["_league"].shift(1)) | (
        work["_pair"] != work["_pair"].shift(1)
    )
    over_gap = (work["_t"] - work["_t"].shift(1)) > cfg.gap
    brk = np.asarray(new_group | over_gap, dtype=bool).copy()
    brk[0] = True

    if cfg.game_col is None or not work["_g"].notna().any():
        return pd.Series(np.cumsum(brk), index=work.index, name="_blk")

    # 二次切分：同一块内 game 号重复 → 在重复处开新 series（见模块 docstring 第 3 条）。
    # 先筛出"块内确实有重复局号"的块——全量 102,084 场里只有 193 个，剩下 59k 个块不必进 Python 循环。
    gno = work["_g"].to_numpy(dtype="float64")
    blk0 = np.cumsum(brk)
    valid = ~np.isnan(gno)
    dup_rows = (
        pd.DataFrame({"b": blk0, "g": gno}).duplicated(subset=["b", "g"], keep=False).to_numpy()
        & valid
    )
    if dup_rows.any():
        dirty = set(blk0[dup_rows].tolist())
        starts = np.flatnonzero(brk)
        for lo, hi in zip(starts, np.append(starts[1:], len(brk)), strict=True):
            if blk0[lo] not in dirty:
                continue
            seen: set[float] = set()
            for i in range(lo, hi):
                if not valid[i]:
                    continue
                if gno[i] in seen:      # 局号回头 = 新的一个系列从这一局开始
                    brk[i] = True
                    seen = {gno[i]}
                else:
                    seen.add(gno[i])
    return pd.Series(np.cumsum(brk), index=work.index, name="_blk")


def _format_ids(work: pd.DataFrame, blk: pd.Series) -> pd.Series:
    """块号 → `LEAGUE::teamA|teamB::YYYYmmddTHHMM` 字符串，并消解碰撞。

    碰撞真实存在：二次切分出来的两个 series 可能**同分同秒**开始（全量 24 例，
    都是 2021 BIG / 2023 EM 这类同一分钟并发开赛的业余赛）。碰撞会让两个不同的
    series 共用一个 key，分组切分再次静默失效——所以第 2 个及以后追加 `#k` 后缀。
    """
    start = work["_t"].groupby(blk).transform("min")
    base = work["_league"].astype(str) + "::" + work["_pair"].astype(str) + "::" + start.dt.strftime(_TS_FMT)
    per_block = base.groupby(blk).first()
    order = per_block.groupby(per_block).cumcount()
    suffix = np.where(order.to_numpy() > 0, "#" + (order.to_numpy() + 1).astype(str), "")
    final = per_block + suffix
    return blk.map(final)


def build_series_id(
    df: pd.DataFrame, cfg: SeriesConfig | None = None, **kwargs: Any
) -> pd.Series:
    """构造 series_id（一场一行的 df → 与 df.index 对齐的 str Series）。

    规则：同 league、同无序队对、相邻局间隔 ≤ `cfg.gap`（默认 6h）归为一个 series；
    额外地，同一块内 game 号重复时在重复处开新 series。

    >>> sid = build_series_id(silver)          # 列名走 §3.10 默认
    >>> sid = build_series_id(df, league_col="league_raw", game_col=None)  # 手册原版行为
    """
    cfg = _resolve(cfg, kwargs)
    if len(df) == 0:
        return pd.Series(index=df.index, dtype=object, name="series_id")
    work = _sorted_view(df, cfg)
    blk = _blocks(work, cfg)
    return _format_ids(work, blk).reindex(df.index).rename("series_id")


def build_game_in_series(
    df: pd.DataFrame,
    series_id: pd.Series | None = None,
    cfg: SeriesConfig | None = None,
    **kwargs: Any,
) -> pd.Series:
    """系列内第几局（1..n，按 kickoff 顺序）。

    **注意它不等于 OE 的 `game` 列**：OE 的 `game` 会缺号（全量 251 个 series 的
    game 号不是 1..n，因为 OE 根本没收录那几局），而本列按位置生成，恒为 1..n 连续。
    做「game_in_series ≤ 5」这类硬断言要用本列；要还原真实局号用 OE 的 `game`。
    """
    cfg = _resolve(cfg, kwargs)
    if len(df) == 0:
        return pd.Series(index=df.index, dtype="int64", name="game_in_series")
    if series_id is None:
        series_id = build_series_id(df, cfg)
    work = _sorted_view(df, cfg)
    sid = series_id.reindex(work.index)
    return (
        (sid.groupby(sid, sort=False).cumcount() + 1)
        .reindex(df.index)
        .astype("int64")
        .rename("game_in_series")
    )


def attach_series(
    df: pd.DataFrame, cfg: SeriesConfig | None = None, **kwargs: Any
) -> pd.DataFrame:
    """便利函数：一次性挂上 `series_id` 与 `game_in_series` 两列（返回副本）。"""
    cfg = _resolve(cfg, kwargs)
    sid = build_series_id(df, cfg)
    return df.assign(series_id=sid, game_in_series=build_game_in_series(df, sid, cfg))


# --------------------------------------------------------------------------- #
# 校验 / 诊断                                                                   #
# --------------------------------------------------------------------------- #


def count_split_series(
    df: pd.DataFrame,
    series_id: pd.Series | None = None,
    cfg: SeriesConfig | None = None,
    *,
    respect_game_no: bool = True,
    **kwargs: Any,
) -> int:
    """手册 §2.2 断言 (c)：存在「同 league 同队对、相邻两局间隔 < gap 却 series_id 不同」的样本数。

    `respect_game_no=True`（默认）会排除 game 号**没有严格递增**的相邻对——那正是
    本模块二次切分**故意**切开的两个独立系列（全量 193 处），把它们算成"被劈开"是
    误报。`respect_game_no=False` 复刻手册的字面判据，用来复现手册那 89 的数字。
    """
    cfg = _resolve(cfg, kwargs)
    if len(df) == 0:
        return 0
    if series_id is None:
        series_id = build_series_id(df, cfg)
    work = _sorted_view(df, cfg)
    sid = series_id.reindex(work.index)
    same = (work["_league"] == work["_league"].shift(1)) & (work["_pair"] == work["_pair"].shift(1))
    near = (work["_t"] - work["_t"].shift(1)) < cfg.gap
    diff = sid != sid.shift(1)
    bad = same & near & diff
    if respect_game_no:
        g, gp = work["_g"], work["_g"].shift(1)
        # game 号已知且没有递增 → 本来就是两个系列，不算"被劈开"
        deliberate = (g.notna() & gp.notna() & (g <= gp)).fillna(False)
        bad = bad & ~deliberate
    return int(bad.sum())


def series_span_hours(
    df: pd.DataFrame,
    series_id: pd.Series | None = None,
    cfg: SeriesConfig | None = None,
    **kwargs: Any,
) -> pd.Series:
    """每个 series 的首末局时间跨度（小时）。手册断言 (b) 要求 ≤ 12h。"""
    cfg = _resolve(cfg, kwargs)
    if series_id is None:
        series_id = build_series_id(df, cfg)
    t = df[cfg.time_col].groupby(series_id)
    return (t.max() - t.min()).dt.total_seconds() / 3600.0


def series_stats(
    df: pd.DataFrame,
    series_id: pd.Series | None = None,
    cfg: SeriesConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """`m̄` / `n_series` / 长度分布——直接喂给 §6.3 的 `deff = 1 + (m̄−1)·ICC`。"""
    cfg = _resolve(cfg, kwargs)
    if series_id is None:
        series_id = build_series_id(df, cfg)
    size = series_id.groupby(series_id).size()
    span = series_span_hours(df, series_id, cfg)
    return {
        "n_games": len(df),
        "n_series": int(size.size),
        "m_bar": float(size.mean()) if size.size else float("nan"),
        "length_dist": {int(k): int(v) for k, v in sorted(size.value_counts().items())},
        "max_len": int(size.max()) if size.size else 0,
        "span_hours_max": float(span.max()) if size.size else 0.0,
        "span_hours_p99": float(span.quantile(0.99)) if size.size else 0.0,
    }


def audit(
    df: pd.DataFrame,
    series_id: pd.Series | None = None,
    cfg: SeriesConfig | None = None,
    *,
    n_examples: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """异常清点：长度 >5 / game 号重复 / game 号不严格递增 / game 号非 1..n / 跨度超限。

    返回的每一类都带计数和最多 `n_examples` 条示例，方便直接贴进 /health 与报告。
    """
    cfg = _resolve(cfg, kwargs)
    if series_id is None:
        series_id = build_series_id(df, cfg)
    work = _sorted_view(df, cfg)
    sid = series_id.reindex(work.index)
    gno = work["_g"]

    out: dict[str, Any] = dict(series_stats(df, series_id, cfg))
    size = sid.groupby(sid, sort=False).size()
    span = series_span_hours(df, series_id, cfg)

    too_long = size[size > MAX_GAMES_PER_SERIES]
    too_wide = span[span > 12]
    out["too_long"] = {"count": int(too_long.size), "examples": list(too_long.index[:n_examples])}
    out["span_gt_12h"] = {"count": int(too_wide.size), "examples": list(too_wide.index[:n_examples])}
    out["span_gt_gap"] = int((span > cfg.gap.total_seconds() / 3600.0).sum())

    dup, non_incr, non_contig, has_nan, gt5 = [], [], [], [], []
    patterns: Counter[tuple[int, ...]] = Counter()
    for s, grp in gno.groupby(sid, sort=False):
        vals = grp.tolist()
        if any(pd.isna(v) for v in vals):
            has_nan.append(s)
            continue
        ns = [int(v) for v in vals]
        if len(set(ns)) != len(ns):
            dup.append((s, ns))
        if any(ns[i + 1] <= ns[i] for i in range(len(ns) - 1)):
            non_incr.append((s, ns))
        if sorted(ns) != list(range(1, len(ns) + 1)):
            non_contig.append((s, ns))
            patterns[tuple(ns)] += 1
        if max(ns) > MAX_GAMES_PER_SERIES:
            gt5.append((s, ns))

    n_ser = int(size.size)
    out["game_no_missing"] = {"count": len(has_nan), "examples": has_nan[:n_examples]}
    out["game_no_duplicated"] = {"count": len(dup), "examples": dup[:n_examples]}
    out["game_no_not_increasing"] = {"count": len(non_incr), "examples": non_incr[:n_examples]}
    out["game_no_not_1_to_n"] = {
        "count": len(non_contig),
        "rate": len(non_contig) / n_ser if n_ser else 0.0,
        "examples": non_contig[:n_examples],
        "top_patterns": patterns.most_common(8),
        "single_game_series": sum(1 for _, ns in non_contig if len(ns) == 1),
    }
    out["game_no_gt_5"] = {"count": len(gt5), "examples": gt5[:n_examples]}
    out["split_pairs_strict"] = count_split_series(df, series_id, cfg, respect_game_no=True)
    out["split_pairs_literal"] = count_split_series(df, series_id, cfg, respect_game_no=False)
    return out


def calendar_day_series_id(
    df: pd.DataFrame, cfg: SeriesConfig | None = None, **kwargs: Any
) -> pd.Series:
    """**【已证伪】v1.0 的按 UTC 日历日构造**——只保留用于回归测试和对照，禁止用于生产。

    留着它是因为"89 个系列被零点劈开"这个数字必须能被任何人一行命令复现，
    而不是活在文档里的一句断言。
    """
    cfg = _resolve(cfg, kwargs)
    _require(df, [cfg.league_col, cfg.time_col])
    return (
        df[cfg.time_col].dt.strftime("%Y-%m-%d")
        + "::"
        + df[cfg.league_col].astype(str)
        + "::"
        + pair_key(df, cfg)
    ).rename("series_id_calendar")
