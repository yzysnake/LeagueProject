"""patch 归一化：原始字符串 → (major, minor) → 全局单调 `patch_idx` → `patch_age_days`。

为什么单独成模块：patch 同时是**时间衰减权重的自变量**（PLAN.md §4.4）和 **meta
特征的分组键**（§4.2）。归一化错了不会抛异常，只会让衰减权重压在错误的样本上，
指标静默变差——PLAN.md §9.4 第 5 条把它列为"最贵的隐形 bug"之一。

================================================================================
本文件所有设计选择的实测依据（2026-09-18，data/raw 13 个年度 CSV，1,225,008 行 /
102,084 场，2014-01-14 → 2026-09-18）
================================================================================

1) **OE 的 patch 格式只有一种**：`^\\d{1,2}\\.\\d{2}$`。
   - 279 个 distinct 非空取值，**100% 命中**该正则；不存在三段式、不存在非零填充
     （没有 `13.1` 这种写法）、不存在前后空白。
   - major ∈ [3, 16]（3.15 → 16.17），minor ∈ [1, 25]。
   - 704 场（0.69%）`patch` 为空，集中在 LSPL(401) / DCup(86) / NA LCS(80) /
     EU LCS(64) / LDL(26) / NA CS(24) / LPL(12) / EU CS(11)，日期 2014-01~2024-04。
   - 同一 `gameid` 内 `patch` 恒唯一（0 例冲突）。

2) **手册 §3.10.1 的 `major >= 25 则 -10` 在 OE 数据上是彻底的 no-op**：
   实测 OE 的 major 最大值是 **16**，13 年里一次都不会触发。这条修正只对
   Leaguepedia / gptilt 系（2025 年起 major = Riot major + 10）有意义。
   → 因此本模块**默认不做 -10**，必须由调用方通过 `source=` 显式声明数据源。
   在 `source="oe"` 下遇到 major ≥ 25 直接报错，因为那意味着喂错了源。
   把 -10 设成默认值是危险的：Riot 真正发布 25.x 是 2035 年，那时它会静默把
   25.01 改写成 15.01，和 2025 年的数据撞在一起。

3) **真正会丢数据的不是 -10，而是"JSON number 吃掉零填充"**：OE 私有 API
   `/patches` 把 patch 作为 JSON number 返回，`16.10` 解析成 float 16.1。
   手册说"必须先 `str()`"——**这个建议无效**：等 Python 拿到 float 16.1，
   `"16.10"` 和 `"16.1"` 的区别已经在 JSON 解析阶段丢掉了，`str(16.1) == '16.1'`
   会被解析成 (16, 1)，也就是 patch **16.01**，比真实值早了 9 个版本。
   唯一正确的做法是在解析层就不要走 float：`json.loads(text, parse_float=str)`
   或 `parse_float=decimal.Decimal`。
   → 本模块对 float 输入**默认直接抛 TypeError**，把这个坑变成显式失败。

4) **patch 序与日期序不一致是常态，不是异常**（跨赛区版本推进不同步）：
   101,380 场有 patch 的比赛里，**33,744 场（33.28%）**的 patch 低于"该时刻之前
   全库已出现过的最高 patch"。落后幅度中位数 1 个版本、p95 = 2、p99 = 3，
   最大值取决于阶梯口径：观测阶梯 6，DDragon 发布阶梯 9（stale 的**场次**两种
   口径完全相同，阶梯只影响"落后几个版本"的刻度）。
   赛区分布：LDL 65.91% / PRMP 58.24% / LMF 53.11% / VCS 52.96% / LPL 49.53% /
   PCS 48.80%；年份上从 2022 的 48.93% 降到 2026 的 14.03%；
   而 **同一赛区内部只有 335 场（0.33%）回退** —— 33% 的乱序 100% 来自赛区之间
   版本推进不同步，不是数据错误。
   ⚠ 这个数字对"同一时刻开赛的比赛"的排序敏感，必须固定成 (date, gameid) 稳定排序
     才可复现（不定序会差几十场）。
   → 推论：**`patch_idx` 绝对不能按日期排序生成**，只能按 (major, minor) 排。
     本模块正是这么做的。

5) 反过来，**按 patch 序排列的"首现日"几乎是单调的**：279 个 patch 里只有
   2 个违反（7.10 首现 2017-05-23 早于 7.09 的 2017-06-01；7.22 首现
   2017-11-20 早于 7.21 的 2017-12-01）。所以 (major, minor) 排序本身可靠。

6) **`patch_age_days` 用"全库首现日"不构成未来信息泄漏**：
   对任意在 patch p 上的比赛 g，first_seen(p) = min(所有 p 上比赛的日期) ≤ date(g)，
   取到最小值的那场比赛本身就在 g 之前。也就是说全表 groupby-min 的结果与
   "只看 ≤ date(g) 的行做 expanding-min" **恒等**。这条写在这里是因为它看起来
   很像泄漏，评审时容易被误杀。
   （唯一例外是 walk-forward 里训练窗被截断的情形 → 用 `first_seen_override=`
   把上一折冻结的首现日传进来。）
   实测分布：中位 12.09 天，p90 23.55，p99 39.34，max 147.74（patch 6.12）；
   0 场为负。分桶 0-7 天 22.43% / 8-14 天 37.88% / 15-30 天 35.42% / 30+ 天 4.27%。

7) **`patch_idx` 用"观测阶梯"还是"DDragon 发布阶梯"**：
   DDragon `versions.json`（499 条）去掉 `lolpatch_` 前缀并压到 (major, minor) 后
   得到 311 个版本（3.6 → 16.18），**OE 观测到的 279 个 patch 全部被它覆盖**
   （0 个例外）。区间内 DDragon 有而 OE 无职业赛的 patch 共 **22 个**，其中
   2018 年之后只剩 3 个（12.22 / 13.23 / 14.24）。
   → 默认用观测阶梯（自包含、不需要联网、可复现）；需要"距今多少个版本"精确到
     发布序时，把 `versions.json` 解析成 `PatchLadder` 传进来即可。
   ⚠ 手册说"`patch_idx` = 完整版本列表**反转**的序号"——**不要这么做**。
     versions.json 是新→旧排列，按它反转出来的序号每发一个新版本就整体 +1，
     昨天算出的 `patch_idx` 今天就变了，特征缓存和已训模型全部对不上。
     本模块一律**升序 0-based**：追加更高的版本不会改动任何已有序号。
   ⚠ 手册还说"用 `versions.json` 给每个 patch 打上真实发布日期"——**做不到**，
     实测 versions.json 是纯字符串数组，不带任何日期字段。发布日期只能用
     "数据中首现日"近似（这也是本模块 `patch_first_seen` 的由来）。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

__all__ = [
    "MEASURED",
    "OE_PATCH_RE",
    "PatchKey",
    "PatchLadder",
    "attach_patch_features",
    "build_patch_table",
    "load_ddragon_ladder",
    "norm_patch",
    "parse_patch_series",
    "patch_order_violations",
    "patch_str",
]

#: 归一化后的 patch 主键：(major, minor)，例如 OE 的 "13.01" → (13, 1)。
PatchKey = tuple[int, int]

#: OE 实测格式：一到两位 major + 点 + **恰好两位零填充** minor。279/279 命中。
OE_PATCH_RE = re.compile(r"^\d{1,2}\.\d{2}$")

#: 宽松解析正则：兼容 OE("13.01") / Leaguepedia("13.1") / DDragon("16.18.1")
#: / livestats("16.17.810.4348") / DDragon 老式("lolpatch_7.20")。
_PARSE_RE = re.compile(r"^(?:lolpatch_)?(\d{1,3})\.(\d{1,3})(?:\.\d+)*$")

#: 数据源类型。决定是否做 Leaguepedia 的 -10 修正，以及越界时是报错还是放行。
SourceKind = Literal["oe", "leaguepedia", "ddragon", "livestats", "auto"]

#: Leaguepedia 系自 2025 年起 major = Riot major + 10。阈值取 25 与手册一致。
_LP_OFFSET_THRESHOLD = 25
_LP_OFFSET = 10

#: 本文件顶部那些实测数字的机读副本。**只作文档与回归对照，不要拿来做运行时断言**
#: （数据每天在长，硬断言会在明天变红）。
MEASURED: dict[str, object] = {
    "snapshot_date": "2026-09-18",
    "n_rows": 1_225_008,
    "n_games": 102_084,
    "n_distinct_patch": 279,
    "n_games_patch_null": 704,
    "major_range": (3, 16),
    "minor_range": (1, 25),
    "oe_regex_hit_rate": 1.0,
    "n_games_stale_patch": 33_744,
    "pct_games_stale_patch": 33.28,
    "n_games_stale_within_league": 335,
    "patch_age_days_median": 12.09,
    "patch_age_days_p99": 39.34,
    "patch_age_days_max": 147.74,
    "ddragon_ladder_size": 311,
    "ddragon_gaps_in_observed_range": 22,
    "first_seen_order_violations": ("7.10", "7.22"),
}


# --------------------------------------------------------------------------- #
# 单值归一化
# --------------------------------------------------------------------------- #
def norm_patch(
    value: object,
    *,
    source: SourceKind = "oe",
    allow_lossy_float: bool = False,
) -> PatchKey | None:
    """把任意来源的 patch 标识归一成 ``(major, minor)``；无法识别时返回 ``None``。

    Args:
        value: 原始值。``None`` / ``NaN`` / 空串 / 纯空白 一律视为缺失 → ``None``。
        source: 数据源。``"oe"`` 不做 -10 且 major ≥ 25 时报错（说明喂错了源）；
            ``"leaguepedia"`` 与 ``"auto"`` 对 major ≥ 25 做 -10；
            ``"ddragon"`` / ``"livestats"`` 不做 -10（它们跟 Riot 编号一致）。
        allow_lossy_float: 是否容忍 ``float`` 输入。默认 ``False`` 并抛
            ``TypeError``——见模块 docstring 第 3 条，float 到手时零填充已经丢了。

    为什么默认不做 -10：实测 OE 13 年 major 最大 16，这条修正一次都不触发；
    而把它设成默认会在 Riot 真正发到 25.x 时静默改写数据。
    """
    if value is None:
        return None
    if isinstance(value, float):
        # float("nan") 也走这里；先放过缺失，再拦真正的数字。
        if math.isnan(value):
            return None
        if not allow_lossy_float:
            raise TypeError(
                f"patch 收到 float {value!r}：JSON number 会把 '16.10' 变成 16.1，"
                "零填充已不可恢复。请在解析层用 json.loads(text, parse_float=str) "
                "或 decimal.Decimal；确认无损时再传 allow_lossy_float=True。"
            )
        s = repr(value)
    elif isinstance(value, int):
        # 纯整数没有 minor，无法定位版本。
        raise TypeError(f"patch 收到 int {value!r}，缺少 minor 部分，无法归一化。")
    else:
        s = str(value)

    s = s.strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "-"}:
        return None

    m = _PARSE_RE.match(s)
    if m is None:
        return None

    major, minor = int(m.group(1)), int(m.group(2))

    if major >= _LP_OFFSET_THRESHOLD:
        if source in ("leaguepedia", "auto"):
            major -= _LP_OFFSET
        elif source == "oe":
            raise ValueError(
                f"source='oe' 却收到 major={major}（{s!r}）。OE 实测 major ∈ [3,16]，"
                "这看起来是 Leaguepedia 编号。请显式传 source='leaguepedia'。"
            )
    return (major, minor)


def patch_str(key: PatchKey | None) -> str | None:
    """把 ``(major, minor)`` 还原成 **OE 规范写法**（minor 零填充到两位）。

    ``(13, 1) -> '13.01'``。用 OE 写法做规范形式，是因为整个项目的事实基准是 OE，
    这样 `patch_raw` 与 `patch_str(norm_patch(patch_raw))` 在 OE 数据上恒等
    （实测 279/279 往返一致）。
    """
    if key is None:
        return None
    return f"{key[0]}.{key[1]:02d}"


# --------------------------------------------------------------------------- #
# 版本阶梯
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatchLadder:
    """升序排列的版本阶梯，`patch_idx` 就是元素在其中的 0-based 下标。

    升序而非"反转序号"是刻意的：追加一个更高的版本不会改动任何已有下标，
    昨天算出的特征今天仍然有效。手册 §3.10.1 的"反转序号"每发一个版本就整体
    平移，会让特征缓存与已训模型静默错位。
    """

    keys: tuple[PatchKey, ...]

    def __post_init__(self) -> None:
        if list(self.keys) != sorted(set(self.keys)):
            raise ValueError("PatchLadder.keys 必须升序且无重复")

    @classmethod
    def from_keys(cls, keys: Iterable[PatchKey]) -> PatchLadder:
        return cls(tuple(sorted(set(keys))))

    @classmethod
    def from_ddragon_versions(
        cls, versions: Sequence[str], *, min_major: int = 3
    ) -> PatchLadder:
        """从 DDragon ``versions.json`` 构造发布阶梯（含无职业赛的版本）。

        实测要点：499 条里有 99 条是老式 ``lolpatch_X.Y``，它们与数字版重复，
        压到 (major, minor) 后不新增任何条目；还有 4 条 ``0.15x.y`` 的史前版本，
        用 ``min_major`` 过滤掉。结果 311 条，覆盖 OE 全部 279 个观测 patch。
        """
        out: set[PatchKey] = set()
        for v in versions:
            k = norm_patch(v, source="ddragon")
            if k is not None and k[0] >= min_major:
                out.add(k)
        return cls.from_keys(out)

    def union_observed(self, keys: Iterable[PatchKey]) -> PatchLadder:
        """并入数据中观测到但阶梯里没有的 patch，保证 `index_of` 不返回 None。"""
        return PatchLadder.from_keys(set(self.keys) | set(keys))

    def index_of(self, key: PatchKey | None) -> int | None:
        if key is None:
            return None
        try:
            return self.keys.index(key)
        except ValueError:
            return None

    def as_mapping(self) -> dict[PatchKey, int]:
        return {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)


def load_ddragon_ladder(path: str = "data/ddragon/versions.json") -> PatchLadder | None:
    """尽力加载本地缓存的 DDragon 版本列表；没有就返回 ``None``（调用方降级到观测阶梯）。

    刻意不联网：patches.py 要能在离线 / CI 里跑。抓取是 ingest/ddragon.py 的事。
    """
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):  # 兼容 {"versions": [...]} 这种包装
        payload = payload.get("versions", [])
    if not isinstance(payload, list) or not payload:
        return None
    return PatchLadder.from_ddragon_versions([str(x) for x in payload])


# --------------------------------------------------------------------------- #
# 表级 API
# --------------------------------------------------------------------------- #
def parse_patch_series(
    s: pd.Series, *, source: SourceKind = "oe", allow_lossy_float: bool = False
) -> pd.DataFrame:
    """向量化（按 distinct 值做字典映射）解析一整列 patch。

    只对 distinct 值调用 `norm_patch`：13 年全量也只有 280 个 distinct 值，
    比逐行 apply 快三个数量级。

    Returns:
        与 ``s`` 同索引的 DataFrame，列 ``patch_raw / patch_major / patch_minor``。
        ``patch_major`` / ``patch_minor`` 为可空整型 ``Int16``。
    """
    mapping: dict[object, PatchKey | None] = {
        v: norm_patch(v, source=source, allow_lossy_float=allow_lossy_float)
        for v in pd.Series(s.dropna().unique())
    }
    keys = [mapping.get(v) for v in s.to_numpy()]
    return pd.DataFrame(
        {
            "patch_raw": s.astype("string"),
            "patch_major": pd.array(
                [k[0] if k is not None else None for k in keys], dtype="Int16"
            ),
            "patch_minor": pd.array(
                [k[1] if k is not None else None for k in keys], dtype="Int16"
            ),
        },
        index=s.index,
    )


def build_patch_table(
    df: pd.DataFrame,
    *,
    patch_col: str = "patch",
    date_col: str = "date",
    game_col: str | None = "gameid",
    source: SourceKind = "oe",
    ladder: PatchLadder | None = None,
) -> pd.DataFrame:
    """由一张（长表或宽表都行）比赛表产出 patch 维表。

    Args:
        df: 至少含 ``patch_col`` 与 ``date_col``。
        game_col: 给出时按 distinct gameid 计数，长表（12 行/场）不会重复计。
        ladder: 版本阶梯。``None`` 则用本表观测到的 patch 自建（自包含、可复现）。

    Returns:
        一行一个 patch，按 ``patch_idx`` 升序，列：
        ``patch_raw_canon / patch_major / patch_minor / patch_idx /
        patch_first_seen / patch_last_seen / n_games / n_rows``。

    `patch_idx` 按 (major, minor) 升序而**不是**按首现日排——实测 33.40% 的比赛
    打在"比全库当时最高版本更旧"的 patch 上（跨赛区推进不同步），按日期排会得到
    一个非单调的伪序号。
    """
    parsed = parse_patch_series(df[patch_col], source=source)
    work = pd.DataFrame(
        {
            "patch_major": parsed["patch_major"],
            "patch_minor": parsed["patch_minor"],
            "kickoff": pd.to_datetime(df[date_col], errors="coerce"),
        },
        index=df.index,
    )
    if game_col is not None:
        work["gameid"] = df[game_col].to_numpy()
    work = work.dropna(subset=["patch_major", "patch_minor"])

    grp = work.groupby(["patch_major", "patch_minor"], observed=True)
    agg = grp.agg(
        patch_first_seen=("kickoff", "min"),
        patch_last_seen=("kickoff", "max"),
        n_rows=("kickoff", "size"),
    )
    agg["n_games"] = (
        grp["gameid"].nunique() if game_col is not None else agg["n_rows"]
    )
    agg = agg.reset_index()

    observed = [
        (int(a), int(b)) for a, b in zip(agg.patch_major, agg.patch_minor, strict=True)
    ]
    lad = PatchLadder.from_keys(observed) if ladder is None else ladder.union_observed(observed)
    idx_map = lad.as_mapping()

    agg["patch_idx"] = [
        idx_map[(int(a), int(b))]
        for a, b in zip(agg.patch_major, agg.patch_minor, strict=True)
    ]
    agg["patch_raw_canon"] = [
        patch_str((int(a), int(b)))
        for a, b in zip(agg.patch_major, agg.patch_minor, strict=True)
    ]
    cols = [
        "patch_raw_canon",
        "patch_major",
        "patch_minor",
        "patch_idx",
        "patch_first_seen",
        "patch_last_seen",
        "n_games",
        "n_rows",
    ]
    return (
        agg[cols]
        .sort_values("patch_idx")
        .reset_index(drop=True)
        .astype({"patch_major": "int16", "patch_minor": "int16", "patch_idx": "int32"})
    )


def attach_patch_features(
    games: pd.DataFrame,
    *,
    patch_col: str = "patch",
    date_col: str = "date",
    source: SourceKind = "oe",
    ladder: PatchLadder | None = None,
    patch_table: pd.DataFrame | None = None,
    first_seen_override: Mapping[PatchKey, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """给一场一行的表补齐 silver 需要的 patch 列（PLAN.md §3.10 数据字典 A 段）。

    新增列：``patch_raw``（规范化写法）、``patch_major``、``patch_minor``、
    ``patch_idx``、``patch_first_seen``、``patch_age_days``。
    patch 缺失的行（实测全库 704 场）这些列全部为空，**不删行**——是否丢弃由
    silver 的整场完整性判定统一决定，这里不越权。

    Args:
        patch_table: 复用已算好的维表（walk-forward 里应当传上一折冻结的那张，
            保证折与折之间 `patch_idx` 一致）。
        first_seen_override: 用外部权威首现日覆盖本表算出来的（同上）。

    `patch_age_days` = kickoff − 该 patch 全库首现日。**不是未来信息**：首现日
    按定义 ≤ 本行 kickoff，全表 groupby-min 与 expanding-min 恒等（见模块 docstring
    第 6 条）。
    """
    out = games.copy()
    if patch_table is None:
        patch_table = build_patch_table(
            games, patch_col=patch_col, date_col=date_col, game_col=None, source=source,
            ladder=ladder,
        )

    parsed = parse_patch_series(games[patch_col], source=source)
    out["patch_major"] = parsed["patch_major"]
    out["patch_minor"] = parsed["patch_minor"]
    out["patch_raw"] = [
        patch_str((int(a), int(b))) if pd.notna(a) and pd.notna(b) else None
        for a, b in zip(parsed.patch_major, parsed.patch_minor, strict=True)
    ]

    idx_map = {
        (int(r.patch_major), int(r.patch_minor)): int(r.patch_idx)
        for r in patch_table.itertuples()
    }
    fs_map: dict[PatchKey, pd.Timestamp] = {
        (int(r.patch_major), int(r.patch_minor)): r.patch_first_seen
        for r in patch_table.itertuples()
    }
    if first_seen_override:
        fs_map.update(dict(first_seen_override))

    keys = [
        (int(a), int(b)) if pd.notna(a) and pd.notna(b) else None
        for a, b in zip(parsed.patch_major, parsed.patch_minor, strict=True)
    ]
    out["patch_idx"] = pd.array(
        [idx_map.get(k) if k is not None else None for k in keys], dtype="Int32"
    )
    out["patch_first_seen"] = pd.to_datetime(
        pd.Series([fs_map.get(k) if k is not None else None for k in keys], index=out.index)
    )
    kickoff = pd.to_datetime(out[date_col], errors="coerce")
    out["patch_age_days"] = (
        (kickoff - out["patch_first_seen"]).dt.total_seconds() / 86400.0
    )
    return out


def patch_order_violations(
    games: pd.DataFrame,
    *,
    patch_idx_col: str = "patch_idx",
    date_col: str = "date",
    group_col: str | None = None,
) -> pd.DataFrame:
    """量化"老 patch 打在新日期"的乱序：按时间扫一遍，标出低于历史最高 patch 的场。

    Args:
        group_col: 给出时在组内（例如 ``league``）各自计算 running max，
            用来区分"赛区之间不同步"与"赛区自己回退"。``None`` 则全库口径。

    Returns:
        与 ``games`` 同索引，列 ``running_max_idx``（此前出现过的最高 patch_idx）、
        ``is_stale``（bool）、``patch_lag``（落后多少个阶梯位次，非 stale 为 0）。

    全库实测（按 (date, gameid) 稳定排序）：33,744 / 101,380 场（33.28%）is_stale，
    lag 中位 1、p95 2、p99 3；LDL 65.91% / PRMP 58.24% / VCS 52.96% / LPL 49.53%；
    2022 年 48.93% → 2026 年 14.03%。改成 ``group_col='league'`` 后只剩 335 场
    （0.33%），说明乱序全部来自赛区之间不同步。

    ⚠ 调用前请先把 ``games`` 按 (date, 主键) 稳定排序：同一时刻开赛的比赛很多，
      不定序会让结果在不同机器上差几十场。
    """
    kickoff = pd.to_datetime(games[date_col], errors="coerce")
    idx = games[patch_idx_col]
    order = kickoff.sort_values(kind="mergesort").index
    s = pd.DataFrame({"idx": idx.reindex(order), "grp": (
        games[group_col].reindex(order) if group_col else 0)})
    running = s.groupby("grp", observed=True)["idx"].cummax().groupby(
        s["grp"], observed=True).shift(1)
    res = pd.DataFrame(
        {
            "running_max_idx": running,
            "is_stale": (s["idx"] < running).fillna(False),
        },
        index=order,
    )
    res["patch_lag"] = (res["running_max_idx"] - s["idx"]).where(res["is_stale"], 0)
    return res.reindex(games.index)
