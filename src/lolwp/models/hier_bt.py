"""分层 Bradley-Terry：选手潜变量 + 队伍 chemistry + 赛区校准（PLAN.md §4.6.2 / §4.6.2b / §9.4 / §9.5）。

为什么要有这个模块
------------------
阶段 3/4 的实测把手册 §9.4「选手 ID 是唯一有强信号的设想（ΔLL≈0.0075）」打了折扣：
`features/priors.py` 里那版 player Elo 只拿到 ΔLL=+0.00160，在任何 ICC 假设下都低于 MDE。
原因不是"选手不重要"，而是那版实现**没有做功劳分配**——5 名选手取算术平均，
并且 5 个人共用队伍层的期望胜率 `E(team)` 做同一次更新。于是同队 5 人永远同涨同跌，
选手 rating 退化成 team rating 的一个噪声副本，转会才能把它们分开，而转会只占极小比例。

分层 BT 修的正是这一点：它把「谁值多少」写成一个**联合方程组**，
同一个人在不同队、不同队友组合下的胜负共同约束他的 r_p，最小二乘/最大后验一次解出来。

模型形式（与 §4.6.2b 一致，但把"队伍层"显式写进层次结构）
--------------------------------------------------------
    logit P(blue win) = b0 + [S_blue − S_red] + [c_Tblue − c_Tred]
    S_side            = Σ_{p∈side 已知名单} r_p + (5 − |已知名单|)·m_{T(side)}

    r_p  ~ N(m_{T(p)}, σ_p²)     选手效应收缩到其所属队伍
    m_T  ~ N(ℓ_{g(T)}, σ_T²)     队伍效应收缩到其所属赛区
    ℓ_g  ~ N(0, σ_L²)            赛区效应收缩到全局
    c_T  ~ N(0, σ_c²)            chemistry 残差：team ≠ Σplayer 的那一部分

注意 `m_T` 对**名单完整的 96.4% 的比赛**只出现在先验里、不出现在似然里，
`c_T` 则只出现在似然里。这样拆开是为了避开"5 个 r_p 同时 +δ" 与 "c_T +5δ" 的共线性：
队伍层的**水平**由 m_T 承担（纯收缩目标），队伍层的**残差**由 c_T 承担（带紧先验）。
剩下 3.6% 名单不全的比赛（silver 实测：某一侧只有 0-4 个非空 playerid）
把缺位按 m_T 计价，理由见 `_build_design`。

`ℓ_g`（= 手册的 `L_region`）只可能被**跨赛区对战**识别出来：同区内战里
ℓ 在两边同时出现、在 logit 里直接抵消。本模块实测的桥接量见 `bridge_census()`。

为什么它是"选手层"而不是"队伍层"
--------------------------------
LoL 转会频繁，teamid 不变而阵容剧变。r_p 跟人走，转会当天就继承；
队伍层 rating 要等新阵容打满一个赛季才反应过来。
team strength 在本模块里是 `Σ(5 个 r_p) + c_T`，正是手册 §9.4 写的那个分解。

时间正确性（§6.1.2 红线 3：本模块属于哪一类）
--------------------------------------------
**结论：属于"as-of 快照序列"，可以全局跑一遍，不需要逐 fold 重估；理由如下。**

红线 3 要管的是"依赖**训练段**全局分布的统计量"——它的危险在于：同一个统计量
对 fold A 是合法的、对 fold B 就偷看了未来。本模块的产物不是这种东西：
`fit_snapshot(df, as_of)` **只吃 kickoff < as_of 的比赛**，
`rolling_ratings()` 沿一条固定日历网格（默认 3 天）依次做快照，
每一场比赛取**严格早于它自己开赛时刻**的最近一张快照。
于是「任何一行的特征都只用了该行开赛前的信息」这条性质是**逐行成立**的，
与 fold 划分无关——这正是 Elo/Glicko 拿到豁免的那条理由，
只不过本模块把"每场一次的在线更新"换成了"每 3 天一次的批量重解"。

`tests/test_hier_bt.py` 里三条测试把这条性质钉死，**全部要求 bit-identical**（§4.7.1 T2 档 A，
比手册给 BT/SVI 的档 B 严一档；MAP 是确定性的，本来就不该退到档 B）：
  * `test_future_deletion_invariance_snapshot` / `_rolling`：删掉 T 之后的全部比赛重跑，T 之前逐位相同；
  * `test_future_label_perturbation_invariance`：只把 T 之后的**胜负翻转**，T 之前逐位相同
    （删除可能被"窗口下界跟着动"掩盖，翻转不会）；
  * `test_refit_is_bit_identical_across_runs`：同输入两次运行逐位相同。
真实数据上另跑过一次同样的未来删除检查（cut=2021-07-01，7,685 行非零特征，
6 列全部 maxabs_diff=0.0；另有一张 31,454 场的快照 r_player 最大差 0.0）。

**为什么不照手册字面的"每个 origin 在训练段内拟合一次"**：那样只能给 valid 段
造出 as-of 特征，train 段的特征要么没有、要么得用"拟合时已经看过这些比赛结果"的
同一张 rating 表。后者是典型的 in-sample plug-in 偏差——主力 logistic 会学到一个
被高估的 β_player，到了 valid 段（真·as-of rating）就系统性过度自信。
滚动网格严格支配它：train / valid 两段的特征口径**完全一致**，且成本更低
（网格点数 ≈ 时间跨度/3 天；单次拟合在 3 万场规模上约 1 秒，全库 1,096 个网格点）。
而且网格比 origin 密得多（3 天 vs 4 周），valid 段拿到的 rating 反而**更新鲜**——
这正是实测里最值钱的一项：调参段上把网格从 28 天收到 3 天，log loss 从 0.632941 掉到 0.628324。

实现选择
--------
MAP（最大后验），scipy L-BFGS-B + 解析梯度，稀疏设计矩阵。
不用 SVI 的理由很简单：这里只要点估计，而 MAP 是**确定性**的——
确定性换来 T2 档 A 的 bit-identical，比档 B 的"跨 seed <1e-3"强一档。
方差成分的后验区间（验收④）该由 NumPyro 那条线出，不在本模块。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

UNKNOWN_REGION = "UNKNOWN"
ROSTER_SIZE = 5


def _pad_roster(ids: Sequence[str | None]) -> list[str | None]:
    """把一侧名单补齐到 5 人：缺位填 `None`（取值时会落到队伍层 m_T）。

    只补不截——真出现 6 个 id 说明上游折叠出了错，那是 silver 的问题，
    在这里悄悄截掉反而会把它藏起来。
    """
    out = list(ids)
    if len(out) < ROSTER_SIZE:
        out.extend([None] * (ROSTER_SIZE - len(out)))
    return out


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HierBTConfig:
    """超参。σ 的量纲全部是 **log-odds**，与手册 §9.4 的"选手层真实效应 SD"同尺度。

    **这些默认值是在 2016-01-01 ~ 2019-01-01 这一段上选出来的**（43 组配置，
    walk-forward 口径与验收段逐字一致，OOS 9,145 场 / L3=0.639869）。
    验收口径是 **2019 起**，两段不重叠 —— 没有在验收数据上调过任何一个数。

    手册 §9.4 给的先验 `sigma_player=0.1294`（PLAYER 层效应 SD，k≈14）**被实测否掉了**：
    在调参段上 0.1294 给 BT_repl=0.634362，放宽到 0.25 给 0.633418；
    `sigma_team` 从 §9.4 的 0.1330 放宽到 0.70 更是从 0.634362 直接跳到 0.631186。
    原因是手册那两个 SD 描述的是"**真实效应**的离散度"，而这里的 σ 是**先验宽度**；
    MAP 收缩本身就会把估计压向中心，再按真实 SD 设先验等于收缩了两次。
    实测最优的先验宽度约为真实效应 SD 的 2 倍（player）到 5 倍（team），这符合预期。

    `sigma_chem` / `sigma_league` 在调参段上几乎无差别（0.628324 / 0.628399 / 0.628317
    / 0.628711 对应 σ_L=0.3,0.6,0.6(σ_c=.15),1.0，跨度 <4e-4，小于任何 MDE），
    因此保留手册给的紧先验，不为 1e-4 的噪声改默认值。
    """

    window_years: float = 3.0       # §4.6.2b 硬性要求：L_region 用 3 年滚动窗口
    half_life_days: float = 270.0   # 调参段实测：180→270 给 ΔLL −0.00068，365 与 540 反弹
    sigma_player: float = 0.2500    # 调参段实测最优（§9.4 的 0.1294 是真实效应 SD，不是先验宽度）
    sigma_team: float = 0.7000      # 同上；队伍层先验必须放得比 §9.4 的 0.1330 松得多
    sigma_chem: float = 0.0500      # chemistry 残差，先验刻意收紧
    sigma_league: float = 0.3000    # 赛区偏移，桥接样本少 → 不能放太松
    # 3 天一次重解：调参段上 28→14→7→3 天单调变好（0.632941→0.630505→0.628924→0.628324）。
    # 这不是"更多信息"，而是**更新鲜的 as-of rating**：对照组 player Elo 是逐场在线更新的
    # （rating 年龄 = 0），BT 每 N 天才重解一次，rating 平均老 N/2 天。粗网格是在自残。
    refit_every_days: int = 3
    max_iter: int = 3000
    gtol: float = 1e-7
    min_games: int = 200            # 窗口内不足这么多场就不拟合（返回全零快照）


# ---------------------------------------------------------------------------
# 赛区归属
# ---------------------------------------------------------------------------
def infer_team_region(df: pd.DataFrame) -> dict[str, str]:
    """队伍 → 主场赛区。**只看 tier<2 的国内赛**，取出场最多的 `league_canon`。

    为什么不能直接用比赛自己的 `league_canon`：国际赛（tier=2）的 league_canon 是
    `WLDs`/`MSI`，它不是任何一支队的"赛区"。而 `L_region` 要回答的恰恰是
    "这支队来自哪个赛区"。用比赛的 league 会让所有国际赛队伍落进同一个桶，
    跨赛区校准直接失效（而且不会报错）。
    """
    dom = df[df.league_tier < 2]
    if not len(dom):
        return {}
    stack = pd.concat(
        [
            pd.DataFrame({"tid": dom[f"{s}_team_id"].to_numpy(),
                          "lg": dom["league_canon"].to_numpy()})
            for s in ("blue", "red")
        ],
        ignore_index=True,
    )
    cnt = stack.groupby(["tid", "lg"]).size().rename("n").reset_index()
    # 并列时取字典序最小的 league，保证确定性（否则 groupby 顺序变了结果就变）
    cnt = cnt.sort_values(["n", "lg"], ascending=[False, True], kind="mergesort")
    top = cnt.drop_duplicates("tid")
    return {str(t): str(g) for t, g in zip(top.tid, top.lg, strict=True)}


def bridge_census(df: pd.DataFrame, team_region: dict[str, str] | None = None) -> dict[str, Any]:
    """实测跨赛区对战的场次与赛区对数量 —— `L_region` 到底识别不识别得了。

    手册 §4.6.1 给的参照系是「2022 年国际赛 235 场中跨赛区 193 场，桥接 43 个赛区对」。
    本函数在全库上重算，返回的 `pairs` 是 (赛区A, 赛区B) → 场次。

    **全库实测（2026-09-19，load_dev 的 98,592 场）——结论是 ℓ_g 识别得了，
    但桥接的来源和手册设想的不是一回事：**

        跨赛区对战       18,418 场，482 个赛区对，88 个赛区
        其中国际赛        2,190 场（tier=2 的 2,564 场里）——手册说的那种桥
        其中国内赛       16,228 场，374 个赛区对 ←**真正的主力**

    手册只数了国际赛，于是得出"每年约 200 场、几乎识别不了"的悲观结论。
    实际上一二队/青训联赛的互打提供了 8 倍于国际赛的桥接量：
    CK–LCK 904 场、LMS–PCS 559、LFL–LFL2 482、NACL–UPL 440、LDL–LPL 400。
    这类桥是**纵向**的（同一国家的一队 vs 二队），国际赛那类是**横向**的（LCK vs LPL，311 场）。
    两类合起来把 88 个赛区连成一张连通图：桥接 ≥100 场的赛区有 66 个，
    ≥30 场的有 80 个，只有 6 个赛区桥接 <30 场、2 个赛区完全没有跨区对战。

    本函数对 2022 年的复算与手册逐字对上（国际赛 235 场 / 跨区 193 场 / 45 对，
    手册写 43 对，差在赛区归属的并列打破规则），可以认为口径一致。

    **给调用方的硬约束**：桥接 <30 场的赛区，其 ℓ_g 基本就是先验（≈0），
    不要拿它当"该赛区实力"解读；真要用，先查 `per_region`。
    """
    tr = team_region if team_region is not None else infer_team_region(df)
    rb = np.array([tr.get(str(t), UNKNOWN_REGION) for t in df.blue_team_id])
    rr = np.array([tr.get(str(t), UNKNOWN_REGION) for t in df.red_team_id])
    cross = rb != rr
    pairs: dict[tuple[str, str], int] = {}
    for a, b in zip(rb[cross], rr[cross], strict=True):
        key = (a, b) if a <= b else (b, a)
        pairs[key] = pairs.get(key, 0) + 1
    per_region: dict[str, int] = {}
    for (a, b), n in pairs.items():
        per_region[a] = per_region.get(a, 0) + n
        per_region[b] = per_region.get(b, 0) + n
    return {
        "n_games": len(df),
        "n_cross": int(cross.sum()),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "per_region": per_region,
        "n_regions": len(set(rb) | set(rr)),
    }


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------
@dataclass
class HierBTSnapshot:
    """某个 `as_of` 时刻的 as-of rating 表。**只由 kickoff < as_of 的比赛产生。**"""

    as_of: pd.Timestamp
    r_player: dict[str, float]
    m_team: dict[str, float]
    c_team: dict[str, float]
    l_region: dict[str, float]
    team_region: dict[str, str]
    player_team: dict[str, str]
    intercept: float
    cfg: HierBTConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # -- 取值的兜底链：见 §5.5 冷启动 --------------------------------------
    def region_of(self, team_id: str) -> str:
        return self.team_region.get(str(team_id), UNKNOWN_REGION)

    def team_level(self, team_id: str) -> float:
        """队伍层水平 m_T。没见过的队 → 该赛区的 ℓ_g → 0。"""
        t = str(team_id)
        if t in self.m_team:
            return self.m_team[t]
        return self.l_region.get(self.region_of(t), 0.0)

    def player_rating(self, player_id: str | None, team_id: str) -> float:
        """选手 rating。没见过的人 → 其队伍的 m_T（**不是 0**）。

        这条兜底是冷启动的关键：新人不该被当成"平均水平的全球选手"，
        而该被当成"这支队的一名普通选手"。LCK 的新人和 LJL 的新人先验不同。
        """
        if player_id is not None and str(player_id) in self.r_player:
            return self.r_player[str(player_id)]
        return self.team_level(team_id)

    def chemistry(self, team_id: str) -> float:
        return self.c_team.get(str(team_id), 0.0)

    def league_strength(self, team_id: str) -> float:
        return self.l_region.get(self.region_of(team_id), 0.0)

    def n_known_players(self, player_ids: Iterable[str | None]) -> int:
        return sum(1 for p in player_ids if p is not None and str(p) in self.r_player)

    def features(
        self,
        *,
        blue_team_id: str,
        red_team_id: str,
        blue_player_ids: Sequence[str | None],
        red_player_ids: Sequence[str | None],
    ) -> dict[str, float]:
        """单场 BT 特征。**训练与推理共用这一个函数**（§7.4 train-serve skew）。

        名单**先补齐到 5 人**再求和（补位按 `player_rating(None, team)` = 队伍层 m_T）。
        为什么必须补：silver 里 3.6% 的比赛某一侧只有 3-4 个非空 playerid
        （评测口径 2019+ tier≥1 里是 97 场，且**全部是 4 打 5 的不对称缺失**）。
        直接对 k 个人求和，等于把缺位那名选手的实力当成 0；而 rating 的零点是
        "全球平均"，不是"该赛区平均"，于是强赛区的队伍会因为**缺数据**而被系统性削弱。
        这类 bug 在总指标上只值 1e-5，但它对着特定的 97 场比赛稳定地给错方向。
        """
        blue_player_ids = _pad_roster(blue_player_ids)
        red_player_ids = _pad_roster(red_player_ids)
        rb = [self.player_rating(p, blue_team_id) for p in blue_player_ids]
        rr = [self.player_rating(p, red_team_id) for p in red_player_ids]
        sb, sr = float(np.sum(rb)), float(np.sum(rr))
        cb, cr = self.chemistry(blue_team_id), self.chemistry(red_team_id)
        lb, lr = self.league_strength(blue_team_id), self.league_strength(red_team_id)
        known = self.n_known_players(blue_player_ids) + self.n_known_players(red_player_ids)
        return {
            "bt_player_diff": sb - sr,
            "bt_league_diff": lb - lr,
            "bt_chem_diff": cb - cr,
            "bt_logit": self.intercept + (sb - sr) + (cb - cr),
            "bt_known_players": float(known),
            "bt_team_level_diff": self.team_level(blue_team_id) - self.team_level(red_team_id),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["as_of"] = str(self.as_of)
        d["cfg"] = asdict(self.cfg)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> HierBTSnapshot:
        """从 artifact 里重建快照。推理侧用它 + features() 复用训练的同一套公式。"""
        d = dict(d)
        d["as_of"] = pd.Timestamp(d["as_of"])
        d["cfg"] = HierBTConfig(**d["cfg"])
        return HierBTSnapshot(**d)


FEATURE_COLUMNS = (
    "bt_player_diff",
    "bt_league_diff",
    "bt_chem_diff",
    "bt_logit",
    "bt_known_players",
    "bt_team_level_diff",
)


# ---------------------------------------------------------------------------
# 设计矩阵
# ---------------------------------------------------------------------------
def _iter_ids(arr: Any) -> Iterable[str]:
    if arr is None:
        return ()
    out = []
    for p in arr:
        if p is None:
            continue
        if isinstance(p, float) and math.isnan(p):
            continue
        out.append(str(p))
    return out


@dataclass
class _Design:
    A: sp.csr_matrix
    y: np.ndarray
    w: np.ndarray
    pid: dict[str, int]
    tid: dict[str, int]
    lid: dict[str, int]
    p_team: np.ndarray       # 每个选手的归属队 index
    t_league: np.ndarray     # 每个队伍的赛区 index
    team_region: dict[str, str]
    player_team: dict[str, str]


def _build_design(win: pd.DataFrame, as_of: pd.Timestamp, cfg: HierBTConfig,
                  target: str = "y_blue_win") -> _Design:
    """列布局：[ r_player | c_team | b0 | m_team ]。l_region 只进先验，不进 A。

    `target` 是似然里的 y 那一列：默认胜负；`y_margin` = 赛后经济差的软标签（models.margin，修订 A6）。
    似然 Σ w·[log(1+e^η) − y·η] 对 [0,1] 里的软标签照样成立，不用改任何别的东西。

    `m_team` 在 A 里的系数是 **(5 − 该场该侧的已知 playerid 数)**，
    对 96.4% 的比赛这个数是 0，于是 m_T 对它们**只出现在先验里**，
    模块顶部那段"用 m 承担水平、用 c 承担残差，避开共线性"的论证原封不动成立。
    只有名单不全的那 3.6% 会让 m_T 露一次面：缺位那名选手被当作
    "这支队的一名普通选手"，正好就是 `player_rating(None, team)` 的兜底口径。
    不这么做的话，那几场的账会被硬塞给在场的 3-4 个人，把他们的 rating 顶高。
    """
    age = (as_of - win.kickoff_utc).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (np.maximum(age, 0.0) / cfg.half_life_days)
    team_region = infer_team_region(win)

    bp = win.blue_player_ids.to_numpy()
    rp = win.red_player_ids.to_numpy()
    bt = win.blue_team_id.astype(str).to_numpy()
    rt = win.red_team_id.astype(str).to_numpy()
    n = len(win)

    pid: dict[str, int] = {}
    tid: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    # 先扫一遍选手列，保证 r 的列块连续
    for i in range(n):
        for arr, sgn in ((bp[i], 1.0), (rp[i], -1.0)):
            for p in _iter_ids(arr):
                rows.append(i)
                cols.append(pid.setdefault(p, len(pid)))
                vals.append(sgn)
    n_p = len(pid)
    for i in range(n):
        for t, sgn in ((bt[i], 1.0), (rt[i], -1.0)):
            rows.append(i)
            cols.append(n_p + tid.setdefault(t, len(tid)))
            vals.append(sgn)
    n_t = len(tid)
    rows.extend(range(n))
    cols.extend([n_p + n_t] * n)
    vals.extend([1.0] * n)

    # 名单不全时的补位：缺几个人就给 m_T 加几份
    m_off = n_p + n_t + 1
    for i in range(n):
        for arr, t, sgn in ((bp[i], bt[i], 1.0), (rp[i], rt[i], -1.0)):
            miss = ROSTER_SIZE - len(list(_iter_ids(arr)))
            if miss > 0:
                rows.append(i)
                cols.append(m_off + tid[t])
                vals.append(sgn * miss)

    A = sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n_p + 2 * n_t + 1),
    )

    # 选手 → 归属队：窗口内衰减加权出场最多的那支
    acc: dict[tuple[int, int], float] = {}
    for i in range(n):
        for arr, t in ((bp[i], bt[i]), (rp[i], rt[i])):
            j = tid[t]
            for p in _iter_ids(arr):
                key = (pid[p], j)
                acc[key] = acc.get(key, 0.0) + w[i]
    best: dict[int, tuple[float, int]] = {}
    for (pj, tj), v in acc.items():
        cur = best.get(pj)
        # 并列时取 index 小的，保证确定性
        if cur is None or v > cur[0] or (v == cur[0] and tj < cur[1]):
            best[pj] = (v, tj)
    p_team = np.zeros(n_p, dtype=np.int64)
    for pj, (_, tj) in best.items():
        p_team[pj] = tj

    lid: dict[str, int] = {}
    t_league = np.zeros(n_t, dtype=np.int64)
    for t, j in tid.items():
        g = team_region.get(t, UNKNOWN_REGION)
        t_league[j] = lid.setdefault(g, len(lid))

    inv_t = {j: t for t, j in tid.items()}
    player_team = {p: inv_t[int(p_team[j])] for p, j in pid.items()}
    return _Design(
        A=A,
        y=win[target].to_numpy(dtype=float),
        w=w,
        pid=pid,
        tid=tid,
        lid=lid,
        p_team=p_team,
        t_league=t_league,
        team_region=team_region,
        player_team=player_team,
    )


def _neg_log_post(theta: np.ndarray, d: _Design, cfg: HierBTConfig) -> tuple[float, np.ndarray]:
    """负对数后验与解析梯度。θ = [ r | c | b0 | m | ℓ ]。"""
    n_p, n_t = len(d.pid), len(d.tid)
    n_l = len(d.lid)
    k = n_p + n_t + 1
    # A 的列跨到 m 块为止（θ 的前 k+n_t 位），ℓ 不进似然
    kk = k + n_t
    eta = d.A @ theta[:kk]
    # logaddexp 形式在 |eta| 很大时也稳定
    val = float(np.sum(d.w * (np.logaddexp(0.0, eta) - d.y * eta)))
    grad = np.zeros_like(theta)
    prob = 1.0 / (1.0 + np.exp(-eta))
    grad[:kk] = d.A.T @ (d.w * (prob - d.y))

    r = theta[:n_p]
    c = theta[n_p:n_p + n_t]
    m = theta[k:k + n_t]
    lg = theta[k + n_t:k + n_t + n_l]

    dev_p = r - m[d.p_team]
    val += float(np.sum(dev_p * dev_p)) / (2.0 * cfg.sigma_player ** 2)
    grad[:n_p] += dev_p / cfg.sigma_player ** 2
    np.add.at(grad, k + d.p_team, -dev_p / cfg.sigma_player ** 2)

    dev_t = m - lg[d.t_league]
    val += float(np.sum(dev_t * dev_t)) / (2.0 * cfg.sigma_team ** 2)
    grad[k:k + n_t] += dev_t / cfg.sigma_team ** 2
    np.add.at(grad, k + n_t + d.t_league, -dev_t / cfg.sigma_team ** 2)

    val += float(np.sum(c * c)) / (2.0 * cfg.sigma_chem ** 2)
    grad[n_p:n_p + n_t] += c / cfg.sigma_chem ** 2

    val += float(np.sum(lg * lg)) / (2.0 * cfg.sigma_league ** 2)
    grad[k + n_t:k + n_t + n_l] += lg / cfg.sigma_league ** 2
    return val, grad


def _empty_snapshot(as_of: pd.Timestamp, cfg: HierBTConfig, n: int) -> HierBTSnapshot:
    return HierBTSnapshot(
        as_of=as_of, r_player={}, m_team={}, c_team={}, l_region={},
        team_region={}, player_team={}, intercept=0.0, cfg=cfg,
        diagnostics={"n_games": n, "fitted": False},
    )


def fit_snapshot(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: HierBTConfig | None = None,
    warm: HierBTSnapshot | None = None,
    *,
    target: str = "y_blue_win",
) -> HierBTSnapshot:
    """在 `[as_of − window_years, as_of)` 上拟合一次分层 BT。

    **`df` 里晚于 `as_of` 的比赛会被物理裁掉**，所以这个函数不可能偷看未来；
    衰减基准时刻也取 `as_of` 而不是"窗口内最后一场"，否则删掉未来数据会让权重变化。

    `warm` 只允许是**更早**的快照（`rolling_ratings` 保证这一点）：暖启动只影响
    L-BFGS 的起点，不改变可行域，且仍然是确定性的 → 未来删除不变性不受影响。
    """
    cfg = cfg or HierBTConfig()
    lo = as_of - pd.DateOffset(days=round(cfg.window_years * 365.25))
    win = df[(df.kickoff_utc >= lo) & (df.kickoff_utc < as_of)]
    if len(win) < cfg.min_games:
        return _empty_snapshot(as_of, cfg, len(win))

    d = _build_design(win, as_of, cfg, target)
    n_p, n_t, n_l = len(d.pid), len(d.tid), len(d.lid)
    k = n_p + n_t + 1
    x0 = np.zeros(k + n_t + n_l)
    if warm is not None:
        for p, j in d.pid.items():
            if p in warm.r_player:
                x0[j] = warm.r_player[p]
        for t, j in d.tid.items():
            if t in warm.c_team:
                x0[n_p + j] = warm.c_team[t]
            if t in warm.m_team:
                x0[k + j] = warm.m_team[t]
        x0[n_p + n_t] = warm.intercept
        for g, j in d.lid.items():
            if g in warm.l_region:
                x0[k + n_t + j] = warm.l_region[g]

    res = minimize(
        _neg_log_post, x0, args=(d, cfg), jac=True, method="L-BFGS-B",
        options={"maxiter": cfg.max_iter, "maxfun": cfg.max_iter * 2,
                 "ftol": 1e-14, "gtol": cfg.gtol, "maxcor": 20},
    )
    th = res.x
    inv_p = {j: p for p, j in d.pid.items()}
    inv_t = {j: t for t, j in d.tid.items()}
    inv_l = {j: g for g, j in d.lid.items()}
    return HierBTSnapshot(
        as_of=as_of,
        r_player={inv_p[j]: float(th[j]) for j in range(n_p)},
        m_team={inv_t[j]: float(th[k + j]) for j in range(n_t)},
        c_team={inv_t[j]: float(th[n_p + j]) for j in range(n_t)},
        l_region={inv_l[j]: float(th[k + n_t + j]) for j in range(n_l)},
        team_region=d.team_region,
        player_team=d.player_team,
        intercept=float(th[n_p + n_t]),
        cfg=cfg,
        diagnostics={
            "n_games": len(win),
            "n_eff_games": float(d.w.sum()),
            "n_players": n_p,
            "n_teams": n_t,
            "n_regions": n_l,
            "nit": int(res.nit),
            "fun": float(res.fun),
            "gmax": float(np.abs(res.jac).max()),
            "converged": bool(res.success),
            "sd_r": float(np.std(th[:n_p])),
            "sd_m": float(np.std(th[k:k + n_t])),
            "sd_c": float(np.std(th[n_p:n_p + n_t])),
            "sd_l": float(np.std(th[k + n_t:k + n_t + n_l])),
            "target": target,
            "fitted": True,
        },
    )


# ---------------------------------------------------------------------------
# 滚动 as-of 特征
# ---------------------------------------------------------------------------
def refit_grid(
    df: pd.DataFrame,
    cfg: HierBTConfig,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """固定日历网格。锚在**数据首日**上而不是 `start` 上，网格点因此与评测口径无关
    —— 换一个评测起点不会让同一场比赛拿到不同的快照（可复现性）。
    """
    t0 = df.kickoff_utc.min().normalize()
    tz = t0.tz

    def _as_ts(v: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(v)
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

    tn = df.kickoff_utc.max() if end is None else min(df.kickoff_utc.max(), _as_ts(end))
    step = pd.Timedelta(days=cfg.refit_every_days)
    lo = t0 if start is None else max(t0, _as_ts(start))
    first = t0 + step * int(np.ceil(max((lo - t0) / step, 0.0)))
    out: list[pd.Timestamp] = []
    t = first
    while t <= tn + step:
        out.append(t)
        t = t + step
    return out


def _to_utc_naive(v: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(v)
    return ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert("UTC").tz_localize(None)


def rolling_ratings(
    df: pd.DataFrame,
    cfg: HierBTConfig | None = None,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    warm_start: bool = True,
    return_snapshots: bool = False,
    verbose: bool = False,
    target: str = "y_blue_win",
) -> pd.DataFrame | tuple[pd.DataFrame, list[HierBTSnapshot]]:
    """沿日历网格重复拟合，给 `df` 的**每一行**贴上 as-of BT 特征。

    每一行取「严格早于该行 kickoff 的最近一张快照」。因为快照 `as_of` 只吃
    `< as_of` 的比赛，所以该行的特征一定只用到它自己开赛前的信息。

    `start` 之前的行拿不到快照（特征为 0 / NaN 语义上都不对），这里返回 0 并把
    `bt_known_players=0`，调用方应当只在 `start` 之后的区间上评测。
    `end` 只是省算力的截断（调参时只需要一段），不影响 `end` 之前任何一行的取值。
    `target`：似然里当 y 的那一列（见 `_build_design`）；某一行的 target 只进开赛晚于它的快照。
    """
    cfg = cfg or HierBTConfig()
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")
    grid = refit_grid(
        df, cfg,
        pd.Timestamp(start) if start is not None else None,
        pd.Timestamp(end) if end is not None else None,
    )

    out = {c: np.zeros(len(df)) for c in FEATURE_COLUMNS}
    bt = df.blue_team_id.astype(str).to_numpy()
    rt = df.red_team_id.astype(str).to_numpy()
    bp = df.blue_player_ids.to_numpy()
    rp = df.red_player_ids.to_numpy()

    snaps: list[HierBTSnapshot] = []
    prev: HierBTSnapshot | None = None
    bounds = np.asarray([np.datetime64(t.tz_convert("UTC").tz_localize(None)) for t in grid])
    last_hi = (
        np.datetime64("2999-01-01")
        if end is None
        else np.datetime64(_to_utc_naive(end))
    )
    kick_naive = pd.DatetimeIndex(df.kickoff_utc).tz_convert("UTC").tz_localize(None).to_numpy()
    for gi, as_of in enumerate(grid):
        snap = fit_snapshot(df, as_of, cfg, warm=prev if warm_start else None, target=target)
        if warm_start and snap.diagnostics.get("fitted"):
            prev = snap
        if return_snapshots:
            snaps.append(snap)
        hi = bounds[gi + 1] if gi + 1 < len(bounds) else last_hi
        sel = np.nonzero((kick_naive >= bounds[gi]) & (kick_naive < hi))[0]
        if verbose and gi % 20 == 0:
            print(f"  [{gi + 1}/{len(grid)}] {as_of.date()} n={snap.diagnostics.get('n_games')} "
                  f"nit={snap.diagnostics.get('nit')} rows={len(sel)}", flush=True)
        for i in sel:
            f = snap.features(
                blue_team_id=bt[i], red_team_id=rt[i],
                blue_player_ids=list(_iter_ids(bp[i])),
                red_player_ids=list(_iter_ids(rp[i])),
            )
            for c in FEATURE_COLUMNS:
                out[c][i] = f[c]
    feats = pd.DataFrame(out, index=df.index)
    if return_snapshots:
        return feats, snaps
    return feats


# ---------------------------------------------------------------------------
# 方差成分的矩估计（给验收④做独立对照，不依赖 NumPyro）
# ---------------------------------------------------------------------------
def variance_components(snap: HierBTSnapshot) -> dict[str, float]:
    """从一张快照上读出各层的经验 SD。

    这是 **MAP 点估计的离散度**，会比真实后验 SD 偏小（收缩把尾巴压扁了），
    所以它只能用来做"同一量级"的对照（验收④要求差异 <3 倍），不能当后验区间用。
    """
    r = np.fromiter(snap.r_player.values(), dtype=float)
    m = np.fromiter(snap.m_team.values(), dtype=float)
    c = np.fromiter(snap.c_team.values(), dtype=float)
    lg = np.fromiter(snap.l_region.values(), dtype=float)
    parent = np.asarray([snap.m_team.get(snap.player_team.get(p, ""), 0.0) for p in snap.r_player])
    return {
        "sd_player_total": float(np.std(r)) if r.size else float("nan"),
        "sd_player_within_team": float(np.std(r - parent)) if r.size else float("nan"),
        "sd_team": float(np.std(m)) if m.size else float("nan"),
        "sd_chem": float(np.std(c)) if c.size else float("nan"),
        "sd_league": float(np.std(lg)) if lg.size else float("nan"),
        "n_players": int(r.size),
        "n_teams": int(m.size),
        "n_regions": int(lg.size),
    }
