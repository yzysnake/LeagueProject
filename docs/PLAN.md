# LoL 职业赛 BP 后胜率预测系统 · 开发方案

> 版本：**v1.1** ｜ 编制 2026-09-18 ｜ 数据源事实复核 2026-09-18 ｜ 读者：本人（data scientist，独立开发）
> v1.1 修订要点：修正 4 处已过期的 [实测] 数据源事实（OE manifest 冻结 / `firstPick` 已回填全历史 / 2026 赛区 code 改回 `LCS`·`CBLOL` / Drive 直连恢复可用）；删掉未经测量的 `n_eff = n/2.5`，改为实测 ICC + block bootstrap；修 `series_id`、`dropna`、holdout 冻结三处"照做会静默出错"的实现点；新增 §8.0「第一天的 20 条命令」。
> 本文是工作手册，不是建议清单。每一节都应当能照着做。
> 数据快照基准（**v1.1 更新**）：v1.0 的基准是本机 `scratchpad/` 下 OE 2022–2025 四个陈旧快照（均 164 列，2025 最后一天 2025-11-16）——**全文凡是基于"本机快照"的 schema 结论都已在 2026-09-18 用线上文件复核并改写**。当前基准：**2026 线上文件 = 70,932,929 bytes / 105,984 行 / 8,832 场 / 165 列 / `max(date) = 2026-09-18 11:51:26`**，且 2014–2026 **全部 165 列、全部含 `firstPick`**。

## 0. 文档约定

**证据等级标记**，全文统一使用：

| 标记 | 含义 |
|---|---|
| **[实测]** | 本机 curl / Python 跑出来的数字，可复现 |
| **[实测·已复核]** | 实测且经过第二次独立核查确认 |
| **[已证伪]** | 曾出现在调研中、但被核查推翻的说法，正文给出正确版本 |
| **[待验证]** | 有理有据的推断或引用，但本项目尚未实测，必须自己跑一遍 |

**证据的保质期纪律（v1.1 新增，血的教训）**：v1.0 里有四条被标 **[实测·已复核]** 的数据源事实在 3 个月内就过期了（OE manifest 冻结、`firstPick` 回填全历史、赛区 code 改回 `LCS`/`CBLOL`、Drive 直连恢复可用），而文档在它们之上建了硬性指令和 CI 断言。所以：

1. 凡涉及 **schema / 赛区词表 / 外部接口行为** 的 [实测] 事实，必须同时记录**测量日期**与 **snapshot sha**，写成 `**[实测 2026-09-18 · snap-xxxx]**`。
2. 这类事实**不得**写成一次性断言硬编码进代码，一律做成 `store/audit_columns.py` 的**每日复核项**，变了就报告，而不是让 pipeline 崩掉。
3. 任何以"某年才有 / 某年没有 / 只有某赛区有"为前提的设计决策，落笔前先跑一次 `audit_columns.py` 复核。

**本文的三条纪律**（后面每一节都是这三条的展开）：

1. 任何数字，先问"这个样本量能不能分辨出这么大的差异"。分辨不出的，不准写成"改进"。
2. 任何特征，先问"它在严格时间外推下、连续两个留出期是否同号"。不同号的，不进模型。
3. 任何数据，先问"这个文件是不是 2009 字节的 Google Drive 错误页"。本机 `scratchpad/` 里此刻就躺着 5 个这样的假 CSV（`oe2021.csv` / `oe2026.csv` / `oe_2025.csv` / `oe_2026.csv` / `t2021.csv`，全部 2009 字节）**[实测]**。

---

# 1. 结论先行

## 1.1 推荐技术路线（一段话）

用 **Oracle's Elixir 年度 CSV 做训练主数据 + lolesports livestats feed 做实时阵容 + Data Dragon 做元数据** 这三层免费栈；主力模型是一个 **参数量在 10¹–10² 量级的分组 L2 正则化 logistic regression**，输入是「按赛区收缩的蓝方先验 + 在线递归的 team rating + player rating + 收缩后的英雄版本强度标量」，所有历史聚合特征由**单遍时序状态机**生成（先读状态写特征、再用赛果更新状态），保证 point-in-time 正确；评估一律走 **series 级分组的 rolling-origin walk-forward**，每一个特征、每一个超参都必须通过「连续两个留出期 Δ 同号且 series 级 t > 2」的准入门禁才准进模型；产品形态是 **Streamlit 单进程本地站点**（第 4 周就上线，手动摆阵容是默认模式，实时读取是增强），第 9 周以后才考虑 FastAPI + docker compose 拆分。整条路线的核心判断是：**这个问题的信息量上限很低，所以项目的价值不在"多准"，而在"每个结论都带置信区间、都可复现、都能被推翻"**。

## 1.2 现实效果预期

BP 结束 / 开赛前 / 单张地图 / 严格时间切分 / tier-1 赛区口径：

| 指标 | 诚实区间 | 点估计 | 依据 |
|---|---|---|---|
| **log loss（主指标）** | **0.615 – 0.645** | ~0.630 | 本机 team+player Elo logistic 实测 0.6274–0.6336 **[实测]** |
| **Brier score** | **0.216 – 0.228** | ~0.221 | 实测 0.2189–0.2220；arXiv 2609.08060 walk-forward 0.2207 |
| **accuracy（仅展示）** | **62% – 66%** | ~64% | 实测 63.5–65.1%；Riot 官方 GPR 公布 65% |
| **AUC（辅助）** | **0.67 – 0.71** | ~0.70 | 实测 0.694–0.708 |
| **ECE（15 等频 bin）** | **≤ 0.03** | ~0.017 | 实测 logistic 天生校准 |
| **calibration slope** | **0.93 – 1.07** | ~1.00 | arXiv 报 0.995 |
| **单次更换 1 个英雄的 \|Δp\|**（产品核心功能的量级） | **0.5 – 1.2 pp**（Top5 升与 Top5 降之间全距约 2–3 pp） | ~0.8 pp | 先验推算：σ_champ=0.0229、k≈476、典型 decayed n_eff 50–150 → 收缩因子 0.10–0.24，× β_champ=0.975 → Δlogit 0.02–0.05 **[待验证，第 4 周用真实 artifact 实测替换]** |

**参照系**：常数预测蓝方基准率（2025 年实测 53.385%）= log loss 0.6912 / Brier 0.2489。所以模型只把 Brier 从 0.249 拉到 0.221，**"解释掉"的方差约 11%**。剩下 89% 是同水平队伍对战的内在随机性，不是模型不够好。

**最重要的一条期望管理**：全部 draft 相关特征（英雄版本强度 + 熟练度 + 协同 + 克制）加起来的预期增量是 **ΔlogLoss ∈ [0, 0.008]，点估计约 0.004，对应 accuracy +0.5 ~ +1.5 pp**，而且**很可能在统计上不显著**。实测证据：

- 队伍 Elo 一项就吃掉 0.0613 的 log loss（占全部可解释信号 0.070 的 87.6%）**[实测]**
- 在 Elo 之上加 171 维英雄 one-hot：0.6336 → **0.6342（变差）** **[实测]**
- 在 Elo 之上加选手该英雄收缩胜率：0.6336 → **0.6345（无增益）** **[实测]**
- 显式 pairwise 协同/克制（29,420 参数）：**0.7756**，比抛硬币的 0.6912 还差 **[实测]**
- **单次换 1 个英雄的 |Δp| 约 0.5–1.2 pp，而 bootstrap 区间宽度约 8.6 pp**（§7.5.2 示例）——即**大多数替换的影响小于本模型的不确定度**。这不是 bug，但必须在 UI 上提前讲清楚（§7.6.2 ①），否则第 4 周上线当天就会被判定为"工具坏了" **[待验证]**

所以产品叙事必须一开始就定为「**队伍/选手实力 + 阵容微调**」，而不是「我们能算出这套阵容值多少胜率」。UI 上必须把两层贡献分开展示。

**红线**（写进 README，用来抓自己的 bug）：BP 后、无赛中数据的职业赛模型，若 **accuracy > 67% 或 log loss < 0.60 或任一 fold AUC > 0.78**，几乎一定有泄漏。对照：随机划分 + 整赛季聚合胜率会把 AUC 从诚实的 0.650 推到 **0.794（+0.144）** **[实测]**。

## 1.3 最容易翻车的三件事

### 翻车点 1：数据静默损坏（发生概率最高，后果最隐蔽）

Oracle's Elixir 的 Google Drive 配额耗尽时返回 **HTTP 200 + 2009 字节的 `<title>Google Drive - Quota exceeded</title>` HTML**，不是 429、不是 404 **[实测·已复核，三个不同出口 IP 同错 → 是文件所有者账号的配额，与你的 IP 无关]**。

**注意这是【间歇性】的，不是永久状态**：2026-09-18 复核时同一端点返回 200 / 70.9 MB 的真 CSV（§3.2.2）。所以正确的姿势是"直连为主 + 硬校验 + 小时级退避"，而不是把第一步锁成人工操作。

更要命的是：`csv.DictReader` 读这个 HTML **不会抛异常**，它会解析成 0 行。而三份候选方案里都写着的校验条件 `rows % 12 == 0` —— **`0 % 12 == 0` 为真，校验会放行**。

本机现状就是现行犯：`scratchpad/oe2021.csv`、`oe2026.csv`、`oe_2025.csv`、`oe_2026.csv`、`t2021.csv` 五个文件，全部 2009 字节，全部是那张 HTML。任何 `glob("oe*.csv")` 的脚本此刻已经在吃它们了。

**修法见 §3.2.3（五重校验）和 §3.12（自建镜像）。**

### 翻车点 2：显著性口径错误，导致选错主力模型

同一场 Bo3/Bo5 里的多局高度相关（同样两队、同样当日状态、共享 BP 博弈），用 game 级配对检验会让标准误系统性偏窄。

**但"偏窄多少"必须实测，不能拍脑袋。** 正确的量是 design effect `deff = 1 + (m̄−1)·ICC`，其中 `m̄` 是每系列平均局数、ICC 是同系列内 per-game log loss 差值 `d_i` 的组内相关：

- `m̄` **[实测 2026-09-18]**：2026 全量 8,832 场 / 4,127 个 series → **m̄ = 2.14**，每系列局数分布 `{1:1274, 2:1479, 3:1037, 4:196, 5:141}`。**不是 v1.0 写的 2.5。**
- `ICC` **【尚未测量，第 3 周必须先测】**：v1.0 用的 `n_eff = n_games/2.5` 等价于隐含假设 **ICC = 1**（系列赛内多局完全相关），这是最保守的一端。

具体到本项目：调研给出的 L3(team+player Elo) → L5(+mastery_diff) 的 `ΔLL = +0.00496, sd = 0.0894, n = 3711, t = 3.38`。**裁决对 ICC 极度敏感**：

| 口径 | deff | n_eff | t | 结论 |
|---|---|---|---|---|
| game 级（ICC=0） | 1.00 | 3,711 | 3.38 | 显著（紧贴下限） |
| ICC = 0.3 | 1.34 | 2,770 | **2.92** | **仍显著 → 裁决翻转** |
| ICC = 0.6 | 1.68 | 2,209 | 2.61 | 边缘 |
| ICC = 1（v1.0 的隐含假设） | 2.14 | 1,734 | **2.31** | 不显著 |

**所以"mastery_diff 不进主力"这条裁决是有条件的：ICC ≥ 0.6 时成立，ICC < 0.35 时翻转。第 3 周必须先用 `eval/power.py::effective_n()` 实测 ICC（§6.3.1），再下裁决**；在此之前一律标 **【裁决待定】**。

更根本的一条：§6.6.1 已经推荐了 series 级 block bootstrap，而 **block bootstrap 本来就直接、正确地估计了聚类后的标准误**，根本不需要 `n_eff` 这个近似。两套尺子必须合一（§6.3.1）。

同理，"半衰期 75 天最优"（网格差 0.0004–0.0025）、"收缩强度 k=476 最优"（网格差 0.0001–0.0005）这些结论，差异比任何口径下的 MDE 小一个数量级，**全部是在拟合噪声**。

**修法见 §6.3（功效纪律）和 §6.7（特征准入门禁）。**

### 翻车点 3：期望落差诱导你去"修"出一个有泄漏的高分模型

这是最危险的，因为它不是技术错误而是心理陷阱。当你做了两个月、发现"加了这么多阵容特征才涨 0.4 个百分点"，最自然的反应是怀疑实现有 bug，然后去"修"——而所有能让数字大幅变好的"修法"，本质上都是泄漏（随机切分、赛季聚合、series 跨 fold、事后 rating）。

**修法：在写第一行模型代码之前，先把 §2.5 的预注册文件 commit 进 git，明确写下"若 draft 特征不显著，我们将如实报告 α̂ ≈ 0 并把它作为主要结论"。** 提前把负结果合法化，是唯一有效的防御。

---

# 2. 问题定义

## 2.1 预测目标与预测时点

**预测目标**：单张地图（single map / single game，不是 Bo3/Bo5 系列赛）的蓝方获胜概率 `P(blue wins) ∈ (0,1)`。

**预测时点**：BP（ban/pick）阶段结束、开局之后、任何游戏内事件发生之前。可用信息严格限于：

- 双方各 5 个英雄 + 各自位置（top/jng/mid/bot/sup）
- 双方各 5 名选手身份、双方队伍身份
- 赛区、赛事级别（常规赛/季后赛/国际赛）、系列赛第几局
- patch 版本
- 蓝红方分配
- 以上全部实体的**历史**（严格早于本场 kickoff 的所有比赛）

**一个必须接受的现实修正**：产品上做不到"BP 结束瞬间"。官方 livestats feed 有**硬性 50 秒延迟下限**（错误原文 `disallowed window with end time less than 50 sec old`），且未开始的 game 返回 **HTTP 204 空体** **[实测]**。实测一次完整翻转：某场 LEC Bo5 的 game4，`getEventDetails` 里 state 在 17:51:27Z 就变成 `inProgress`，但 window 端点**连续 204 长达 20 分 50 秒**，直到 18:12:17Z 才转 200——而那时游戏时钟 0 是 18:12:13Z，即**端点可用时刻比开局还晚 2–4 秒** **[实测·已复核]**。

所以产品文案写「**开局即刻胜率**」，不写「BP 结束胜率」。这对模型零影响（frame 0 与 BP 结束只差几十秒，没有任何赛中信息），只影响轮询逻辑与文案。

## 2.2 样本单位与标签

**样本单位 = 一张地图（one map / one gameid）**，不是一行 OE 记录。

OE 的原始结构是**每场严格 12 行**（`participantid` 1–10 是选手行，100/200 是两队汇总行，后者的 `position` 字面量就是 `'team'`）**[实测：2023 全量 130,764 行 / 10,897 gameid，rows-per-game 分布 `Counter({12: 10897})`，无一例外]**。

这意味着 **silver 层必须先把 12 行折叠成 1 行**，且 `df.game_id.nunique() == len(df)` 要写成断言。按行随机切分 = 标签直接泄漏，这是三条泄漏通道里最蠢也最常见的一条。

**标签** `y_blue_win ∈ {0, 1}`，来自 OE 的 `result` 列（蓝方 team 行的 result）。`result` **只能作为 label，绝不能进特征**。

**series_id（分组键，比 game_id 重要得多）**：OE 没有 series id，本地构造。

**【已证伪】按日历日构造是错的**：

```python
# ✗ v1.0 的写法：跨 UTC 零点的系列赛会被劈成两个 series
series_id = f"{date.date()}::{league_canon}::{'|'.join(sorted([blue_team_id, red_team_id]))}"
```

**[实测 2026-09-18]** 2026 全量 8,832 场里 **89 个系列被 UTC 零点劈开**（判据：同 league 同队对、相邻两局间隔 <6 小时、却落在两个 series_id）。示例：LCS `2026-01-24 23:14:34`(g1) vs `2026-01-25 00:22:15`(g2)；LCS `2026-02-14 23:52:02`(g4) vs `2026-02-15 00:41:15`(g5)；CBLOL `2026-08-07 23:48:44` → `08-08 00:46:12`。**美洲赛区（LCS/CBLOL/LLA）因北美晚间开赛必然高发。** 反方向也有 3 例：同一天同一对队伍打了两个独立系列被错误合并（`2026-05-27::NLC::…` 出现 game 号 [1,1]；`2026-08-13::EBL::…` 出现 [1,1,2]）。

**[正确写法] 基于时间邻接的会话切分**（OE 实测同系列相邻局间隔基本在 30–70 分钟）：

```python
GAP = pd.Timedelta(hours=6)

def build_series_id(df: pd.DataFrame) -> pd.Series:
    """df: 一场一行，含 kickoff_utc / league_canon / blue_team_id / red_team_id"""
    key = list(zip(df.league_canon,
                   [frozenset(p) for p in zip(df.blue_team_id, df.red_team_id)]))
    out = pd.Series(index=df.index, dtype=object)
    for k, g in df.assign(_k=key).groupby('_k', sort=False):
        g = g.sort_values('kickoff_utc')
        blk = (g.kickoff_utc.diff() > GAP).fillna(True).cumsum()
        start = g.groupby(blk).kickoff_utc.transform('min')
        out.loc[g.index] = (f"{k[0]}::{'|'.join(sorted(k[1]))}::"
                            + start.dt.strftime('%Y%m%dT%H%M'))
    return out
```

**三条必须写成 pytest 的断言**（替换 v1.0 那两条抓不到问题的验收）：

```python
# (a) 同一 series 内 game 号是从 1 开始、无重复的连续序列
for sid, g in silver.groupby('series_id'):
    assert sorted(g.game.tolist()) == list(range(1, len(g) + 1)), sid
# (b) 同一 series 内跨度 ≤ 12 小时
assert (silver.groupby('series_id').kickoff_utc.agg(lambda x: x.max() - x.min())
        <= pd.Timedelta(hours=12)).all()
# (c) 全库不存在「同 league 同队对、相邻两局间隔 <6h 却 series_id 不同」的样本
assert count_split_series(silver) == 0     # 上面那段检测代码，直接写成测试
```

**为什么这条这么重要**：series_id 是全文最 load-bearing 的键——分组切分、embargo、block bootstrap、ICC / `n_eff`、MDE 全挂在它上面。被劈开 = fold 分组失效（同一系列的 g1-3 与 g4-5 落进不同 fold，正是 §4.8.2 第 4 条列为"比标签泄漏更严重"的那条）+ `n_series` 被高估 + MDE 被低估。而 v1.0 阶段 2 的验收④（"series_id 非空率 100%、`game_in_series` ≤5"）对被劈开的 Bo5 两段（game 号分别是 1-3 和 4-5）**两条都通过**，完全抓不到。

注意：**不要把 series_id 的来源押在 Leaguepedia 的 `MatchId` 上**。评估协议的正确性不能挂在一个 5 次/60 秒配额的第三方 API 上。但用它做**覆盖率交叉校验**正合适——把"series 切分一致率"（本地切分 vs `ScoreboardGames.MatchId`）做成 Tab 3 的一个 gauge。

## 2.3 为什么输出概率而不是分类

四条理由，前两条是硬的：

1. **accuracy 检测不出模型失败。** 有先例记录：某未校准模型 Brier **0.2536**，比"永远输出基准率"的常数预测器 **0.2495** 还差——而 accuracy 看起来完全正常。accuracy 把 0.51 和 0.99 当成同一件事，把置信度信息全丢掉了。
2. **这个问题的预测分布极窄。** BP 后大量比赛的真实胜率落在 0.45–0.65 区间。在 53/47 的基准率下，63% 和 65% 的 accuracy 差别在 800 场上完全在噪声里，accuracy 根本不是一个可用于选模的量。
3. **log loss 是严格适当（strictly proper）的评分规则**，且与 logistic 的训练目标一致，训练和评估口径统一。
4. **产品需要的就是概率。** 用户要看的是"58%"和"换掉这个英雄变成 60%"，不是"预测蓝方赢"。

**因此：主指标 log loss，副指标 Brier + 校准四件套，AUC 只作 sanity check，accuracy 只作展示、永不用于选模。**

## 2.4 蓝红方对称性处理

这个问题**不对称**，必须显式建模，不能做数据增强式的对称化。

**实测依据**：2022 全库 12,549 场蓝方胜率 **52.47%**，但赛区间差异巨大 **[实测]**：

| 赛区 | 蓝方胜率 | 场次 |
|---|---|---|
| PCS | 60.44% | 273 |
| LPL | 54.96% | 786 |
| CBLOL | 53.91% | 243 |
| LEC | 53.50% | 243 |
| LCK | 50.75% | 467 |
| VCS | 49.54% | 325 |
| LCS | 49.35% | 306 |
| LLA | 49.20% | 187 |

PCS 的 60.4% 基于 273 场，SE ≈ 3.0 pp，与全局 52.5% 差约 2.7σ——可能是真实差异（赛区节奏/meta 不同），也可能是小样本。

**正确处理：按赛区收缩的 as-of 估计**，三种错误做法都要避免：

```python
# ✗ 错误 1：硬编码常数 0.52         → 欠拟合，忽略赛区差异
# ✗ 错误 2：每个赛区一个自由参数     → 过拟合小赛区
# ✗ 错误 3：把蓝红方镜像做数据增强   → 直接抹掉真实的 side 效应
# ✓ 正确：empirical Bayes 收缩
side_prior_league = (blue_wins_asof + k * p_global_asof) / (n_games_asof + k)
# k ∈ {200, 300, 400}，含义是"需要几百场才相信这个赛区和全局不一样"
```

进模型时用 `logit(side_prior_league)` 作为一个连续特征，而不是 one-hot。

**额外要求**：把 side 系数的漂移做成监控指标。它动了通常意味着版本发生了结构性改动（地图改动、中立资源改版），是该重训的早期信号。

**特征的蓝方视角约定**：所有 diff 类特征一律 `blue − red`，模型输出 `P(blue wins)`。这样 what-if 面板里"蓝方换个英雄"的方向性是自然的。

## 2.5 项目成功的可证伪判据（预注册，第一天就要 commit）

在写任何模型代码之前，创建 `experiments/preregistration.md`，commit 并记录 git sha。内容：

```markdown
# 预注册 v1  (commit 前不得评估 holdout)

## Primary comparison（唯一一个）
M_main(全特征) vs L3(side_prior + team rating + player rating)
指标：log loss
样本：tier-1 严格子集的 walk-forward OOS
检验：series 级 block bootstrap + Diebold-Mariano (Harvey 修正)
其余所有比较一律标记 exploratory，用 Benjamini-Hochberg FDR 控制。

## 成功判据
(a) walk-forward Brier ≤ 0.228 且 calibration slope ∈ [0.93, 1.07]
(b) draft 特征组相对无 draft 基线的配对 ΔlogLoss ≥ 0.004，
    且在 series 级聚类下 |t| > 2.5，且连续两个留出期同号

## 失败判据（红线，CI 自动 fail）
任一 fold AUC > 0.78  或  全样本 accuracy > 67%  或  log loss < 0.60
→ 强制走完 §6.9 的八条泄漏自查清单，结论写进报告

## 提前声明
若 draft 特征不显著，我们将如实报告 α̂ ≈ 0 及其 95% CI，
并把"BP 阶段在职业赛数据量下贡献不可测量"作为本项目的主要结论。
这不是失败，这是结果。
```

最后那段是整个文档里最重要的三行字。

---

# 3. 数据层

## 3.1 数据源总表

| 层 | 数据源 | 用途 | 更新频率 | 许可 | 是否已核实 | 是否硬依赖 |
|---|---|---|---|---|---|---|
| L0 | **Oracle's Elixir 年度 CSV** | 训练主数据（唯一免费的「选手×英雄×赛果」源） | 官方每天 1 次 | 免费，须遵守 Riot 条款 | **[实测 2026-09-18]** Drive 直连 200，13 个年度文件全可下 | **是** |
| L1 | **lolesports persisted API** | 赛程 / 进行中比赛 / 全量队伍名单 | 事件驱动 | 非官方，无 SLA | **[实测]** 全部端点 200 | 否（可降级） |
| L1 | **feed.lolesports.com livestats** | 实时阵容（10 英雄 + 选手 ID + patch） | 30s 轮询 | 非官方，无 SLA | **[实测]** | 否（可降级） |
| L2 | **Data Dragon** | 英雄元数据、patch 时间轴、前端图标 | 版本变更时 | 免费 CDN | **[实测]** 16.18.1 / 173 英雄 | **是** |
| L2 | CommunityDragon | 新英雄补位 | 同上 | 免费 | **[实测]** | 否 |
| L3 | **Leaguepedia Cargo** | first-pick 归属 / roster / 交叉校验 | 每日增量 | CC BY-SA | **[实测]** 匿名可用但需限速 | 否（补充） |
| L3 | **HF `gptilt/lol-esports-matches`** | 低级别赛事补量 / 交叉校验 | 每周 | CC BY-SA 3.0 | **[实测]** 136,956 场 | 否（补充） |
| L4 | Polymarket | 外部基准（市场共识） | 每日 | 公开 API | **[实测]** 无鉴权可用 | 否（v2） |

## 3.2 Oracle's Elixir（主数据，最关键也最脆弱）

### 3.2.1 唯一活的官方渠道

`oracleselixir.com` 是 React SPA，HTML 里没有下载链接。从 JS bundle `https://oracleselixir.com/static/js/main.4629d2d3.js` 里挖出页面原文 "Access files via Google Drive."，指向：

```
folder id = 1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
列目录: https://drive.google.com/embeddedfolderview?id=<FOLDER>#list
正则:   id="entry-([A-Za-z0-9_-]+)" ... flip-entry-title">([^<]*) ... flip-entry-last-modified"><div>([^<]*)
```

实测返回 HTTP 200 / 11,617 bytes / 13 个 entry（2014–2026）**[实测]**。

**已知的 13 个 fileId（仅供应急，代码里不要硬编码，每次重新发现）**：

| 年份 | fileId | 目录显示的最后修改 |
|---|---|---|
| 2014 | `12syQsRH2QnKrQZTQQ6G5zyVeTG2pAYvu` | Jun 8（冻结） |
| 2015 | `1qyckLuw0-hJM8XqFhlV9l1xAbr3H78T_` | Jun 8（冻结） |
| 2016 | `1muyfpaIqk8_0BFkgLCWXDGNgWSXoPBwG` | Jun 8（冻结） |
| 2017 | `11fx3nNjSYB0X8vKxLAbYOrS2Bu6avm9A` | Jun 8（冻结） |
| 2018 | `1GsNetJQOMx0QJ6_FN8M1kwGvU_GPPcPZ` | Jun 8（冻结） |
| 2019 | `11eKtScnZcpfZcD3w3UrD7nnpfLHvj9_t` | Jun 8（冻结） |
| 2020 | `1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d` | Jun 8（冻结） |
| 2021 | `1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA` | Jul 2（冻结） |
| 2022 | `1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy` | Jul 2（冻结） |
| 2023 | `1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2` | Jul 2（冻结） |
| 2024 | `1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN` | Jul 2（冻结） |
| 2025 | `1v6LRphp2kYciU4SXp0PCjEMuev1bDejc` | Jul 2（冻结） |
| **2026** | `1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm` | **每天变** |

**这张表里最有价值的信息是最后一列**：2014–2025 是冻结文件，冷启动下一次即可；日更只需要碰 2026 那一个约 70 MB 的文件。

**但【已证伪】不要拿 `flip-entry-last-modified` 这个串做身份键。** **[实测 2026-09-18]** 它返回的是**本地化显示串**，不是时间戳：2014–2020 是 `Jun 8`、2021–2025 是 `Jul 2`（**均无年份**），2026 是 `12:03 pm`（**无日期**——Google 对"今天修改"的文件只显示时间）。而 §3.12.1 的 manifest 示例却写成 `"drive_modified":"2026-09-18T06:03:00Z"` 这种 ISO 时间戳，中间没有任何转换代码，也没说时区基准。

风险恰好落在唯一每天变的那个文件上：OE 官方是每天定时更新一次，cron 时间稳定的话，连续两天的显示串都会是同一个 `12:03 pm` → 比对"未变" → **静默跳过当天的新数据**；反过来在更新前后取，同一个逻辑时间戳会先显示 `Sep 17` 后显示 `12:03 pm`，串本身不稳定。

**[正确做法] 用 `Content-Length` + `sha256` 做去重，不用显示串**：

| 文件 | 策略 |
|---|---|
| **当年文件（2026）** | **无条件每天下载一次**。70 MB/天成本可忽略，且 §3.12.2 本来就要求每天 1 次。下载后算 `sha256`，与 manifest 上次相同则不落新快照（`raw/<sha256>.csv.zst` 的内容寻址布局天生支持这个，v1.0 只是没用上） |
| **历史年份（2014–2025）** | 先 `curl -sI -L ".../download?id=<ID>&export=download&confirm=t"` 取 **`Content-Length`**，与 manifest 上次比对，变了才下载；下载后同样用 sha256 复核。这 12 个文件的显示串确实是稳定的日期，但 Content-Length 无歧义且可程序化 |

若仍要留存显示串，字段名必须写成 `drive_modified_raw`（原始串）+ `drive_modified_parsed`（带解析规则与时区假设），**不要把一个有歧义的串伪装成 ISO 时间戳**（§3.12.1）。

### 3.2.2 【已证伪】两条流传很广的下载路径，以及正确做法

**[已证伪 1] S3 直链**。网上一半教程（包括 22 star 的 `MRittinghouse/ProjektZero-LoL-Model/src/oracles_elixir.py`）还在用：

```
https://oracleselixir-downloadable-match-data.s3-us-west-2.amazonaws.com/...
→ <Code>NoSuchBucket</Code>   # 桶已彻底删除
https://oracles-elixir.s3-us-west-2.amazonaws.com/...
→ 301 → s3.amazonaws.com → 403 AccessDenied   # 新桶存在但已非公开
```
**[实测]** 两个桶都不可匿名访问。fork 任何 OE loader 之前先 `curl -I` 它的 BASE_URL。

**[已证伪 2] "用 GitHub LFS 镜像做自动兜底"**。这条在三份候选方案里都出现了，**必须删掉**。核查结论：

- 不存在任何官方或持续维护的镜像
- 现存的都是个人项目里的一次性快照：`Matthew-Paoletta/The-Snowballing-Effect` 的 2026 文件最后提交 **2026-03-22**，`yagoandradelima/data-analysis-python` 是 **2026-02-12**
- 而且它们多是普通 git blob 不是 LFS（`raw.githubusercontent.com` 直接吐出真 CSV 表头）
- 同一年份不同镜像的 schema 还不一样：`LouisPoutrain/Lol-ML/raw_data/2025_*.csv` 无 `firstPick`，`Matthew-Paoletta/.../data/2025_*.csv` 有 **[实测]**

**对一个按 patch 近期加权的模型，一份 3 月份的快照等于静默投毒**：它会通过所有内容校验（合法 CSV、正确表头、行数整除 12），然后让你的时间衰减权重把最大权重压在半年前的 meta 上。

**[正确版本 · v1.1 修订] 直连是主路径，Make-a-copy 是兜底**

v1.0 写的"今天从任何 IP 都必然返回 2009 字节 quota HTML，唯一真能拿到数据的路径是浏览器手动 Make-a-copy"，**把一次瞬时配额耗尽固化成了永久事实**，并据此把项目第一步锁死成一个无法自动化的人工操作；同时它与 §3.12.3 的 daily.sh（第 3–4 步本来就是直连 Drive 下载）自相矛盾。正确的、可证伪的表述是：

> **Drive 配额耗尽是【间歇性】的**（按文件所有者账号、按天计）。**直连通常可用**；失败时的特征是 **HTTP 200 + `content-type: text/html` + 2009 字节**，而不是 429/404。

**[实测 2026-09-18 复核]** 同一个端点现在完全可用：

```bash
curl -sI -L "https://drive.usercontent.google.com/download?id=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm&export=download&confirm=t"
# → HTTP/2 200 / content-type: application/octet-stream / content-length: 70932929
```

下载下来是真 CSV：**8,832 场 / 105,984 行 / 165 列 / `max(date) = 2026-09-18 11:51:26`**。2025/2024/2023/2022/2019/2014 的 Range 请求也都返回 206 + 真表头。

**daily.sh 真正该敲的命令**（这就是 §3.12.3 第 3–4 步里此前从未写明的下载端点）：

```bash
FILE_ID=1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm     # 每次从 embeddedfolderview 重新发现，别硬编码
curl -fsSL --retry 0 \
  -A "lolwp/1.0 (qxing@uchicago.edu)" \
  "https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t" \
  -o "${YEAR}.csv.tmp"
python -m lolwp.ingest.oracles_elixir --validate "${YEAR}.csv.tmp"   # 五重校验，§3.2.3
```

**降级链（按顺序，前两级全自动）**：

```
1. 直连 drive.usercontent.google.com                 ← 主路径，通常可用
2. 五重校验失败 / 网络失败 → 按 §3.12.2 小时级退避（1h / 3h / 6h）当天最多重试 3 次
   仍失败 → 保留旧文件 + 告警，绝不覆盖（§3.12.3 第 4 步）
3. 连续 2 天全失败 → 人工兜底：浏览器登录 → 打开
   https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
   → 全选 → "创建副本 / Make a copy" 到自己的 Drive → 从自己的副本下载
   （配额记在文件所有者 Tim Sevenhuysen 账号上，你的副本走你自己的配额）
   ⚠ 70 MB × 13 个文件，手工做一次很痛，所以它是兜底不是主路径
4. Streamlit 侧栏的 `st.file_uploader` 始终是一等公民：任何时候都能手工塞一份 CSV 进来
5. 下载成功后一律推进自建镜像（按 snapshot_date 归档），训练代码只读自建镜像
```

**关于 `gdown`**：gdown ≥5 走的就是上面这个端点，所以它**没有任何绕过配额的能力**——配额耗尽时它一样失败。不用它的理由是"多一层依赖、错误信息更难读"，不是"它不管用"。配额确实与你的出口 IP 无关：配额耗尽窗口内，三个互不相干的出口（家宽 Comcast / 服务器出口 / r.jina.ai 代理）实测同错 **[实测]**，换 IP / 挂代理 / 换机器都没用，只能等或走兜底。

**长期自动化方案（可选，第二阶段再做）**：Google Drive API v3 + OAuth（scope `drive.readonly` + `drive.file`），程序化地把文件 copy 进自己 Drive 再下载、下完即删。端点 `https://www.googleapis.com/drive/v3/files/{id}?alt=media&key={API_KEY}` 实测存在（无 key 时返回结构化 403）**[实测]**。凭证进 `.gitignore`，绝不打进 docker 镜像。

### 3.2.3 五重校验（少一条就会被静默污染）

```python
import csv, os
from datetime import date, datetime, timedelta

EXPECTED_PREFIX = ("gameid,datacompleteness,url,league,year,split,"
                   "playoffs,date,game,patch")

def validate_oe_csv(path: str, year: int, prev: dict) -> None:
    """五条断言全过才准 os.replace()，任一失败 -> 保留旧文件 + 非零退出 + 告警

    prev = 上一次成功 ingest 的 manifest.jsonl 里同年那条记录（冷启动时为 {}）。
    ★ 参照系是【我们自己上一次的快照】和【数据本身的日期】，不是任何第三方接口。
    """

    # 闸 1: MIME / 内容特征 —— 抓 quota HTML
    head = open(path, 'rb').read(4096)
    assert b'Quota exceeded' not in head, "Drive 配额错误页（HTTP 200 伪装）"
    assert not head.lstrip().startswith(b'<'), "拿到的是 HTML 不是 CSV"

    # 闸 2: 体积 —— quota 页恰好 2009 字节
    assert os.path.getsize(path) > 1_000_000, "文件过小"

    # 闸 3: 表头前缀
    first = head.decode('utf-8', 'replace').split('\n')[0]
    assert first.startswith(EXPECTED_PREFIX), f"表头异常: {first[:80]}"

    # 闸 4: 结构 —— 注意 n_games > 0 这一半！0 % 12 == 0 会放行空文件
    n_rows, games, max_date = 0, set(), ''
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            n_rows += 1
            games.add(row['gameid'])
            if row['date'] > max_date:      # 'YYYY-MM-DD HH:MM:SS'，字典序 == 时间序
                max_date = row['date']
    assert len(games) > 0,                "解析出 0 场比赛"
    assert n_rows == 12 * len(games),     f"非 12 行/场: {n_rows} 行 / {len(games)} 场"

    # 闸 5: 新鲜度 —— 抓"内容合法但陈旧"的镜像，这是前四条抓不到的
    #   ★ 两条都是【单侧】的：只拦"比已知的更旧 / 更少"，绝不拦"比已知的更新 / 更多"
    if year == date.today().year:           # 只有当年文件才该是新鲜的
        age = date.today() - datetime.strptime(max_date[:10], '%Y-%m-%d').date()
        assert age <= timedelta(days=3), f"数据陈旧: max(date)={max_date}，距今 {age.days} 天"
    if prev.get('n_games'):                 # OE 只回填、不删档 → 同年场次单调不减
        assert len(games) >= prev['n_games'] * 0.95, \
            f"场次倒退: {len(games)} < 上次快照 {prev['n_games']} 的 95%"
```

**闸 4 的括号内容是整份文档里最容易被跳过、但最该记住的一行**：调研里三份方案都只写了 `rows % 12 == 0`，而 `0 % 12 == 0` 为真。

#### 【已证伪】闸 5 不能用 `oe.datalisk.io` 的 games 计数做参照系

v1.0 的闸 5 是 `assert abs(n_games − manifest_games)/manifest_games < 0.05`，参照 `https://oe.datalisk.io/matchData`。**这会让每日 pipeline 开箱即死，而且是被"最新鲜的正确文件"触发的。**

**[实测 2026-09-18]** 该接口返回 2026 年 `games=5439, updatedAt=2026-06-02T19:04:15Z`，其余年份 `updatedAt` 全部停在 `2026-05-22`（2014 停在 `2026-01-18`）——**这份 manifest 自己已经冻结了三个半月**。而同一时刻从 Drive 实下的 2026 文件是 **8,832 场**。`drift = |8832−5439|/5439 = 62.4%`，远超 `< 0.05`。按 §3.12.3"第 4 步失败时绝不能继续往下走"，ingest 每天 `sys.exit(1)`，**永远用不上新数据**。方向判断也反了：v1.0 阶段 1 验收②要求"注入 8,040 场 vs manifest 的更高值，闸 5 必须拦下"，而实际 manifest 远**低于**真实值；同时阶段 0 验收③"2026 场次 ≥ 5,439"会通过——**同一份数据被两条验收给出相反裁决**。

三条修正，已体现在上面的代码里：

1. **单侧下界**：改成 `n_games >= 上一次快照的 n_games × 0.95`。只拦"比已知快照还少"（真正的截断/投毒），不拦"比参照值多"（正常增长）。
2. **校验数据本身而不是第三方计数**：`assert max(date) >= today − 3 days`。本次实测 `max(date)=2026-09-18`，这条能直接抓住"3 月快照投毒"（GitHub 镜像那类），**且不依赖任何外部接口**。
3. **参照系换成 `manifest.jsonl` 自己**：OE 只回填、不删档，所以同年 `n_games` 单调不减是一条可靠且自洽的不变量。

**`oe.datalisk.io` 整个从 `daily.sh` 和闸 5 里移除**，降级为本文档里的一条可选参考（无需下载就能大致知道各年样本量）。这也兑现了 v1.0 自己写下的合规承诺——它本来就写着"只用作低频探针，不写进生产关键路径"，而闸 5 恰恰**就是**生产关键路径。

> ⚠️ **合规提醒（若你仍要手动查它）**：`x-api-key: f561197a-82ea-4e54-acd2-386979018a7a` 是从站点 JS bundle 里扒出来的私有常量，属灰色地带。**每天最多一次，不进生产路径，不高频轮询**，请求带上含你邮箱的 User-Agent。返回体里的 S3 link 是已失效的遗留字段，忽略。
>
> ```bash
> curl -H "x-api-key: f561197a-82ea-4e54-acd2-386979018a7a" https://oe.datalisk.io/matchData
> ```

### 3.2.4 CSV 真实结构

**[实测·本机全量逐行统计]**

- 每场严格 12 行（10 选手 + 2 队伍汇总），`position` 取值 `top/jng/mid/bot/sup/team` 各占 1/6
- **[实测 2026-09-18]** 线上 2014 / 2019 / 2022 / 2023 / 2024 / 2025 / 2026 **全部 165 列，全部含 `firstPick`**（OE 已回填整个历史档案）。v1.0 记录的"本机四个快照全部 164 列、无 `firstPick`"是**陈旧快照上的观测，已失效**，见 §3.2.5
- 前 18 列固定为：`gameid, datacompleteness, url, league, year, split, playoffs, date, game, patch, participantid, side, position, playername, playerid, teamname, teamid, firstPick`，随后 `champion, ban1..ban5, pick1..pick5, gamelength, result`，之后全是赛后统计。**`firstPick` 插在 `teamid` 与 `champion` 之间——这正是"绝不按列位置索引"的原因**（§3.12.4）
- `pick1..pick5` 和 `ban1..ban5` **只写在队伍行**（participantid 100/200），选手行恒为空
- `date` 格式 `YYYY-MM-DD HH:MM:SS`（UTC，游戏创建时间）
- `patch` 是字符串，OE 用零填充（`'13.01'`）
- `playerid` 形如 `oe:player:38e0af7278d6769d0c81d7c4b47ac1e`，`teamid` 形如 `oe:team:...`。**一律用 id 不用 name**：2022 年有 3,921 个 distinct playerid 但 4,429 个 distinct playername，说明改名/同名很常见

### 3.2.5 【已证伪 → 正确版本】关于 `datacompleteness` 和 `firstPick`

**[已证伪] "LPL 100% 是 partial，所以要特殊处理整个 LPL"**

正确版本随年份变化，必须按年核实：

| 年份 | LPL partial 比例 | 说明 |
|---|---|---|
| 2022 | **100%**（1,572 个 team row 全 partial） | 此年成立 |
| 2026 | **19.09%**（613 场里 117 场 partial） | 已大幅改善；且 2026 年 LPL 是唯一含 partial 的赛区 |

**但结论不变，而且更强**：无论哪一年，`datacompleteness == 'partial'` 的行里，**BP 时点可见的信息 100% 完整**：

| 字段 | partial 行覆盖率 |
|---|---|
| champion / result / side / position / date | **100%** |
| patch | 99.5% |
| playerid | 98.4% |
| teamid | 98.5% |
| `pick1..pick5` | **0%（全缺）** |
| `goldat10/15/20/25` 等时点统计 | **0%（全缺）** |

**所以，绝对不要写这一行代码**：

```python
df = df[df.datacompleteness == 'complete']   # ✗ 2022 年这一行删掉整个 LPL
```

**但 v1.0 给的"正确写法"同样是错的**：

```python
# ✗ 仍然错：它作用在 12 行/场的长表上，会静默产出"只有 4 个选手的一方"
df = df.dropna(subset=['champion','side','result','patch','date','playerid','teamid'])
```

**[实测 2026-09-18]** 2026 全量：**675 / 8,832 场（7.6%）至少有一个选手行的 `playerid` 为空**，照这行 dropna 之后这 675 场**全部**只剩 <10 个选手行；另有 **380 场（4.3%）的 team 行 `teamid` 为空**。这些畸形场进 silver 后 `blue_player_ids` 会短于 5，`player_rating_diff` 的分母 `/5` 就错了，而模型照样输出一个数——**不报错，只让模型悄悄变差**。

**[正确写法] 把过滤顺序倒过来：先按 gameid 聚合判定完整性，整场保留或整场丢弃**

```python
def keep_complete_games(df: pd.DataFrame) -> pd.DataFrame:
    """df 是 12 行/场的长表。整场判定，绝不按行 dropna。"""
    def ok(g):
        return (len(g) == 12
                and g.loc[g.position != 'team', 'playerid'].notna().sum() >= 10
                and g.loc[g.position == 'team', 'teamid'].notna().all()
                and g[['champion', 'side', 'result', 'date']].notna().all().all())
    good = df.groupby('gameid').filter(ok)
    dropped = df.gameid.nunique() - good.gameid.nunique()
    log.warning("整场丢弃 %d 场；按 league × year 的明细写进 ingest 报告", dropped)
    return good
```

**silver 写出前的硬断言**：

```python
assert (df.blue_player_ids.str.len() == 5).all()
assert (df.red_player_ids.str.len()  == 5).all()
assert df.blue_team_id.notna().all() and df.red_team_id.notna().all()
```

被丢弃的场次数必须**按 league × year 打进 ingest 报告**（当前 2026 约 675 + 380 场）。冷启动侧"场内选手数 <10"的处理见 §5.5。

保留 partial（而不是 `datacompleteness=='complete'` 过滤）这件事本身仍然成立：2023 年可用样本因此从 9,232 → **10,897（+18%）**。

**并且禁止把 `datacompleteness` 本身当特征**——它与赛区强相关，等于偷偷泄漏赛区。

---

**[已证伪 → v1.1 再次修订] 关于 `firstPick`：本机快照上的观测 ≠ 数据源的不变量**

v1.0 写的是"本机 2022–2025 四个快照全部 164 列、无 `firstPick`，只有 2026 是 165 列"，并在其上建了三条"必须改"的硬指令。**这条被标 [实测·已复核] 的事实已经过期。**

**[实测 2026-09-18 · 直接取各年线上文件表头]**：2014 / 2019 / 2022 / 2023 / 2024 / 2025 / 2026 **全部 165 列，全部含 `firstPick`，且都插在第 18 列（`teamid` 与 `champion` 之间）**。OE 已经把这一列**回填进整个历史档案**。

于是 v1.0 那三条后果全部失效并撤回：(1)"与 `year==2026` 完全共线"不成立；(2)"无法在 2026 之前任何 fold 上验证"不成立；(3)"把 `firstPick` 当 baseline 协变量（对标 arXiv 2609.08060）在本地数据上立不住，必须改"——**这条是在指挥读者去推翻 SOTA 论文的 baseline 设计，而依据已不存在，撤回。**

**结论碰巧还成立，但理由完全换掉。** 真正该测的是**每年的 blue-first-pick 比例**，不是列是否存在。**[实测 2026-09-18]**：

| 年份 | blue 先手比例 | 含义 |
|---|---|---|
| 2025 | 2542 / 2542 = **100%** | `firstPick ≡ (side == Blue)`，与截距完全共线，**零信息** |
| 2026（全量） | 6132 / 8735 = **70.2%** | 解耦到 70/30，**才开始有信息量** |

所以正确表述是：**2026 之前 `firstPick` 与 side 完全共线、零信息；2026 起解耦，才值得用。** 做法不变——作为 **2026+ 的可缺失特征**，配 `has_first_pick` 指示列（2026 有 97 场为空），增量只在 2026 子集上评估。

**根因值得记住**：v1.0 把"本机陈旧快照上的观测"写成了"数据源的不变量"——这恰好是 §3.12.1 自己用来论证快照必要性的那个现象。纪律见 §0：schema 类 [实测] 事实必须带测量日期 + snapshot sha，并做成 `audit_columns.py` 的**每日复核**，而不是一次性断言。

同类跨年语义漂移（必须写 `audit_columns.py` 做专项审计，因为**任何统计方法都发现不了这些**）：

| 列 | 漂移 | 后果 |
|---|---|---|
| `league` | 2025: LCS → `LTA N`，CBLOL → `LTA S`，新增 `LCP`；**2026 又改回 `LCS` / `CBLOL`** | 不做双向 canonical 映射就静默丢掉整年北美/巴西比赛（两个方向都会丢） |
| `atakhans` | 2025 有值，2026 恒为 0（机制移除，列还在、非空） | 语义已死但看起来正常 |
| `turretplates` | 天花板从 15 变 45 | 同名不同分母 |
| `firstPick`（列本身 2014–2026 全有） | 2025 蓝方先手 **100%**，2026 降到 **70.2%** | 2026 之前与 `side` 完全共线、零信息；只有 2026+ 有增量 |

### 3.2.6 【已证伪】赛区集合与样本量

调研里出现了几组互相矛盾的 MAJOR 场次数字。**[实测·本机重算]** 的正确版本：

**2025 年唯一 league code 数 = 43；2024 年 = 51。**（词表本身在变，见 §3.11 的处理方式）

2025 年 ≥30 场的赛区（前 20）**[实测]**：

| 赛区 | 场次 | | 赛区 | 场次 |
|---|---|---|---|---|
| LPL | 805 | | PRM | 303 |
| LCK | 555 | | PCS | 303 |
| LCKC | 542 | | CD | 301 |
| LAS | 489 | | AL | 293 |
| EM | 440 | | LCP | 291 |
| LJL | 374 | | NACL | 276 |
| LFL | 318 | | LVP SL | 264 |
| LEC | 306 | | NLC | 249 |
| … | | | **LTA S** | 222 |
| **LTA N** | 213 | | VCS | 141 |
| WLDs | 96 | | MSI | 80 |
| FST | 35 | | EWC | 33 |
| **LTA** | **40** ← 注意 | | | |

**[已证伪] 三处**：

1. 调研里的 MAJOR 集合写了 `FS`（First Stand）——**不存在这个 code，实际是 `FST`** **[实测]**。
2. 2025 年 `LTA N` / `LTA S` 之外，**还存在一个独立的 `LTA`（40 场）**，三份方案的映射表都漏了它 **[实测]**。
3. **【v1.1 新增，最严重的一条】上面那张表是 2025 的词表，而 2026 已经改回去了。** **[实测 2026-09-18，2026 全量 8,832 场共 40 个 league code]**：有 **`LCS`（227 场）和 `CBLOL`（250 场）**，**没有 `LTA N`、没有 `LTA S`、没有独立的 `LTA`，也没有 `IC`**。v1.0 多处把严格 tier-1 写成 "LPL/LCK/LEC/**LTA N/LTA S**/LCP/PCS/VCS + MSI/WLDs/FST/**IC**/EWC"（§3.2.6、§4.2 G0 的 `is_international`、§6.8 分层维度 1），**照抄原始 code 过滤会把 2026 的 477 场 tier-1 比赛整体排除在主结论之外**——正是 §9.5 警告的"不做 canonical 映射会静默丢掉一整年北美比赛"那个坑，只是方向反了。

**2026 年出现的全部 40 个 league code 及 tier 归属** **[实测 2026-09-18]**（`LEAGUE_CANON` 必须逐个显式覆盖，见 §3.11.5）：

| tier | code（括号内为 2026 场次，缺省表示未逐个统计） |
|---|---|
| **2 · 国际赛** | `MSI`(71)、`WLDs`(8)、`FST`、`EWC` |
| **1 · 一线赛区** | `LPL`(674)、`LCK`、`LEC`、`LCS`(227)、`CBLOL`(250)、`LCP`、`PCS`、`VCS` |
| **0 · 二级 / 区域 / 青训 / 杯赛** | `LIT`(274)、`ROL`(257)、`HLL`(235)、`EBL`(213)、`RL`(208)、`LES`(193)、`LRS`(187)、`LRN`(176)、`HM`(171)、`LPLOL`(162)、`HC`(139)、`NL`(137)、`HW`(103)、`Asia Master`(63)、`KeSPA Cup`(61)、`CCWS`(40)、`AC`(19)，以及 `LCKC` / `LAS` / `EM` / `LJL` / `LFL` / `PRM` / `CD` / `AL` / `NACL` / `LVP SL` / `NLC` / `LLA` 等 |

（其中 17 个 code —— `LIT`/`ROL`/`HLL`/`RL`/`LES`/`LRS`/`LRN`/`HM`/`LPLOL`/`HC`/`NL`/`HW`/`EBL`/`Asia Master`/`KeSPA Cup`/`CCWS`/`AC` —— 在 v1.0 全文从未出现过，读者拿不到可直接用的映射表。）

**[已证伪] "宽定义赛区 → 4,952 场/年、每 patch 225–237 场、复杂度上限放宽 60%"**：实测宽定义（MAJOR + CBLOL/LLA/LJL/TCL）是 **3,480 场/年**（不是 4,952），每 patch mean 174 / median 183（不是 225–237），样本增幅 **+26%**（不是 +60%）。

**更重要的是这个"赛区范围"决策本身被高估了**。实测对比英雄搭配对的稀疏度：

| 口径 | 场次 | 观测到的同队 pair | pair 场次中位数 | ≤5 场占比 |
|---|---|---|---|---|
| CLAIM-MAJOR | 5,541 | 6,712 / 13,530 (50%) | **5** | 54% |
| WIDE（+4 赛区） | 6,960 | 7,310 / 14,196 (51%) | **5** | 53% |

**扩了 26% 样本，pair 中位数纹丝不动（5→5）。** 真正的瓶颈是 pair 级稀疏，不是总场次。所以赛区范围不是"比调参重要得多"的决策，不要在这上面纠结。

**本方案采用的口径**：

- **训练**用宽集合（全部赛区，含二级联赛），加 `league_tier` 特征。多出的样本对 player / champion 主效应有帮助。
- **评估主结论**只报严格 tier-1，且**一律用 canonical 值表达，绝不用任何一年的原始 code**：

  ```python
  TIER1_CANON = {'LPL','LCK','LEC','LCS','CBLOL','LCP','PCS','VCS'}   # 一线赛区
  INTL_CANON  = {'MSI','WLDs','FST','EWC'}                            # 国际赛
  # ↑ 'LCS' 同时覆盖 2025 的 'LTA N'；'CBLOL' 同时覆盖 2025 的 'LTA S'
  #   'IC' 在 2025/2026 都不存在，已从集合中删除
  ```

  全赛区结果只作 secondary analysis 单列。理由见 §6.3：把次级联赛堆进来提升的是"混合总体平均效应"的功效，不是产品要服务的 tier-1 效应的功效，而 draft 效应在 tier 之间本来就异质。
- **两条必须进 CI 的断言**（防止这个坑再来一次，实现见 §3.11.5）：`set(silver.league_raw) - set(LEAGUE_CANON) == set()`；以及"每年 canonical 后的 tier-1 场次不得比上一年掉超过 30%"（这条会直接抓到 `LTA N`/`LTA S` 消失）。

## 3.3 lolesports 实时 API（L1）

### 3.3.1 端点清单（全部本机实测 200）

```
base   = https://esports-api.lolesports.com/persisted/gw
header = x-api-key: 0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z    # 不带 → 403
镜像   = https://prod-relapi.ewp.gg/persisted/gw                # 实测同样可用

GET /getLeagues?hl=en-US                 → 48 个联赛
GET /getSchedule?hl=en-US[&leagueId=]    → 每页 80 events，pages.older/newer 分页
GET /getLive?hl=en-US                    → 当前 state=inProgress 的比赛
GET /getEventDetails?hl=en-US&id={matchId} → match.games[] 每个含 {id, number, state}
GET /getTeams?hl=en-US                   → 不带 id 返回全部 1,578 队 / 4,770 选手
```

关键 leagueId：`worlds 98767975604431411` / `msi 98767991325878492` / `first_stand 113464388705111224` / `lck 98767991310872058` / `lpl 98767991314006698` / `lec 98767991302996019` / `lcs 98767991299243165` / `lcp` / `pcs 104366947889790212` / `vcs 107213827295848783` / `lta_n 113470291645289904` / `lta_s 113475181634818701`。注意 slug 是 `cblol-brazil`、`ljl-japan`。

**`getTeams` 不带 id 一次返回全量**（1.55 MB，1,185 支 active 队伍、每名选手 `{id, summonerName, firstName, lastName, role}` + `homeLeague`）。这是选手选择器"自动带出常规首发"的现成数据源，也是 ID crosswalk 的桥。每周落地一次。

### 3.3.2 livestats window feed（实时阵容的唯一来源）

```
GET https://feed.lolesports.com/livestats/v1/window/{gameId}?startingTime={RFC3339}
无需任何鉴权
```

返回结构：

```json
{
  "esportsGameId": "...", "esportsMatchId": "...",
  "gameMetadata": {
    "patchVersion": "16.17.810.4348",
    "blueTeamMetadata": {
      "esportsTeamId": "103461966965149786",
      "participantMetadata": [
        {"participantId": 1, "esportsPlayerId": "110350224593953686",
         "summonerName": "MKOI Myrwn", "championId": "KSante", "role": "top"},
        ...
      ]
    },
    "redTeamMetadata": { ... }
  },
  "frames": [ ... ]   ← 赛中数据，绝不能进 pre-game 模型
}
```

**`gameMetadata` 一次性给全模型需要的所有输入。** 这正是产品要的"读取当前比赛及其阵容"。

### 3.3.3 四条硬约束（全部本机实测）

| # | 约束 | 实测证据 |
|---|---|---|
| 1 | `startingTime` 必须被 10 秒整除 | 传 `...T16:30:07Z` → 400，body: `{"errorCode":"BAD_QUERY_PARAMETER","message":"startingTime must be evenly divisible by 10 seconds. Was 2026-09-18T16:30:07Z"}` |
| 2 | 硬性 50 秒延迟下限 | off=20/30/45s → 400 `disallowed window with end time less than 50 sec old`；off=60s → 200（实测 lag 56.3s） |
| 3 | 未开始返回 **HTTP 204 空体** | game4/game5（unstarted）→ 204 / 0 bytes；`/details/` 端点同样 204 |
| 4 | **不带 `startingTime` 返回的是本局最早的帧，不是最新帧** | 16:47 请求返回的 frames 全是 16:24，落后 1401 秒，**但长得像当前状态** |

### 3.3.4 【关键修正】不要用 `state` 字段做触发器

这是一个很容易写错、错了之后前端会永远转圈的地方。

**[实测·已复核]** 某场 LEC Bo5 的 game4（`115548681803406325`）：

```
17:51:27Z  getEventDetails 里 state: unstarted → inProgress   ← 此时是局间休息 + BP，比赛没开始
17:53:06Z  两个独立轮询器以 2-3s 间隔开打 window  → 连续 204
18:11:30Z  仍然 204
18:12:17.959745Z  首次 200   ← 空窗持续 20 分 50 秒
           回查该局：最早帧 18:12:07.070Z（blueGold=0，读条画面）
                    游戏时钟 0  18:12:13.860Z
           即：端点可用时刻比开局还晚 2-4 秒
```

**结论**：

- `state == 'inProgress'` 标记的是"局间休息 + BP 开始"，不是"开打"
- **绝不能写 `if state == 'unstarted': skip`** —— 那会跳过整个真正该密集轮询的时段
- **唯一权威触发器是 window 端点的 204 → 200 跳变**
- 判断比赛结束以 `frames[-1].gameState` 为准，`schedule` 的 state 会滞后
- Bo5 后续局的 `teams[].side` 在未开始时是**占位值不可信**（边选权取决于前一局结果），必须以 window feed 的 `blueTeamMetadata.esportsTeamId` 为准

**另外，"BP 结束后 60 秒的可取窗口"这个假设不存在**，不需要去复测：BP 结束到开局之间是**纯空窗**。跨 12 个赛区实测的"最早帧 → 游戏时钟 0"间隔稳定在 **4–7 秒**（LCK 4s / LPL 5s / CBLOL 5s / LEC 6-7s / LJL 7s / PCS 7s / Worlds 2024 6s），而且这个窗口只在**回溯查询**时看得到，实时拿不到。

### 3.3.5 【意外收获】用 window 端点批量回填 ID crosswalk

对 **completed** 的 gameId 不带 `startingTime` 调 window，会返回开局首帧 + **完整的 gameMetadata**（实测 200 / 16,799 bytes）。

这意味着：**对所有历史 completed gameId 批量跑一遍，就能自动拿到 (esportsPlayerId, championId, role, esportsTeamId) 四元组，再按「同一场比赛、同一位置、同一英雄」与 OE 硬 join，自动生成 OE playerid ↔ esportsPlayerId 对照表。**

这比"名字规范化 + 模糊匹配"（实测裸匹配率仅 77.2%）可靠一个数量级，而且完全自动。**这是整个工程里性价比最高的一个技巧**，见 §3.11.3。

**⚠️ 但"零 fuzzy"只对约 80% 的样本成立，LPL 是例外**（§3.11.6 已实测这个事实，此处显式引用，v1.0 漏了这条交叉引用）：

- **有 Riot `platformGameId` 的赛区**：`OE.gameid ↔ lolesports gameId` 是确定性对照，join 零 fuzzy。
- **LPL**：OE 的 gameid 是腾讯格式 `11715-11715_game_1`，**没有 Riot platformGameId**。必须先用 `(date_utc ± 1d, 两队 canonical 名, game number)` 建立 gameId 对照表——**这一步是模糊的**；命中之后的"同场 + 同位置 + 同英雄"硬 join 仍然是确定性的。
- LPL 是最大单一赛区（2025 年 805 场、2026 年 674 场），所以必须写一个 **LPL 专用的对照构建脚本**，并把它的匹配成功率**单独**打进 `/health`（§3.11.6）。验收指标也要分赛区（§8 阶段 2 验收⑤）。

## 3.4 Data Dragon / CommunityDragon（L2）

```
https://ddragon.leagueoflegends.com/api/versions.json
  → 200，最新 16.18.1，完整版本列表可直接当 patch 主数据
https://ddragon.leagueoflegends.com/cdn/16.18.1/data/en_US/champion.json
  → 200，159,008 bytes，173 个英雄
https://ddragon.leagueoflegends.com/cdn/16.18.1/img/champion/{alias}.png
  → 方形头像，约 28 KB/张
https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-summary.json
  → 200，比 DDragon 更早跟进新英雄（按 id>0 过滤掉占位项）
```
全部 **[实测]**。

**英雄命名：21 个 alias ≠ display name**（调研里说 9 个，偏少）**[实测]**：

```
AurelionSol, Belveth, Chogath, DrMundo, JarvanIV, KSante, Kaisa, Khazix,
KogMaw, Leblanc, LeeSin, MasterYi, MissFortune, MonkeyKing(→Wukong), Nunu,
RekSai, Renata, TahmKench, TwistedFate, Velkoz, XinZhao
```

今天的实时 feed 里就直接出现了 `JarvanIV` 和 `KSante`。**名字对不上不会报错，只会让那个英雄的特征变成缺失值，而模型照样输出一个数。**

**处理规则（写死）**：

1. **内部英雄唯一 ID 一律用 ddragon 的数字 `key`**（Cho'Gath=31、Wukong=62、Jarvan IV=59）
2. 映射表由 `champion.json` **程序生成**写成 `champion_map.parquet`，一行手写都不要
3. 配一个"173 个英雄三体系往返映射"的单元测试，命中率必须 100%
4. **一个已验证的好消息**：对 2022–2025 全部 43,274 场求 `set(OE champion names) − set(ddragon display names)` = **空集**。即 `feed.championId --champion.json--> OE champion` 这条链路 100% 打通，零手工映射 **[实测]**

**图标离线化**：build 时把 173 张 PNG + `champion.json` 拉到 `frontend/public/dd/16.18.1/`，运行时只引本地路径。**这是"离线可用"的必要条件**，不要在运行时打 CDN。

## 3.5 Leaguepedia Cargo（L3，补充源，非硬依赖）

### 3.5.1 【已证伪】三处流传的错误说法

| 说法 | 核查结论 |
|---|---|
| "匿名 cargoquery 完全不可用，怎么重试都 ratelimited" | **[已证伪]** 匿名可用。空闲 75–90 秒后单发查询稳定成功，一次能拿满 500 行 |
| "必须申请 bot 用户组权限" | **[已证伪]** 只需注册账号 → `Special:BotPasswords` 自助勾选 Cargo 权限 → 确认邮箱。**全程自助无审批，5 分钟** |
| "bot 组 = 5,000 行/页" | **[已证伪]** 5,000 是 **admin 管理员**专属。普通账号（含 bot password）仍是 **500 行/页硬上限**，`limit=5000` 服务端直接报错 |

### 3.5.2 真实的限流行为（这才是"感觉永远被限流"的原因）

服务器自己公布的配额（`action=query&meta=userinfo&uiprop=ratelimits`）：`cargo-query: {anon: {hits:5, seconds:60}, ip: {hits:5, seconds:60}}`。

但**实测的真实行为比公布值更严**：

- 连发 6 条 → 第 2~6 条全部 ratelimited，且**毫秒级立即返回**
- 14 秒间隔连打 8 条（理论 ~4.3 次/分，低于公布的 5/60s）→ **0 成功 / 8 全败**
- 紧接着空闲 80 秒单发 → **仍然 ratelimited**

**机制：被限流的请求本身也会计数并续期惩罚。** 一旦打爆就进入长惩罚期。所以"多试几次"只会让情况更糟。

**正确姿势**：严格串行、单线程、每次**成功后** `sleep 75~90s`、遇 ratelimited 走指数退避（90→180→360→封顶 600s）、绝不并发。有效吞吐约 **500 行 / 75–90 秒 ≈ 2–2.4 万行/小时**。

另外：HTML 页面被 Cloudflare 挡（`cf-mitigated: challenge`），但**跟数据采集完全无关**——走 `api.php` 即可，连 wiki 正文都能用 `action=query&prop=extracts` 读出来。

### 3.5.3 可用表与 join

```
ScoreboardGames:  GameId, MatchId, Tournament, DateTime UTC, Patch,
                  Team1, Team2, Winner('1'/'2'), Team1Picks(逗号串,按 top/jng/mid/adc/sup),
                  Team1Bans, Team1Players, RiotPlatformGameId, Gamelength
ScoreboardPlayers: SP.Side 取 '1'/'2'
Tournaments:      Name, League, Region(International/China/Korea/...),
                  TournamentLevel(Primary/Secondary), IsPlayoffs, DateStart
PicksAndBansS7:   join on PicksAndBansS7.GameId = ScoreboardGames.GameId
```

**两个陷阱**：

1. 官方文档明确警告 **Row ID 会在 Cargo 数据重建时改变，不要当主键**。用 `GameId` / `OverviewPage` 做业务主键。
2. **`RiotPlatformGameId` 在刚入库的比赛里是空字符串** **[实测]**，与 OE `gameid` 的 join 会静默丢行。必须准备 `(date, team1, team2, game_number)` 的 fallback 并**监控 join 命中率**。

**本项目对 Leaguepedia 的定位：补充维度，不是硬依赖。** 用它补 first-pick 归属、选手转会历史、赛事层级、OE 覆盖不到的冷门赛事，以及交叉校验。**series_id 本地构造，不依赖它的 MatchId。**

## 3.6 HuggingFace `gptilt/lol-esports-matches`（L3，补充）

**[实测]** 下载 `games.parquet`（9,170,813 bytes）用 pandas 统计：

- shape = **(136,956, 17)**，日期范围 2011-06-18 → 2026-09-12（更新滞后仅几天）
- 列：`game_id, match_id, tournament, game_number, datetime_utc, patch, team_blue, team_red, winner, picks_blue, picks_red, bans_blue, bans_red, gamelength, riot_game_id, vod, source_url`
- `picks_blue/picks_red` 是**长度恰为 5 且按 top→jng→mid→bot→sup 排序**的英雄名列表，长度 != 5 的行数为 **0**（100% 完整）
- `patch` 非空 94.7%，`riot_game_id` 非空 **80.9%**（这是跟 OE 对接的钥匙）
- 2025 年 14,118 场、2026 年至今 10,640 场（都比 OE 多，因为覆盖更多层级赛事）
- 许可 cc-by-sa-3.0（继承 Leaguepedia）

```python
from datasets import load_dataset
ds = load_dataset('gptilt/lol-esports-matches', name='games', split='train')
```

**用途**：(a) 补 OE 缺的低级别赛事，扩大 champion embedding 预训练语料；(b) 交叉校验 patch / winner / 日期。

**缺点**：只有**队伍级** picks，**没有"哪个选手用哪个英雄"**。所以选手层特征仍然只能来自 OE。

**配套的 `gptilt/lol-esports-entities`** 含 `entity_aliases`（75,717 行）、`public_figures`（20,797 行）、`teams`（3,562 行），用于名字规范化，能省掉大部分手工 crosswalk 工作。

> ⚠️ **使用它做 embedding 预训练时的泄漏陷阱见 §6.9 第 9 条**——它覆盖 2011–2026，与你的评测期完全重叠，且与 OE 是同一批职业赛，不做去重 + fold 内重估就是把测试集直接喂进预训练。

## 3.7 Polymarket（L4，外部基准，v2 再做）

```
1. GET https://gamma-api.polymarket.com/public-search?q=LCK&limit_per_type=5
   → events，找 slug 以 'lol-' 开头的
2. event.markets[0]: slug='lol-hle1-t1-2026-09-02-game1',
   outcomes=["Hanwha Life Esports","T1"], outcomePrices=["0","1"]  ← 自带结算标签
   gameStartTime='2026-09-02 08:00:00+00', clobTokenIds=[...], volumeNum=1626192.38
3. GET https://clob.polymarket.com/prices-history?market={clobTokenId}&interval=max&fidelity=1
   → 419 个 {t, p} 点
4. 取 t < gameStartTime 的最后一个 p = 开赛前市场隐含概率
```
全部 **[实测]**，无需鉴权。注意必须用 curl / requests 带 User-Agent，urllib 裸调 403。

**覆盖范围**：扫描 2,000 个已结束的 esports event，其中 206 个 slug 以 `lol-` 开头，最早 endDate **2025-09-15**。所以它只是"近期子集上的参照上界"，不是全样本 baseline。在它上面比较时**必须把所有模型都在同一子集重算一遍**。

**判读规则**：arXiv 2609.08060 在 928 张匹配地图上与市场统计不可区分（市场仅微弱领先 ΔBrier=+0.0093, t=+2.24, p=0.025）。**如果你显著赢市场超过 0.015 Brier，第一反应应该是去找泄漏，不是庆祝。**

## 3.8 明确排除的数据源（写进 README，防止以后手痒）

| 源 | 排除理由 |
|---|---|
| **gol.gg** | 站内公告原文 "stats are not updated since 4th April because of a bug with Riot Match History"，已停更 5 个月；且 robots.txt 明确 `Disallow: /ranking/, /meta/`（最有价值的两个路径）**[实测]** |
| **GRID Open Access** | 页面明确只含 **CS2 和 Dota 2**，LoL 属单独付费的 Competitor Portals **[实测]** |
| **Bayes / BEDEX** | 不公开价格，B2B 商业授权；唯一免费口子是仅供联调的 "training wheels" 回放流 |
| **Riot LoL Esports Data Portal (LDP)** | 只面向 Riot 运营赛区和 Leaguepedia/OE 这类社区伙伴，无个人申请通道 |
| **solo queue 自采（v1）** | Riot dev key 限流 20 req/1s 且 100 req/2min、**24 小时失效**，官方明文 "You may not run your application for public consumption using a personal key"。采 50 万场是多周独立工程 + production key 审批不可控 |
| **lolalytics / u.gg / op.gg** | 沙箱实测全部 000/403，有反爬保护，ToS 不明 |
| **Kaggle `lauffing/oracles-elixir-...`** | 许可 **CC BY-NC-SA 4.0（NonCommercial）**，将来若想商业化要避开；直接用 OE 原始文件 |
| **GitHub LFS 镜像作为生产源** | 见 §3.2.2，陈旧 6–7 个月、schema 不一致、随时消失 |

## 3.9 样本量估算表

**OE 官方 manifest 的权威每年 games（全赛区）** **[实测]**：

| 年 | games | 年 | games |
|---|---|---|---|
| 2014 | 918 | 2021 | 12,305 |
| 2015 | 1,815 | 2022 | 12,529 |
| 2016 | 4,214 | 2023 | 11,106 |
| 2017 | 5,505 | 2024 | 10,195 |
| 2018 | 6,736 | 2025 | 10,038 |
| 2019 | 8,125 | 2026 (manifest 停更于 06-02) | 5,439 ⚠ **已过期，实测 8,832** |
| 2020 | 9,747 | **2014–2025 合计** | **93,233** |

**分层后的可用样本（这才是决定模型复杂度上限的数字）**：

| 口径 | 场次/年 | 近 24 个月 | 每 patch 场次 |
|---|---|---|---|
| tier-1（MAJOR + 国际赛） | ~2,800 – 3,100 | ~5,500 | mean 142 / median 130（剔除碎片 patch 后 157/143，range 8–285） |
| 广义 tier-1（+CBLOL/LLA/LJL/TCL） | ~3,480 | ~6,960 | mean 174 / median 183 |
| 全赛区 | ~10,000 | ~20,000 | mean 419 / median 446 |

**稀疏性硬约束（决定"能建什么、不能建什么"）** **[实测]**：

| 交互层 | 理论单元数 | 实际观测到 | 中位场次 | 结论 |
|---|---|---|---|---|
| player × champion（2025 MAJOR） | 675 × 159 = **107,325** | 观测仅 30,800 条 → **0.29 条/格** | 2 | 不可直接查表 |
| player × champion（2025 全赛区） | 28,213 个组合 | — | 2（均值 3.51） | 39.6% 的选择来自 <5 场的组合 |
| 同队 champion pair（173 英雄） | C(173,2) = **14,878** | 7,310 (51%) | **5** | 53% ≤5 场，必须低秩 |
| 跨队 counter pair | 173² = **29,929** | 6,701 (2025 Q1 一线) | **2** | 62.6% ≤3 场 |
| champion × position | 173 × 5 = 865 | 231（2025 Q1 一线） | 8 | 仅 77 个单元 ≥30 次 |

**跨赛区校准的样本瓶颈**（国际赛是唯一锚点）：2022 年国际赛 235 场中，**跨赛区仅 193 场**，桥接 43 个赛区对；最多的 LCK-LPL 只有 **30 场**。→ 赛区对之间的 log-odds 差 SE ≈ 2/√30 ≈ 0.37（约 **±65 Elo**）**[实测]**。

> ⚠️ **上表来自 `oe.datalisk.io` 的 manifest，而它已冻结在 2026-06-02**（§3.2.3）。**2026 的真实值是 8,832 场（实测 2026-09-18）**，不是 5,439。上表只作历史年份的粗略参照，**不得用于任何校验或断言**。

**⚠️ 一个必须立刻补的数据缺口**：本机旧快照 2025 数据最后一天是 **2025-11-16**，而今天是 2026-09-18，**缺了整整 10 个月的 2026 赛季**。好消息是它**现在就能自动补上**：Drive 直连可用，§8.0 的第 9–12 条命令半小时内跑完（2026 = 8,832 场 / 70.9 MB / `max(date)` 到今天）。一个立项前提是"侧重近期数据"的模型，其半衰期、收缩强度、赛区强度若全部在缺最近一整季的窗口上标定，**最大权重会压在已经过时的 meta 上**——所以所有超参网格必须在补齐 2026 后重跑。

## 3.10 数据字典：`silver/games.parquet`（一场一行，蓝方视角）

**A. 键与元数据（不进模型）**

| 字段 | 类型 | 来源 / 构造 |
|---|---|---|
| `game_id` | str | OE `gameid` |
| `series_id` | str | **6 小时会话切分**（§2.2 的 `build_series_id`），形如 `LCS::oe:team:a\|oe:team:b::20260124T2314`。⚠ 不是按日历日 |
| `kickoff_utc` | ts | OE `date`（UTC） |
| `asof_ts` | ts | 特征冻结时刻，必须 ≤ kickoff_utc |
| `league_raw` / `league_canon` / `league_tier` | str/str/int | 原值 / canonical 表映射 / {0=二级, 1=一级, 2=国际} |
| `patch_raw` / `patch_major` / `patch_minor` / `patch_idx` | str/int/int/int | 见 §3.10.1 |
| `patch_age_days` | float | kickoff − 该 patch 在数据中首次出现日 |
| `blue_team_id` / `red_team_id` | str | OE `teamid` |
| `blue_player_ids[5]` / `red_player_ids[5]` | list | OE `playerid`，按 top/jng/mid/bot/sup 排序 |
| `blue_champ_keys[5]` / `red_champ_keys[5]` | list[int] | ddragon 数字 key |
| `has_pick_order` / `has_ban_data` / `has_first_pick` | bool | 缺失指示 |
| `datacompleteness` | str | **仅记录，禁止进特征** |
| `data_snapshot_sha` | str | 该行来自哪个快照，用于复现 |
| `y_blue_win` | int8 | **label** |

**B–I 的特征列见 §4.2 完整特征表。**

### 3.10.1 patch 归一化（不做会静默算错衰减权重，且不报错）

各源编号完全不一致 **[实测]**：

| 源 | 2023 | 2025 | 2026 |
|---|---|---|---|
| OE | `13.01`（零填充） | `15.09` | `16.01` |
| Leaguepedia / gptilt | `13.1`（不填充） | `25.09` | `26.17` |
| Data Dragon | — | — | `16.18.1` |
| livestats feed | — | — | `16.17.810.4348` |
| **OE 私有 API `/patches`** | — | — | **JSON number → `16.10` 变成 `16.1`** |

即：**2025 年起 Leaguepedia 主版本号 = Riot 主版本号 + 10**；OE 零填充而 Leaguepedia 不填充，**字符串直接比较必错配**；OE 私有 API 返回的是 number 类型会丢零。

```python
def norm_patch(s) -> tuple[int, int]:
    """统一输出 (major, minor) 整数元组。注意：OE 私有 API 返回 number，必须先 str()"""
    parts = str(s).split('.')
    major, minor = int(parts[0]), int(parts[1])
    if major >= 25:          # 只对 Leaguepedia 系做 -10 修正
        major -= 10
    return (major, minor)
```

然后用 `versions.json` 给每个 patch 打上真实发布日期，`patch_idx` = 完整版本列表反转的序号。**时间衰减权重按"距今多少个 patch"算比按"多少天"更贴合版本变化的本质**（但两种都要在网格里试，见 §4.4）。

## 3.11 ID 规范化与三张映射表

**总原则：全部显式，绝对禁止 difflib / fuzzy 兜底。** 有实测先例：`difflib` 把 `Beijing JDG Esports` 静默映射成了 `LNG Esports`——**另一支队伍，无报错**。

### 3.11.1 表 1：`champion_map.parquet`

由 `champion.json` 程序生成，字段 `{ddragon_key(int, 内部主键), alias, display_name, oe_name, lp_name}`。配 173 个英雄的往返映射单元测试（见 §3.4）。

### 3.11.2 表 2：`role_map`（常量，5 条）

```python
ROLE_MAP = {'top':'top', 'jungle':'jng', 'mid':'mid', 'bottom':'bot', 'support':'sup'}
```
feed 用左边，OE 用右边。

### 3.11.3 表 3：`player_crosswalk.parquet`（最麻烦，也最关键）

OE 用 `oe:player:<hash>`（如 Faker = `oe:player:e1edfc5cea461399a63cb813cf795cc`），lolesports 用数字雪花 `esportsPlayerId`（如 Elyoya = `103877629120745120`）。**两套 ID 毫无关系，没有官方对照表。**

**三路构建，按优先级**：

1. **【首选，自动化，可靠性最高】用历史 completed gameId 批量回填**（见 §3.3.5）。对每个有 livestats 覆盖的历史 gameId 调一次 window，拿 `(esportsPlayerId, championId, role)`，与 OE 同场同位置同英雄硬 join。这是**确定性对齐，零 fuzzy**。
2. **`getTeams` 全量**给出干净 handle（不带队伍前缀，如 `Paz`、`NightSlayer`），作为规范化的参照。
3. **名字规范化兜底**：feed 的 `summonerName` **通常**形如 `'MKOI Elyoya'`（队伍缩写 + 空格 + ID）。

   **⚠️ 但 LPL 没有空格分隔符** **[实测 2026-09-18]**：完赛局 `116957100120526831`（WE vs JDG）返回的是 `['WECube','WEMonki','WEKaris','WEAbout','WEErha']`，而同期 LCS 是 `['SR Fudge','SR Contractz', …]`。v1.0 那句"去前缀后规范化"**在最大赛区上无法执行**，因为无法确定前缀边界。

   **正确做法：先用该队的 `code` 剥离前缀，再规范化。** `code` 从 `getTeams` 拿（`WE` / `SR` / `MKOI`），一次同时覆盖"有空格"和"无空格"两种格式：

```python
import unicodedata, re

def strip_team_code(name: str, team_code: str) -> str:
    return re.sub(rf'^{re.escape(team_code)}\s*', '', name, flags=re.I)

def norm_name(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))   # 去变音符
    return re.sub(r'[^a-z0-9]', '', s.lower())

# 单测必须含 LPL 真实样例：
assert strip_team_code('WECube', 'WE')        == 'Cube'
assert strip_team_code('SR Fudge', 'SR')      == 'Fudge'
assert strip_team_code('MKOI Elyoya', 'MKOI') == 'Elyoya'
```
配 `gptilt/lol-esports-entities` 的 `entity_aliases`（75,717 行）。剩余进 `configs/player_overrides.yaml` 人工维护。

**裸名字匹配率实测**：对 `getTeams` 的 top-12 赛区在册选手（644 人）vs OE 2024+2025 playername = **497/644 = 77.2%**；对一场真实 LEC 比赛的 10 名首发 = **9/10**（未命中的是 2026 才升上一线的新人）。所以路径 1 不是可选项，是必需。

### 3.11.4 【关键设计】覆盖率监控绝不能做成启动 assert

调研里两份方案都写了"crosswalk 匹配率 < 90% 则启动 assert 失败"或"映射命中率 < 100% 则 CI fail"。**这是设计错误**：转会期和新人首发**必然**把命中率打下来，结果是整站起不来。

**正确做法**：

```python
# ✗ assert coverage > 0.90        → 转会期整站崩溃
# ✓ 三层处理：
#   1. /health 暴露 gauge: crosswalk_coverage
#   2. 低于阈值 → 告警（日志 / 通知），不阻塞
#   3. 单个选手未命中 → 退化到「该赛区 × 该位置的 player rating 中位数」
#      并在 UI 上把该选手标黄 + API 返回 warning
```

**唯一该做成 CI 断言的是"确定性映射"**：英雄名往返映射必须 100%（因为它是程序生成的，不 100% 说明代码有 bug）。

### 3.11.5 league canonical 表（必须带 UNKNOWN 兜底桶）

**[实测]** 唯一 league code 数：2024 = **51**，2025 = **43**。词表本身不稳定，且 2025 年发生了破坏性改名。

**映射必须【双向且年份无关】地收敛**：2025 年 `LCS→LTA N`、`CBLOL→LTA S`，而 **2026 年又改回了 `LCS` / `CBLOL`** **[实测 2026-09-18]**。所以两边都要写，任何一年的原始 code 都必须落到同一个 canonical 值上。v1.0 只给了 4 个键就写"… 显式列全"，读者拿不到可用的表——下面是完整版（2026 实际出现的 40 个 code 全部覆盖）：

```python
# configs/league_canonical.yaml 的内容（此处以 dict 形式给出）
LEAGUE_CANON = {
    # ── tier 1：一线赛区（canonical 名固定，不随改名漂移）────────────────
    'LPL': 'LPL',     'LCK': 'LCK',     'LEC': 'LEC',
    'LCS': 'LCS',     'LTA N': 'LCS',           # 2025=LTA N；2024/2026=LCS
    'CBLOL': 'CBLOL', 'LTA S': 'CBLOL',         # 2025=LTA S；2024/2026=CBLOL
    'LTA': 'LTA(2025-only)',                    # 2025 独立存在的 40 场跨区赛，不并入上面两个
    'LCP': 'LCP',     'PCS': 'PCS',     'VCS': 'VCS',
    # ── tier 2：国际赛（注意：没有 'IC'，2025/2026 均不存在）─────────────
    'MSI': 'MSI',     'WLDs': 'WLDs',   'FST': 'FST',   'EWC': 'EWC',
    # ── tier 0：二级 / 区域 / 青训 / 杯赛（2026 实际出现的全部 code）─────
    'LCKC': 'LCKC',   'LAS': 'LAS',     'EM': 'EM',     'LJL': 'LJL',
    'LFL': 'LFL',     'PRM': 'PRM',     'CD': 'CD',     'AL': 'AL',
    'NACL': 'NACL',   'LVP SL': 'LVP SL', 'NLC': 'NLC', 'LLA': 'LLA',
    'TCL': 'TCL',     'LDL': 'LDL',     'LIT': 'LIT',   'ROL': 'ROL',
    'HLL': 'HLL',     'EBL': 'EBL',     'RL': 'RL',     'LES': 'LES',
    'LRS': 'LRS',     'LRN': 'LRN',     'HM': 'HM',     'LPLOL': 'LPLOL',
    'HC': 'HC',       'NL': 'NL',       'HW': 'HW',
    'Asia Master': 'Asia Master', 'KeSPA Cup': 'KeSPA Cup',
    'CCWS': 'CCWS',   'AC': 'AC',
}
LEAGUE_TIER = {                                  # canonical → {0 二级, 1 一线, 2 国际}
    **{k: 1 for k in ['LPL','LCK','LEC','LCS','CBLOL','LCP','PCS','VCS']},
    **{k: 2 for k in ['MSI','WLDs','FST','EWC']},
}                                                # 其余一律 0，由 .get(x, 0) 兜底

def canon_league(x: str) -> str:
    if x in LEAGUE_CANON:  return LEAGUE_CANON[x]
    log.warning("未知 league code: %s，归入 UNKNOWN 桶", x)   # 告警而非静默新增实体
    return 'UNKNOWN'
```

**两条必须进 CI 的断言**：

```python
# ① 新 code 出现时 CI 红，而不是悄悄进 UNKNOWN 桶
missing = sorted(set(silver.league_raw) - set(LEAGUE_CANON))
assert missing == [], f"未登记的 league code: {missing}"

# ② 覆盖率断言：每年 canonical 后的 tier-1 场次不得比上一年掉超过 30%
#    （这条会直接抓到"2026 年 LTA N / LTA S 消失"这类改名事故）
for y in years[1:]:
    cur, prev = tier1_games[y], tier1_games[y - 1]
    assert cur >= prev * 0.70, f"{y} 年 tier-1 场次 {cur} 相对 {y-1} 年 {prev} 掉了 >30%"
```

**绝不能用 `pd.get_dummies(df.league)` 现场生成** —— 新 code 出现时会静默新增一个维度，且训练/推理的列不一致。

### 3.11.6 LPL 的跨源 join 例外

**[实测]** OE 的 gameid 前缀分布：`LOLTMNT01/02/03/04/05/06_*` 是 Riot 的 platformGameId 格式，但 **LPL 完全不同**：形如 `11715-11715_game_1`，`url` 列指向 `https://lpl.qq.com/es/stats.shtml?bmid=11715`——说明 LPL 数据是从腾讯官方站抓的，**没有 Riot platformGameId**。

后果：非 LPL 赛区可以用 `OE.gameid ↔ Leaguepedia.RiotPlatformGameId ↔ gptilt.riot_game_id` 做精确 join；**LPL 只能走 `(date, team1, team2, game_number)` 模糊匹配**。LPL 是最大单一赛区（2025 年 805 场），这块必须单独统计匹配成功率。

## 3.12 增量更新流程与快照管理

### 3.12.1 存储布局（内容寻址，永不覆盖）

```
data/
  raw/<sha256>.csv.zst                    # 内容寻址，永不覆盖、永不删除
  manifest.jsonl                          # 每行一条 ingest 记录
  bronze/source=oe/year=YYYY/*.parquet    # 30 列白名单裁剪（164→30，文件小 5 倍）
  bronze/source=gptilt/*.parquet
  silver/games/year=YYYY/*.parquet        # 一场一行的 draft 宽表
  gold/features_pregame.parquet
  artifacts/<YYYY-MM-DD>/                 # 推理用的预计算包
snapshots/<snapshot_id>.json              # {snapshot_id, 各年 sha256, git_sha, created_at}
```

`manifest.jsonl` 每行：

```json
{"source":"oe","year":2026,
 "url":"https://drive.usercontent.google.com/download?id=1hnp...&export=download&confirm=t",
 "sha256":"...","bytes":70932929,"content_length":70932929,
 "n_rows":105984,"n_games":8832,"n_cols":165,"header_sha1":"...",
 "max_date":"2026-09-18 11:51:26",
 "drive_modified_raw":"12:03 pm",
 "drive_modified_parsed":null,
 "fetched_at":"2026-09-18T09:12:03Z","mirror_used":"drive_direct"}
```

> **字段说明（v1.1 修正）**：`drive_modified_raw` 是 Drive 目录页的**本地化显示串**（`Jun 8` / `Jul 2` / `12:03 pm`，无年份或无日期），**不是 ISO 时间戳、不可作身份键**（§3.2.1）。去重一律用 `content_length` + `sha256`；`max_date` 是闸 5 的新鲜度依据；`n_games` 是闸 5 的单调不减参照。

**为什么必须做快照**：OE 会**持续回填修订历史数据**。实测同一年份不同快照场次能差 4%（2024 年 GitHub 镜像 9,804 场 vs OE manifest 10,195 场）。**"重跑相同代码得到不同结果"是常态**，不锁快照的话报告里每个数字都不可复现。

不用 DVC：日期分区 + manifest 已足够复现，不引入 remote 的运维成本。

### 3.12.2 更新节奏

| 源 | 频率 | 说明 |
|---|---|---|
| OE 当年文件 | **每天 1 次**（建议 09:00 本地时间） | 官方原文 "Data files are updated ONCE PER DAY. There is no value in downloading the files more frequently than this." 且配额挂在维护者个人账号上，高频拉取会影响整个社区 |
| OE 历史年份 | 冷启动 1 次 + 每季度 diff 一次 | 会被回填修订 |
| `getTeams` | 每周 | 队伍/选手名单 |
| Data Dragon `versions.json` | 每天（几 KB） | 检测新 patch |
| Leaguepedia | 每天增量（按 `DateTime UTC` 分页，严格串行） | 补充维度 |
| gptilt | 每周 | 交叉校验 |
| livestats | 只在有比赛时：`getLive` 60s / `window` 30s | 见 §7.6 |

**重试策略：小时级退避（1h / 3h / 6h），不要秒级。** Drive 配额是按天恢复的，秒级重试纯属浪费且加剧问题。

### 3.12.3 每日任务流程

```
scripts/daily.sh
 1. GET versions.json                     → 检测新 patch
 2. 从 embeddedfolderview 重新发现 13 个 fileId（正则见 §3.2.1）
 3. 决定下哪些文件：
      · 当年文件（2026）：无条件下载
      · 历史年份：curl -sI 取 Content-Length，与 manifest 上次比对，变了才下
 4. curl 直连下载到 *.tmp（端点与完整命令见 §3.2.2）→ validate_oe_csv() 五重校验
    → 失败: 1h / 3h / 6h 退避，当天最多重试 3 次
    → 仍失败: 保留旧文件 + sys.exit(1) + 告警（绝不静默）
 5. 算 sha256：与 manifest 上次相同 → 不落新快照，直接结束
                不同 → os.replace() 原子替换 + 写 manifest.jsonl
                （含 content_length / sha256 / max_date / n_games）
 6. 用 gameid 集合差集找出新增场次 → 追加进 bronze/silver parquet
 7. 跑 store/audit_columns.py（schema / league 词表 / firstPick 比例的每日复核，§0）
 8. 重建 artifacts/<today>/
 9. 触发重训 + 门禁（见 §7.8）

★ v1.1 已从流程中移除：oe.datalisk.io 探针（参照系已冻结，见 §3.2.3）
★ v1.1 已从流程中移除：用 drive_modified 显示串做跳过判定（有歧义，见 §3.2.1）
```

**关键：第 4 步失败时绝不能继续往下走。** 宁可用昨天的模型，也不要用被 quota 页污染的数据训出的新模型。

### 3.12.4 Schema 演进的容错

```python
# ✓ 按列名取值
df = pd.read_csv(path, usecols=lambda c: c in WHITELIST_30)
# ✗ 绝不按列位置索引 —— OE 会静默增删列，而且会【回填历史年份】：
#   firstPick 插在第 18 列（teamid 与 champion 之间），2026-09 已回填进 2014-2026 全部年份，
#   同一个年份文件在不同快照下列数可能是 164 也可能是 165（§3.2.5）
```

缺列显式补空并告警，新增列默认忽略而非报错。加一个 schema 契约测试：断言 `gameid/datacompleteness/league/date/patch/side/result/champion/playerid/teamid/ban1..5` 这批必需列存在，缺任一就中断。

同时订阅 OE Discord 的 `oe-data-updates` 频道（`https://discord.gg/GsauAGAr6`）获知 schema 变更。

## 3.13 被核查证伪的内容汇总（速查）

| # | 流传的说法 | 正确版本 | 出处小节 |
|---|---|---|---|
| 1 | OE 的 S3 直链可用 | 两个桶都已 `NoSuchBucket` / `403 AccessDenied` | §3.2.2 |
| 2 | 用 GitHub LFS 镜像做自动兜底 | 不存在可靠镜像，现存的陈旧 6–7 个月且 schema 不一致；正解是 curl 直连 + 五重校验 + 自建镜像，Make-a-copy 只是连续 2 天失败时的人工兜底 | §3.2.2 |
| 3 | `rows % 12 == 0` 足以校验文件 | `0 % 12 == 0` 会放行空的 quota HTML；必须五重校验且含 `n_games > 0` | §3.2.3 |
| 4 | gdown 能绕过配额 | gdown 走的就是同一个端点，配额耗尽时它一样失败（但配额是**间歇性**的，直连通常可用） | §3.2.2 |
| 5 | 配额是本机 IP 被限流 | 三个互不相干的出口同错，是文件所有者账号的配额 | §3.2.2 |
| 6 | Leaguepedia 匿名完全不可用 | 匿名可用，空闲 75–90s 后单发稳定成功，500 行/次 | §3.5.1 |
| 7 | 需要申请 bot 用户组 | `Special:BotPasswords` 自助生成，5 分钟，无审批 | §3.5.1 |
| 8 | bot 组 = 5,000 行/页 | 5,000 是 admin 专属；普通账号含 bot password 仍是 500 行硬上限 | §3.5.1 |
| 9 | `firstPick` 只有 2026 文件有（并据此推翻 arXiv 的 baseline 设计） | **[2026-09-18 复核] v1.0 这条已过期**：OE 已把 `firstPick` 回填进 2014–2026 全部年份（全部 165 列）。真正的限制是 2026 之前 `firstPick ≡ (side==Blue)`（2025 实测 100%）、与截距共线零信息，2026 才解耦到 70.2% | §3.2.5 |
| 10 | LPL 100% partial（当成永久事实） | 2022 年成立，2026 年只有 19.09%；但"partial 保留全部 BP 信息"这个结论永远成立 | §3.2.5 |
| 11 | MAJOR 集合含 `FS` | 不存在，实际是 `FST` | §3.2.6 |
| 12 | 赛区映射只需 LTA N / LTA S | 2025 还有一个独立的 `LTA`（40 场）；且 **2026 已改回 `LCS` / `CBLOL`，`LTA N`/`LTA S`/`LTA`/`IC` 全部不存在**。tier-1 集合必须用 canonical 值表达，否则 2026 的 477 场一线赛被静默丢掉 | §3.2.6 |
| 13 | 宽赛区 = 4,952 场/年，复杂度上限放宽 60% | 实测 3,480 场/年，+26%；且对 pair 稀疏度零改善 | §3.2.6 |
| 14 | 选手×英雄残差 SD = 0.019 → 可以砍特征 | 那是 2022 单年点估计，95% CI = [0, 0.051]；四年合并 0.036–0.044。结论碰巧对但证据无效，必须换成样本外门禁 | §4.3 |
| 15 | pair 残差方差 ≤0 只是单年功效不足 | 四年 24k team-side 复现后结论更强：扣掉英雄主效应后点估计为负，95% CI 全负 | §4.5 |
| 16 | livestats 的 `state` 字段可做触发器 | state=inProgress 时 feed 可连续 204 达 20 分 50 秒；唯一触发器是 204→200 跳变 | §3.3.4 |
| 17 | BP 结束后有 60 秒可取窗口 | 不存在，BP 到开局是纯空窗 | §3.3.4 |
| 18 | L3→L5 的 ΔLL=0.00496 显著（t=3.38） | 那是 game 级聚类；但"修正多少"取决于尚未测量的 ICC → **【裁决待定】**，见下一行 | §6.3 |
| 19 | 闸 5 用 `oe.datalisk.io` 的 games 计数做新鲜度参照 | 该 manifest 已冻结在 2026-06-02（2026 games=5439），而真实文件 8,832 场，62% drift 会让 pipeline 每天拒收**最新鲜的正确数据**。改为单侧下界 + `max(date)` 自检 | §3.2.3 |
| 20 | Drive 必然返回 quota HTML，唯一路径是手动 Make-a-copy | 配额耗尽是**间歇性**的；2026-09-18 实测直连 200 / 70.9 MB / 真 CSV。直连是主路径，Make-a-copy 是兜底 | §3.2.2 |
| 21 | `series_id` 按 UTC 日历日构造 | 2026 实测 **89 个系列被零点劈开**（LCS/CBLOL 高发）。改为 6 小时间隔的会话切分 | §2.2 |
| 22 | 在 12 行长表上 `dropna(subset=[...])` | 2026 实测产出 **675 场（7.6%）只有 <10 个选手**的畸形样本。必须整场判定、整场保留或整场丢弃 | §3.2.5 |
| 23 | `n_eff = n_games / 2.5` | 从未测量、从未推导；等价于假设 series 内 **ICC=1**。实测 **m̄=2.14**；ICC 必须实测，而 block bootstrap 本来就给出正确的聚类标准误 | §6.3.1 |
| 24 | feed 的 `summonerName` 一律是"队伍缩写 + 空格 + ID" | **LPL 无空格**（`WECube`）。必须用 `getTeams` 的 team `code` 剥离前缀 | §3.11.3 |
| 25 | holdout 用"第一天枚举 gameid 列表"冻结 | 开发期内新产生的 holdout 期比赛不在列表里，会漏进训练集。改用日期截止线 `HOLDOUT_START` | §6.1.3 |

---

# 4. 特征工程

## 4.1 设计原则（三条硬约束）

**约束 A —— 泄漏**：BP 时点可见白名单以外的列，硬编码禁止进特征矩阵。见 §4.8。

**约束 B —— 稀疏**：任何"直接查表的高基数交互"必然是噪声。player×champion 平均每格 0.29 条观测，champion pair 中位 5 场。所有高基数量必须走 empirical Bayes 收缩或低秩分解。见 §4.3、§4.5。

**约束 C —— 时间尺度分离**：meta 漂移（半衰期 60–90 天）和队伍实力漂移（半衰期 ~180 天）是两个不同的过程，而"特征内聚合衰减"和"训练样本权重衰减"又是两个不同的对象。四者必须分开且联合调参。见 §4.4。

**贯穿三条约束的实现手段：单遍时序状态机。**

```python
state = init_state()
rows = []
for game in sorted(games, key=lambda g: (g.kickoff_utc, g.game_id)):
    rows.append(emit_features(game, state))   # ① 先读状态，写出这一场的特征
    update_state(state, game, game.result)    # ② 再用赛果更新状态
```

这不是"好习惯"，是**唯一能结构性杜绝未来信息**的写法。

**但 v1.0 那句"walk-forward 时不需要重算特征"只对一部分特征成立**，且与 §6.1.2 红线 3（"所有全局统计量必须在每个 fold 的训练段内重新估计"）直接冲突。下面这张分类表决定了回测的真实成本，**必须先看它再排期**：

| 类别 | 成员 | 为什么 | 每个 fold 的重估成本 |
|---|---|---|---|
| **fold 内可复用**（在线递归类：全局跑一遍状态机，所有 fold 共用） | team / player Elo、Glicko RD、side 先验的 as-of 累加器、h2h、mastery 计数、streak、rest days、champion EB 累加器的**计数部分** | 严格因果递归：第 t 场的值只依赖 `<t` 的比赛，与"训练段到哪结束"无关 | **0** |
| **每个 fold 必须重估**（依赖训练段的全局分布） | EB 收缩强度 `k`、`θ̄_pos` 基准、`L_region` / 分层 BT、champion embedding、standardizer 的 mean/std、Platt / temperature 校准器、target encoding | 用到了训练段的**聚合分布**，跨 fold 复用 = 用未来数据标定超参 | `k` / `θ̄`：秒级（矩估计）<br>standardizer / 校准器：亚秒级<br>**BT（SVI）：分钟级**<br>**embedding 预训练：十分钟级** ← 主要成本 |

**结论**：在线递归类特征确实"算一次用到底"；但 BT 和 embedding 让每个 origin 的成本从"亚秒级拟合"变成"分钟到十分钟级"。**而超参网格搜索会再乘一个 `N_grid`——计算预算见 §4.4.3，这是全项目最大的一笔开销，v1.0 完全没算。**

## 4.2 完整特征表

> 约定：所有 `*_diff` 一律 `blue − red`。所有带 ⏱ 的特征在物化时同时写一列 `<feat>_last_event_ts`。

### 组 G0：场次上下文（100% 可得，零聚合）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| `side_blue` | 常数 1（截距吸收） | — | — | 蓝方基准优势 |
| `league_tier` | {0 二级, 1 一级, 2 国际} | — | OE `league` + canonical 表 | 不同 tier 的可预测性与 rating 尺度不同 |
| `is_playoffs` | OE `playoffs` 列 | — | OE | 季后赛强度分布更窄、BO 更长 |
| `event_tier` | {0 二级, 1 常规赛, 2 季后赛, 3 国际赛} | — | OE `league`+`playoffs` | 比 is_playoffs 更细 |
| `is_international` | `league_canon ∈ INTL_CANON = {MSI, WLDs, FST, EWC}` ⚠ 用 canonical 值不用原始 code；`IC` 在 2025/2026 均不存在，已删 | — | OE + canonical 表 | 跨赛区场次需单独校准 |
| `game_in_series` | OE `game` 列 | — | OE | 后续局有前局信息 |
| `series_score_diff` | 本系列蓝方已胜局 − 红方 | **仅本系列已结束的局** | 自算 | 决胜局的状态不同 |
| `patch_age_days` | kickoff − 该 patch 首现日 | — | OE + versions.json | 新 patch meta 不稳定，模型应降低对英雄项的信心 |
| `has_pick_order` / `has_ban_data` / `has_first_pick` | 缺失指示 | — | OE | 缺失非随机（与赛区强相关），必须显式建模 |

### 组 G1：side 先验（收缩，as-of）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| ⏱ `side_prior_logit` | `logit((w_asof + k·p_glob_asof)/(n_asof + k))`，k∈{200,300,400} | expanding + 半衰期 180d | OE | 赛区蓝方优势实测 49.2%–60.4%，既不能硬编码也不能自由参数化 |

### 组 G2：队伍实力（**贡献 87.6% 的信号**）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| ⏱ `team_rating_diff` | `(TE[blue] − TE[red]) / 400` | 在线递归，半衰期 180d | OE | 实测单独一项把 LL 从 0.6912 拉到 0.6411 |
| ⏱ `team_rd_blue/red` | Glicko-2 的 RD | 同上，休赛期膨胀 | OE | "我有多不确定"本身就是有用的元特征 |
| ⏱ `h2h_decayed_diff` | 两队历史交手 decayed 胜率差，k=10 收缩到 0.5 | 半衰期 365d | OE | 克制关系的直接证据 |
| ⏱ `games_played_90d_diff` | 近 90 天场次差 | 90d | OE | rating 可靠度的代理 |

Elo 更新（**纯函数，训练与推理共用同一份代码**）：

```python
KT, KP, INIT = 24.0, 16.0, 1500.0          # 季后赛 K×1.15，国际赛 K×1.3
# ---- 预测时刻：先快照 ----
td = (TE[bt] - TE[rt]) / 400.0
pd_ = (sum(PE[x] for x in blue5) - sum(PE[x] for x in red5)) / 5.0 / 400.0
# ---- 比赛结算后才更新（顺序绝不能反）----
et = 1/(1+10**(-td));  TE[bt] += KT*(y-et);  TE[rt] -= KT*(y-et)
ep = 1/(1+10**(-pd_))
for x in blue5: PE[x] += KP*(y-ep)
for x in red5:  PE[x] -= KP*(y-ep)
# 休赛期不重置：r ← 1500 + 0.65*(r - 1500)，同时 RD 膨胀
```

> **注意**：不要直接用 Elo 公式出概率。把 Elo 当 logistic 的**连续特征**，让模型端到端学缩放。实测系数是 **1.28–1.33 而不是 1.0**，说明 400 分制对职业赛偏保守。这正是 arXiv 2609.08060 的做法。

### 组 G3：选手实力（rating 建在选手层，不是队伍层）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| ⏱ `player_rating_diff` | `(Σ5 PE[blue] − Σ5 PE[red]) / 5 / 400` | 在线递归，半衰期 180d | OE | 实测 player Elo (0.6362) 优于 team Elo (0.6411)，两者并用 0.6336 |
| ⏱ `player_rating_min_diff` | 双方最弱环节之差 | 同上 | OE | 5v5 里短板效应真实存在 |
| ⏱ `n_substitutes_blue/red` | 首发中不在该队近 10 场主力名单的人数 | 近 10 场 | OE | 换人时 team rating 失真 |
| ⏱ `days_since_roster_change` | 距上次首发变动天数 | — | OE | 新阵容磨合期 |
| ⏱ `roster_games_together_decayed` | 当前 5 人组合的 decayed 同场次数 | 半衰期 365d | OE | 团队默契的代理 |

**为什么建在选手层**：LoL 转会频繁，`teamid` 不变但阵容可能剧变。选手层 rating 转会后自动继承，而队伍层会滞后一整个赛季。`team_rating ≈ Σ(5 个 player rating) + chemistry 残差`。

### 组 G4：跨赛区校准

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| ⏱ `league_strength_diff` | `L_region(blue) − L_region(red)`，由全局分层 BT 联合估计 | **3 年滚动窗口** | OE（国际赛识别） | 同一 Elo 分在 LPL 和 LJL 不是一回事 |

见 §4.6。

### 组 G5：状态与赛程

| 特征名 | 口径公式 | 时间窗口 | 依赖源 |
|---|---|---|---|
| ⏱ `rest_days_diff` | 距上一场结束天数（>30 截断） | — | OE |
| ⏱ `games_last_7d_diff` | 近 7 天场次差 | 7d | OE |
| ⏱ `streak_diff` | 连胜为正、连败为负，±5 截断 | — | OE |
| `is_back_to_back` | 当日是否已打过 | 当日 | OE |

### 组 G6：英雄版本强度（**draft 层唯一稳定有增量的特征**）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| ⏱ `champ_meta_diff` | `Σ5 logit(θ̂_blue) − Σ5 logit(θ̂_red)` | 半衰期 **75d**（待重新验证） | OE **全赛区** | 实测 ΔLL = 0.0044，是 draft 层 88% 的贡献 |
| ⏱ `n_offmeta_picks_diff` | decayed presence < 2% 的英雄数之差 | 半衰期 75d | OE 全赛区 | 用于 what-if 的 OOD 警告 |

**【v1.1 修正：v1.0 的代码与它自己的设计意图相反】** v1.0 给的 `champ_strength` 里 `cs[(c,pos)]` 是**单一累加器**，结算时把所有赛区的**胜负**等权拍进同一个池子——那恰恰就是"借了 win rate"，与紧邻的设计要点 1 自相矛盾。而"只借 exposure"到底怎么落到 `p_hat = (w + k·0.5)/(n + k)` 这个式子里，v1.0 没给任何公式（是用全赛区的 n 调 k？分层收缩？还是只用全赛区估 pick 率当另一个特征？三种做法结果完全不同）。下面给出明确的两层 EB 公式并替换原代码：

```python
LAM_C = log(2) / 75.0          # 天，待在联合网格里重新确认

# ── 两层 EB 的确切公式 ───────────────────────────────────────────────
#   θ̂_{c,p} = ( Σ_{i ∈ tier1} w_i·y_i + k₁·θ̄_p ) / ( Σ_{i ∈ tier1} w_i + k₁ )
#
#   · 分子/分母里的【胜负】只吃 tier-1  → 同质池子，不被次级联赛胜率污染
#   · 全赛区池子只用来标定两样东西（这就是"只借 exposure"的确切含义）：
#       θ̄_p = 该 position 的基准 logit（全赛区样本估，稳得多）
#       k₁  = σ²_noise / σ²_effect，由【全赛区】样本的矩估计给出（当前标定 ≈476）
#   · 直觉：全赛区告诉你"这个位置的基准在哪、该收缩多狠"，但不告诉你"谁更强"

def champ_strength(champ_keys, positions, now, cs_t1, cs_all, k1, theta_bar):
    """cs_t1  : (c,pos) -> [w_wins, w_games, last_ts]   仅 tier-1 比赛更新
       cs_all : (c,pos) -> [w_wins, w_games, last_ts]   全赛区更新，仅用于标定
       k1     : 由 cs_all 的矩估计每个 fold 内重算（§4.1 分类表：必须 fold 内重估）
       theta_bar[p] : 该 position 的基准 logit，同样由 cs_all 估出"""
    total = 0.0
    for c, p in zip(champ_keys, positions):
        w_wins, n_eff, _ = decay_to(cs_t1[(c, p)], now, LAM_C)   # 先衰减到 now
        if n_eff > 0:
            logit_obs = logit_safe(w_wins / n_eff)
            logit_hat = (logit_obs * n_eff + k1 * theta_bar[p]) / (n_eff + k1)
        else:
            logit_hat = theta_bar[p]
        total += logit_hat
    return total
# 结算后：全赛区比赛更新 cs_all；tier-1 比赛【同时】更新 cs_t1 与 cs_all
#         蓝方 5 个 (c,pos) 各 +y，红方各 +(1-y)，games 各 +1
```

**备选（等价、更贝叶斯的写法）**：分层版本 `θ_{c,p} ~ N(θ^{allregion}_{c,p}, τ²)`——把全赛区估计当先验均值、只用 tier-1 数据更新，`τ` 由 §5.4 的离线 NumPyro 读出。**两者二选一，并在 §6.5 消融矩阵里加一行"全赛区 prior vs 纯 tier-1 prior"的对照**，让数据裁决。

**三个设计要点**：

1. **用全赛区（约 10 万场）标定 prior，胜负只吃 tier-1。** 次级联赛与主赛区的 pick 分布余弦相似度 0.93–0.98（meta 完全可迁移），但英雄**胜率**跨层级相关只有 **r = 0.312**。所以**只借 exposure，不借 win rate**——全赛区用来扩大有效样本、标定 `k₁` 与 `θ̄_p`，θ̂ 的分子仍然只在同质池子里估。**⚠️ §5.3.4 实测的 `β(champ_meta_diff) = 0.975` 是在 v1.0 的【单池等权】口径下得到的，换成两层 EB 后必须重新拟合，不要直接沿用。**
2. **按 (champion, position) 建 key，做加性分解** `θ_{c,p} = θ_c + δ_{c,p}`，δ 加强 L2（5–10 倍）。实测 162 个出场英雄里 140 个出现在 >1 个位置，但按"第二位置 ≥20%"标准真正的 flex 只有 **27 个**（Graves 0.53/0.46、Sejuani 0.66/0.32、Akali 0.67/0.33、Taliyah 0.79/0.21）。纯 champion×position 交叉会把 865 个单元的样本打散。
3. **收缩强度 k≈476 的来源**：empirical Bayes 矩估计，CHAMPION 层真实效应 SD 仅 **0.0229** → `k = p(1-p)/σ² ≈ 476`。**英雄层几乎全是噪声**，因为 draft 是自适应均衡（强势英雄会被 ban 或被针对，胜率被推回 50%）。对照：PLAYER 层 SD 0.1294 → k≈14，TEAM 层 SD 0.1330 → k≈13——**选手/队伍层才是真信号**。

### 组 G7：选手×英雄（**用计数，不用胜率**）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 为什么有用 |
|---|---|---|---|---|
| ⏱ `mastery_diff` | `Σ5 min(g_decayed,14)/14 / 5`，双方之差 | 半衰期 180d | OE | 0→14 场舞台局有效果、之后饱和 |
| ⏱ `n_first_time_champ_diff` | 首次在舞台上使用该英雄的人数之差 | 全历史 | OE | 首秀风险 |
| ⏱ `min_champ_games_diff` | 最生疏那一手的 decayed 场次之差 | 半衰期 180d | OE | 短板效应 |
| ⏱ `champ_pool_entropy_diff` | 近 20 场英雄分布的 Shannon 熵之差 | 近 20 场 | OE | pool 宽度 |
| 🖥 `p_shrunk(player,champ)` | 三层 EB 收缩胜率 | 半衰期 180d | OE | **仅 UI 展示，不进特征矩阵** |

见 §4.3 的详细论证。

### 组 G8：协同 / 克制（**必须先过可靠性闸门**）

| 特征名 | 口径公式 | 时间窗口 | 依赖源 | 状态 |
|---|---|---|---|---|
| `synergy_diff` | `Σ_{i<j∈blue}⟨u_i,u_j⟩ − Σ_{i<j∈red}⟨u_i,u_j⟩` | 预训练期 | 全赛区 + gptilt | **默认关闭**，预期被闸门砍掉 |
| `counter_score` | `(Σu_blue)ᵀ M (Σu_red)`，约束 `M = −Mᵀ` | 预训练期 | 全赛区 + gptilt | 条件启用 |
| `lane_counter` | `Σ_{5 positions} u_{blue,p}ᵀ M u_{red,p}` | 预训练期 | 全赛区 | 条件启用 |

见 §4.5。

### 组 G9：pick order / ban（可选，2026+ 才完整）

| 特征名 | 口径公式 | 依赖源 | 注意 |
|---|---|---|---|
| `first_pick_is_blue` | OE `firstPick` | OE **全年份均有此列**（已回填 2014–2026） | **2026 之前 `firstPick ≡ (side==Blue)`（2025 实测 100%）→ 与截距完全共线、零信息；2026 起解耦到 70.2% 才有增量。** 配 `has_first_pick`（2026 有 97 场为空），增量只在 2026 子集上评估 |
| `blue_b1_champ` / `red_r5_champ` | pick1（盲选）/ pick5（完全信息反选） | OE `pick1..5` | 覆盖率仅 81.2% 且缺失与赛区强相关 |
| `n_bans_on_opp_pool_diff` | 对手 5 个 ban 落在我方核心选手 as-of decayed pick top-5 的个数 | OE `ban1..5` | as-of 计算，注意时点 |

**pick order 语义已验证** **[实测]**：20,354 个 team-side 的 `{pick1..pick5}` 集合与实际出场 5 个 champion 完全一致（仅 26 个不一致 = 0.13% 脏数据，丢弃）；`pick1` 的位置分布 bot 8,161 / jng 5,446 / mid 3,087 / top 2,038 / sup 1,622——40% 首选是 ADC，符合 2022 职业赛优先级，佐证它确实是顺序而非任意排列。

**但 v1 不用它**：81.2% 覆盖率且缺失非随机，做重了等于偷偷给"有 pick order 数据的赛区"一个不同的模型。v2 再做，届时必须配指示列并**分层看有/无两组表现是否一致**。

## 4.3 选手-英雄的 empirical Bayes 收缩

### 4.3.1 用户最看重的这个特征，为什么不进模型

**这是全文最需要仔细说明的一个决策**，因为它直接否定了原始需求里的核心设想。

**稀疏性（硬事实，无争议）** **[实测]**：

| 口径 | 单元数 | 观测数 | 每格平均 | 中位场次 | ≤3 场占比 |
|---|---|---|---|---|---|
| 2025 MAJOR | 675×159 = 107,325 | 30,800 | **0.29** | 2 | — |
| 2022 主赛区 | — | — | — | 2 | **64.3%**（35.0% 只有 1 场） |
| 2025 全赛区 | 28,213 组合 | — | 3.51 | 2 | 39.6% 的选择来自 <5 场的组合 |

**信号强度（这里有一处必须纠正的证据错误）**：

**[已证伪]** 调研里反复引用的"扣掉选手基准后残差真实 SD = 0.019，等价 k≈700 场"，**这个数字站不住**：

- 那是 **2022 单年**的点估计，按选手 cluster bootstrap，**95% CI = [0, 0.0508]** —— 当时根本无法区分 0 和 0.05
- 分年重算：2023 SD=0.0512、2024 SD=0.0442、2025 SD=0.0253
- **2022–2025 合并：SD = 0.0436，bootstrap 95% CI = [0.0343, 0.0515]（不含 0）**
- 用 player-year 基准（去掉跨年状态漂移）：SD=0.0369；再扣掉英雄主效应（真正的交互）：SD=0.0357
- 而且该矩估计本身有**正偏**：把英雄标签在 player-year 内置换后重跑，零假设下仍得到 var 0.00021–0.00090（SD 0.015–0.030），即约 +0.0005 的向上偏差。观测值 0.00137 扣掉零分布均值后只剩 ~0.0008（SD≈0.029）

**所以"方差很小所以砍掉"这条推理链是无效的。** 必须换成唯一有效的依据：

**样本外检验（这才是判据）** **[实测]**。在线（严格时间前向、无泄漏）构造特征 + Elo 基线，滚动两次留出：

| 配置 | 2024 留出 | 2025 留出 |
|---|---|---|
| 基线 elo | LL 0.61695 / AUC 0.7165 | LL 0.62172 / AUC 0.7083 |
| + 英雄主效应 | 0.61482 / 0.7194 | 0.61958 / 0.7106 |
| **+ 选手×英雄残差（K=10/30, 半衰期 365d）** | **Δ = +0.00159 (t=+2.82)** | **Δ = −0.00060 (t=−0.81)** |
| + 同位对位（lane counter） | Δ = +0.00026 (t=0.42) | Δ = −0.00012 (t=−0.20) |
| + 同队 pair 协同 | Δ = −0.00005 (t=−0.46) | Δ = +0.00002 (t=0.34) |

**选手×英雄在两个留出期符号相反、净增益 ≈ 0**（而训练集 logloss 反而下降，典型过拟合签名）。这与另一组独立实测（Elo + 选手该英雄收缩胜率 = 0.6345 vs 纯 Elo 0.6336）一致。

**结论：不作为预测特征。但理由是"两折符号翻转、净增益≈0"，不是"方差小"。**

> 这个区别很重要：如果你用"方差阈值"做决策，下次有人重跑矩估计拿到 0.044，就会把这个没用的特征加回来。**用样本外门禁，结论才稳定。**

### 4.3.2 如果一定要用：正确的三层收缩公式

```python
p_hat = (Σ w_i·y_i + k1·p_player + k2·p_champ_pos + k3·p_global) / (Σ w_i + k1 + k2 + k3)
```

- `w_i = exp(-ln2·Δt/180d)`，且 **EB 收缩的有效样本量用 `Σw_i` 而不是原始 count**（这一点极易写错）
- `k` 由一次离线 NumPyro 分层模型读出方差成分换算：`k = σ²_noise / σ²_effect`
- 标定参考值 **[实测]**：champion k≈476、champion×position k≈416、player k≈14、team k≈13。player×champ 残差按合并后 SD=0.036–0.044 换算 → **k ≈ 129–193**（不是 700）

**可靠度换算**（这张表直观说明为什么它没用）：

| σ | k | 中位 3 场时的可靠度 | 可靠度 0.5 需要多少场 |
|---|---|---|---|
| 0.044 | 129 | 2.3% | 129 |
| 0.036 | 193 | 1.5% | 193 |
| 0.019 | 693 | 0.4% | 693 |

**中位 3 场的单元，可靠度只有 1.5–2.3%。** 收缩后它 98% 等于父层，而父层（player rating）已经在模型里了。

### 4.3.3 但它在产品上必须保留

**把 `p_shrunk` 做成纯 UI 展示字段**：

```
Faker · Azir    23 胜 9 负（原始 71.9%） → 收缩后 61%  ⓘ 样本量偏小
```

用户想看的那个数字还在，而且加了收缩和样本量提示后**比裸胜率更诚实**。这比把它塞进模型有价值得多。

## 4.4 时间衰减：两种衰减必须分开

这是整个特征工程里最容易搞混、也是对原始需求"侧重近期数据"回答得最准的地方。

### 4.4.1 两组互相冲突的证据

**支持快衰减**（meta 确实在快速漂移）**[实测]**：按 patch 计算英雄 pick 分布的余弦相似度随间隔天数衰减：

| 间隔 | patch 对数 | 余弦相似度 |
|---|---|---|
| 0–29 天 | 13 | **0.897** |
| 30–59 天 | 17 | 0.754 |
| 60–89 天 | 15 | **0.638** |
| 90–119 天 | 13 | 0.569 |
| 120–149 天 | 12 | 0.529 |
| 150–179 天 | 9 | 0.506 |
| 180+ 天 | 6 | **0.438** |

从 0.90 衰减到长期基线 ≈0.44，中点 0.67 出现在约 **75 天** → **meta 半衰期 60–90 天**。

**反对快衰减**（对训练样本整体加权反而有害）**[实测]**：

| 训练样本权重半衰期 | Elo+player 的 test log loss |
|---|---|
| 45 天 | 0.6351 |
| 90 天 | 0.6341 |
| 180 天 | 0.6336 |
| **365 天** | **0.6335** |
| 730 天 | 0.6336 |
| 无衰减 | 0.6336 |

另有独立来源的配对门禁实验直接**拒绝**了三种半衰期的时间衰减和滚动训练窗口，结论是 "**Data volume beats recency, decisively**"。

### 4.4.2 调和：它们是两个不同的对象

| | **特征内聚合衰减** | **训练样本权重衰减** |
|---|---|---|
| 作用对象 | 算聚合特征时给**历史比赛**加权 | 给训练集中**每一场比赛整体**加权 |
| 效果 | 让**特征值**反映"近期状态" | 让**模型系数**偏向近期 |
| 公式 | `w_i = exp(−λ_feat·Δt)`，聚合写成 `Σw_i·y_i / Σw_i`，**收缩分母用 `Σw_i`** | `sample_weight = exp(−λ_train·(T−t_game))` |
| 实测最优 | meta 类 **75d**，rating **180d**，熟练度 **180–365d** | **365d 或不加** |
| 为什么 | meta 真的在快速漂移 | 队伍强度漂移已被 Elo/EWMA 的递归更新吸收，再衰减 = 二次丢数据 |

**关键洞察**：漂移主要发生在"队伍强度"上，而这已经被在线 Elo 的递归更新（K 值）吸收了。所以**"近期性"应该放进 rating 的递归更新参数和特征内聚合的 λ 里，而不是对训练样本二次加权**。

**用户的"侧重近期数据"直觉：在 champion meta 层是对的（半衰期约 75 天），在训练样本权重上是错的。这两层必须分开说。**

### 4.4.3 联合网格搜索（不能分别调）

四个 λ 会互相影响，**必须在同一次 walk-forward CV 里联合搜**：

```
H_meta   ∈ {45, 75, 150, ∞}      天（champion 强度、offmeta 判定）
H_rating ∈ {90, 180, 365}        天（team/player Elo）
H_mastery∈ {180, 365}            天（选手熟练度计数）
H_train  ∈ {365, 730, ∞}         天（训练样本权重）
→ 完整 72 组；先跑粗网格 12 组（各取 2-3 档）
```

**同时试一个变体**：`H_meta` 按 **patch 数**而非天数计（`0.5 ** (patch_distance / half_life_patches)`），因为版本变化本质上是离散事件而非连续时间。

#### ⚠️ 计算预算：这是全项目最大的一笔开销，v1.0 完全没算

**每换一组 λ，整个单遍时序状态机都必须在约 9.3 万场上重跑一遍**（λ 改变的是**特征值本身**，不是模型系数），再乘以 ≥10 个 origin。§5.3.3 那句"sklearn 亚秒级，没有任何借口用重训太贵来偷懒"**只覆盖了拟合，完全没算特征重建**——读者会在第 4 周才发现网格搜索是主要成本，而 10 周排期没给它留时间。

```
总时长 ≈ N_grid × N_origin × T_state
  T_state = 单次全量状态机构建的 wall-clock  ← 【第 1 周就能实测，填进这里：______ 秒】
  粗网格：12 组 × 10 origin × T_state
  完整：  72 组 × 10 origin × T_state
```

**第 1 周测出 `T_state` 后立刻做这道算术。** 若超过一个通宵（约 8 小时），按顺序降级：

1. **固定 `H_rating=180` / `H_train=∞`**（这两档的网格差本来就低于 MDE，见 §6.3.4），只搜 `H_meta` 的 4 档 × `H_mastery` 的 2 档 = **8 组**。
2. 仍超 → 在 **20% 子样本**上搜粗网格定出量级，再在全量上只跑最优的 2–3 组确认。
3. 把 `T_state` 与网格总时长写进 `experiments/runs.jsonl` 的 `wall_clock_s`，每周复核。

### 4.4.4 【重要】最优半衰期本身就是一个交付物

**但必须带 MDE 约束。** 实测的半衰期网格差异是 **0.0004–0.0025 log loss**，而任何口径下的 MDE 至少是 0.004。**这些差异全部低于可检出下限。**

所以报告里应当这样写，而不是"75 天最优"：

> 在 H_meta ∈ {45, 75, 150, ∞} 上，walk-forward OOS log loss 分别为 0.6235 / 0.6230 / 0.6234 / 0.6255。
> 本网格 n_series = ____，MDE = ____。
> 除"无衰减"一档外，其余三档的差异（≤0.0005）**低于本样本量的可检出下限，不可分辨**。
> 可以支持的结论只有一条：**"完全不衰减"确实更差（Δ=0.0025，接近但未超过 MDE，标记为 suggestive）**。
> 实践上取 75 天，理由是它与独立观测到的 pick 分布相似度中点（~75 天）一致，而非因为它在网格上最优。

**这才是对用户问题的诚实回答。** 把 0.0004 的网格差写成设计结论，会让整个项目围绕噪声做决策。

## 4.5 协同/克制的低秩处理 + 准入闸门

### 4.5.1 为什么不能查表（三重证据）

**证据 1 · 稀疏性** **[实测]**：

| 层 | 理论空间 | 观测到 | 中位场次 | 最大 |
|---|---|---|---|---|
| 同队 pair（2025 Q1 一线，1,198 场） | C(152,2)=11,476 | 3,590 | **3** | 90，**没有一个达到 100** |
| 跨队 counter（同上） | — | 6,701 | **2** | 55 |
| 同队 pair（2022–2025 主赛区，11,879 场） | 13,861 | 8,779 | **7** | — |
| (champ,pos)×(champ,pos) | — | 6,343 | **2** | — |

30 场样本的胜率标准误约 9pp。中位 3 场 = 纯噪声。

**证据 2 · 信号量** **[实测·已复核]**：

- 只扣队伍年度胜率（宽松口径）：4 年 24,194 个 team-side，var = 0.000224（SD=0.015），95% CI 的 SD 上界 0.021
- **正确口径（扣队伍基准 + 双方英雄主效应）：var = −0.000456，bootstrap 95% CI = [−0.000614, −0.000293]，整段为负**
- 独立来源的 split-half reliability：**r ≈ 0.00**（695 对）

即：扣掉英雄各自强度后，**纯 pair 协同在四年主赛区数据上仍然测不出来**。

**证据 3 · 直接建模全面崩盘** **[实测，同一 SGD 优化器公平比较]**：

| 模型 | 参数量 | log loss |
|---|---|---|
| 仅 Elo | 2 | **0.6345** |
| + 英雄主效应 | 171 | 0.6585 |
| + 英雄×位置 | 528 | 0.6526 |
| **+ 队内协同 pair** | **9,990** | **0.7165** |
| **+ 跨队克制 pair** | **29,420** | **0.7756** |
| **仅协同+克制** | 29,418 | **0.8242** |
| FM rank=4 | ~700 | 0.6601 |
| FM rank=8 | ~1,400 | 0.6699 |

**最后三行都比"全预测 53%"的常数预测 0.6912 还差。** FM 的 rank 从 4 涨到 8 结果继续变差，是过拟合的典型签名。

### 4.5.2 准入闸门（在建模之前跑，让数据裁决）

**不要凭我的断言砍掉这个特征，也不要凭直觉加上它。** 建立一个闸门：

```
阶段 1：split-half reliability
  把数据对半分（按 series 随机分，不是按时间），两边各算一次统计量，求 Pearson r
  用 Spearman-Brown 修正到全样本：r_full = 2r / (1 + r)
  参考值（独立来源实测）：单英雄胜率 r ≈ 0.25-0.36
                        champion scaling index r ≈ 0.05-0.11
                        champion pair synergy  r ≈ 0.00 (695 对)
  门槛：r_full ≥ 0.20 才允许进入阶段 2；否则记录"不可测量"并砍掉

阶段 2：低秩建模 + 样本外两折门禁（§6.7）
  连续两个留出期 Δ 同号 且 series 级 t > 2 才准进主力模型
```

**预期结果：synergy 会在阶段 1 被砍掉**（r≈0.00）。counter 有一条独立的职业侧佐证值得试：iTero 实测打野 solo queue counter multiplier <0.93x 时职业胜率 **45.6%**、>1.07x 时 **54.4%**（每列 ≥500 场职业数据），且给出迁移强度"每 +100G 的 SQ counter 在职业约对应 +40G，即约 **4 折**"；**中路是唯一例外，关系反向**。

### 4.5.3 如果闸门通过：低秩结构

```python
# 每个英雄一个 d=8 维 embedding（最多 16）
synergy_blue = sum(dot(u[i], u[j]) for i<j in blue5)      # 10 对
synergy_red  = sum(dot(u[i], u[j]) for i<j in red5)
counter      = (u[blue5].sum(0)) @ M @ (u[red5].sum(0))   # M = A - A.T 反对称
lane_counter = sum(u[blue5[p]] @ M @ u[red5[p]] for p in range(5))
```

- **参数量**：173×8 + 28 = **1,412**（反对称 M 在 d=8 时只有 `d(d-1)/2 = 28` 个自由参数），对比查表的 29,929
- **反对称约束 `M = −Mᵀ` 的意义**：保证 counter 的方向性（A 克制 B 等价于 B 被 A 克制），这是领域知识注入
- **喂进模型的是 4 个标量**（`synergy_diff`, `counter`, `lane_counter`, embedding 范数和），不是 embedding 本身
- **必须在联合模型里与队伍强度同时估计**，绝不能先算边际胜率再喂进去——否则估到的是被 draft 自适应抹平的边际效应
- **embedding 按 champion 建，position 做加性偏移**：`u_{c,p} = u_c + ε_{c,p}`，ε 的 L2 权重设为 u 的 5–10 倍

### 4.5.4 zero-init α（最重要的一个工程技巧）

所有 draft 侧组件都乘一个**初始化为 0 的可学标量**再加到 logit 上：

```python
logit = (base_terms
         + alpha_theta * champ_meta_diff
         + alpha_syn   * synergy_diff
         + alpha_cnt   * counter_score)
# alpha_* 全部 init = 0.0
```

三重收益：

1. **保证最差退化为纯实力 baseline**（优化器会把没用的 α 留在 0 附近）
2. **一键回滚**：线上出问题就 `α := 0`，系统退化成 AUC 0.70 的可用模型，不用重新部署
3. **`α̂` 的 series 级 bootstrap CI 直接就是"BP 到底值多少胜率"这个问题的带区间答案** —— 这是整个项目最有科研价值的单个数字

### 4.5.5 【严重】预训练 embedding 的泄漏陷阱

调研里的一份方案主张"在 gptilt 13 万场 + OE 全赛区 9 万场上预训练 embedding，然后**冻结**，只在 tier-1 上学 4 个 α"。

**这是全折泄漏，冻结一点都没解决。** 因为：

- gptilt 覆盖 **2011–2026**，OE 全赛区语料同样覆盖你的整个评测期
- 预训练时，**被评测的那些比赛的胜负标签已经进了 `u_c`**
- 之后冻结、只学 4 个参数，泄漏一点没少，只是被藏进了一个"看起来参数量只有 4"的外壳里
- 而且 gptilt 与 OE 是**同一批职业赛**，不按 `gameid` / `riot_game_id` 去重就是把测试集直接喂进预训练

**正确做法（二选一）**：

```
方案 A（推荐，简单）：
  只用严格早于最早 walk-forward origin 的固定前置期预训练。
  例如 origin 从 2023-01 开始，则 embedding 只用 2016-2022 的数据训，此后冻结。
  代价：embedding 反映的是旧 meta。但由于 pick 分布相似度 180 天就衰减到 0.44，
        这基本等于说 embedding 只能表达"机制性"协同（控制链、射程、开团接团），
        而那恰恰是唯一可能跨版本迁移的部分。

方案 B（严谨，昂贵）：
  每个 fold 内用该 fold 训练段重训 embedding。
  代价：walk-forward 成本乘以 embedding 训练时间。

两方案共同要求：跨源按 (riot_game_id 或 规范化的 date+两队+局序) 去重。
```

**任何全局统计量——包括 embedding 预训练语料——必须在每个 fold 的训练段内重估或严格前置。** 这条要写进 CI。

## 4.6 跨赛区 Elo 校准

### 4.6.1 问题的硬约束

同一个 Elo 分在 LPL 和 LJL 完全不是一回事，而国际赛（MSI/Worlds/First Stand）恰恰是跨赛区对战。

**锚点极少** **[实测]**：2022 年国际赛 235 场中跨赛区仅 **193 场**，桥接 **43 个赛区对**：

| 赛区对 | 场次 |
|---|---|
| LCK–LPL | 30 |
| LCS–LEC | 14 |
| LCK–LEC | 13 |
| LEC–LPL | 12 |
| LCS–LPL | 11 |
| 其余 38 对 | 大多 <10 |

某两个赛区之间只有 ~20–30 场直接对局时，log-odds 差的标准误 ≈ 2/√25 ≈ 0.40，换算 Elo 约 **±70–80 分**。

### 4.6.2 【先把角色定死】BT 在本项目里是"rating 生产器"，不是主力模型

v1.0 让 BT 同时扮演了两个互斥的角色，读者无法判断第 5–6 周到底要建什么：本节把它写成一个**完整的独立模型**（"所有赛区的比赛在一个模型里联合估计"），而 §5.3.1 的主力 logistic 里 `league_strength_diff` 只是一个**输入特征**（§4.2 G4 也写"由全局分层 BT 联合估计"）。若 BT 已经联合估了选手潜变量 `r_i`，那主力 logistic 里的 `team_rating_diff` / `player_rating_diff` 就与它重复；若 BT 只用来产 `L_region`，那"联合估计、不做事后拉尺度"就无从谈起——把 `L_region` 拆出来喂给另一个 logistic，**本质就是两阶段**。

**本方案的选择（写死）：BT 作为 as-of rating 的【生产器】。**

```
每个 walk-forward origin：
  1. 在该 origin 的【训练段】上拟合一次分层 BT（NumPyro AutoNormal SVI）
  2. 导出快照 (r_player, L_region)  ← 这就是该 origin 的 as-of rating 表
  3. 主力 logistic 只消费两个标量：player_rating_diff、league_strength_diff
```

**这就是两阶段，我们明确承认并接受这个代价**：arXiv 2609.08060 实测一阶段 vs 两阶段的 ΔBrier 仅 **0.0027**，远低于本项目任何口径的 MDE（§6.3.3）；换来的是"主力模型仍然是 12–16 参数、可解释、能做 what-if 逐项分解"这个产品必需品。

**若要保留"BT 即模型"的路线**：把它作为对比表里**独立的一行 M2**，与 M1 并列而不是嵌套（§6.4.2），走同一套 walk-forward 裁决。

**BT 的 as-of 协议（必须写清，否则违反红线 3）**：`L_region` 用 3 年滚动窗口，**每个 origin 在训练段内重拟合**，绝不跨 origin 复用。成本见 §4.1 分类表（分钟级/origin）与 §4.4.3 预算；随机拟合的可复现性标准见 §4.7.1 的 T2 档 B 豁免。

### 4.6.2b 模型形式：单一全局分层 Bradley-Terry，不做事后拉尺度

```
logit P(A beats B) = (r_A + L_region(A)) − (r_B + L_region(B)) + β·side_prior_league
r_i       ~ N(0, σ_r²)      选手潜变量
L_region  ~ N(0, σ_L²)      赛区偏移，收缩先验
```

**所有赛区的比赛在一个模型里联合估计，`L_region` 只能被国际赛的数据识别出来，天然完成校准。** 这比"先各赛区独立算 Elo、再事后乘一个缩放系数"干净得多。

`σ_r` 的先验 scale 起点：arXiv 2609.08060 报告队伍强度在 log-odds 尺度上 **σ̂_θ = 0.5**。

**`L_region` 必须用 3 年滚动窗口**（Riot 官方 GPR 的做法：team Elo 用 2 年窗口、league Elo 用 3 年窗口、80/20 加权、国际赛驱动 league Elo、按赛区缩放 K factor）。

### 4.6.3 如果坚持用在线 Elo（v1 的简化路径）

两套 rating：

- **国内赛**：只更新 team/player Elo，`L_region` 冻结
- **国际赛**：同时更新两者，且 K 更大
- K 值建议：常规赛 20 / 季后赛 24 / 国际赛 32；`L_region` 的 K 取 team 的 1/4（因为它聚合整个赛区）
- 赛季间衰减而非重置：`r ← μ + ρ(r − μ)`，ρ ≈ 0.6–0.7；同时 rating 不确定度回升（Glicko-2 的 RD 膨胀 / TrueSkill 的 σ 膨胀天然做这件事——**阵容变动大时优先 Glicko-2 或 openskill，而不是裸 Elo**）

**v1 的最简做法**：不做单独的 league Elo，全赛区所有比赛在同一个 Elo 池里跑，让国际赛自然完成跨区锚定。代价是收敛慢、二级联赛队伍分数偏高——用 `league_tier` 特征 + `has_new_team` 警告兜住。第 5 周以后再升级到分层 BT。

### 4.6.4 UI 必须体现的不确定性

国际赛的预测置信区间要**明显放宽**，并在界面上说明原因：

```
⚠ 跨赛区对战。赛区强度锚点每年仅约 200 场（LCK–LPL 历史最多也只有约 30 场），
  该预测的不确定性显著高于区内赛。
```

## 4.7 point-in-time 正确性的工程做法

### 4.7.1 四个必须写进 CI 的测试

```python
# ---- T1  时间戳断言（最便宜也最有效的一道闸）----
# 每个聚合特征物化时同时写 <feat>_last_event_ts
for f in AGGREGATED_FEATURES:
    assert (df[f'{f}_last_event_ts'] < df['kickoff_utc']).all(), f"{f} 用了未来数据"

# ---- T2  未来删除不变性（golden test，最强的一道闸）----
# 随机抽 200 场历史比赛 G(kickoff=T)
for g in random.sample(games, 200):
    v_full = featurize(g, build_state(all_history))
    v_cut  = featurize(g, build_state([h for h in all_history if h.date < g.kickoff]))
    np.testing.assert_array_equal(v_full, v_cut)   # 确定性组件：必须 bit-identical
#
# ★ 分两档标准 —— 随机拟合的组件不可能通过 bit-identical（v1.1 修正）：
#   档 A · bit-identical：所有在线递归类特征（Elo / Glicko / h2h / mastery 计数 /
#          side 先验 / champion EB 累加器）。它们是纯确定性递归，没有任何借口不通过。
#   档 B · 固定 seed 下 bit-identical + 跨 seed max|Δlogit| < 1e-3：
#          分层 BT / SVI、champion embedding 预训练、bootstrap 系数。
#          （SVI 有随机初始化与随机小批量，逐位相等是不可能达到的要求。）

# ---- T3  train-serve skew golden test ----
# 用 mock 的 livestats window JSON 走线上代码路径
v_online  = api_path_featurize(json.load(open('tests/fixtures/window_sample.json')))
v_offline = offline_table.loc[SAME_GAME_ID].values
assert np.max(np.abs(v_online - v_offline)) < 1e-9
# 这条能抓出三类必然会发生的 bug：
#   role 命名 (bottom vs bot)
#   英雄名   (LeeSin vs Lee Sin, KSante vs K'Sante)
#   patch 解析 ('16.17.810.4348' -> (16,17))

# ---- T4  purged walk-forward + embargo ----
# 按 series_id 分组、fold 边界前后各 7 天 embargo
# 任一 fold 的 AUC > 0.78 -> CI fail
```

**T3 的真实事故案例**（值得记住）：某同类项目的 `api.py` 加载 2 年数据而 `train.py` 训练用 5 年，**没有任何报错**，唯一症状是选手-英雄熟练度统计系统性偏低；修复后"已知 player-champion 组合从 5,841 涨到 11,587"。这类 skew bug 不崩溃、只让模型悄悄变差。

### 4.7.2 artifact 自描述

```json
// artifacts/<date>/meta.json
{
  "feature_names": ["side_prior_logit", "team_rating_diff", ...],
  "code_version": "<git tree hash of src/lolwp/features/>",
  "train_start": "2019-01-01", "train_end": "2026-09-17",
  "n_games": 84213, "snapshot_id": "snap-2026-09-18",
  "model_sha": "...",
  "calibrator": null,            // 主力 logistic 不加校准层（§5.6.1 / §5.6.2）
  "calib_range": null            // ★ 不是常数！无校准层时为 null
}
```

服务启动时断言 `model.feature_names == artifact.feature_names` 且 `code_version` 匹配，**不一致拒绝启动**。`/health` 全部暴露——上面那个 skew 事故如果 `/health` 打印了 `n_games`，一眼可见。

**【v1.1 修正】`calib_range` 必须由 artifact 在训练时计算并写入，不得硬编码。** v1.0 把 `[0.18, 0.84]` 写死进 meta.json、`/predict` 响应和 UI 的"阴影死区"规范，但 §5.6.1 / §5.6.2 的结论是**主力 logistic 不加校准层**——既然没有 Platt，就不存在这个截断：UI 会画出一段并不存在的不可达区间，API 会在 `p>0.84` 时返回一条假 warning（而 §5.6.1 的可靠性表里 0.9–1.0 区间本来就有 35 个样本、预测均值 0.930）。这是本文最强调的那类错误（"不要假装 0.84 就是 0.84"）的**镜像版本：不要假装存在一个其实没有的边界**。

```python
meta['calibrator']  = None if calibrator is None else type(calibrator).__name__
meta['calib_range'] = None if calibrator is None else [
    float(calibrator.transform(0.0)), float(calibrator.transform(1.0))]
# 若确实想记录经验范围，另起一个字段 p_oos_range = [min(p_train_oos), max(p_train_oos)]，
# 并明确标注它是经验分布范围而非截断边界。
```

UI 侧**只在 `calib_range != null` 时**渲染死区阴影（§7.6.2 ②）；API 侧只在它非 null 且触顶时返回校准上界 warning。

### 4.7.3 特征层是训练与推理的唯一共用入口（含 loader）

```
src/lolwp/features/
  priors.py     build_priors(games_df, as_of: datetime) -> Artifacts
  encode.py     featurize(draft: DraftInput, art: Artifacts) -> np.ndarray[F]
                featurize_batch(drafts: list[DraftInput], art) -> np.ndarray[N, F]
  artifacts.py  读写 + feature_names/code_version 断言
  whitelist.py  BP 时点白名单 + 黑名单正则
  loader.py     ← 数据加载器也必须在这个包里
```

**loader 在包外就是上面那个事故的根因。**

## 4.8 泄漏黑名单

### 4.8.1 列级（硬编码进 `features/whitelist.py`，CI 里跑）

```python
import re

# 白名单：BP 时点可见的 OE 原始列
ALLOWED_RAW = {
    'gameid','datacompleteness','league','year','split','playoffs','date','game','patch',
    'participantid','side','position','playername','playerid','teamname','teamid',
    'champion','ban1','ban2','ban3','ban4','ban5',
    'pick1','pick2','pick3','pick4','pick5','firstPick',
}
# 注意：'result' 只能作 label；'datacompleteness' 只能记录不能进特征
#      （它与赛区强相关，等于偷偷泄漏赛区）

# 黑名单正则：约 60 列
FORBIDDEN_RE = re.compile(r'(at10|at15|at20|at25)$|^(opp_|golddiff|xpdiff|csdiff)')

# 黑名单手写清单（全部来自实测表头）
FORBIDDEN_EXPLICIT = {
    'gamelength','kills','deaths','assists','teamkills','teamdeaths',
    'doublekills','triplekills','quadrakills','pentakills',
    'firstblood','firstbloodkill','firstbloodassist','firstbloodvictim',
    'team kpm','ckpm',
    'firstdragon','dragons','elementaldrakes','infernals','mountains','clouds',
    'oceans','chemtechs','hextechs','dragons (type unknown)','elders',
    'firstherald','heralds','void_grubs','firstbaron','barons','atakhans',
    'firsttower','towers','firstmidtower','firsttothreetowers',
    'turretplates','inhibitors',
    'damagetochampions','dpm','damageshare','damagetakenperminute',
    'damagemitigatedperminute','damagetotowers',
    'wardsplaced','wpm','wardskilled','wcpm','controlwardsbought','visionscore','vspm',
    'totalgold','earnedgold','earned gpm','earnedgoldshare','goldspent','gspd','gpr',
    'total cs','minionkills','monsterkills','monsterkillsownjungle',
    'monsterkillsenemyjungle','cspm','url',
}

def assert_no_leakage(raw_cols_used: set[str]) -> None:
    bad = (raw_cols_used - ALLOWED_RAW) | (raw_cols_used & FORBIDDEN_EXPLICIT)
    bad |= {c for c in raw_cols_used if FORBIDDEN_RE.search(c)}
    assert not bad, f"泄漏列: {sorted(bad)}"
```

**外加 leak canary（一次性，验证检测灵敏度）**：故意往特征表放一列 `_leak_canary = gamelength`，跑一次训练确认 AUC 显著跳升（> +0.10），证明你的检测阈值确实灵敏，然后从生产管道删掉并在 CI 里断言它不存在。

### 4.8.2 非列级泄漏（更隐蔽、更常见，也更致命）

| # | 泄漏形式 | 实测虚高幅度 |
|---|---|---|
| 1 | 用整赛季 / 整 split 聚合的胜率 | AUC +0.041（0.6498 → 0.6908） |
| 2 | 随机划分 train/test | 再 +0.10（→ 0.7942） |
| 3 | 同一场的 12 行被拆到不同 fold | 标签直接泄漏 |
| 4 | **同一 series（Bo3/Bo5）的多局被拆到不同 fold** | 比 3 更严重（共享 BP 博弈与当日状态） |
| 5 | Elo 先跑完整个赛季再 join 回去（最常见的 bug） | 巨大 |
| 6 | 用 patch 上线后的**全 patch**聚合算该 patch 内早期比赛的英雄强度 | 中等 |
| 7 | 用"选手本赛季最终所属队伍"做特征（转会窗后才知道） | 中等 |
| 8 | scaler / 归一化在全量数据上 fit | 小但真实 |
| 9 | target encoding 不做 out-of-fold + as-of | 中等 |
| 10 | 用 `datacompleteness` 当特征 | 偷偷泄漏赛区 |
| 11 | **embedding 预训练语料覆盖评测期** | 见 §4.5.5，可能很大 |
| 12 | 跨源未去重（gptilt 与 OE 是同一批比赛） | 同上 |

---

# 5. 模型

## 5.1 选型第一性原理：参数量 / 样本量

**这条曲线是本项目全部模型决策的唯一依据** **[实测，同一优化器、同一数据]**：

| 参数量 | log loss | 每参数样本数 |
|---|---|---|
| 2（仅 Elo） | **0.6345** | 12,290 |
| 174（Elo + 英雄指示） | **0.6336** | 141 |
| 528（+英雄×位置） | 0.6526 | 47 |
| 9,990（+队内协同 pair） | 0.7165 | 2.5 |
| 29,420（+跨队克制 pair） | **0.7756** | 0.8 |
| 29,418（仅协同+克制） | **0.8242** | 0.8 |

**单调恶化，且在"每参数样本数 < 50"处开始崩溃。**

配合硬数字：tier-1 近 24 个月 5,500 场；每 patch 仅 142 场；**整个可解释信号总量只有 0.070 log loss / 0.028 Brier**。

**结论：主力必须是 10¹–10² 参数量级的正则化线性模型。不上深度学习不是保守，是样本量的算术结论。**

## 5.2 Baseline 阶梯（严格按序，每级必须通过准入门禁才准进下一级）

| ID | 内容 | 特征组 | 实测 / 预期 log loss |
|---|---|---|---|
| **L0** | 常数 0.5 | — | 0.6931 / Brier 0.2500 |
| **L1** | 训练窗口内蓝方基准率（**as-of，不是全样本**） | — | 0.6912–0.6924 |
| **L1b** | 按赛区收缩的 side 先验 | G1 | ~0.690 |
| **L2** | + team Elo | G2 | **0.6311–0.6411** / Brier 0.2208 / acc 63.5% / AUC 0.694 |
| **L3** | **+ player Elo ← 真正的对手，很可能就是最终答案** | G2+G3 | **0.6274–0.6336** / Brier 0.2189 / acc 64.7% / AUC 0.701 |
| **L4** | **+ champ_meta_diff ← 推荐的产品模型** | +G6 | **0.6230** / Brier 0.2171 / acc 64.9% / AUC 0.707 |
| L5 | + mastery_diff | +G7 | 0.6224（Δ=0.00496；**是否超过 MDE 取决于实测 ICC：ICC≥0.6 不显著、ICC<0.35 显著 →【裁决待定，第 3 周测完 ICC 再定】**，见 §6.3.4） |
| L6 | L3 + 英雄 170/171 维 one-hot | — | 0.6269 / 0.6342（**反面教材，变差**） |
| L7 | L4 + 低秩 draft（若闸门通过） | +G8 | 待验证 |
| E1 | CatBoost，同一套工程化标量 | — | 上限探索，预期 0.620–0.638 |
| E2 | zero-init residual DeepSet | — | 上限探索 |

**把这张表原样放进 README——它本身就是最有说服力的成果。**

**为什么产品模型【暂定】停在 L4**：L4→L5 的 `ΔLL = 0.00496` 落在 MDE 附近，而 MDE 取决于尚未测量的 series 级 ICC——ICC=1（v1.0 的隐含假设）→ MDE 0.0060–0.0071，不显著；ICC≈0.3 → MDE 约 0.0049，**裁决翻转**。**第 3 周先用 `eval/power.py::effective_n()` 实测 ICC，并用 series 级 block bootstrap 直接反解 MDE（§6.3.1），再定 L4 还是 L5。** 在此之前 `mastery_diff` 只做 UI 展示，且不得把"停在 L4"写成已定论。

## 5.3 主力模型：分组 L2 正则化 logistic regression

### 5.3.1 形式

```
logit P(blue win) =
    β₀
  + β_side  · side_prior_logit
  + β_team  · team_rating_diff
  + β_plyr  · player_rating_diff
  + β_rd    · (team_rd_blue − team_rd_red)
  + β_lg    · league_strength_diff
  + Σ_g w_g · x_context_g            (is_playoffs, event_tier, game_in_series, ...)
  + α_θ     · champ_meta_diff        ← zero-init
  [+ α_syn  · synergy_diff]          ← zero-init，仅在闸门通过时启用
  [+ α_cnt  · counter_score]         ← zero-init，同上
```

参数量约 **12–16**，训练样本 tier-1 约 15,000 / 全赛区约 60,000 → 样本每参数比 **1,000–5,000**。这个比例是整个方案能成立的原因。

### 5.3.2 分组 L2 的实现技巧

sklearn 的 `LogisticRegression` 只支持单一 `C`。要对不同特征组施加不同强度的 L2，用列缩放：

```python
# 推导：X̃β = Σ_g X_g·β_g/√λ_g ，令 γ_g = β_g/√λ_g ，
#       则 ½‖β‖² = ½ Σ_g λ_g‖γ_g‖²  → 等价于分组岭
X_scaled = np.hstack([X_g / np.sqrt(lam_g) for X_g, lam_g in groups])
clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                         max_iter=1000, fit_intercept=True)
clf.fit(X_scaled, y, sample_weight=w_time)
# 还原
coef = np.concatenate([clf.coef_[0][sl] / np.sqrt(lam_g) for sl, lam_g in slices])
```

**正确性验证（写成单测）**：所有 `λ_g` 设成同一值时，还原系数与直接 sklearn L2 的结果 `max|diff| < 1e-8`。

稀疏矩阵用 `scipy.sparse.csr_matrix`。

### 5.3.3 为什么是它（三条，两条是实测）

1. **实测已达 SOTA 档**。L3/L4 落在 0.6224–0.6336，与 arXiv 2609.08060 的 Brier 0.2207（2026-09 发表，5,135 场，与 Polymarket 统计不可区分）同档。
2. **参数效率正好在可学范围**。每加一层交互就掉到 <1 样本/参数，直接崩（§5.1）。
3. **产品属性决定性**。线性模型每个 pick 的贡献可直接读为 log-odds 加项，what-if 页面能显示 "+1.2% / −0.8%"，**GBDT 和 Transformer 给不了**；且对没见过的阵容组合，线性模型优雅外推，树模型会落到叶子给出荒谬常数。

**不用 elastic net**：你不需要变量选择（171 个英雄全都要留，而且它们已经被压成 1 个标量了），L1 只会随机砍掉低频英雄。要做"选择"的对象是"要不要加这一整组特征"，靠嵌套模型的配对检验决定。

**训练成本**：sklearn **拟合**是亚秒级，所以 walk-forward 每个 origin 全量重拟合完全可行，**没有任何借口用"重训太贵"来偷懒**。这也是选线性模型而非 GBDT 作主力的一个隐性理由。

**⚠️ 但"训练成本"≠"回测成本"**：真正的开销是**特征重建**（单遍状态机在约 9.3 万场上重跑）与 fold 内必须重估的组件（BT / embedding）。超参网格搜索会把它乘以 `N_grid × N_origin`。分类见 §4.1，预算算法见 §4.4.3。

### 5.3.4 实测的系数（供 sanity check）

train 2022-06 → 2024-12（MAJOR 口径）：

```
β₀ = +0.164
β(team_rating_diff)   = +1.282
β(player_rating_diff) = +1.330
β(champ_meta_diff)    = +0.975
β(mastery_diff)       = +0.603
β(is_playoffs)        = -0.023
```

两个 rating 项的系数各约 1.3（不是 1.0），印证了 §4.2 的说明：**Elo 的 400 分制对职业赛偏保守，必须让 logistic 端到端学这个缩放。**

## 5.4 探索模型（各跑一次，做好"无显著提升"的心理准备并诚实报告）

### E1 · CatBoost（`catboost 1.2.10`）

```python
CatBoostClassifier(
    has_time=True,            # ★ 必须，强制 ordered target statistics 按时间序
    one_hot_max_size=8,
    max_ctr_complexity=1,     # 限制在 1-2
    depth=5, l2_leaf_reg=20,
    iterations=500, learning_rate=0.03,
)
```

**输入只能是工程化后的标量**：`team_rating_diff`、`player_rating_diff`、`team_rd_*`、`champ_meta_diff`、`mastery_diff`、`patch_age_days`、`league_tier`、`event_tier`、`game_in_series`、`n_substitutes_*`、`rest_days_*`。

**绝不直接喂 `playerid` / `champion` 当 category**：2,560 个 playerid × 每人中位 26 场，target statistic 方差巨大；而且这本质上和 29k 参数 pairwise 模型是同一个失败换个包装。

（若用 LightGBM 4.7.0 更危险：默认 `min_data_per_group=100` 会把大量低频英雄/选手合并进同一组，`max_cat_threshold=32`、`cat_smooth=10.0` 都必须往强正则方向大幅调。）

**诚实预期**：落在 0.620–0.638，与 logistic 的差距远小于 MDE，即**测不出差别**。若真如此，选 logistic——它能给可解释性，GBDT 不能。

### E2 · zero-init residual DeepSet

10 个 `(player, champion, role)` token → φ = 2 层 16 维 MLP → sum pooling → ρ 线性 → 乘 α（init=0）加到 logistic 的 logit 上。参数量控制在 3–5k。

zero-init 保证最差退化为 baseline。**这是简历/论文上最好看的诚实负结果**——用 rolling-origin 配对检验报告"没有显著提升"，本身就是一个有价值的结论。

### 离线定标工具（跑一次，不上线）· 分层贝叶斯

`numpyro 0.22.0` + **AutoNormal SVI**（不是 NUTS——NUTS 在 3–5 千个随机效应上要跑几十分钟到几小时，且无法在线更新），或 `bambi 0.21.0` 一行起步：

```python
result ~ side + (1|league) + (1|team) + (1|player) + (1|champion) + (1|player:champion)
```

**用途只有两个，都不是线上服务**：

1. 读出方差成分 `σ_player / σ_champion / σ_player:champ` 的后验 → 换算 EB 收缩强度 `k = σ²_noise/σ²_effect` → **硬编码进线上快速模型**。当前标定值：champion k≈476、champion×position k≈416、player k≈14、team k≈13、player×champ k≈129–193。
2. 画英雄/选手效应森林图，作为网站的"数据洞察"内容。

**这是把贝叶斯的严谨性以零推理成本带进生产的正确方式。**

### 明确不做（写进 README，防止反复）

| 方案 | 为什么不做 |
|---|---|
| 显式 pairwise 查表 | 实测 0.7756，比常数预测还差 |
| FM 库依赖 | `xlearn` 停在 2018-12、`torchfm` 停在 2020-01，py3.12+ 大概率编译失败。要做就 PyTorch 手写 30 行 |
| player/champion entity embedding + MLP 从头训 | 参数量 4 万 vs 样本 3 万 |
| Transformer 主力路线 | 最小配置 3–5 万参数，而它要学的恰是信号最弱的那部分 |
| **stacking / 学权重的集成** | 需再切一层 OOF 缩短有效训练窗口；即使真带来 0.002 提升也需 ~31,000 场才能检出；且摧毁 what-if 页面赖以生存的可解释性 |
| 独立的 Elo "模型" | Elo 应该是 logistic 的输入特征，直接用 Elo 公式出概率会丢掉那个 1.28 的缩放系数 |
| solo queue 自采（v1） | dev key 24h 失效 + 100 req/2min，采百万场是多周工程 + production key 审批不可控。**v2 再说** |
| conformal prediction | 二分类下给的是标签集合 `{}/{蓝}/{红}/{蓝,红}`，对"胜率是多少"没用 |

**唯一允许的"集成"**：2–3 个结构不同模型的**固定等权 logit 平均**，不学权重。成本几乎为 0，降低单模型崩掉的风险，解释性退回主模型。且必须满足 `ΔLL > MDE` 才上线。

## 5.5 冷启动与新英雄新选手

**冷启动是常态不是例外**：43% 的选手整季不到 20 场；每年转会期后大量换队；新赛季有新战队；每个大版本有新英雄。**回测里大部分主体是见过的，线上不是。**

| 场景 | fallback | 额外处理 |
|---|---|---|
| 未见过的 `playerid` | 该赛区 × 该位置的 player rating **中位数**（不是 1500） | UI 标黄 + API warning；二级联赛升上来的应给更低先验 |
| 未见过的 `teamid` | 该赛区 team rating 中位数 − 半个标准差 | `has_new_team` 特征；Glicko RD 设为最大 |
| 未见过的 `champion` | `θ = 0`（收缩到全局 0.5），`n_eff = 0` | 新英雄上线头两周强制显示低置信度 |
| 首发大幅变动 | Glicko RD 膨胀 + `n_substitutes` 特征 | 临时提高 K 值 |
| 未知 league code | `UNKNOWN` 桶 + 全局先验 | 告警 |
| **场内选手数 < 10**（`playerid` 根本缺失，不是"没见过"） | **离线：整场丢弃**（§3.2.5 的 `keep_complete_games`）<br>**线上 / what-if：缺的位置用"该赛区 × 该位置 player rating 中位数"补位，`player_rating_diff` 分母仍为 5** | 2026 实测 675 场缺选手行、380 场缺 team 行；API 返回 warning，UI 把该位置标黄 |

**这条 fallback 路径必须写单元测试**，且在评估时切出"至少含 1 名训练集未见选手"的子集单独报指标（§6.8）。

## 5.6 概率校准

### 5.6.1 主力 logistic 大概率不需要额外校准层

**[实测]** L3 在 6,145 场测试集上的可靠性曲线：

| 预测区间 | n | 平均预测 | 实际频率 | 偏差 |
|---|---|---|---|---|
| 0.0–0.1 | 31 | 0.075 | 0.032 | −0.043 |
| 0.1–0.2 | 168 | 0.159 | 0.202 | +0.043 |
| 0.2–0.3 | 391 | 0.256 | 0.292 | +0.036 |
| 0.3–0.4 | 776 | 0.354 | 0.369 | +0.015 |
| **0.4–0.5** | 1186 | 0.453 | 0.433 | −0.020 |
| **0.5–0.6** | 1432 | 0.550 | 0.548 | −0.002 |
| **0.6–0.7** | 1172 | 0.647 | 0.639 | −0.008 |
| **0.7–0.8** | 681 | 0.745 | 0.767 | +0.022 |
| 0.8–0.9 | 273 | 0.843 | 0.832 | −0.011 |
| 0.9–1.0 | 35 | 0.930 | 0.971 | +0.041 |

**中段（0.3–0.9，占 92% 样本）偏差全在 2pp 以内，ECE = 0.0174。** 两端偏差在 n=31/35 的样本上毫无统计意义。

原因：**用 log loss 直接训练的 logistic 天生就是校准的**（arXiv 报 walk-forward calibration slope 0.995，"natively calibrated out of sample"）。

**但必须测了才说，不能假设。**

### 5.6.2 方法选择由校准集样本量决定，不由方法优劣决定

**[实测的方法-样本量交互，非常重要]**：同一个 isotonic vs Platt 的改动，

| 校准集规模 | 胜者 | 配对 t |
|---|---|---|
| ~35,000 个 in-game 样本 | **isotonic** | t = +6.91 |
| 1,716 场 pre/post-draft 样本 | **Platt** | t = −4.74 |

**你的 BP 后模型属于后者量级 → 默认 Platt，绝不用 isotonic**（几千场校准集上它会严重过拟合台阶）。

| 情况 | 方法 |
|---|---|
| 主力 logistic | **不加校准层**（但持续监控 slope） |
| CatBoost / DeepSet | **单参数 temperature scaling**（最省数据），其次 Platt（`CalibratedClassifierCV(method='sigmoid', cv='prefit')`） |
| 需要三参数的折中 | beta calibration（`netcal 1.4.0`） |

### 5.6.3 反面教材必须记住

某先例的未校准模型 **Brier 0.2536，比"永远输出基准率"的常数预测器 0.2495 还差**，而 accuracy 看起来完全正常。Platt 后 0.2276、ECE 0.119 → 0.021。**accuracy 根本检测不出这个失败。**

### 5.6.4 Platt 的截断区间必须在 UI 上诚实表达（**仅适用于需要 Platt 的探索模型**）

> ⚠️ **适用范围（v1.1 补）**：本条只适用于 **CatBoost / DeepSet 这类加了 Platt 的探索模型**。**主力 logistic 不加校准层（§5.6.1 / §5.6.2），因此根本不存在这个截断区间**——对它 `calib_range = null`，UI 不得渲染死区、API 不得返回触顶 warning。v1.0 把 `[0.18, 0.84]` 写死进 meta.json / `/predict` 响应 / UI 规范，是"假装存在一个其实没有的边界"，与本节要防的错误恰好互为镜像。

对**确实加了 Platt 的模型**，它会把输出范围截断到约 **[0.18, 0.84]**（具体值由训练时实测写入 `calib_range`，不是常数）。这是真实缺陷不是 bug：

- UI 在 `calib_range != null` 时把不可达区间画成**阴影死区**
- API 触顶时返回 `warning: "预测概率已触及模型校准上界，真实概率可能更高"`
- **不要假装 0.84 就是 0.84；同样，也不要假装存在一个 0.84**

### 5.6.5 分层校准比全局校准重要

跨区 rating 锚定不准会在国际赛上产生**系统性偏差**。必须分别检查并单独校准：

- tier-1 vs tier-2
- 常规赛 vs 季后赛
- **国际赛 vs 区内赛**（单独一层）

用分组 ECE 和分组 calibration slope 监控。

### 5.6.6 不确定性区间

**不要用标准 conformal prediction**（见 §5.4）。两个可行方案：

**方案 A（推荐）· bootstrap ensemble**

```python
# 每个 walk-forward origin 上，对训练集做 20 次 series 级 bootstrap 重采样并重拟合
# logistic 亚秒级 × 20 = 秒级，成本可忽略
coefs = [fit(resample_by_series(train)) for _ in range(20)]
ps = [sigmoid(c @ x) for c in coefs]
p_lo, p_hi = np.percentile(ps, [10, 90])
```
同时给出"区间宽度主要来自哪"（team rd / 阵容支持度），这对 UI 很有价值。

**方案 B · Venn-ABERS**（`venn-abers 1.5.4`，2026-08 活跃维护），给 `[p_lo, p_hi]` 且有有效性保证。

## 5.7 依赖版本（2026-09-18 PyPI 实测）

**可用且活跃**：`catboost 1.2.10`、`lightgbm 4.7.0`(py≥3.10)、`xgboost 3.4.1`(py≥3.12)、`pymc 6.3.2`、`numpyro 0.22.0`、`bambi 0.21.0`、`arviz 1.3.0`、`cmdstanpy 1.3.0`、`MAPIE 1.5.0`、`netcal 1.4.0`、`venn-abers 1.5.4`、`crepes 0.9.1`、`openskill 6.2.0`、`glicko2 2.1.0`、`streamlit 1.64.0`、`shap 0.52.0`、`onnxruntime 1.30.0`、`skl2onnx 1.20.0`、`duckdb 1.x`

**已死 / 慎用**：`xlearn 0.40a1`(2018-12)、`torchfm 0.7.0`(2020-01)、`trueskill 0.4.5`(2018-09 → 用 openskill)、`betacal 1.1.0`(2021-04)、`pyro-ppl 1.9.1`(2024-06 → 用 numpyro)

**环境**：本机 conda env `LeagueOfLegendWinRate` 是 **Python 3.13.15 且 numpy/pandas/sklearn/torch 全部未安装** **[实测]**。Python 3.13 对 torch 和部分包的 wheel 可能还不全。**第一天就新建一个 Python 3.11 环境**，用 `uv` 管理并提交 `uv.lock`，别在依赖安装上浪费第一周。


---

# 6. 评估与模型对比

> **这是全文最重要的一节。** 用户明确问了"需要做什么样的对比"，而这个项目的成败几乎完全取决于裁判尺子对不对——因为整个可解释信号只有 0.070 log loss，尺子稍微歪一点，你就会围绕噪声做三个月决策。

## 6.1 切分方案（文字版图示）

### 6.1.1 三层 nested 结构

```
时间轴 ─────────────────────────────────────────────────────────────────────────▶
2019            2023-01        2024-01        2025-01     2026-06      2026-09
  │               │              │              │            │            │
  │◀─────── 预训练 / 冷启动期 ────▶│                           │            │
  │               │                                          │            │
  │               │◀═══════ 中层：rolling-origin walk-forward ═══════▶│    │
  │               │                                          │            │
  │               │  origin₁  origin₂  origin₃  ...  origin_N │            │
  │               │    │        │        │              │     │            │
  │               │    ▼        ▼        ▼              ▼     │            │
  │               │  ┌────┐   ┌─────┐  ┌──────┐      ┌──────┐│            │
  │               │  │train│  │train │  │train │      │train ││            │
  │               │  └────┴┬─┐└─────┴┬─┐└─────┴┬─┐   └──────┴┬─┐          │
  │               │        │V│       │V│       │V│           │V│          │
  │               │       7d└─┘     7d└─┘     7d└─┘         7d└─┘         │
  │               │      embargo   embargo   embargo       embargo        │
  │               │                                                        │
  │               │                                          │◀ FINAL ────▶│
  │               │                                          │  HOLDOUT    │
  │               │                                          │ (冻结，开发  │
  │               │                                          │  期一次都不看)│

每个 origin 的内部（内层超参选择）：
  ┌──────────── train 段（expanding 或 sliding-W）────────────┐
  │                                              │ 最后 10%  │  ← inner valid
  │          用这段拟合                            │ 只用它选超参│
  └──────────────────────────────────────────────┴───────────┘
  选完超参后，用整个 train 段重拟合，再去打 valid 窗口
```

参数：

| 项 | 取值 |
|---|---|
| origin 推进步长 | **4 周** |
| valid 窗口 | **4 周** |
| `min_train` | **500 场**（与 arXiv 论文一致） |
| embargo | **fold 边界前后各 7 天** |
| 累计折数 | **≥ 10** |
| 汇总方式 | 把所有 fold 的 per-game 预测**拼成一条长向量**算总 log loss（不是各 fold 平均） |
| train 窗口类型 `W` | `{expanding, 36mo, 24mo, 18mo, 12mo, 9mo, 6mo}` **当作超参用中层选**，并把"W 的最优值"本身作为一个结论汇报（它量化了版本漂移速度） |

### 6.1.2 三条红线

**红线 1 · 分组键是 `series_id`，不是 `game_id`。**
同一 Bo3/Bo5 的多局共享双方实力、当日状态、BP 博弈状态，甚至"是否打到第 5 局"这件事本身就携带系列赛结果信息。**构造必须用 §2.2 的 6 小时会话切分，不能用 UTC 日历日**（2026 实测 89 个系列被零点劈开），并跑通 §2.2 的三条 pytest。

**红线 2 · embargo 7 天。**
Elo/rating 是递归状态，会跨 fold 边界渗透。

**红线 3 · 所有【依赖训练段全局分布】的统计量必须在每个 fold 的训练段内重新估计。**
包括但不限于：EB 收缩强度 `k` 与基准 `θ̄_pos`、target encoding、standardizer 的 mean/std、Platt 校准器、`L_region` / 分层 BT、**以及 embedding 预训练语料**（见 §4.5.5）。

**与 §4.1 的边界（v1.1 澄清，v1.0 这两处直接冲突）**：在线递归类特征（Elo / Glicko / h2h / mastery 计数 / side 先验的 as-of 累加器）**不在此列**——它们逐场只用 `<t` 的信息，全局跑一遍状态机即可、所有 fold 共用，重估反而是浪费。哪些豁免、哪些必须重估，**以 §4.1 的分类表为唯一准绳**。

### 6.1.3 final holdout 的冻结

**【v1.0 的方式有洞】"第一天枚举 gameid 列表并 commit"在这里行不通**：holdout 窗口（2026-06-01 起，含 2026 Worlds）在项目开发期内**还在持续产生新比赛**——这些新比赛不在列表里 → 不会被排除 → 会流进训练集，而它们的日期又落在 holdout 期内，**时间隔离直接被破坏**。**[实测 2026-09-18]** 当前 2026 文件里 `WLDs` 只有 **8 场**、`MSI` 71 场，`max(date)` 就是今天——第一天根本枚举不到 2026 Worlds 的绝大部分比赛。而 v1.0 阶段 0 验收④（"行数 >500 且已 commit"）只查行数不查完整性，发现不了。

**[正确方式] 用日期截止线冻结；gameid 列表只作存档，不作排除机制**

```python
# src/lolwp/features/loader.py —— 唯一真源，硬编码
HOLDOUT_START = pd.Timestamp('2026-06-01', tz='UTC')

def load_train(df):   return df[df.kickoff_utc <  HOLDOUT_START]
def load_holdout(df): return df[df.kickoff_utc >= HOLDOUT_START]
```

```bash
# 第一天仍然 commit 一份成员快照，但它是【记录】不是【排除机制】
python - <<'PY' > eval/holdout_gameids_20260918.txt
import pandas as pd
d = pd.read_csv('data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv',
                usecols=['gameid', 'date'])
print('\n'.join(sorted(d[d.date >= '2026-06-01'].gameid.unique())))
PY
git add eval/holdout_gameids_20260918.txt
git commit -m "snapshot holdout membership at freeze time (record only)"
```

**一条必须进 CI 的断言**（每次 ingest 后自动跑）：

```python
assert (train_df.kickoff_utc < HOLDOUT_START).all(), "训练集里混进了 holdout 期的比赛"
```

整个开发期一次都不许评估 holdout。**揭盲时（阶段 9）先打印 holdout 的 `n_games` / 日期范围 / 赛事构成，与预注册时的预期对照**；若 Worlds 2026 尚未打完，如实记录 holdout 的实际覆盖，**绝不事后扩窗**（扩窗 = 第二次看 holdout）。

## 6.2 指标定义与主指标

### 6.2.1 主指标：log loss

```
LL = -1/N · Σ [ y·log(p̂) + (1-y)·log(1-p̂) ]
```

选它的三条理由：唯一**严格适当（strictly proper）**且对极端错误重罚；与 logistic 的训练目标一致（训练评估口径统一）；对概率质量而非排序敏感。

### 6.2.2 Brier + Murphy 三分解（必配）

```
BS = 1/N · Σ (p̂ - y)²  =  Reliability − Resolution + Uncertainty
```

| 分量 | 定义 | 本项目的量级 | 含义 |
|---|---|---|---|
| **Uncertainty** | `p̄(1-p̄)` | ≈ **0.2489**（蓝方 53.4%） | 无论如何降不下去的部分 |
| **Resolution** | `Σ(n_b/N)(ȳ_b − p̄)²` | ≈ **0.028** | **你的真实增量** |
| **Reliability** | `Σ(n_b/N)(p̄_b − ȳ_b)²` | 应 ≈ **0** | 校准误差 |

**三项分开报的价值**：模型变差时能一眼看出是"校准坏了"还是"区分力丢了"。**版本漂移通常先体现为 Reliability 上升**，这是监控的关键（§6.11）。

### 6.2.3 校准四件套

```python
# 1. reliability diagram：15 个【等频】bin（不要等宽，尾部会空）
# 2. ECE = Σ (n_b/N) · |acc_b - conf_b|
# 3. MCE = max_b |acc_b - conf_b|
# 4. calibration slope / intercept：
#    把 logit(p̂) 作为唯一自变量重做一次 logistic
#    slope 应 ≈ 1（arXiv 报 0.995），intercept ≈ 0
#    slope < 1  = 过度自信 = 过拟合的直接指纹
#    slope < 0.85 或 > 1.15 → 立即触发重校准（§6.11）
```

ECE 自己写 10 行即可，不必引 netcal。

### 6.2.4 辅助指标

| 指标 | 角色 | 为什么不能当主指标 |
|---|---|---|
| **AUC** | sanity check | 对任何**单调变换不变**。一个把概率全压到 [0.48, 0.52] 的模型和完美校准的模型 AUC 完全一样，而 log loss 差很多。**用户要的是"胜率"不是"排序"** |
| **accuracy** | 仅展示 | 把 0.51 和 0.99 当成同一件事；对阈值和基准率敏感；63% 和 65% 在 800 场上完全在噪声里。**永不用于选模** |
| **McFadden pseudo-R²** | 给非技术读者 | `1 − LL_model/LL_null`，SOTA 约 0.081 |

## 6.3 功效分析：先算清楚你能分辨多大差异

> **这一小节是整节评估设计的地基。** 不先算这个，后面的所有对比表都是自欺。

### 6.3.1 核心公式（**v1.1 重写：删掉 `/2.5`，两套尺子合一**）

**【已证伪】v1.0 的 `n_eff = n_games / 2.5`** 是一个**从未测量、从未推导、全文只出现为断言**的常数，而全项目最重要的裁决（mastery_diff 不进主力、产品模型停在 L4）就挂在它上面。三个问题：

1. **它等价于假设 ICC = 1。** 正确的量是 design effect `deff = 1 + (m̄−1)·ICC`；`n/2.5` 只在系列赛内多局**完全相关**时才成立。
2. **`m̄ = 2.5` 本身就是错的。** **[实测 2026-09-18]** 2026 全量按 §2.2 构造 series：8,832 场 / 4,127 个 series → **m̄ = 2.14**，局数分布 `{1:1274, 2:1479, 3:1037, 4:196, 5:141}`。
3. **它与 §6.6.1 推荐的 series 级 block bootstrap 是两套不自洽的尺子。** block bootstrap 本来就直接、正确地估计了聚类后的标准误，根本不需要 `n_eff` 这个近似。用一个手工常数算 MDE、再用 bootstrap 算 CI，两把尺子并存。

**[做法 A · 报告用] 直接用 block bootstrap 的经验分布反解 MDE**

```python
# eval/power.py —— 一把尺子量到底，不再出现任何 /2.5
boot = paired_block_bootstrap(preds_a, preds_b, series_ids, B=10000)   # §6.6.1
MDE  = 2.8 * boot.std()
```

**[做法 B · 诊断用] 实测 ICC 与 deff，让读者看到裁决对它有多敏感**

```python
def effective_n(d: np.ndarray, series_ids) -> dict:
    """d = per-game 的 log loss 差值 d_i。单向随机效应 ANOVA 估 series 级 ICC。"""
    g = pd.Series(d).groupby(series_ids)
    m_bar, k, n = g.size().mean(), g.ngroups, len(d)
    MS_between = (g.size() * (g.mean() - d.mean()) ** 2).sum() / (k - 1)
    MS_within  = g.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum() / (n - k)
    icc  = float(np.clip((MS_between - MS_within) /
                         (MS_between + (m_bar - 1) * MS_within), 0.0, 1.0))
    deff = 1 + (m_bar - 1) * icc
    return {"m_bar": m_bar, "n_series": k, "icc": icc,
            "deff": deff, "n_eff": n / deff}
```

**`m̄` / `ICC` / `deff` 这三个数必须和 `n_games` / `n_series` / `σ_d` / `MDE` 一起打进每一张表的表脚**（§6.3.6）。

作为保守上界仍可写 `MDE = 2.8·σ_d/√n_eff`，但 `n_eff` **必须来自上面实测的 `deff`，不得再用任何手工常数**。

### 6.3.2 σ_d 的实测值

| 场景 | σ_d |
|---|---|
| 两个结构差异大的模型（噪声相关 ρ≈0.5） | ~0.20 |
| 两个同质模型（ρ≈0.8） | ~0.11 |
| **本项目实测（L3 vs L5）** | **0.0894** |
| 跨模型族的工作值 | **0.126** |

### 6.3.3 MDE 速查表（σ_d = 0.126 的保守工作值；**这是上界，实测 ICC 后必须重算**）

下表用实测 `m̄ = 2.14` + **ICC = 1**（最保守的一端，即 v1.0 的隐含假设）算出 `deff = 2.14`。**真实 MDE 一定 ≤ "MDE 上界"这一列**，ICC 越小 MDE 越小。

| n_games | n_eff (ICC=1, ÷2.14) | **MDE 上界** | 参考：ICC=0.3 时（deff=1.34） |
|---|---|---|---|
| 1,000 | 467 | **0.0163** | 0.0141 |
| 2,000（tier-1 一年） | 935 | **0.0116** | 0.0100 |
| 3,700（本项目当前测试集） | 1,729 | **0.0085** | 0.0073 |
| 6,145 | 2,871 | **0.0066** | 0.0057 |
| 20,000（tier-1 全历史 walk-forward） | 9,346 | **0.0037** | 0.0032 |
| 60,000（全赛区 walk-forward） | 28,037 | **0.0021** | 0.0018 |

（ICC=0.3 一列仅作敏感度示意。**报告里的正式数字一律来自 block bootstrap 的经验分布，§6.3.1 做法 A。**）

**参照系**：整个可解释信号总量只有 **0.070 log loss**。arXiv SOTA 与 dynamic Bradley-Terry 的差距 ΔBrier = 0.0144、与 static 的差距只有 0.0038。

### 6.3.4 【本项目的具体裁决】L3 → L5 不显著

调研里把 `mastery_diff` 写成主力模型的立项理由，复算如下：

| 口径 | deff | n | n_eff | t | MDE | ΔLL = 0.00496 是否显著 |
|---|---|---|---|---|---|---|
| **game 级**（调研口径，ICC=0） | 1.00 | 3,711 | 3,711 | **3.38** | 0.0041 | 显著（但紧贴下限） |
| **★ ICC 实测值**（`deff = 1 + 1.14·ICC`） | **____** | 3,711 | **____** | **____** | **____** | **【第 3 周填；此前一律标"裁决待定"】** |
| ICC = 0.3（示意） | 1.34 | 3,711 | 2,770 | **2.92** | 0.0049 | **是 → 裁决翻转** |
| ICC = 0.6（示意） | 1.68 | 3,711 | 2,209 | 2.61 | 0.0055 | 边缘 |
| **ICC = 1**（v1.0 的隐含假设） | 2.14 | 3,711 | 1,734 | **2.31** | **0.0060** | 否 |
| ICC = 1 + 严格 tier-1（剔除 LJL 374 / PCS 303 / TCL 240 / VCS 141 等 28.5%） | 2.14 | 2,653 | 1,240 | ~1.95 | **0.0071** | 否 |

**裁决（有条件）：本裁决在 ICC ≥ 0.6 时成立（`mastery_diff` 不进主力、产品模型停在 L4），在 ICC < 0.35 时翻转。必须先测。** `eval/power.py::effective_n()` 的实测 ICC 一出来就把上表第二行填满、给出最终裁决并写进 `reports/<date>/`。在此之前 `mastery_diff` 保留为 UI 展示（"该 ADC 这个英雄打过 12 场舞台局"），但**不得把"停在 L4"写成已定论**。

同理，以下"最优值"全部是在拟合噪声（差异 ≤0.0025，比任何口径的 MDE 小一个数量级）：

- 半衰期网格：30d 0.6235 / 75d 0.6230 / 180d 0.6234 / 365d 0.6241 / 730d 0.6247 / 无衰减 0.6255
- 收缩强度网格：k=100 0.6235 / 250 0.6234 / 476 0.6234 / 1000 0.6235 / 2000 0.6239

**唯一能从这两个网格里提取的结论**：`k` 在 100–1000 区间完全不敏感（别浪费时间调它）；"完全不衰减"确实更差（Δ=0.0025，标记为 suggestive 而非 confirmed）。

### 6.3.5 【重要的方法论警告】不要靠混池买功效

调研里有一份方案主张"把评估拉到全 49 个赛区、n≈60,000，把 MDE 压到 0.0015"。**这有两重问题**：

1. **算错了**：60,000 场按 series 修正后（实测 m̄=2.14、ICC=1 的保守端）`n_eff ≈ 28,000`，MDE = **0.0021** 不是 0.0015。
2. **更根本的是它换了估计对象**：把次级联赛堆进来提升的是"**混合总体平均效应**"的功效，不是产品要服务的 **tier-1 效应**的功效。而 draft 效应在 tier 之间本来就异质（次级联赛水平参差更大，draft 的相对重要性不同）。**为一个混合总体的参数攒功效，然后拿去给 tier-1 产品下结论，是答非所问。**

**正确做法**：

```
主结论：只报严格 tier-1，诚实承认 MDE ≈ 0.007–0.012
      → 接受"在 tier-1 上就是测不出 0.002 的差异"这个事实
Secondary analysis：全赛区结果单列，作为"效应在更宽总体上是否一致"的旁证
异质性检验：报 tier-1 vs tier-2 的效应差及其 CI
```

### 6.3.6 强制纪律：每张表都要打印 MDE

**`eval/power.py` 必须自动在每张表（包括超参网格表、消融表、分层表）脚下打印**：

```
本表 n_games = ____，n_series = ____，m̄ = ____，ICC = ____，deff = ____，n_eff = ____，
σ_d = ____
80% 功效下可检出的最小差异 MDE（由 series 级 block bootstrap 经验分布反解
= 2.8·sd(boot)）= ____
凡 |Δ| < MDE 的行已标注【低于可检出下限】，不得解读为改进。
```

**这一条纪律会直接砍掉项目里 80% 的无效工作。**

## 6.4 模型对比表模板

### 6.4.1 表格结构（行 = 模型，列 = 13 项）

| 列名 | 说明 | 为什么需要 |
|---|---|---|
| `n` | 该行评估的场次 / series 数 | 没有 n 就无法判断小数差 |
| **`LogLoss`** | 全量 walk-forward OOS，mean ± **series 级 block bootstrap 95% CI** | 主指标 |
| **`ΔLL vs L3`** | 配对差 + CI + Diebold-Mariano p 值 | primary comparison |
| **`是否超过 MDE`** | ✓ / ✗（自动判定） | 防止过度解读 |
| `Brier` | + REL / RES / UNC 三分量 | 看清变差的原因 |
| `AUC` | + DeLong p vs L3 | sanity check |
| `ECE` | 15 等频 bin | 产品输出的是百分比 |
| `Calib slope` | 应 ≈ 1.0 | 过拟合指纹 |
| **`\|Δrating\|<50 子集 LL`** | 势均力敌那一档 | **真正体现能力的 KPI** |
| **`国际赛子集 LL`** | MSI/WLDs/FST/IC/EWC | 模型最容易崩的地方 |
| `训练耗时` | 全量 walk-forward 重放的 wall-clock，**必须拆成三段分别记录**：`T_state`（特征重建）+ `T_refit`（fold 内必须重估的 BT / embedding / 校准器）+ `T_fit`（模型拟合）。v1.0 只写"wall-clock"容易被理解成只含拟合，而拟合恰恰是最便宜的一段（§4.1 / §4.4.3） | 决定你能不能每周重训、以及跑不跑得起超参网格 |
| `推理延迟 p50/p99` | 单次 + 1730 行批量 | what-if 要求 p99 < 200ms |
| `可解释性` | 系数级 / SHAP / 黑盒 | 决定 what-if 页面能不能做 |
| `冷启动退化` | 新选手/新队/新英雄各自的 ΔLL | 线上表现的预告 |
| `在线更新难度` | O(1) 增量 / 需全量重训 | 决定重训节奏 |

### 6.4.2 行（严格按实现顺序）

```
L0   常数 0.5
L1   训练窗口内蓝方基准率（as-of）
L1b  按赛区收缩的 side 先验
L2   + team Elo
L3   + player Elo                    ← primary comparison 的对照组
L4   + champ_meta_diff               ← 推荐的产品模型
L5   + mastery_diff                  （预期：低于 MDE）
L6   L3 + 英雄 170 维 one-hot        （反面教材）
L7   L4 + 低秩 draft（闸门通过时）
M1   分组 L2 logistic 全特征 + zero-init α   ← 主力（消费 BT 导出的 player_rating_diff / league_strength_diff）
M2   分层 BT 即模型（直接用 BT 的 logit 出概率，不经 logistic） ← 与 M1 并列的对照行，不嵌套（§4.6.2）
E1   CatBoost（同一套工程化标量）
E2   zero-init residual DeepSet
ENS  M1 + E1 的等权 logit 平均
B5   Polymarket 隐含概率            （仅交集子集，另出一张表）
```

### 6.4.3 填好后的样子（示例，数字待实测替换）

| 模型 | n_games / n_series | LogLoss (95% CI) | ΔLL vs L3 (DM p) | >MDE? | Brier (R/Res/U) | AUC | ECE | slope | \|Δr\|<50 LL | 国际赛 LL | p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 20,000 / 8,000 | 0.6931 (—) | +0.066 (<0.001) | ✓ | 0.2500 | 0.500 | — | — | 0.693 | 0.693 | <1ms |
| L1 | 20,000 / 8,000 | 0.6912 (±0.0008) | +0.064 (<0.001) | ✓ | 0.2489 | 0.500 | 0.004 | — | 0.691 | 0.692 | <1ms |
| L2 | 20,000 / 8,000 | 待测 | 待测 | | | | | | | | |
| **L3** | 20,000 / 8,000 | **基准** | — | — | | | | | | | |
| **L4** | 20,000 / 8,000 | 待测 | 待测 | | | | | | | | |
| L5 | 20,000 / 8,000 | 待测 | 预期 −0.005 | **预期 ✗** | | | | | | | |
| … | | | | | | | | | | | |

> 本表 n_games = 20,000，n_series = 8,000，σ_d = ____，**MDE = 0.0039**。
> 凡 |ΔLL| < 0.0039 的行标注【低于可检出下限】。

## 6.5 消融矩阵（双向，两张表对照）

在 walk-forward 的**全量 OOS 预测**上做（目标 ≥ 3,000 场，最好 ≥ 20,000），**不是在 final holdout 上**。

### 6.5.1 两个方向

**方向 A · leave-one-out**：固定 `M_full`，每次只去掉一组 → `ΔLL_k = LL(M_−k) − LL(M_full)`

**方向 B · 正向增量**：从 L0 起逐组往上加 → 每一步的 ΔLL

**两表不一致的地方 = 特征共线**（某项 LOO 显示无用但正向增量显示有用，说明它的信息被别的特征替代了）。这是消融实验里最有信息量的部分，单向做会漏掉。

### 6.5.2 13 个消融项与先验预期

| # | 消融项 | 预期 ΔLL | 置信度 | 依据 |
|---|---|---|---|---|
| 1 | team rating（G2） | **−0.049** | 极高 | 实测 0.6912 → 0.6411 |
| 2 | player rating（G3，替代 team 层） | **−0.0075** | 极高 | 实测 0.6411 → 0.6336 |
| 3 | 分层 BT 联合估计（替代 plug-in Elo） | −0.002 ~ −0.005 | 中高 | arXiv 一阶段 vs 两阶段 Brier 0.2230 vs 0.2257 |
| 4 | `L_region` 赛区偏移（3 年窗口） | −0.001 ~ −0.004 | 中 | 主要作用在国际赛，全样本上被稀释 |
| 5 | side×league 收缩 + event_tier | −0.001 ~ −0.003 | 中高 | 赛区间蓝方胜率实测 49.2%–60.4% |
| 6 | Glicko RD / roster 变动 / rest days | −0.001 ~ −0.003 | 中 | 冷启动与转会期子集上增益更大 |
| 7 | **英雄主效应 `champ_meta_diff`** | **−0.000 ~ −0.0044** | 中 | 实测直接加 171 维指示是 **+0.0006（变差）**；靠"控制队伍强度后的偏效应 + 收缩 + 大样本"才转正 |
| 8 | 选手×英雄熟练度计数 | −0.000 ~ −0.002 | 低 | 实测两折符号翻转 |
| 9 | 低秩 counter | −0.000 ~ −0.003 | 低 | iTero 有独立佐证但迁移系数约 0.4 |
| 10 | 低秩 synergy | **−0.000 ~ −0.001** | 极低 | pair 真实方差 ≤0；split-half r≈0.00。**大概率被闸门砍掉** |
| 11 | 特征内时间衰减（关掉 = 等权） | −0.001 ~ −0.0025 | 中 | 网格实测，紧贴 MDE |
| 12 | 训练样本权重衰减 | **~0 或为正（有害）** | 中高 | 实测 45d/90d 反而更差 |
| 13 | **全赛区 prior vs 纯 tier-1 prior**（`champ_meta_diff` 的两层 EB，§4.2 G6） | −0.000 ~ −0.002 | 中 | 检验"只借 exposure、不借 win rate"这个设计是否真的有增量；同时对照 v1.0 的"单池等权"写法 |

**必须诚实写进报告的一句话**：前两项（队伍/选手实力）就吃掉总增益的 **~93%**，全部 draft 相关项加起来预期只值 **0.003–0.012**，**且其中一半以上的项预期低于 MDE**。

## 6.6 显著性检验做法

### 6.6.1 Series 级 block bootstrap（首选，最易解释）

```python
def paired_block_bootstrap(preds_a, preds_b, series_ids, B=10000, seed=0):
    """以 series 为重抽单位，不是 game，更不是行"""
    rng = np.random.default_rng(seed)
    ll_a = per_game_logloss(preds_a)    # 逐场的 -[y log p + (1-y) log(1-p)]
    ll_b = per_game_logloss(preds_b)
    d = ll_a - ll_b
    groups = pd.Series(d).groupby(series_ids)
    keys = list(groups.groups)
    sums  = groups.sum().values
    counts= groups.size().values
    out = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        out[i] = sums[idx].sum() / counts[idx].sum()
    return out.mean(), np.percentile(out, [2.5, 97.5])
```

**以 game 为单位重抽会让 CI 系统性偏窄。**

### 6.6.2 Paired Diebold-Mariano（学术标配）

`dieboldmariano`（PyPI，纯 Python 无依赖，**支持自定义 cost function——必须传 log loss 而不是默认的平方误差**），或 vendor `github.com/johntwk/Diebold-Mariano-Test`（含 **Harvey et al. 1997 小样本修正**，n < 几百时必须开）。

### 6.6.3 DeLong（仅 AUC，审稿人会问的形式要求）

sklearn 和 scipy 都没有内置。vendor `yandexdataschool/roc_comparison` 的 `compare_auc_delong_xu.py`（不在 PyPI，直接复制进仓库）。**记住 AUC 不是主指标，这只是形式要求。**

### 6.6.4 决策门槛：用 t 统计量，不用效应量

```python
t = delta_mean / (paired_SD / sqrt(n_eff))
门槛: |t| > 2.5
```

**`ratio = |Δ| / SD` 是错的**——那是效应量，**不随 n 改善，永远无法累积证据**。

**配对逐 fold 相减，不要比均值**：消掉"这段时期有多难"这个共同因子，同样样本量下敏感度约 **5 倍**。

噪声量级参考 **[实测]**：

| 噪声源 | SD | 能否通过加数据降低 |
|---|---|---|
| **seed SD**（模型估计方差） | ≈ 0.014 | **能**（加训练数据） |
| **fold SD**（验证窗口难度） | ≈ 0.023 | **不能**，只有加大验证窗口。从 2 年/7% 窗口扩到 5 年/15%，fold SD 从 0.104 降到 0.023 |

**边界情况**：per-fold delta 全部同号且相同时 `paired SD = 0`，要报 `t = ±inf` 并按符号判定，不能当成噪声。

### 6.6.5 多重比较

8 模型 × 5 赛区 × 3 指标 = 120 次检验，α=0.05 下期望 **6 个假阳性**。

```
Primary comparison：预注册一个（§2.5）
其余：全部标记 exploratory，用 Benjamini-Hochberg FDR 控制
     （比 Bonferroni 功效高得多，探索性分析更合适）
```

**额外警告**：如果你尝试了 20 个模型然后报告最好的那个在测试集上的分数，那个分数不可信——按功效分析，20 次尝试里出现一个 0.008 的虚假"提升"是大概率事件。

## 6.7 特征准入门禁（本方案的核心机制）

**不要用"方差阈值"或"作者断言"决定特征去留，用样本外门禁。**

### 6.7.1 门禁流程

```
┌─ 阶段 0：可靠性预检（仅对交互类特征）───────────────────────┐
│  split-half reliability，Spearman-Brown 修正 r_full ≥ 0.20   │
│  参考值：单英雄胜率 0.25-0.36 / champion scaling 0.05-0.11   │
│          pair synergy ≈ 0.00                                │
│  不通过 → 记录"不可测量"，砍掉，写进报告                       │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ 阶段 1：滚动起点两折门禁（所有特征都要过）─────────────────┐
│  至少连续 2 个留出期（建议按赛季/季度切）                      │
│  准入条件（三条同时满足）：                                    │
│    (a) 两期的 Δ 符号相同                                      │
│    (b) 合并后 series 级 |t| > 2.5                             │
│    (c) |Δ| > 该样本量下的 MDE                                 │
│  不通过 → 不进主力模型；可保留为 UI 展示字段                    │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌─ 阶段 2：全量 walk-forward 消融确认 ─────────────────────────┐
│  双向消融矩阵 + block bootstrap CI + DM p                     │
└──────────────────────────────────────────────────────────────┘
```

### 6.7.2 已有的门禁结果（可直接采信，但补 2026 后要重跑）

| 特征 | 2024 留出 Δ (t) | 2025 留出 Δ (t) | 裁决 |
|---|---|---|---|
| 英雄主效应 | −0.00213 | −0.00214 | **✓ 通过**（两折同号且稳定） |
| 选手×英雄残差 | +0.00159 (t=+2.82) | −0.00060 (t=−0.81) | **✗ 符号翻转，拒绝** |
| 同位对位 counter | +0.00026 (t=0.42) | −0.00012 (t=−0.20) | **✗ 拒绝** |
| 同队 pair 协同 | −0.00005 (t=−0.46) | +0.00002 (t=0.34) | **✗ 拒绝**（回归系数 −0.086 / −0.035，接近 0 甚至为负） |

（注：Δ 为正表示"加了这个特征后 log loss 上升"，即变差。）

### 6.7.3 为什么这个机制比"方差阈值"好

方差阈值（如 "SD < 0.02 就砍"）有三个问题：

1. **估计量本身不稳**：SD=0.019 的 95% CI 是 [0, 0.051]，四年合并变成 0.044，且矩估计有 +0.0005 正偏。
2. **方差大 ≠ 有预测价值**：0.044 的方差分量在严格时间外推里只贡献 |ΔLL| ≤ 0.0016 且跨年变号。
3. **类别错误**：调研里出现过"若选手×英雄 SD 升到 0.04+ 则应保留显式 pairwise 项"这样的推理——**选手×英雄的方差和英雄配对项是两个完全不同的参数，没有任何逻辑关系。**

**样本外门禁直接测的就是你关心的东西（预测能力），而且结论稳定。**

## 6.8 分层评估（6 个维度，总体指标会掩盖系统性失效）

| # | 维度 | 分桶 | 为什么必须看 |
|---|---|---|---|
| 1 | **赛区** | **用 canonical 值分桶**：LPL / LCK / LEC / **LCS**（含 2025 的 `LTA N`）/ **CBLOL**（含 2025 的 `LTA S`）/ LCP / PCS / VCS / 国际 / 二级汇总。⚠ 绝不用任何一年的原始 code | LPL/LCK 样本多、强度分布宽 → LL 更低；VCS/LLA/CBLOL 样本少爆冷多 → 接近 0.69。**暴露"模型其实只在两个大区有用"** |
| 2 | **赛事级别** | 常规赛 / 季后赛 / **国际赛单列** | 国际赛跨区、rating 不可直接比较，是最容易崩的地方。arXiv 也发现市场优势集中在 cross-region Worlds 与决胜局 |
| 3 | **`patch_age_days`** | 0–7 / 8–14 / 15–30 / 30+ 天 | 新 patch 头两周 LL 应显著更高。这条曲线**直接量化版本漂移成本**，也是重训触发条件与 `H_meta` 的实证依据。**不要按单个 patch 分桶**（一个 patch 跨全赛区可能只有几百场，噪声极大），用滚动 200 场窗口 |
| 4 | **`\|rating_diff\|`** | 0–50 / 50–100 / 100–200 / 200+ Elo | 强弱悬殊 LL 低是应该的。**真正体现能力的是 <50 那一档**——很多模型在这一档退化到 0.69 等于没用。**这一档单独作为一个 KPI** |
| 5 | **BO 局序** | game1 / game2 / game3+ / 决胜局 | 若特征不含系列赛内状态，game3+ 会明显更差 |
| 6 | **冷启动子集** | 含 ≥1 名训练集未见选手 / 未见战队 / 未见英雄 | 43% 的选手整季不到 20 场。**回测里大部分主体是见过的，线上不是** |

**每个分层单元报**：`n_games / n_series / LL / Brier / ECE / calib slope / ΔLL vs L3 + bootstrap CI + MDE`。

**额外一维（若做了 pick order）**：按 `has_pick_order` 分层，检查有/无两组的表现是否一致。不一致说明你在偷偷给"有数据的赛区"一个不同的模型。

## 6.9 如何自查泄漏（12 条清单，写成 CI 测试，不靠自觉）

### 6.9.1 四个负控制（自动化测试）

```python
# ---- N1  label shuffle ----
# 打乱 y 后跑【完整 pipeline】（含特征构造），log loss 必须回到 0.693 ± 0.01
def test_label_shuffle():
    df2 = df.copy(); df2['y_blue_win'] = rng.permutation(df2['y_blue_win'])
    ll = run_full_pipeline(df2)
    assert 0.685 < ll < 0.700, f"结构性泄漏：shuffle 后 LL={ll}"

# ---- N2  时间反转对照 ----
# 用未来数据预测过去，若性能反而更好 → 特征携带未来信息
def test_time_reversal():
    ll_fwd = walk_forward(df)
    ll_rev = walk_forward(df.iloc[::-1])
    assert ll_rev >= ll_fwd - 0.01, "时间反转后更准 = 特征有未来信息"

# ---- N3  leak canary（一次性，验证检测灵敏度）----
def test_canary_sensitivity():
    auc_base = walk_forward(df).auc
    auc_leak = walk_forward(df.assign(_leak_canary=df.gamelength)).auc
    assert auc_leak - auc_base > 0.10, "检测阈值不够灵敏"
def test_no_canary_in_prod():
    assert '_leak_canary' not in production_feature_names

# ---- N4  未来删除不变性 golden test ----  （见 §4.7.1 的 T2）
```

### 6.9.2 八条人工自查清单

| # | 检查项 | 判定 |
|---|---|---|
| 1 | accuracy > 67% / LL < 0.60 / 任一 fold AUC > 0.78 | **几乎肯定泄漏**。锚点：SOTA 且与市场持平是 LL 0.637 / acc 约 64% |
| 2 | 特征里有任何 §4.8.1 黑名单列 | 直接 fail |
| 3 | 用了全样本统计量（选手胜率、英雄胜率、target encoding、scaler、校准器未在 fold 内估计） | 实测虚高 +0.041 AUC |
| 4 | 随机划分 | 实测再虚高 +0.10 AUC |
| 5 | 同一 series 的多局跨 train/test | 最隐蔽，网上那些 72%+ 多半栽在这里 |
| 6 | 同一场 12 行被拆到不同 fold | `assert df.game_id.nunique() == len(df)` |
| 7 | rating 先跑完整个赛季再 join 回去 | 最常见的 bug |
| 8 | 重复行 / 缺唯一约束 | `df.gameid.nunique() vs len(df)`、跨年文件的 gameid 交集 |
| **9** | **embedding 预训练语料覆盖评测期** | 见 §4.5.5，可能非常大 |
| **10** | **跨源未去重**（gptilt 与 OE 是同一批比赛） | 按 `riot_game_id` 或 `(date,两队,局序)` 去重 |
| 11 | 用 `datacompleteness` 当特征 | 偷偷泄漏赛区 |
| 12 | 用"选手本赛季最终所属队伍"做特征 | 转会窗后才知道 |

### 6.9.3 为什么网上 75%+ 都不可信（具体案例，值得记住）

| 来源 | 声称 | 真实原因 |
|---|---|---|
| **Do et al., FDG 2021**（arXiv:2108.02799，33 引用） | 75.1% | 5,000 场 **NA 排位 Iron→Diamond**（不是职业赛）+ **80/20 随机切分** + 核心特征 "player-champion win rate" 定义为"该选手 **2020 整赛季**该英雄胜率"，而被预测的对局本身就在 2020 赛季、统计是事后从 API 拉的 → **被预测那一局的结果被算进了特征里**。论文 Limitations 完全没提泄漏 |
| **LoLDraftAI** | 曾宣称 62% | 官方更正：Riot API client 未指定 region 时默认 EUW1，导致四个区数据其实是同一批 EUW 打了不同标签；加上数据库无 match id 唯一约束 → 训练测试重复。**修正后 ≈55%** |
| 某 GitHub 仓库 | 职业赛 AUC 0.797 / 72.1% | 单一特征 `patch_wr_delta_diff` 占 25.3% 重要性、V3→V4 让 AUC 从 0.698 跳到 0.797 —— 典型泄漏指纹。最可能是 Bo3/Bo5 系列内跨 fold |
| 各类 90%+ / 97.42% | — | 全部使用赛中 / 赛后特征 |

**还有一条读文献的陷阱**：arXiv 2609.08060 的 **0.2207 / 0.2230 是 Brier**，网上多个二手摘要写成了 log loss。Brier 0.22 ≈ log loss 0.63–0.64 ≈ accuracy 63–65%；而 log loss 0.22 对应约 93% accuracy——**差了一个数量级**。对标前先确认指标定义。

## 6.10 滚动回测与可重放

### 6.10.1 评估层必须与建模完全解耦

```
src/lolwp/eval/
  walkforward.py   walk_forward(model_factory, df, origins, window,
                                min_train=500, embargo_days=7)
                   -> DataFrame[game_id, series_id, date, league, patch_idx,
                                tier, rating_diff, y, p_hat, fold]
  metrics.py       log_loss / brier + murphy 分解 / auc / ece(15 等频) / mce
                   / calib slope + intercept
  compare.py       读 preds/*.parquet -> 对比表 + block bootstrap CI + DM + BH-FDR
  ablation.py      双向消融矩阵
  power.py         自动在每张表脚打印 n_games / n_series / σ_d / MDE
  negctrl.py       label shuffle / 时间反转 / leak canary
  strata.py        六维分层报告
  reliability.py   split-half + Spearman-Brown
  gate.py          两折同号门禁
preds/<model_id>.parquet
reports/<date>/
```

**任何模型只要实现 `fit(df_train)` 和 `predict_proba(df_test)` 两个方法，就能自动进入同一套回测、同一套分层报表、同一套配对检验。**

**硬性要求：加一个新模型进对比表的成本必须 ≤ 20 行代码。** 否则你不可能真的跑完 14 行 × 13 列的对比表。第 3 周用 L1 演示验证一次。

### 6.10.2 可重放脚本

把 walk-forward 做成**输入 = 截止日期、输出 = 该日期之前所有场次的 OOS 预测 CSV** 的可重放脚本，每周自动跑一次全量重放。

**任何代码改动都能立刻看到对历史曲线的影响——这是防止无意中引入泄漏的最有效手段。**

### 6.10.3 复现三件套

```bash
make reproduce SNAPSHOT=snap-2026-09-18
# 固定 snapshot_id（各年 CSV 的 sha256）
# → 固定 git sha
# → 固定随机种子列表
# → 重跑 walk-forward
# → 生成 reports/<date>/ 下的全部表和图
# → 与已归档版本 diff，应为空
```

`experiments/runs.jsonl` 每行：

```json
{"run_id":"...","git_sha":"...","snapshot_id":"...","model_id":"M1",
 "hyperparams":{...},"seeds":[0,1,2],"metrics":{...},
 "n_games":20134,"n_series":8042,"mde":0.0039,"wall_clock_s":1420}
```

**这一条做到了，你的结论才叫"经得起同行评议"。**

## 6.11 漂移监控与重训门禁

### 6.11.1 监控三条曲线（挂在 slope 上，不挂在 log loss 上）

横轴 patch 或周：

1. **滚动 200 场 log loss**（逐场推进）
2. **Brier 的 Reliability 分量**
3. **calibration slope**（滚动窗口内重拟合 `logit(p̂) → y`）

**漂移的典型顺序**：

```
新版本上线 → 英雄层特征失效 → 模型仍然自信
          → Reliability 先涨、slope 从 1.0 掉到 0.7-0.8
          → 【1-2 周后】总 log loss 才开始涨
```

**所以预警挂在 slope 和 reliability 上，比挂在 log loss 上早 1–2 周。**

### 6.11.2 触发条件（全部相对历史 walk-forward 分布的分位数，不用绝对阈值）

| 条件 | 动作 |
|---|---|
| 滚动 200 场 LL 超过历史 90 分位，持续 2 个 patch | 全量重训 |
| **calibration slope < 0.85 或 > 1.15** | **立即重跑 Platt / temperature 重校准**（比全量重训便宜得多，且往往就够） |
| 某赛区分层 LL > 0.693 连续 100 场 | 该赛区降级为"仅显示 rating 基线" |
| crosswalk 覆盖率 < 95% | 告警（**不阻塞**，见 §3.11.4） |
| livestats 204 率异常升高 | 告警，检查上游 |
| `data_age_days` > 3 | 前端显著提示 + 告警 |

### 6.11.3 重训门禁（新模型必须证明不劣才准上线）

```
触发：每周新数据 / 检测到新 patch（silver 的 patch 列出现新值，或 versions.json 首位变化）

流程：
 1. ingest → 五重校验
    ★ 失败直接退出不训练（宁可用旧模型，也不要用被 quota 页污染的数据训出的新模型）
 2. walk-forward 多 fold × 多 seed
 3. 【逐 fold 配对相减】，不是比均值
 4. 门禁（全满足才 promote）：
      log loss 配对 t > -2.5     （不显著变差）
      Brier    配对 t > -2.5
      ECE ≤ 0.03
      calib slope ∈ [0.90, 1.10]
      feature_names 与上一版一致（或有可解释的变更记录）
      n_games 与上一版可解释地一致
      四个负控制全部通过
 5. promote = 写 models/{run_id}/ + 原子切换 models/current 软链 + 重启
 6. 回滚 = 软链指回上一 run_id + 重启，30 秒内完成
 7. 所有 run 的 metrics 追加进 models/leaderboard.jsonl，永不删
```

**验证门禁真的会拒绝**：人为制造一个更差的模型（比如把 Elo 的 K 改成 200，或把 rating 特征置零），`make gate` 必须以 `t < -2.5` 拒绝 promote。这个测试要写进 CI。

---

# 7. 系统架构与本地部署

## 7.1 架构决策：v1 刻意做小

**v1 不要 FastAPI、不要 Docker、不要 DuckDB、不要 React、不要 ONNX、不要 Redis、不要 Postgres、不要 DVC。**

理由：模型是 **12–16 个浮点系数 + 三张几十 MB 的状态表**，全部进程启动时载入内存，推理是一次点积。给它配 FastAPI + DuckDB + Docker + React 是纯粹的仪式感，每一样都要 1–3 天且不产生任何用户价值，同时每一样都是演示时可能挂掉的点。

**Streamlit 单进程 + parquet + crontab 能做到完全一样的功能，而且第 4 周就能上线。**

**但要保留未来拆分的接缝**（这是关键）：

- `predict.py` 是**纯函数**（输入 draft dict → 输出 dict）
- `features/` 是训练与推理的唯一共用入口
- artifact 用 parquet / json / npz，不用 pickle
- API 契约用 pydantic 提前定死

**v2 想拆 FastAPI 就是包一层 `@app.post("/predict")`，零重构。**

**一条硬约束决定了推理路径的形态**：DuckDB 官方并发模型是"单进程写、多进程只读"（`access_mode='READ_ONLY'` 只在无写者时有效）。所以即使到了 v2，**推理进程也不应该连 DuckDB**——retrain 持有可写连接时 API 会直接连不上。把所有需要查的东西预计算成 artifact，既绕开并发限制，又把 p99 打到亚毫秒。DuckDB 只服务离线训练与特征计算，读 hive 分区 parquet。

## 7.2 目录树

> **读法**：下面是**最终形态**。**阶段 0 只需创建 4 个文件**——`experiments/preregistration.md`、`eval/HOLDOUT_START`、`eval/holdout_gameids_<date>.txt`、`scripts/day0_*.py`（见 §8.0），其余全部留到对应阶段再建，不要一次性 `mkdir` 出 40 个空文件。各文件的创建阶段见 §8.1 的"交付"列。

```
LeagueProject/
├─ pyproject.toml            # uv 管理，Python 3.11（阶段 0）
├─ uv.lock                   # ★ 提交进 git
├─ Makefile
├─ .env.example              # LOLWP_LOLESPORTS_KEY / LOLWP_LP_BOT_USER / ...
├─ .gitignore                # data/, models/, *.json 凭证
├─ docs/PLAN.md              # 本文件
├─ experiments/
│  ├─ preregistration.md     # ★ 第一天 commit
│  └─ runs.jsonl
├─ eval/HOLDOUT_START        # ★ 阶段 0 创建：holdout 日期截止线（唯一排除机制，§6.1.3）
├─ eval/holdout_gameids_<date>.txt  # ★ 阶段 0 创建：成员快照，仅作记录
├─ src/lolwp/
│  ├─ config.py              # pydantic-settings BaseSettings, env_prefix="LOLWP_"
│  ├─ ingest/
│  │  ├─ oracles_elixir.py   # 发现 fileId → 下载 → 五重校验 → 原子替换
│  │  ├─ lolesports.py       # getLeagues / getSchedule / getTeams 全量落地
│  │  ├─ ddragon.py          # champion.json → champion_map.parquet
│  │  ├─ leaguepedia.py      # mwrogue + bot password + 严格串行限速
│  │  ├─ gptilt.py           # HF datasets
│  │  └─ normalize.py        # 三张显式映射表（no fuzzy!）+ league canonical
│  ├─ store/
│  │  ├─ bronze.py           # CSV → parquet（30 列白名单裁剪）
│  │  ├─ silver.py           # 12 行 → 1 行/局的 draft 宽表 + series_id
│  │  └─ audit_columns.py    # ★ 跨年语义审计
│  ├─ features/              # ★ 训练与推理唯一共用入口（含 loader）
│  │  ├─ loader.py
│  │  ├─ priors.py           # build_priors(games, as_of) -> Artifacts（单遍时序状态机）
│  │  ├─ encode.py           # featurize / featurize_batch
│  │  ├─ artifacts.py        # 读写 + feature_names + code_version 断言
│  │  └─ whitelist.py        # 白名单 + 黑名单正则
│  ├─ models/
│  │  ├─ train.py            # 分组 L2 logistic
│  │  ├─ bootstrap.py        # 20 次 series 级 bootstrap 重拟合
│  │  ├─ calibrate.py
│  │  └─ registry.py         # models/current 软链管理
│  ├─ eval/                  # 见 §6.10.1
│  ├─ live/
│  │  ├─ poller.py           # getLive 60s / window 30s，204→200 状态机
│  │  └─ session.py          # (gameId, metadata_hash) 幂等缓存
│  ├─ explain.py             # 字段名 → 自然语言
│  └─ app/
│     ├─ main.py             # Streamlit 三个 tab
│     └─ predict.py          # 纯函数，v2 直接给 FastAPI 用
├─ frontend/public/dd/16.18.1/   # 173 张 PNG + champion.json（build 时落盘）
├─ configs/
│  ├─ player_overrides.yaml
│  ├─ league_canonical.yaml
│  └─ shrinkage_constants.yaml   # 由离线 NumPyro 读出后硬编码
├─ data/                     # gitignore
│  ├─ raw/<sha256>.csv.zst
│  ├─ bronze/ silver/ gold/ artifacts/<YYYY-MM-DD>/
│  └─ manifest.jsonl
├─ snapshots/<snapshot_id>.json
├─ models/{run_id}/ + models/current -> 软链 + leaderboard.jsonl
├─ scripts/daily.sh
└─ tests/
   ├─ fixtures/              # 小样本 parquet + mock window JSON（CI 绝不联网）
   ├─ test_ingest_validation.py   # 注入 quota HTML 必须非零退出
   ├─ test_normalize.py           # 173 英雄三体系往返映射
   ├─ test_features_pit.py        # T1/T2 point-in-time
   ├─ test_skew_golden.py         # T3 训练/推理一致性
   ├─ test_leakage.py             # N1-N4 负控制
   ├─ test_group_l2.py            # 分组 L2 实现正确性
   ├─ test_coldstart.py           # 未知选手/队伍/英雄的 fallback
   └─ test_e2e_smoke.py
```

## 7.3 各模块职责

| 模块 | 职责 | 绝不做 |
|---|---|---|
| `ingest/` | 发现 → 下载 → 五重校验 → 原子替换 → 写 manifest | 不做任何清洗或业务逻辑 |
| `store/bronze` | CSV → parquet，30 列白名单裁剪 | 不折叠行 |
| `store/silver` | 12 行 → 1 行/局，构造 series_id、canonical league、patch 归一化 | 不算聚合特征 |
| **`features/`** | **训练与推理的唯一共用入口**，单遍时序状态机产 as-of 特征；数据加载器也在这里 | 不碰模型 |
| `models/` | fit / predict_proba / calibrate / registry | 不碰特征构造 |
| `eval/` | 与建模完全解耦，只吃 `preds/*.parquet` | 不引用任何模型代码 |
| `live/` | 轮询状态机 + 幂等缓存 | 不做特征构造（调 `features.featurize`） |
| `app/` | 展示层 | **不写任何领域逻辑**，warnings / reasons 全从后端来 |

## 7.4 训练与推理共用特征代码

### 7.4.1 接口

```python
# features/priors.py
def build_priors(games_df: pd.DataFrame, as_of: datetime) -> Artifacts:
    """单遍时序状态机，产出截至 as_of 的全部聚合状态"""

# features/encode.py
def featurize(draft: DraftInput, art: Artifacts) -> np.ndarray:        # shape (F,)
def featurize_batch(drafts: list[DraftInput], art) -> np.ndarray:      # shape (N, F)
```

- **训练时**：每场用自己的 `as_of`（point-in-time）
- **推理时**：用 `as_of = now` 的那份 artifact
- **两条路径调同一个 `featurize`**

### 7.4.2 三道保险

```python
# 1. artifact 自描述 + 启动断言
assert model.feature_names == artifact.feature_names
assert artifact.code_version == git_tree_hash('src/lolwp/features')
# 不一致 → 拒绝启动（不是警告）

# 2. /health 暴露数据范围
{"run_id": "...", "train_start": "...", "train_end": "...", "n_games": 84213,
 "snapshot_id": "...", "code_version": "...", "crosswalk_coverage": 0.962}
# 上面那个 skew 事故（api 用 2 年、train 用 5 年，无任何报错）
# 如果 /health 打印了 n_games，一眼可见

# 3. golden test（§4.7.1 的 T3）
```

### 7.4.3 artifact 包

```
data/artifacts/<YYYY-MM-DD>/
  model.json            # 12-16 个系数 + 分组信息
  bootstrap_coefs.npz   # 20 组 bootstrap 系数，用于出区间
  calibrator.joblib     # Platt（若需要）
  champ_theta.parquet   # (champ_key, position) -> theta, n_eff
  player_rating.parquet # (oe_player_id) -> r, rd, n_games, last_seen
  team_state.parquet    # (team_id) -> glicko_mu, rd, streak, last_roster_hash
  league_effect.parquet # (league_canon) -> L_region, side_prior, n
  player_champ.parquet  # (player_id, champ_key) -> games_decayed, wr_shrunk, n  【仅 UI】
  support.npz           # float32 [173,173] decayed 共现计数，用于 OOD 警告
  synergy.npz counter.npz  # 仅在闸门通过时存在，各 120 KB
  crosswalk.parquet     # esportsPlayerId <-> oe_player_id
  champion_map.parquet
  meta.json             # 见 §4.7.2
```

整包几 MB，进程启动时全量载入内存，**推理零 DB 查询**。

## 7.5 FastAPI 接口设计（v2 实现，v1 用 pydantic 提前定死契约）

### 7.5.1 端点清单

```
GET  /health              服务与模型状态
GET  /champions           173 个英雄（含本地图标路径）
GET  /players?league=     选手选择器数据（含常规首发预填）
POST /predict             单个阵容
POST /predict/batch       N 个阵容（what-if 敏感度用）
POST /explain             逐项 log-odds 贡献 + 自然语言
GET  /live                当前进行中的比赛及缓存预测
```

### 7.5.2 `POST /predict`

**Request**

```json
{
  "league": "LEC",
  "patch": "16.17",
  "is_playoffs": true,
  "game_in_series": 3,
  "series_score_diff": 1,
  "blue": [
    {"champion": 150, "role": "top", "player_id": "oe:player:8a8f18e4b7d0f30c19c5fb99a526951"},
    {"champion": 64,  "role": "jng", "player_id": null},
    {"champion": 103, "role": "mid", "player_id": null},
    {"champion": 429, "role": "bot", "player_id": null},
    {"champion": 888, "role": "sup", "player_id": null}
  ],
  "red": [
    {"champion": 266, "role": "top", "player_id": null},
    {"champion": 76,  "role": "jng", "player_id": null},
    {"champion": 34,  "role": "mid", "player_id": null},
    {"champion": 15,  "role": "bot", "player_id": null},
    {"champion": 12,  "role": "sup", "player_id": null}
  ],
  "blue_team_id": "oe:team:4acc40654379b4da680be1bd3129dfa",
  "red_team_id": null
}
```

> `champion` 用 **ddragon 数字 key**（内部唯一 ID）。`player_id` / `team_id` 可为 null，缺失时走 §5.5 的 fallback。

**Response**

```json
{
  "p_blue": 0.583,
  "p_lo": 0.541,
  "p_hi": 0.627,
  "interval_method": "bootstrap_ensemble_20",
  "contributions": [
    {"name": "intercept",           "label": "基准（LEC 蓝方优势）", "delta_logit":  0.052, "delta_pct":  1.3},
    {"name": "team_rating_diff",    "label": "队伍实力 (MKOI vs NAVI)", "delta_logit": 0.241, "delta_pct":  5.9},
    {"name": "player_rating_diff",  "label": "选手实力",           "delta_logit":  0.061, "delta_pct":  1.5},
    {"name": "league_strength_diff","label": "赛区强度",           "delta_logit":  0.000, "delta_pct":  0.0},
    {"name": "champ_meta_diff",     "label": "英雄版本强度",       "delta_logit":  0.081, "delta_pct":  2.0},
    {"name": "context",             "label": "季后赛 / 第3局",     "delta_logit": -0.100, "delta_pct": -2.4}
  ],
  "layers": {
    "strength":  {"delta_pct": 7.4, "label": "队伍与选手实力"},
    "draft":     {"delta_pct": 2.0, "label": "阵容"},
    "context":   {"delta_pct": -1.1, "label": "赛区与赛况"}
  },
  "support": {
    "min_pair_decayed_games": 3.2,
    "n_unseen_players": 1,
    "n_unseen_champ_pos": 0,
    "min_champ_pos_n_eff": 18.4
  },
  "warnings": [
    "选手 NAVI SamD 未在训练数据中出现，已退化为 LEC 该位置的中位水平",
    "该阵容中有英雄组合的历史有效样本量仅 3.2 场，预测不确定性显著放大",
    "Renekton@mid 在近 75 天内有效样本量仅 18.4 场，版本强度估计不可靠"
  ],
  "reasons": [
    "MKOI 近期 Glicko 评分高于 NAVI 约 92 分，这是本次预测的主要来源（占 5.9 个百分点）",
    "蓝方阵容的版本强度略优（+2.0 个百分点）",
    "阵容对胜率的总贡献为 +2.0 个百分点；历史上这个量级通常在 ±1 至 ±3 个百分点之间"
  ],
  "meta": {
    "run_id": "2026-09-18T09:12:03Z-a3f9c1",
    "artifact_date": "2026-09-18",
    "data_age_days": 1,
    "calibrator": null,
    "calib_range": null,
    "ood": false
  }
}
```

**契约要点**：

- `contributions` 必须满足 `sum(delta_logit) + intercept == logit(p_blue)`，误差 < 1e-9。**写成单元测试**。
- `layers` 把「实力 / 阵容 / 赛况」三层分开——这是产品诚实性的核心（见 §7.6.2）。
- **`calib_range` 对主力模型是 `null`**（不加校准层，§5.6.1）。前端**只在它非 null 时**渲染死区阴影、只在它非 null 且触顶时显示校准上界 warning。**不要假装存在一个其实没有的边界。**
- **`warnings` 和 `reasons` 现在就加**，前端不写任何领域逻辑。
- `explain.py` 负责把 `b_bot_champ_games_decayed=2.1` 翻译成人话。**用户看不懂字段名，瀑布图配原始字段名等于没做解释。**

### 7.5.3 `POST /predict/batch`

```json
// Request
{"common": { /* league, patch, 其余 9 个位置固定 */ },
 "vary": {"side": "blue", "slot": 2, "candidates": [1,2,3, ... 173 个 champion key]}}

// Response
{"base_p": 0.583,
 "results": [{"champion": 92, "p": 0.611, "delta_pct": +2.8, "n_eff": 44.1},
             {"champion": 64, "p": 0.583, "delta_pct":  0.0, "n_eff": 61.7},
             ...],
 "top_up":   [92, 141, 234, 60, 254],
 "top_down": [83, 27, 72, 16, 3]}
```

### 7.5.4 `GET /live`

```json
[{"match_id": "115548681803406321",
  "game_id":  "115548681803406325",
  "game_number": 4,
  "league": "LEC",
  "patch": "16.17",
  "blue": {"team": "MKOI", "team_id": "103461966965149786",
           "players": [{"name":"Myrwn","role":"top","champion":"K'Sante","champion_key":897,
                        "esports_player_id":"110350224593953686","matched":true}, ...]},
  "red":  { ... },
  "p_blue": 0.612, "p_lo": 0.570, "p_hi": 0.655,
  "layers": {...}, "warnings": [...],
  "source_lag_s": 62,
  "predicted_at": "2026-09-18T18:12:20Z"}]
```

无实时数据时返回 `[]` + 一个顶层 `warning`，`/predict` 完全不受影响。

## 7.6 前端交互设计

### 7.6.1 三个 tab

**Tab 1「实时」** —— 读取当前正在进行的比赛（见 §7.7）

**Tab 2「What-if 模拟」** —— 核心功能

- 5v5 选人面板：10 个 `st.selectbox`，`format_func` 显示英雄名，配 `frontend/public/dd/16.18.1/img/champion/{alias}.png` **本地图标**
- 可选：选手下拉（从 `getTeams` 落盘的 1,185 支 active 队伍 / 4,770 名选手里按队伍联动带出常规首发）+ 赛区下拉
- 概率条 + **bootstrap 区间带** + **校准截断死区阴影**

**Tab 3「数据健康」**

`last_ingest_success` / `data_age_days`（OE 配额挂掉时这个会涨，一眼可见）/ `run_id` / `train_range` / `n_games` / `crosswalk_coverage` / 最近 10 次 leaderboard 记录 / 侧栏的 `st.file_uploader` 手动兜底上传。

### 7.6.2 三个必做的 UI 特性

**① 逐英雄替换敏感度（这类产品最有价值也最容易做的功能）**

固定其他 9 位，遍历该位全部 173 个英雄，按 |Δp| 排序展示 Top5 升 / Top5 降。

**⚠️ 先看清这个功能的真实量级，否则上线当天就会被判定为"工具坏了"。** 用户原始需求里最具体的一句是"换掉这个英雄变成 60%"，而 v1.0 只量化了整个 draft 层（±1~3 pp，5 个英雄合计）和 ΔlogLoss（[0, 0.008]），读者从这两个数推不出单次替换的幅度。按本文自己的参数推算：champion 层真实效应 SD = 0.0229（§4.2 G6），k≈476，若某 (champ,pos) 的 decayed `n_eff ≈ 50–150`，收缩因子只有 **0.10–0.24**，再乘 `β_champ = 0.975` → **单次替换的 Δlogit 约 0.02–0.05，即 p 在 0.5 附近只动 ±0.5~1.2 pp**；Top5 升与 Top5 降之间的全距大概只有 **2–3 pp**。而 bootstrap 区间宽度约 **8.6 pp**（§7.5.2 示例）——**也就是说大多数替换的影响小于本模型的不确定度** **[待验证，第 4 周用真实 artifact 实测，见阶段 5 验收⑦]**。

**因此 UI 规范（强制）**：

- Top5 升 / Top5 降**必须同时显示 `Δp` 与 bootstrap 区间宽度**，不能只给一个排名
- 列表下方固定一行说明：**"多数替换的影响小于本模型的不确定度（约 ±4 pp）。排序反映的是方向性倾向，不是可靠的差距。"**
- 这与 ② 的分层分解是同一个逻辑：**把这个事实做成产品诚实性的一部分，而不是等用户自己发现。**

**必须向量化，不能 for 循环调 predict**：

```python
# champ_meta 是可加的 → 换一个英雄只改一项
base_logit = coefs @ x_base
delta = beta_champ * (logit_theta[cand_keys, pos] - logit_theta[cur_key, pos])
p_all = 1/(1+np.exp(-(base_logit + delta)))    # 173 个候选一次算完，微秒级
```

若启用了 synergy/counter（非可加），用 numpy 花式索引批量算：

```python
syn = emb[blue_ids[:,:,None]] @ emb[blue_ids[:,None,:]].transpose(0,2,1)  # (N,5,5)
cnt = np.einsum('nid,de,nje->n', emb[blue], M, emb[red])
```

**延迟目标**：单次 p99 < 20 ms；1,730 行批量 p99 < 200 ms（含 Streamlit rerun 则 < 500 ms）。到不了先 profile `featurize_batch`，不要换模型。Streamlit 里必须 `@st.cache_data` 包住。

**② 分层胜率分解（产品诚实性的核心）**

```
┌───────────────────────────────────────────────────┐
│  蓝方胜率   58.3%   [54.1% ── 62.7%]              │
│  ████████████████████░░░░░░░░░░░░░                │
│  ░░ = 校准边界外的不可达区间（仅 calib_range≠null 时画）│
├───────────────────────────────────────────────────┤
│  基准（LEC 蓝方优势）           +1.3%   ▏          │
│  ── 实力层 ──────────────────  +7.4%              │
│     队伍实力 (MKOI vs NAVI)     +5.9%   ██████     │
│     选手实力                    +1.5%   ██         │
│     赛区强度                     0.0%              │
│  ── 阵容层 ──────────────────  +2.0%              │
│     英雄版本强度                +2.0%   ██         │
│  ── 赛况层 ──────────────────  -2.4%              │
│     季后赛 / 第 3 局            -2.4%   ███        │
└───────────────────────────────────────────────────┘
  ⓘ 阵容对胜率的贡献通常在 ±1 至 ±3 个百分点。
     总胜率主要由队伍与选手实力决定。
```

**为什么这一条是必需的**：总胜率主要由实力决定，用户换一个英雄可能只动 1–2 pp。**如果只显示总胜率，用户会觉得工具坏了。** 把两层分开显示，用户才能理解"阵容确实起作用了，只是量级就这么大"。

**③ OOD（分布外）警告**

draft 是**自适应均衡**：队伍只选自己练过、且当前版本强势的英雄，所以观测到的英雄胜率被推回 50%（champion 层真实胜率离散度 SD 只有 0.023）。用户手搓一个没人会选的阵容，本质是**分布外的因果外推**。

```python
min_support = support[blue_ids][:, blue_ids].min()   # decayed 共现计数
if min_support < 5 or n_offmeta >= 2:
    warnings.append(f"该阵容中有英雄组合的历史有效样本量仅 {min_support:.1f} 场，"
                    f"预测不确定性显著放大")
    p_lo, p_hi = widen(p_lo, p_hi, factor=1.8)
    reasons.append("这是基于历史相关性的外推，不是因果反事实——"
                   "模型无法回答「如果这支队伍真的这样选会怎样」")
```

**这一条是把玩具和作品分开的地方。**

### 7.6.3 离线可用的必要条件

```bash
# build 时把 173 张图 + champion.json 拉到本地，运行时只引本地路径
python -m lolwp.ingest.ddragon --download-assets --out frontend/public/dd/
```

**运行时打 CDN 就不叫"离线可用"。** 备用源 `raw.communitydragon.org/latest/.../champion-icons/{numericKey}.png`。

### 7.6.4 v1 Streamlit vs v2 Next.js

| | Streamlit 1.64.0（v0/v1） | Next.js + React（v2） |
|---|---|---|
| 出活速度 | **一天** | 一周+ |
| 英雄选择器 | `st.selectbox` + `st.column_config.ImageColumn` | 带搜索 + 键盘导航的网格（Streamlit 做不了） |
| 瀑布图 | `st.pyplot(shap.plots.waterfall(...))`，`shap 0.52.0` TreeExplainer 对树模型是精确解 | Recharts / visx |
| 交互 | 每次交互 rerun 整个脚本，必须 `@st.cache_data` | 局部更新 |
| 移动端 | 基本不能看 | 可控 |
| 极致离线 | 需要后端进程 | 可把模型转 ONNX（`skl2onnx 1.20.0` + `onnxruntime-web`）塞进浏览器，改英雄零延迟、后端可完全不存在 |

**建议**：Streamlit 做 v1 验证模型与交互流程，同时把 `/predict`、`/predict/batch`、`/explain` 契约用 pydantic 定死。v2 换 Next.js 时后端一行不改。

若做 ONNX 转换，务必加一个 "ONNX 输出 vs 原生输出 `np.allclose(atol=1e-5)`" 的测试。

## 7.7 实时比赛读取方案与降级策略

### 7.7.1 状态机（完整逻辑）

```python
# ── 每 60 秒 ──────────────────────────────────────────────
live = GET f"{BASE}/getLive?hl=en-US"                 # header: x-api-key
for ev in live["data"]["schedule"]["events"]:
    detail = GET f"{BASE}/getEventDetails?hl=en-US&id={ev['match']['id']}"
    for g in detail["data"]["event"]["match"]["games"]:
        if g["state"] != "completed":
            watchlist.add(g["id"])
        # ★ 绝不要写 if g["state"] == "unstarted": continue
        #   实测 state 在 BP 开始时就翻成 inProgress，而 feed 可连续 204 达 20 分 50 秒

# ── 每 30 秒，对 watchlist 里的每个 gameId ──────────────────
ts = floor_to_10s(utcnow() - timedelta(seconds=70))
r  = GET f"https://feed.lolesports.com/livestats/v1/window/{gid}" \
         f"?startingTime={ts:%Y-%m-%dT%H:%M:%SZ}"

match r.status_code:
    case 400 if "evenly divisible by 10 seconds" in r.text:
        raise RuntimeError("客户端时间戳对齐 bug")   # 永不该出现，CI 应拦住
    case 400 if "less than 50 sec old" in r.text:
        backoff(gid)                                 # startingTime 太新，退避重试
    case 204:
        state[gid].n204 += 1
        if state[gid].n204 * 30 > 15 * 60:           # 连续 204 超过 15 分钟
            mark_no_coverage(gid)                    # 该赛事无 livestats 覆盖
            ui_message = "该场次不支持实时读取，请到 what-if 页手动摆阵容"
        else:
            ui_message = "等待开局读条（通常 1-2 分钟）…"
    case 200:
        key = (gid, hash_of(r.json()["gameMetadata"]))
        if key in predicted_cache:
            continue                                  # ★ 幂等，同一局只预测一次
        draft = parse_metadata(r.json()["gameMetadata"])
        predicted_cache[key] = predict(draft)
        watchlist.discard(gid)                        # BP 结束后阵容不变，不必再轮询
```

### 7.7.2 三处必须的转换

```python
champion_key = ALIAS_TO_KEY[m["championId"]]          # 'KSante' -> 897
position     = ROLE_MAP[m["role"]]                    # 'bottom' -> 'bot'
patch        = norm_patch(meta["patchVersion"])       # '16.17.810.4348' -> (16,17)
player_id    = CROSSWALK.get(m["esportsPlayerId"])    # 数字雪花 -> oe:player:<hash>
```

全部由 §7.4.3 的 golden test 覆盖。

### 7.7.3 降级策略（架构级要求，不是 nice-to-have）

| 失效场景 | 降级行为 | 影响面 |
|---|---|---|
| `x-api-key` 被轮换 / 封禁 | `/live` 返回 `[]` + warning | what-if 完全不受影响 |
| feed 连续 204 > 15 分钟 | 标记无覆盖，提示手动摆阵容 | 单场 |
| OE ingest 失败 | 沿用上一份快照，Tab 3 显示 `data_age_days` 变红 | 模型略微过时，仍可用 |
| crosswalk 未命中某选手 | 退化到赛区×位置中位 rating + UI 标黄 | 单个选手 |
| 整机断网 | **全部功能正常**（本地图标 + 本地 artifact） | 只失去实时 tab |

**核心设计：手动摆阵容是默认模式，实时是增强。实时源整个消失时整站不受影响。**

这不是"额外实现的降级路径"——在 v1 的 Streamlit 单进程架构里它天然成立；到了 v2 docker compose 也要通过 `profiles: ["online"]` 保持这个性质（见 §7.8）。

### 7.7.4 轮询礼貌与安全

- `getLive` ≥ 60 s，`window` ≥ 30 s，带指数退避
- 每个请求带含你邮箱的 `User-Agent`，出问题对方能联系你而不是直接封
- **key 放 `.env`（`.gitignore`），不进公开仓库**
- **服务 bind `127.0.0.1`**，不要让浏览器用那个公开 key 高频 fan-out——**把 key 打到限流等于砍掉你唯一的免费实时数据源**

## 7.8 docker compose（v2）

```yaml
services:
  api:
    image: ghcr.io/astral-sh/uv:python3.11-bookworm-slim
    command: >
      sh -c "uv sync --frozen --no-dev &&
             uv run uvicorn lolwp.api.main:app --host 0.0.0.0 --port 8000 --workers 1"
    #                                                                     ^^^^^^^^^^^
    # ★ 必须单 worker：artifact 每 worker 一份，多 worker 内存线性翻倍
    #   （有先例记录其 feature store 常驻 2-4 GB）。横向扩就加容器实例。
    volumes: ["./data:/app/data:ro", "./models:/app/models:ro"]
    ports: ["127.0.0.1:8000:8000"]      # ★ 只绑本地
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      start_period: 300s                # ★ 冷启动要载入多年数据建索引
      # 有先例的 systemd 默认 90s 导致重启死循环
      interval: 30s
      retries: 3

  web:
    image: ghcr.io/astral-sh/uv:python3.11-bookworm-slim
    command: sh -c "uv sync --frozen --no-dev && uv run streamlit run src/lolwp/app/main.py"
    ports: ["127.0.0.1:8501:8501"]
    depends_on: {api: {condition: service_healthy}}

  scheduler:
    profiles: ["online"]                # ★ docker compose up 默认不起它
    image: ghcr.io/astral-sh/uv:python3.11-bookworm-slim
    command: sh -c "uv sync --frozen --no-dev && uv run python -m lolwp.live.scheduler"
    volumes: ["./data:/app/data", "./models:/app/models"]
    env_file: [.env]
```

```bash
docker compose up                      # 纯离线：api + web，手动摆阵容完全可用
docker compose --profile online up     # 加上每日 ingest + 实时轮询
```

**`profiles: ["online"]` 这一行的意义**：让"离线降级"成为**默认状态**，而不是一条需要额外实现和测试的路径。

**Makefile**（v1.1：每个 target 都给出真实 body，不再只留注释）：

```makefile
PY       := uv run python
YEAR     ?= 2026
SNAPSHOT ?= latest
RUN      ?=
DATE     := $(shell date +%F)

bootstrap:                 ## 环境 + 下载全量历史 + 建 silver + 训一版（§8.0）
	uv sync --frozen
	$(PY) -m lolwp.ingest.oracles_elixir --all-years
	$(PY) -m lolwp.ingest.ddragon --download-assets --out frontend/public/dd/
	$(PY) -m lolwp.store.bronze && $(PY) -m lolwp.store.silver
	$(PY) -m lolwp.models.train --model L3 --out models/bootstrap

daily:                     ## 每日 ingest（crontab 调这个）
	bash scripts/daily.sh

up:
	uv run streamlit run src/lolwp/app/main.py --server.address 127.0.0.1

up-online:
	docker compose --profile online up -d

train:
	$(PY) -m lolwp.models.train --config configs/main.yaml \
	      --out models/$(shell date +%Y%m%dT%H%M%S)

gate:                      ## 重训门禁，不通过则非零退出（§6.11.3）
	$(PY) -m lolwp.eval.gate --candidate models/candidate --baseline models/current

rollback:                  ## make rollback RUN=20260918T091203
	ln -sfn $(RUN) models/current.tmp && mv -Tf models/current.tmp models/current
	docker compose restart api web || true

backtest:
	$(PY) -m lolwp.eval.walkforward --models L0,L1,L1b,L2,L3,L4,L5,M1 --out preds/

ablation:
	$(PY) -m lolwp.eval.ablation --both-directions --out reports/$(DATE)/ablation.md

eval-report:
	$(PY) -m lolwp.eval.compare --preds preds/ --out reports/$(DATE)/

test:                      ## CI 绝不联网，只用 tests/fixtures/
	uv run pytest -q -m "not network"

reproduce:                 ## make reproduce SNAPSHOT=snap-2026-09-18
	$(PY) -m lolwp.eval.walkforward --snapshot $(SNAPSHOT) --seeds 0,1,2 \
	      --out reports/repro/
	diff -r reports/repro/ reports/archived/$(SNAPSHOT)/

backfill:                  ## make backfill YEAR=2026
	$(PY) -m lolwp.ingest.oracles_elixir --year $(YEAR) --force
	$(PY) -m lolwp.store.bronze --year $(YEAR)
	$(PY) -m lolwp.store.silver --year $(YEAR)
```

**CI（GitHub Actions）**：`uv sync --frozen` → ruff + mypy → pytest（**用 fixtures，绝不联网拉 OE/Leaguepedia**）→ 构建镜像。

另设 **nightly `data-health` workflow**：只对 `ddragon versions.json` / `getLive` / `embeddedfolderview` / `drive.usercontent.google.com`（HEAD 取 Content-Length）/ `cargoquery` 各打一次探活，失败自动开 issue。**这样数据源挂掉当天就知道，而不是下次 retrain 时才发现。**（`oe.datalisk.io` 不在探活清单里——它自己已冻结，探活会永远"成功"却毫无信息量，见 §3.2.3。）

## 7.9 再训练与上线门禁

见 §6.11.3。补充部署细节：

```bash
# promote
mkdir -p models/${RUN_ID} && cp -r artifacts/${DATE}/* models/${RUN_ID}/
ln -sfn ${RUN_ID} models/current.tmp && mv -Tf models/current.tmp models/current  # 原子
docker compose restart api web
echo "${METRICS_JSON}" >> models/leaderboard.jsonl

# rollback（30 秒内）
ln -sfn ${PREV_RUN_ID} models/current.tmp && mv -Tf models/current.tmp models/current
docker compose restart api web
curl -s localhost:8000/health | jq .run_id   # 确认已切回
```

`models/leaderboard.jsonl` **永不删**。

## 7.10 合规（公开部署前必做）

1. **页脚必须写 Riot 官方要求的免责声明原文**：
   > This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. League of Legends is a registered trademark of Riot Games, Inc.
2. **署名并链回 Oracle's Elixir**：
   > Data provided by Oracle's Elixir (oracleselixir.com), aggregated and released by Tim Sevenhuysen.
   用了 Leaguepedia / gptilt 要加 CC BY-SA 署名。
3. **绝不添加任何投注 / 赔率相关功能**。Riot General Policies 原文：
   > Your product cannot feature betting or gambling functionality.

   Riot API 政策也明确禁止将数据用于博彩用途。部分司法辖区对提供概率/赔率的服务有牌照要求。
4. **API bind `127.0.0.1`**，凭证进 `.gitignore` 且不打进 docker 镜像（用 volume / env 挂载）。
5. **OE 每天只 ingest 一次**。官方原文 "There is no value in downloading the files more frequently than this."，且配额挂在维护者个人账号上——**你的滥用会影响整个社区**。Leaguepedia 的 API 原文是 "made available as a courtesy"，同理。
6. 若将来商业化：避开 Kaggle 上 CC BY-NC-SA 4.0 的那个镜像；lolesports 的公开 key 在商业用途下应改走正式授权。

---

# 8. 分阶段开发流程

## 8.0 第一天的 20 条命令（逐行可复制）

v1.0 有 3,500 行代码片段和验收标准，却没有一条可以直接粘的起步命令；`make bootstrap` 只有注释没有 body；而 §6.1.3 让你第一天运行一个第 3 周才存在的包（`lolwp.eval.freeze_holdout`），形成一个解不开的依赖环。这一节把它补上——**这 20 条命令不依赖 `lolwp` 包里的任何东西**，半小时内跑完。

```bash
# ── 环境（1-3）─────────────────────────────────────────────
cd ~/PycharmProjects/LeagueProject
uv init --python 3.11 --no-workspace                                     # 1
uv add pandas numpy scikit-learn scipy pyarrow requests streamlit \
       duckdb catboost numpyro matplotlib pyyaml pydantic-settings       # 2
uv add --dev pytest ruff mypy                                            # 3

# ── 冒烟（4）───────────────────────────────────────────────
uv run python -c "import pandas,sklearn,numpy;print(pandas.__version__, sklearn.__version__)"   # 4

# ── 目录（5-6）：§7.2 的 40 多个文件里，阶段 0 只需要这 4 个 ──
mkdir -p data/raw docs experiments eval scripts src/lolwp tests/fixtures  # 5
printf 'data/\nmodels/\n*.csv\n.env\n' >> .gitignore                    # 6
#   阶段 0 只创建：experiments/preregistration.md、eval/holdout_gameids_<date>.txt、
#   scripts/day0_discover.py、scripts/day0_validate.py。其余全部留到对应阶段。

# ── 发现 13 个 fileId（7-8）────────────────────────────────
cat > scripts/day0_discover.py <<'PY'                                    # 7
import re, json, requests
FOLDER = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
html = requests.get(f"https://drive.google.com/embeddedfolderview?id={FOLDER}#list",
                    headers={"User-Agent": "lolwp/1.0 (qxing@uchicago.edu)"},
                    timeout=30).text
pat = re.compile(
    r'id="entry-([A-Za-z0-9_-]+)".*?flip-entry-title">([^<]*).*?'
    r'flip-entry-last-modified"><div>([^<]*)', re.S)
out = []
for fid, title, mod in pat.findall(html):
    m = re.match(r'(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv', title)
    if m:
        out.append({"year": int(m.group(1)), "file_id": fid,
                    "title": title, "drive_modified_raw": mod.strip()})
out.sort(key=lambda r: r["year"])
assert len(out) == 13, f"期望 13 个年度文件，实际 {len(out)} 个"
json.dump(out, open("data/oe_files.json", "w"), indent=2, ensure_ascii=False)
for r in out:
    print(r["year"], r["file_id"], repr(r["drive_modified_raw"]))
PY
uv run python scripts/day0_discover.py                                   # 8

# ── 下载 13 个 CSV（9-10）──────────────────────────────────
uv run python -c "
import json
rows = json.load(open('data/oe_files.json'))
open('data/oe_files.tsv','w').write(
    ''.join(f\"{r['year']}\t{r['file_id']}\n\" for r in rows))
print(len(rows), 'files listed')
"                                                                        # 9
cat > scripts/day0_fetch.sh <<'SH'                                       # 10
set -euo pipefail
UA="lolwp/1.0 (qxing@uchicago.edu)"
while IFS=$'\t' read -r YEAR FID; do
  OUT="data/raw/${YEAR}_LoL_esports_match_data_from_OraclesElixir.csv"
  if [ -s "$OUT" ]; then echo "skip $YEAR"; continue; fi
  echo "fetch $YEAR ..."
  curl -fsSL --retry 0 -A "$UA" \
    "https://drive.usercontent.google.com/download?id=${FID}&export=download&confirm=t" \
    -o "${OUT}.tmp"
  mv "${OUT}.tmp" "$OUT"
  sleep 2
done < data/oe_files.tsv
SH
bash scripts/day0_fetch.sh
#   任一年失败（拿到 2009 字节 HTML）→ 等 1h/3h/6h 重试；连续 2 天失败才走 §3.2.2 的人工兜底

# ── 五重校验（11-12）：§3.2.3 的函数整段落成可执行脚本 ─────
cat > scripts/day0_validate.py <<'PY'                                    # 11
import csv, os, sys, json, hashlib
from datetime import date, datetime, timedelta
EXPECTED_PREFIX = ("gameid,datacompleteness,url,league,year,split,"
                   "playoffs,date,game,patch")

def validate(path, year, prev):
    head = open(path, 'rb').read(4096)
    assert b'Quota exceeded' not in head, "Drive 配额错误页（HTTP 200 伪装）"
    assert not head.lstrip().startswith(b'<'), "拿到的是 HTML 不是 CSV"
    assert os.path.getsize(path) > 1_000_000, "文件过小"
    first = head.decode('utf-8', 'replace').split('\n')[0]
    assert first.startswith(EXPECTED_PREFIX), f"表头异常: {first[:80]}"
    n_rows, games, max_date, ncol = 0, set(), '', 0
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        rd = csv.DictReader(f); ncol = len(rd.fieldnames)
        for row in rd:
            n_rows += 1; games.add(row['gameid'])
            if row['date'] > max_date: max_date = row['date']
    assert len(games) > 0, "解析出 0 场比赛"
    assert n_rows == 12 * len(games), f"非 12 行/场: {n_rows} / {len(games)}"
    if year == date.today().year:
        age = date.today() - datetime.strptime(max_date[:10], '%Y-%m-%d').date()
        assert age <= timedelta(days=3), f"数据陈旧: max(date)={max_date}"
    if prev.get('n_games'):
        assert len(games) >= prev['n_games'] * 0.95, "场次倒退"
    return {"year": year, "n_rows": n_rows, "n_games": len(games), "n_cols": ncol,
            "max_date": max_date, "bytes": os.path.getsize(path),
            "sha256": hashlib.sha256(open(path,'rb').read()).hexdigest(),
            "fetched_at": datetime.utcnow().isoformat() + "Z"}

prev = {}
if os.path.exists("data/manifest.jsonl"):
    for line in open("data/manifest.jsonl"):
        r = json.loads(line); prev[r["year"]] = r
with open("data/manifest.jsonl", "a") as mf:
    for r in json.load(open("data/oe_files.json")):
        p = f"data/raw/{r['year']}_LoL_esports_match_data_from_OraclesElixir.csv"
        rec = validate(p, r['year'], prev.get(r['year'], {}))
        rec["drive_modified_raw"] = r["drive_modified_raw"]
        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(rec["year"], rec["n_games"], "games,", rec["n_cols"], "cols, max_date",
              rec["max_date"])
PY
uv run python scripts/day0_validate.py                                   # 12

# ── schema / 赛区 / firstPick 的首次复核（13-15）───────────
uv run python -c "
import pandas as pd, glob, os, re
for p in sorted(glob.glob('data/raw/*.csv')):
    year = re.match(r'(\d{4})_', os.path.basename(p)).group(1)
    c = list(pd.read_csv(p, nrows=1).columns)
    print(year, 'cols=%d' % len(c), 'firstPick=%s' % ('firstPick' in c),
          'idx=%d' % (c.index('firstPick') if 'firstPick' in c else -1))
"                                                                        # 13
uv run python -c "
import pandas as pd
d = pd.read_csv('data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv',
                usecols=['gameid','league'])
print(d.groupby('league').gameid.nunique().sort_values(ascending=False).to_string())
"                                                                        # 14
uv run python -c "
import pandas as pd
d = pd.read_csv('data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv',
                usecols=['gameid','side','firstPick','position'])
t = d[d.position=='team']
print('blue-first-pick 比例:', (t[t.firstPick==1].side=='Blue').mean())
"                                                                        # 15

# ── 冻结 holdout（16-17）：一条 pandas 命令，不依赖 lolwp 包 ─
uv run python -c "
import pandas as pd
d = pd.read_csv('data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv',
                usecols=['gameid','date'])
ids = sorted(d[d.date >= '2026-06-01'].gameid.unique())
open('eval/holdout_gameids_20260918.txt','w').write('\n'.join(ids))
print(len(ids), 'games in holdout window')
"                                                                        # 16
#   ★ 排除机制是 loader 里的 HOLDOUT_START 日期截止线，这个文件只是成员快照（§6.1.3）
echo "HOLDOUT_START = 2026-06-01  (UTC)" > eval/HOLDOUT_START             # 17

# ── 预注册 + 提交（18-20）─────────────────────────────────
$EDITOR experiments/preregistration.md      # 抄 §2.5 的模板                # 18
git add .gitignore pyproject.toml uv.lock scripts/ experiments/preregistration.md \
        eval/holdout_gameids_20260918.txt eval/HOLDOUT_START docs/PLAN.md   # 19
git commit -m "day0: env + OE full download + 5-gate validation + preregistration + holdout freeze"   # 20
git rev-parse HEAD   # ← 把这个 sha 抄进 preregistration.md 的头部
```

## 8.1 里程碑表（10 周，单人）

### 阶段 0 · 第 0 周（**约半小时，照 §8.0 的 20 条命令敲**，别跳过）

| 项 | 内容 |
|---|---|
| **目标** | 环境就位 + **自动跑通一次全量下载 + 五重校验** + 冻结 holdout + 预注册 |
| **交付** | Python 3.11 环境 + `uv.lock`；`scripts/day0_discover.py` / `day0_fetch.sh` / `day0_validate.py`；`data/raw/` 里 13 个年度 CSV（**必须含 2026**）+ `data/manifest.jsonl`；`eval/HOLDOUT_START` + `eval/holdout_gameids_<date>.txt`；`experiments/preregistration.md` 提交并记录 git sha |
| **验收** | ① `uv run python -V` 输出 3.11.x，`import pandas,numpy,sklearn,scipy,streamlit,pyarrow,requests` 无报错<br>② 13 个文件全部通过五重校验（非 HTML / >1MB / 表头前缀 / `rows == 12×n_games` 且 `n_games>0` / **当年文件 `max(date)` 距今 ≤3 天 + 同年 `n_games` 不低于上次快照的 95%**）<br>③ **2026 文件的 `max(date)` 距今 ≤3 天**（v1.0 的"场次 ≥5,439"作废——那个参照来自已冻结的 `oe.datalisk.io`，见 §3.2.3）<br>④ `eval/HOLDOUT_START` 已 commit，且 `holdout_gameids_<date>.txt` 行数 >500（**注意它只是成员快照，排除机制是 loader 里的日期截止线，§6.1.3**）<br>⑤ `manifest.jsonl` 13 条各有 `sha256` / `content_length` / `n_games` / `n_cols` / `max_date` / `fetched_at`<br>⑥ 跑完 §8.0 的第 13–15 条：记录下今天各年的**列数 / 是否含 firstPick / league code 全集 / blue-first-pick 比例**，作为 §0 纪律要求的基线快照 |
| **退路** | 直连当天失败（拿到 2009 字节 HTML）→ 1h/3h/6h 退避重试；**连续 2 天失败**才走 §3.2.2 的人工兜底（浏览器 Make-a-copy）或联系 Tim Sevenhuysen / 订阅 Patreon。若最终只拿到 2014–2025（冻结年份，陈旧无害），可先做前 6 周开发，**但必须在 Tab 3 显著标注"2026 数据缺失"，所有超参网格结论标记为 provisional**，绝不在缺 2026 的情况下宣布任何半衰期结论。 |

### 阶段 1 · 第 1 周

| 项 | 内容 |
|---|---|
| **目标** | 数据层做稳：发现 → 下载 → 五重校验 → 原子替换 → bronze |
| **交付** | `ingest/oracles_elixir.py`（含四级降级链与五重校验）、`ingest/ddragon.py`、`store/bronze.py`、`store/audit_columns.py`、`scripts/daily.sh` + crontab |
| **验收** | ① **注入一个 2009 字节的 quota HTML 作为 mock 下载响应，ingest 必须保留旧文件并 `exit 1`，且旧文件 sha256 不变**（写成 pytest）<br>② 注入一个"内容合法但陈旧"的 2026 镜像（**`date` 只到 2026-03 的陈旧快照**），**闸 5 的日期闸必须拦下它**；再注入一个"场次比上次快照少 20%"的截断文件，**单调不减闸必须拦下**（v1.0 那条"8,040 场 vs manifest 更高值"的用例作废，方向是反的）<br>③ 连跑 7 天 crontab：当年文件**每天都下载**；2014–2025 至少一天全部命中"Content-Length 未变 → 跳过下载"；且当年文件 sha256 未变的那天**不落新快照**<br>④ `audit_columns.py` 报告至少检出：2026 的 40 个 league code 全部在 `LEAGUE_CANON` 里（未登记则 CI 红）、**各年 `firstPick` 的列存在性与 blue 先手比例**（2025≈100% / 2026≈70.2%）、`atakhans` 2026 恒 0<br>⑤ 173 个英雄三体系往返映射单测 100% 通过 |
| **退路** | Drive 配额连续多日不恢复 → 切"周更 + 手动上传"模式，Streamlit 侧栏 `st.file_uploader` 作为一等公民而非兜底；模型对 1–2 天数据延迟极不敏感（老队伍 Elo 一天动不了 5 分），这个降级的实际代价接近 0。 |

### 阶段 2 · 第 2 周

| 项 | 内容 |
|---|---|
| **目标** | silver 宽表 + ID crosswalk + 单遍时序状态机 |
| **交付** | `store/silver.py`（12 行 → 1 行、series_id、league canonical + UNKNOWN、patch 归一化）、`ingest/normalize.py`、`features/priors.py` 状态机骨架、`configs/{league_canonical,player_overrides}.yaml` |
| **验收** | ① `df.game_id.nunique() == len(df)`<br>② **`silver` 里 LPL 场次 > 0**（专门防 `datacompleteness=='complete'` 过滤 bug）<br>③ **12 行/场通过率 = 100%；折叠后每场 10 个非空 `playerid` 的比例 ≥99%；`blue_player_ids`/`red_player_ids` 长度恒为 5、两个 team_id 恒非空（硬断言）**；并明确记录因缺 id 被整场丢弃的绝对数（2026 当前约 675 + 380 场）<br>④ **series 切分三条 pytest 全绿**（§2.2）：同 series 内 game 号为从 1 起的无重复连续序列 / 同 series 跨度 ≤12h / **全库不存在"同 league 同队对、相邻两局间隔 <6h 却 series_id 不同"的样本**（v1.0 那两条"非空率 100%、`game_in_series` ≤5"抓不到被劈开的系列，作废）<br>⑤ **crosswalk 分赛区验收**：`非 LPL 自动对齐率 ≥95%` + `LPL 自动对齐率 ≥85%` + `整体 ≥93%`；LPL 的匹配成功率**单独**打进 `/health`（§3.11.6）<br>⑥ `set(silver.league_raw) - set(LEAGUE_CANON) == set()`（未登记的 code 让 CI 红，而不是静默进 UNKNOWN 桶）；且每年 canonical 后 tier-1 场次不低于上一年的 70% |
| **退路** | window 批量回填因配额/限流走不通 → 退到"`getTeams` 干净 handle + entity_aliases 规范化"（裸匹配率 77.2%），缺口靠 overrides.yaml 人工补 200–400 条（约半天）。**注意：无论哪条路，覆盖率都只做 gauge + 告警，绝不做启动 assert。** |

### 阶段 3 · 第 3 周

| 项 | 内容 |
|---|---|
| **目标** | **先有裁判再有选手**：评估基础设施 + 四个负控制 + 功效计算 |
| **交付** | `eval/{walkforward,metrics,compare,power,negctrl,strata,gate,reliability}.py`、vendor 的 `compare_auc_delong_xu.py`、`features/whitelist.py` |
| **验收** | ① **label shuffle 负控制：完整 pipeline 跑完，log loss ∈ [0.685, 0.700]**（写成 pytest）<br>② **leak canary：注入 `gamelength` 后 AUC 提升 >0.10**，随后断言生产管道无此列<br>③ **未来删除不变性：随机 200 场 bit-identical 通过**<br>④ walk-forward 产出 ≥10 折、OOS ≥3,000 场，**任意 series 不跨 fold 边界**（断言）<br>⑤ `power.py` 在每张表脚自动打印 `n_games / n_series / m̄ / ICC / deff / n_eff / σ_d / MDE`，**其中 MDE 由 series 级 block bootstrap 的经验分布反解**（§6.3.1）<br>⑤b **`effective_n()` 在 L3-vs-L5 的 per-game 差值上产出实测 ICC，并据此把 §6.3.4 的裁决表第二行填满、给出 `mastery_diff` 的最终裁决**（这条裁决在此之前一律标"待定"）<br>⑥ 任一 fold AUC >0.78 时 CI 自动 fail（用注入泄漏特征的假模型验证门禁真的会 fail）<br>⑦ **加一个新模型进对比表 ≤20 行代码**（用 L1 演示） |
| **退路** | walk-forward 太慢（>1 小时/次）导致迭代不下去 → 先用"3 个固定 origin"的简化版跑通流程，第 6 周再补满 10 折；**但负控制和 MDE 打印这两条绝不能省**，它们是零成本的。 |

### 阶段 4 · 第 3–4 周（与阶段 3 部分并行）

| 项 | 内容 |
|---|---|
| **目标** | 钉死诚实基线 L0–L4，产出第一版对比表 |
| **交付** | `models/train.py`（分组 L2）、`preds/L0..L4.parquet`、`reports/<date>/model_comparison.md` |
| **验收** | ① 分组 L2 实现正确性：所有 λ 相同时与直接 sklearn L2 的 `max\|diff\| < 1e-8`<br>② **L3（team+player rating）在 tier-1 walk-forward OOS 上 log loss ∈ [0.620, 0.645]、Brier ∈ [0.216, 0.228]、AUC ∈ [0.67, 0.71]**——落在区间外说明管道有问题，回去查<br>③ 若任一档 log loss < 0.60 或 AUC > 0.75，视为失败，逐列排查泄漏后重跑<br>④ 对比表 5 行 × 13 列填满，每行带 series 级 bootstrap CI、DM p、**以及"是否超过 MDE"的自动判定**<br>⑤ **L4→L5 的判定被明确记录**（预期：低于 MDE，`mastery_diff` 不进主力） |
| **退路** | L3 打不到 0.620–0.645 → 按顺序查：(a) 是否误过滤了 partial；(b) series_id 是否正确；(c) Elo 更新顺序是否先快照后更新；(d) league canonical 映射是否漏了 code（`LTA`/`LTA N`/`LTA S` 以及 2026 改回的 `LCS`/`CBLOL`，见 §3.11.5）。**这四条覆盖了 90% 的情况。** |

### 阶段 5 · 第 4 周（关键：产品上线）

| 项 | 内容 |
|---|---|
| **目标** | **Streamlit 网站可用**：手动摆阵容 + what-if 敏感度 + 分层分解。**此时系统已能天天用** |
| **交付** | `app/main.py`（3 个 tab）、`app/predict.py`（纯函数）、`explain.py`、`frontend/public/dd/16.18.1/`（173 张 PNG） |
| **验收** | ① **断网状态下 `streamlit run` 完整走通「摆阵容 → 出胜率 → 看 what-if」，浏览器 Network 面板零外部请求**<br>② what-if 敏感度：遍历 173 个英雄，**p99 端到端 < 500 ms**（必须向量化）<br>③ `sum(contributions) + intercept == logit(p)`，误差 <1e-9（单测）<br>④ 三种情况各触发对应 warning 且不抛异常：未知选手 / 未知战队 / 低样本英雄组合（单测）<br>⑤ **分层分解 UI 把「实力 / 阵容 / 赛况」三层分开显示**<br>⑥ 页脚含 Riot 免责声明原文 + oracleselixir.com 署名链接<br>⑦ **对 100 个真实 draft 各跑一次全 173 英雄敏感度，报告单次替换 `Δp` 的分布（p50 / p95 / max）**，与 §1.2 的先验（0.5–1.2 pp）对照；若 p95 显著小于 bootstrap 区间宽度，UI 必须按 §7.6.2 ① 的规范同时展示 Δp 与区间宽度及那句免责说明 |
| **退路** | Streamlit 的 rerun 性能不可接受 → 把 `featurize_batch` 结果 `@st.cache_data(ttl=3600)`；仍不行则提前切 Next.js（但这会吃掉 1 周缓冲）。**绝不为了性能牺牲"第 4 周必须有可用产品"这个目标**——宁可先只做 Top5 敏感度不做全 173。 |

### 阶段 6 · 第 5–6 周

| 项 | 内容 |
|---|---|
| **目标** | 实力层做到最好（分层 BT + L_region）+ 实时接入 |
| **交付** | 全局单一 Bradley-Terry（选手潜变量 + `L_region` 3 年滚动 + team chemistry）、Glicko-2 双轨、一次性 NumPyro SVI 读出方差成分、`live/poller.py`、`app` 的实时 tab |
| **验收** | ① **对照口径写死**：BT 走"as-of rating 生产器"路线（§4.6.2），即每个 origin 在训练段内重拟合一次 BT、导出 `(r_player, L_region)`，主力 logistic 只消费 `player_rating_diff` + `league_strength_diff` 两个标量；与 **L3（plug-in Elo 版）** 配对比较，`ΔlogLoss ≤ −0.002` 且 series 级 `\|t\| > 2.5`（tier-1 口径）。若另建"BT 即模型"版本，作为对比表里独立的 **M2** 行报告，不替代本条<br>② **国际赛子集的 calibration slope ∈ [0.85, 1.15]**<br>③ 冷启动子集 log loss 不劣于该子集的 L1 蓝方基准率<br>④ NumPyro 输出 σ_player / σ_champion / σ_player:champ 后验 95% CI，写入 `configs/shrinkage_constants.yaml`，且与独立矩估计在同一量级（差异 <3 倍）<br>⑤ **实时：四种 HTTP 状态各有单测；对一场真实进行中的比赛端到端跑通并截图存档；从 window 首次 200 到页面显示 ≤35 秒**<br>⑥ **连续 204 超过 15 分钟自动切「无覆盖」状态**<br>⑦ `x-api-key` 置为无效值后，`/live` 返回空 + warning，what-if 不受影响<br>⑧ rating 更新逻辑通过 golden test：**在线 Elo/Glicko 走 T2 档 A（bit-identical）；BT / SVI 走 T2 档 B（固定 seed bit-identical + 跨 seed `max\|Δlogit\| < 1e-3`）**（§4.7.1） |
| **退路** | 分层 BT 不收敛或不优于 plug-in Elo → 保留 Elo 作为生产路径，BT 降级为 secondary analysis（它对国际赛仍有价值，单独在国际赛子集上用）。实时链路走不通 → 保持 v1 的纯手动模式，在 README 里记录"实时源不可用"及具体原因。 |

### 阶段 7 · 第 7–8 周（证伪区）

| 项 | 内容 |
|---|---|
| **目标** | draft 层：可靠性闸门 → 预训练 → zero-init α → 两折门禁 |
| **交付** | `reports/<date>/reliability_and_shrinkage.md`（split-half 表）、英雄 embedding 预训练（**严格前置期或 fold 内**）、zero-init α 的联合模型、`preds/L7.parquet` |
| **验收** | ① **split-half reliability 报告产出，并据此做出书面 kill/keep 裁决**：r_full < 0.20 的项直接砍掉并记录（预期 pair synergy 被砍）<br>② **embedding 预训练语料与评测期零重叠**（断言：预训练集的 max(date) < 最早 walk-forward origin），且跨源按 `riot_game_id` 去重<br>③ zero-init 回滚验证：α 全置 0 后指标退回阶段 6 水平，误差 <1e-6<br>④ 两折门禁：连续两个留出期 Δ 同号 + series 级 `\|t\|>2.5` + `\|Δ\| > MDE`，三条全满足才准进主力<br>⑤ **无论是否通过都算验收通过**，但必须给出 `α̂` 点估计 + 95% CI + MDE，且若 `\|Δ\| < MDE` 必须在 README 写明【低于可检出下限】 |
| **退路** | 这一阶段**本来就预期大概率失败**，这不是退路而是主线。若 synergy/counter 全部被砍，产品模型就是 L4 + 实力层升级，如实报告 `α̂ ≈ 0` 并把它作为项目的主要结论（预注册里已经写好了）。**省下的两周直接进阶段 8。** |

### 阶段 8 · 第 9 周

| 项 | 内容 |
|---|---|
| **目标** | 对照模型 + 完整消融矩阵 + 分层校准 + 外部对标 |
| **交付** | CatBoost（`has_time=True`）、zero-init DeepSet、等权 logit 平均、三组分层校准、Polymarket 采集器、`reports/<date>/full_comparison.md` + `ablation.md` |
| **验收** | ① **完整对比表 ≥12 行 × 13 列**，每行含 CI / DM p / 是否超 MDE / 三个分层 LL / 延迟<br>② **双向消融矩阵各 10–12 行**，全部带 CI 与 MDE 标注<br>③ 主力总体 ECE ≤0.02，三个分层组各 ≤0.03、slope 各 ∈ [0.90, 1.10]<br>④ Polymarket 交集子集 ≥200 张地图，所有模型在同一子集重算，配对 ΔBrier ∈ [−0.015, +0.015]（**超出正区间要先查泄漏**）<br>⑤ 集成若 ΔLL < MDE 则**不上线**，只保留主力，并在报告中写明原因<br>⑥ BH-FDR 应用于所有 exploratory 检验，报告明确标注 primary / exploratory |
| **退路** | CatBoost / DeepSet 训不动或结果诡异 → 这是"跑一次就够"的探索，直接如实报告"未能超越 logistic"并转下一阶段。Polymarket 交集样本不足 200 → 降级为定性对照，在报告里说明样本量限制。 |

### 阶段 9 · 第 10 周

| 项 | 内容 |
|---|---|
| **目标** | final holdout 一次性揭盲 + 监控 + 重训门禁 + 交付文档 |
| **交付** | `reports/final/technical_report.md`、三条监控曲线、`scripts/daily.sh` 完整门禁、`models/leaderboard.jsonl`、nightly data-health workflow、README |
| **验收** | ①b **揭盲前先打印 holdout 的 `n_games` / 日期范围 / 赛事构成，与预注册时的预期对照**；若 Worlds 2026 尚未打完，如实记录 holdout 的实际覆盖，**绝不事后扩窗**（扩窗 = 第二次看 holdout）。同时确认 CI 里 `assert (train.kickoff_utc < HOLDOUT_START).all()` 全程为真（§6.1.3）<br>① **holdout 评估脚本只运行一次**，运行前后 git sha 记录在案；**holdout 结果与 walk-forward 结果差异 <0.02 log loss**（差异过大说明过拟合了模型选择过程）<br>② 最终数字落在预注册区间：log loss ∈ [0.615, 0.645]、Brier ∈ [0.216, 0.228]、ECE ≤0.03、slope ∈ [0.93, 1.07]<br>③ 若触及红线，按预注册强制走完八条泄漏自查并记录结论<br>④ **报告明确写出 draft 特征组的增量估计 `α̂` 及其 95% CI，无论显著与否**<br>⑤ 门禁验证：人为注入退化模型（rating 置零），`make gate` 必须以 `t < −2.5` 拒绝 promote<br>⑥ 回滚演练 ≤60 秒完成且 `/health` 的 run_id 随之变化<br>⑦ `make reproduce SNAPSHOT=<id>` 从零跑完，与归档报告关键数字 diff 为空<br>⑧ nightly data-health 手动触发一次，人为改坏一个 URL 能自动开 issue |
| **退路** | holdout 结果明显差于 walk-forward（>0.02）→ **不要调模型去救它**，那是二次过拟合。如实报告这个 gap，并把它解释为"模型选择过程本身的乐观偏差"，这是一个诚实且有价值的观察。 |

**缓冲**：前面任何一周超期，吃掉第 7–8 周（证伪区）的时间——那一阶段本来就预期产出负结果，缩短它的代价最小。

## 8.2 全局退路矩阵

| 失败点 | 概率 | 退路 | 代价 |
|---|---|---|---|
| Drive 数据拿不到 | 中 | 周更 + 手动上传 + GitHub 镜像补冻结年份 | 数据延迟 1–7 天，几乎无影响 |
| 2026 数据始终缺失 | 低 | 所有时间衰减结论标记 provisional，README 显著说明 | 半衰期结论不可信，但模型仍可用 |
| crosswalk 覆盖率上不去 | 中 | 退化到"只看队伍不看人"，player rating 全部用赛区中位 | ΔLL 约 +0.004（失去 player 层增量） |
| 实时源被封 | 中 | 纯手动模式（v1 的默认状态） | 失去一个增强功能，核心功能不受影响 |
| 分层 BT 不收敛 | 中 | 保留 plug-in Elo | ΔLL 约 +0.002–0.005，国际赛校准变差 |
| draft 特征全部不显著 | **高（预期）** | 产品模型 = L4 + 实力层，如实报告 α̂≈0 | **这是预期结果，不是失败** |
| holdout 明显差于 walk-forward | 中 | 如实报告 gap，不调模型救它 | 诚实性 > 数字好看 |
| 整体延期 | 中 | 砍阶段 7（证伪区）和阶段 8 的探索模型 | 少两个负结果，核心交付不变 |

---

# 9. 对原始想法的逐条评价

> 你的原始设想是：「选手历史该英雄胜率 + 队友协同 + 对手克制 + 选手 ID + 赛区 + 时间衰减」。
> 方向完全正确——这六项恰好覆盖了这个问题的主要维度。但**实测下来，六项里只有两项半真正有预测价值**。逐条说。

## 9.1 选手历史该英雄胜率

| 维度 | 结论 |
|---|---|
| **是否有用** | ❌ **作为预测特征无用** ✅ **作为 UI 展示极有价值** |
| **预期边际贡献** | ΔLL ∈ [−0.002, +0.002]，**两个留出期符号相反**（+0.00159 / −0.00060），净增益 ≈ 0 |
| **正确实现方式** | 用**场次**不用胜率：`min(decayed_games, 14)/14`、`is_first_time_on_champ`、`min_champ_games`、`champ_pool_entropy` |

**为什么没用**（三条，按证据强度排序）：

1. **稀疏性（硬事实）**：2025 MAJOR 有 675 选手 × 159 英雄 = **107,325 个格子**，但只有 30,800 条观测，**平均每格 0.29 条**。全赛区口径 2022 主赛区 (选手,英雄) 单元中位数 = **2 场**，64.3% 只有 ≤3 场。
2. **样本外检验（决定性）**：滚动两折，Δ = +0.00159 (t=+2.82) 和 −0.00060 (t=−0.81)，**符号翻转**。训练集 logloss 反而下降——典型过拟合签名。
3. **收缩后退化**：无论 k 取 129 还是 700，中位 3 场的单元可靠度只有 1.5%–2.3%，收缩后 98% 等于"选手整体水平"，而那已经被 player rating 吃掉了。

**但你的直觉里有一半是对的**：「这个选手会不会玩这个英雄」确实重要，只是**信息不在胜率里，在场次里**。0→14 场舞台局有明显效果、之后饱和——这是 iTero（曾与 LEC 队伍教练组合作一个赛季）的实测结论。胜率的分母太小所以是噪声，而场次本身是个干净的计数。

**产品上必须保留**：

```
Faker · Azir    23 胜 9 负（原始 71.9%） → 收缩后 61%  ⓘ 样本量偏小
```

这个数字用户想看，而且加了收缩和样本量提示后**比裸胜率更诚实**。它进 UI，不进模型。

**⚠️ 一个必须纠正的证据错误**：调研里用"残差 SD = 0.019"作为砍掉这个特征的依据，**这个数字站不住**（2022 单年点估计，95% CI = [0, 0.051]；四年合并是 0.036–0.044，且矩估计有 +0.0005 正偏）。结论碰巧是对的，但**依据必须换成样本外门禁**——否则下次有人重跑矩估计拿到 0.044，就会把这个没用的特征加回来。

## 9.2 队友英雄协同

| 维度 | 结论 |
|---|---|
| **是否有用** | ❌ **在职业赛数据量下不可测量** |
| **预期边际贡献** | ΔLL ∈ [0, 0.001]，大概率为 0 |
| **正确实现方式** | 不建。若坚持要建：d=8 低秩 embedding + zero-init α + **必须先过 split-half r ≥ 0.20 的闸门** |

**三重独立证据，全部指向"测不出来"**：

| 证据 | 数字 |
|---|---|
| **稀疏性** | 173 英雄 → C(173,2) = **14,878 个 pair**。2025 Q1 一线 1,198 场里只观测到 3,590 个组合，中位 **3 场**，**没有一个达到 100 场**。四年 11,879 场主赛区也只到中位 7 场 |
| **方差分解** | 控制队伍基准 + 双方英雄主效应后，四年 24,194 个 team-side 的 pair 真实方差 **= −0.000456，bootstrap 95% CI = [−0.000614, −0.000293]，整段为负** |
| **split-half reliability** | **r ≈ 0.00**（695 对，独立来源实测） |
| **直接建模** | 显式 pair（9,990 参数）→ log loss **0.7165**；加克制（29,420 参数）→ **0.7756**；仅协同+克制 → **0.8242**。**全部比抛硬币的 0.6912 还差** |
| **样本外两折** | Δ = −0.00005 (t=−0.46) / +0.00002 (t=0.34)，回归系数 −0.086 / −0.035（接近 0 甚至为负） |

**注意这个结论的边界**：功效上限只允许说「效应 < 2–3 个百分点」，**不能说「效应 = 0」**。所以我不是直接删掉它，而是设了一道 split-half 闸门（§4.5.2），让数据在阶段 7 自己裁决。预期它会被砍掉，但那时你手里会有一份可以写进报告的 reliability 表，而不是我的断言。

**为什么职业赛测不出来，而玩家直觉觉得它很重要**：draft 是**自适应均衡**。队伍只会选自己练过、且当前版本能配合的英雄组合，所以观测到的组合胜率被推回 50%。**"协同重要"和"协同有边际预测力"是两回事**——前者可能为真，后者在观测数据上为假。

## 9.3 对手英雄克制

| 维度 | 结论 |
|---|---|
| **是否有用** | ⚠️ **比协同稍有希望，但仍需先过闸门** |
| **预期边际贡献** | ΔLL ∈ [0, 0.003] |
| **正确实现方式** | **不查表，用反对称双线性** `M = −Mᵀ`（d=8 时仅 28 个自由参数）；优先做**同位置对位**（5 个 lane matchup）而不是 25 个 cross pair 全开 |

**为什么比协同稍好**：有一条独立的职业侧实测佐证——iTero 用高分单排数据算出的 "Counter Multiplier"（对位胜率 / 各自基准胜率），在职业赛上有真实分化：

| 打野 SQ counter multiplier | 职业胜率 |
|---|---|
| < 0.93x | **45.6%** |
| > 1.07x | **54.4%** |

（每列至少 500 场职业数据。**中路是唯一例外，关系反向**。）

对线金币层面的迁移强度："每 +100G 的 SQ counter 在职业约对应 **+40G**"，即约 **4 折**。

**但代价很高**：这条路要么自采 solo queue（Riot dev key 24h 失效 + 100 req/2min，采 50 万场是多周独立工程 + production key 审批不可控），要么爬第三方聚合站（lolalytics/u.gg/op.gg 实测全部 403，ToS 不明）。**所以 v1 不做，放 v2 backlog。**

**v1 的替代**：用低秩 embedding 的 `lane_counter` 项，embedding 从职业赛数据预训练（严格前置期），过两折门禁才启用。

**如果你只能保留一个 draft 特征，不要选克制，选英雄版本强度**（见 9.7）。

## 9.4 选手 ID

| 维度 | 结论 |
|---|---|
| **是否有用** | ✅✅ **这是你六项设想里唯一有强信号的** |
| **预期边际贡献** | **ΔLL ≈ −0.0075**（相对纯 team rating），是全部 draft 特征加起来的 1–2 倍 |
| **正确实现方式** | **不要 one-hot，不要 embedding。把 rating 建在选手层。** |

**实测对比**：

| 模型 | log loss | Brier | acc | AUC |
|---|---|---|---|---|
| team Elo only | 0.6411 | 0.2255 | 62.1% | 0.675 |
| **player Elo only** | **0.6362** | 0.2231 | 63.6% | 0.684 |
| **team + player** | **0.6336** | **0.2220** | **63.8%** | **0.689** |

**为什么必须建在选手层而不是队伍层**：LoL 转会频繁，`teamid` 不变但阵容可能剧变。选手层 rating 转会后**自动继承**，而队伍层会滞后一整个赛季。`team_rating ≈ Σ(5 个 player rating) + chemistry 残差`。

**为什么不能 one-hot / embedding**：职业选手一年打 40–80 场。2,560 个 playerid × dim=16 = **40,960 个参数**，而训练样本才 3 万——必过拟合。正确做法是**层次部分池化**：选手效应 ~ N(队伍效应, σ²)，队伍效应 ~ N(赛区效应, σ²)。arXiv 实测队伍强度在 log-odds 尺度上 σ̂_θ = 0.5，可以拿它当先验 scale 的起点。

**实测的收缩强度**：PLAYER 层真实效应 SD = **0.1294** → k ≈ **14**；TEAM 层 SD = 0.1330 → k ≈ 13。对比 CHAMPION 层 SD 只有 0.0229 → k ≈ 476。

**这组数字是整个项目最重要的定量结论之一**：选手/队伍层的真实离散度是英雄层的 **5.6 倍**。你的直觉"选手很重要"完全正确，而且比你想的还重要。

**⚠️ 最大的工程风险也在这里**：OE 的 `oe:player:<hash>` 与 lolesports 的 `esportsPlayerId` 是两套独立体系。**匹配失败不会报错，只会让 player rating 变成先验值——模型悄悄退化成"只看队伍不看人"，指标只轻微下滑。** 必须走 §3.3.5 的自动回填 + 覆盖率监控。

## 9.5 赛区

| 维度 | 结论 |
|---|---|
| **是否有用** | ✅ **有用，但作用方式和你想的不一样** |
| **预期边际贡献** | 直接作为特征：ΔLL ≈ −0.001 ~ −0.003；**作为跨区强度校准：对国际赛子集是决定性的** |
| **正确实现方式** | 三个层次分开做，不要混成一个 one-hot |

**赛区在这个问题里有三个完全不同的角色**：

**① side 效应的载体**（最直接）。蓝方胜率实测 PCS 60.44% vs LLA 49.20%，差 11 个百分点。必须做成按赛区收缩的 as-of 估计（k=200–400），不能硬编码也不能自由参数化。

**② 强度尺度的锚**（最重要）。同一个 Elo 分在 LPL 和在 LJL 完全不是一回事。这是**国际赛预测的全部难点**。正确做法是单一全局分层 BT，`L_region` 由国际赛数据识别，**3 年滚动窗口**（Riot 官方 GPR 也这么做）。

**硬约束**：2022 年跨赛区比赛只有 **193 场**，桥接 43 个赛区对，最多的 LCK-LPL 只有 **30 场** → 赛区对之间的 log-odds 差 SE ≈ 0.40（约 **±70–80 Elo**）。**所以国际赛的预测区间必须在 UI 上明显放宽。**

**③ 可预测性的分层维度**（评估用）。LPL/LCK 样本多、强度分布宽 → LL 更低；VCS/LLA/CBLOL 接近 0.69。**必须分赛区报指标**，否则"模型其实只在两个大区有用"会被总体数字掩盖。

**一个实操陷阱（v1.1 加强）**：赛区代码在 2025 年发生了**破坏性变更**（LCS→`LTA N`、CBLOL→`LTA S`、新增 `LCP`，**还有一个独立的 `LTA`** 40 场），而 **2026 年又改回了 `LCS` / `CBLOL`**（`LTA N`/`LTA S`/`LTA`/`IC` 在 2026 全部不存在）**[实测 2026-09-18]**。所以这个坑有**两个方向**：不做映射，跨 2024/2025 会被切成两个互不相干的实体；而**照抄 2025 的 code 写 tier-1 集合，又会把 2026 的 477 场一线赛（LCS 227 + CBLOL 250）整体丢掉**。league code 总数本身也在变（2024=51 → 2025=43 → 2026=40）。**必须用双向、年份无关的显式映射表（§3.11.5 已给全 40 个 code）+ UNKNOWN 兜底桶 + 两条 CI 断言，绝不能用 `get_dummies` 现场生成，也绝不能在任何地方用原始 code 定义 tier-1。**

## 9.6 时间衰减（"侧重近期数据"）

| 维度 | 结论 |
|---|---|
| **是否有用** | ⚠️ **一半对一半错，而且错的那一半正是最直觉的那一半** |
| **预期边际贡献** | 特征内衰减：ΔLL ≈ −0.0025（紧贴 MDE）；训练样本权重衰减：**ΔLL ≈ 0 或为正（有害）** |
| **正确实现方式** | **把"近期性"放进 rating 的递归参数和特征内聚合的 λ 里，不要对训练样本整体加权** |

**你直觉里对的部分**：meta 确实在快速漂移。实测英雄 pick 分布的余弦相似度从 0–29 天的 **0.897** 衰减到 180+ 天的 **0.438**，中点 0.67 落在约 **75 天** → meta 半衰期 60–90 天。

**你直觉里错的部分**：对**训练样本整体**做时间衰减反而有害。实测半衰期 45 天 → 0.6351，90 天 → 0.6341，180 天 → 0.6336，**365 天 → 0.6335**，无衰减 → 0.6336。另有独立来源的配对门禁实验直接拒绝了三种半衰期和滚动训练窗口，结论是 "**Data volume beats recency, decisively**"。

**为什么**：漂移主要发生在"队伍强度"上，而这已经被在线 Elo/EWMA 的**递归更新**吸收了。再对样本做衰减 = **二次丢数据，纯亏**。

**而且样本量不允许激进衰减**：tier-1 每个 patch 只有 **142 场**，近 4 个 patch 仅 600–950 场。"只用近 4 个 patch"这个做法在数据上根本不成立。

**正确的四层结构**：

| 层 | 半衰期 | 作用 |
|---|---|---|
| champion meta（强度、offmeta 判定） | **75 天** | 特征值反映当前版本 |
| team / player rating | **180 天** | 人比 meta 稳 |
| 选手熟练度计数 | **180–365 天** | 熟练度衰减最慢 |
| **训练样本权重** | **365 天或不加** | 让系数偏向近期，但不能激进 |

**四个 λ 必须在同一次 walk-forward CV 里联合网格搜**（各 2–3 档，粗网格 12 组 / 完整 72 组），不能分别调。

**⚠️ 但必须带 MDE 约束**：实测的半衰期网格差异是 **0.0004–0.0025**，低于任何口径的 MDE。所以报告里能支持的结论只有一条：**"完全不衰减"确实更差（Δ≈0.0025，标记为 suggestive）**，而 45 vs 75 vs 180 天之间**不可分辨**。取 75 天的理由是它与独立观测到的 pick 分布相似度中点一致，不是因为它在网格上最优。

**额外建议**：加一个 `patch_age_days` 特征（距该 patch 上线的天数）。patch 刚上线时 meta 不稳定，模型应当自动降低对英雄项的信心——这比调 λ 更直接。同时把衰减按 **patch 数**而非天数计算作为一个变体去试，因为版本变化本质是离散事件。

## 9.7 你没想到、但更重要的东西

### ⭐ ① 概率校准（边际收益排第一，成本几乎为零）

你的需求里完全没提这一项，但它比所有 draft 特征加起来都重要。

**反面教材**：某未校准模型 Brier **0.2536**，比"永远输出基准率"的常数预测器 **0.2495** 还差——而 accuracy 看起来完全正常。Platt 后 0.2276、ECE 0.119 → 0.021。

**你的产品输出的是给人看的百分比**。AUC 0.70 对应的模型，大部分预测会落在 45%–65% 区间；不做校准很容易输出 80%+ 的过度自信预测，在直播/演示场景下会非常难看。

好消息：**用 log loss 直接训练的 logistic 天生就是校准的**（实测 ECE 0.0174，arXiv 报 slope 0.995），所以这一项对你**几乎零成本**——你只需要**测它、监控它**，而不需要额外做什么。

### ⭐ ② 队伍/选手实力项（占全部信号的 87.6%）

你的需求里只提到"选手 ID"，没有明确提"队伍实力"。但实测 team rating 一项就把 log loss 从 0.6912 拉到 0.6411，**占全部可解释信号 0.070 的 87.6%**。

**这意味着：如果你只有一周时间，全部投在实力建模上，一点 draft 特征都不做，你会得到 90% 的最终效果。**

这对产品定位是根本性的：叙事必须是「**队伍/选手实力 + 阵容微调**」。

### ⭐ ③ 评估协议本身（这个项目最大的技术含量）

网上 90% 的同类项目都没做对，这是你能拉开差距的地方，也是唯一能"经得起同行评议"的部分：

- series 级分组 + embargo
- 单遍时序状态机保证 as-of
- 四个负控制进 CI
- 每张表打印 MDE
- 内容寻址的快照 + `make reproduce`

### ⭐ ④ ID crosswalk（最容易毁掉项目的隐形工程）

OE 与 lolesports 是两套完全独立的 ID 体系。这个链路断了，模型会**悄悄**退化成"只看队伍不看人"——不报错、不崩溃、指标只轻微下滑。

而这恰恰是"实时读到阵容 → 查该选手历史"这个核心产品链路的必经之路。**第 2 周就要做，不能拖。**

### ⭐ ⑤ what-if 的分布外诚实性（产品差异化）

你的 what-if 功能本质上是在做**分布外的因果外推**。draft 是自适应均衡，队伍只选练过且版本强势的英雄；用户手搓一个没人会选的阵容时，模型给出的概率可信度很低。

**大多数同类产品对此只字不提，直接给一个数。** 你显示有效样本量 + 加宽区间 + 明确说明"这是相关性外推，不是因果反事实"，是把玩具和作品分开的地方。

### ⭐ ⑥ 分层展示胜率分解（产品体验的关键）

如果只显示总胜率，用户换一个英雄看到只动 1–2pp，会觉得工具坏了。**把「实力层 / 阵容层 / 赛况层」三层分开显示**，用户才能理解"阵容确实起作用了，只是量级就这么大"。

这一条既是产品设计，也是**诚实性**——它让用户自己看到 draft 的真实贡献量级，而不是被一个大号百分数误导。

## 9.8 你的六项设想 · 汇总评分表

| 你的设想 | 有用性 | 预期 ΔlogLoss | 是否超过 MDE | 正确实现 | 在 v1 中的角色 |
|---|---|---|---|---|---|
| **选手 ID** | ⭐⭐⭐⭐⭐ | **−0.0075** | ✅ 远超 | 选手层 rating + 层次池化 | **核心特征** |
| **赛区** | ⭐⭐⭐⭐ | −0.001 ~ −0.004 | ⚠️ 边缘 | 三个角色分开：side 先验 / L_region 校准 / 分层评估 | **核心特征 + 评估维度** |
| **时间衰减** | ⭐⭐⭐ | −0.0025（仅特征内） | ⚠️ 边缘 | 四层分开，联合网格；**训练样本权重不要衰减** | 特征构造参数 |
| 选手历史该英雄胜率 | ⭐（UI 上 ⭐⭐⭐⭐） | 0（两折符号翻转） | ❌（两折门禁口径）<br>⚠️ `mastery_diff` 那条单独的 MDE 裁决**取决于实测 ICC，第 3 周才定**（§6.3.4） | 用**场次**不用胜率 | **仅 UI 展示**（若 ICC 实测 <0.35 则重新评估是否进 L5） |
| 对手英雄克制 | ⭐⭐ | 0 ~ −0.003 | ❌ 大概率 | 反对称双线性 + lane matchup；需 soloQ 迁移才有希望 | v2 backlog，过闸门才启用 |
| 队友英雄协同 | ⭐ | 0 ~ −0.001 | ❌ | 低秩 embedding；**预期被 split-half 闸门砍掉** | v2 backlog |
| **（你没提）概率校准** | ⭐⭐⭐⭐⭐ | 校准误差从 0.119 → 0.021 | ✅ | logistic 天生校准，但要测 + 监控 | **必做，成本≈0** |
| **（你没提）队伍实力** | ⭐⭐⭐⭐⭐ | **−0.049** | ✅ 远超 | Glicko-2 / BT | **最核心** |
| **（你没提）评估协议** | ⭐⭐⭐⭐⭐ | 防止虚高 0.144 AUC | — | §6 全节 | **项目的技术含量所在** |

**一句话总结**：你设想的六项里，**「选手 ID」和「赛区」是真金，「时间衰减」半真半假，另外三项（选手英雄胜率、协同、克制）在职业赛数据量下基本不可测量**。而你没想到的「队伍实力 + 概率校准 + 评估协议」三项，才是决定这个项目成败的东西。

**另一条必须写进产品预期的数字**：你最想要的那个交互——"换掉这个英雄看胜率怎么变"——**单次替换的预期 |Δp| 只有 0.5–1.2 pp，全距 2–3 pp，小于模型自身的不确定度（约 ±4 pp）**。这不是工具坏了，这就是这个问题的信息量上限。UI 必须按 §7.6.2 ① 的规范把 Δp 和区间宽度一起显示，并把这句话直接印在界面上。

---

# 10. 附录

## A. 关键参考链接

### 数据源

| 名称 | URL | 状态 |
|---|---|---|
| Oracle's Elixir 下载页 | https://oracleselixir.com/tools/downloads | SPA，无静态链接 |
| **OE Google Drive 文件夹** | https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH | **唯一官方渠道** |
| OE 列名释义 | https://oracleselixir.com/definitions | |
| OE 数据字典 | https://lol.timsevenhuysen.com/matchdata/match-data-dictionary/ | 滞后，无 pick1-5 / datacompleteness |
| OE Discord（schema 变更通知） | https://discord.gg/GsauAGAr6 | 订阅 `oe-data-updates` |
| OE Patreon | https://www.patreon.com/oracleselixir | 建议支持 |
| lolesports API 社区文档 | https://github.com/vickz84259/lolesports-api-docs | openapi.yaml 3,183 行 |
| lolesports API 文档站 | https://vickz84259.github.io/lolesports-api-docs/ | |
| Data Dragon versions | https://ddragon.leagueoflegends.com/api/versions.json | |
| Data Dragon champion.json | https://ddragon.leagueoflegends.com/cdn/16.18.1/data/en_US/champion.json | |
| CommunityDragon | https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-summary.json | |
| Leaguepedia API 帮助 | https://lol.fandom.com/wiki/Help:Leaguepedia_API | 网页 403，用 `action=parse&prop=wikitext` |
| Leaguepedia 避免限流 | https://lol.fandom.com/wiki/User:Arbolitoloco/Avoiding_cargo_ratelimits | |
| HF gptilt matches | https://huggingface.co/datasets/gptilt/lol-esports-matches | 136,956 场 |
| HF gptilt entities | https://huggingface.co/datasets/gptilt/lol-esports-entities | 75,717 行别名 |
| Polymarket Gamma | https://gamma-api.polymarket.com/public-search | 无鉴权 |
| Polymarket CLOB | https://clob.polymarket.com/prices-history | 无鉴权 |

### 文献与基准

| 名称 | URL | 关键数字 |
|---|---|---|
| **arXiv 2609.08060**（最接近的 SOTA） | https://arxiv.org/abs/2609.08060 | 5,135 场，walk-forward **Brier 0.2207**，slope 0.995，与 Polymarket 不可区分 |
| Riot Global Power Rankings | https://lolesports.com/en-GB/news/dev-diary-unveiling-the-global-power-rankings | 准确率 **65%**；Elo-based、80/20、league Elo 3 年窗口 |
| Riot Win Probability (AWS) | https://lolesports.com/en-US/news/dev-diary-win-probability-powered-by-aws-at-worlds | **不含阵容数据**；开局固定 50% |
| ProjektZero-LoL-Model | https://github.com/MRittinghouse/ProjektZero-LoL-Model | ensemble 63.66% / LL 0.6425 / Brier 0.2255（无 draft）|
| iTero · draft 影响 | https://medium.com/the-esports-analyst-club-by-itero-gaming/lol-how-to-win-draft-in-pro-play-47470eed32e1 | 庄家约 63%；约 10% 的比赛受 draft 影响 |
| iTero · counter 迁移 | https://medium.com/the-esports-analyst-club-by-itero-gaming/lol-the-impact-of-counters-in-pro-play-28a2e21ed2e0 | SQ counter 迁移约 4 折；中路反向 |
| **Do et al., FDG 2021**（泄漏反面教材） | https://arxiv.org/abs/2108.02799 | 声称 75.1%，实为排位+随机切分+赛季胜率泄漏 |
| Semenov et al. 2017 (Dota2) | https://link.springer.com/chapter/10.1007/978-3-319-52920-2_3 | AUC 按技能分层 0.706/0.670/**0.660** |
| DraftRec (WWW'22) | https://github.com/dojeon-ai/DraftRec | 层级 Transformer，28 万场 soloQ，有开源代码+数据 |
| The Art of Drafting (RecSys'18) | https://web.cs.ucla.edu/~yzsun/papers/2018_recsys_drafting.pdf | MCTS 解 BP 组合博弈（v2 的"推荐下一手"参考）|
| DraftGap（可读源码） | https://github.com/vigovlugt/draftgap | MIT；logit 空间可加 + beta-binomial 收缩 + "减期望"定义 |
| CatBoost 论文 | https://arxiv.org/pdf/1706.09516 | ordered target statistics |
| LightGBM 参数 | https://lightgbm.readthedocs.io/en/latest/Parameters.html | `min_data_per_group=100` 等默认值很危险 |

### 工具

| 用途 | 链接 |
|---|---|
| Diebold-Mariano | https://pypi.org/project/dieboldmariano/ |
| DM + Harvey 修正 | https://github.com/johntwk/Diebold-Mariano-Test |
| DeLong（vendor 进仓库）| https://github.com/yandexdataschool/roc_comparison |
| Leaguepedia 客户端 | `mwrogue` + `pip install git+https://github.com/arbolitoloco1/mwcleric@empty_string_fix` |
| uv docker 集成 | https://docs.astral.sh/uv/guides/integration/docker/ |
| DuckDB 并发模型 | https://duckdb.org/docs/current/connect/concurrency.html |
| Riot 开发者政策 | https://developer.riotgames.com/policies/general |

## B. 常见坑清单

### B.1 数据坑（按危险程度排序）

| # | 坑 | 症状 | 防御 |
|---|---|---|---|
| 1 | **Drive 配额页伪装成 HTTP 200** | 2009 字节 HTML 覆盖整年 CSV，`csv.DictReader` 解析成 0 行不抛异常 | 五重校验（§3.2.3）；**注意 `0 % 12 == 0` 会放行** |
| 2 | **陈旧镜像通过全部内容校验** | 3 月的快照让时间衰减权重压在过期 meta 上 | 闸 5 单侧：`max(date)` 距今 ≤3 天 + 同年 `n_games` 不低于上次快照 95%。**绝不用 `oe.datalisk.io`（它自己已冻结）** |
| 2b | **新鲜度参照系本身过期** | 每天拒收最新鲜的正确数据，pipeline 开箱即死 | 参照系只能是"数据自身的日期"和"我们自己上次的快照"，不能是第三方接口 |
| 3 | `df[df.datacompleteness=='complete']` | 2022 年删掉整个 LPL | 整场判定：`groupby('gameid').filter(...)`（§3.2.5） |
| 3b | **在 12 行长表上 `dropna(subset=[...])`** | 产出"只有 4 个选手的一方"，`/5` 的分母算错，模型照样输出一个数（2026 实测 675 场） | 先聚合判定完整性、整场保留或整场丢弃 + silver 写出前的长度断言 |
| 4 | 按列位置索引 | OE 静默增删列（164↔165） | 一律按列名 + 白名单 |
| 5 | patch 编号跨源差 10 / 零填充不一致 | 衰减权重算错，**指标只略微变差不报错** | `norm_patch()`（§3.10.1） |
| 6 | 赛区改名（2025 LCS→LTA N，**2026 又改回 LCS**）+ 独立的 `LTA` | **两个方向都会静默丢数据**：不映射丢 2025，照抄 2025 code 丢 2026 的 477 场 | 双向年份无关的 canonical 表（§3.11.5 全 40 个 code）+ UNKNOWN 桶 + 两条 CI 断言 |
| 7 | 跨年列语义漂移（atakhans 恒 0、turretplates 15→45、firstPick 蓝方先手 100%→70.2%） | **任何统计方法都发现不了** | `audit_columns.py` **每日**复核（不是一次性断言，§0） |
| 7b | **把"本机快照上的观测"当成"数据源的不变量"** | v1.0 据此要求读者推翻 arXiv 的 baseline 设计（`firstPick` 那条） | schema 类 [实测] 必须带测量日期 + snapshot sha；OE 会**回填历史年份** |
| 8 | 英雄名 21 处 alias ≠ display | 特征变缺失值，模型照样输出 | 数字 key 做主键 + 往返单测 |
| 9 | 队名映射（40 个里 18 个需映射） | difflib 把 `Beijing JDG Esports` 映射成 `LNG Esports` | 显式 dict，**禁止 fuzzy** |
| 10 | LPL 的 gameid 不是 Riot platformGameId | 跨源 join 静默丢行 | `(date, 两队, 局序)` fallback + 命中率监控 |
| 11 | `RiotPlatformGameId` 新入库为空串 | 同上 | 同上 |
| 12 | OE 回填修订历史（同年不同快照差 4%） | "重跑相同代码得到不同结果" | 内容寻址快照 + snapshot_id |
| 13 | Leaguepedia 被限流后重试续期惩罚 | "怎么试都不行" | 成功后 sleep 75–90s，指数退避 |
| 14 | Leaguepedia Row ID 会变 | 主键失效 | 用 `GameId` / `OverviewPage` |

### B.2 建模坑

| # | 坑 | 实测虚高 | 防御 |
|---|---|---|---|
| 1 | 随机划分 + 赛季聚合 | AUC 0.650 → **0.794** | walk-forward + as-of |
| 2 | 按行切分（12 行/场） | 标签直接泄漏 | 先折叠成一场一行 |
| 3 | **series 跨 fold** | 最隐蔽 | `series_id` 分组 + 7 天 embargo |
| 3b | **`series_id` 按 UTC 日历日构造** | 跨零点的系列被劈成两半（2026 实测 89 个），fold 分组静默失效、n_series 虚高 | 6 小时间隔的会话切分 + §2.2 的三条 pytest |
| 4 | rating 先跑完赛季再 join | 巨大 | 单遍状态机：先快照后更新 |
| 5 | **embedding 预训练覆盖评测期** | 可能很大 | 严格前置期或 fold 内重估 + 跨源去重 |
| 6 | scaler / 校准器在全量 fit | 小但真实 | fold 内估计 |
| 7 | 用 `datacompleteness` 当特征 | 泄漏赛区 | 禁用 |
| 8 | 参数量 > 样本量/50 | 29,420 参数 → LL 0.7756 | §5.1 的曲线 |
| 9 | 直接喂高基数 ID 给 GBDT | target statistic 方差巨大 | 只喂工程化标量 |
| 10 | 依赖 `xlearn` / `torchfm` | py3.12+ 编译失败 | PyTorch 手写 30 行 |

### B.3 评估坑

| # | 坑 | 后果 | 防御 |
|---|---|---|---|
| 1 | **game 级而非 series 级配对** | t 虚高（幅度取决于 ICC），可能选错主力模型 | series 级 block bootstrap 直接出 CI；`deff = 1+(m̄−1)·ICC` **实测**，绝不用手工常数 |
| 2 | **不打印 MDE** | 把 0.0004 的网格差写成设计结论 | `power.py` 强制页脚（含 m̄ / ICC / deff / n_eff） |
| 2b | **用一个未经测量的常数当有效样本量** | 整条最重要的裁决可能是错的（本文 v1.0 就栽在 `/2.5` 上） | 任何进入决策的常数，要么有实测、要么有推导，两者皆无就不准写 |
| 3 | 用效应量 `\|Δ\|/SD` 做门槛 | 永远无法累积证据 | 用 `t = Δ/(SD/√n)` |
| 4 | 比 fold 均值而非逐 fold 相减 | 敏感度降 5 倍 | 配对 |
| 5 | 混池买功效（49 赛区）| 换了估计对象，答非所问 | tier-1 报主结论 |
| 6 | 只报 accuracy | 检测不出校准失败（Brier 比常数预测还差） | log loss + Brier + ECE |
| 7 | 把别人的 Brier 当 log loss | 0.22 Brier ≈ 0.63 LL，差一个数量级 | 确认指标定义 |
| 8 | 多重比较失控（120 次检验） | 期望 6 个假阳性 | 预注册 primary + BH-FDR |
| 9 | 尝试 20 个模型报最好的 | 虚假"提升"是大概率事件 | 预注册 + 独立 holdout |
| 10 | 单 patch 分桶做因果解读 | 噪声极大 | 滚动 200 场窗口 |
| 11 | isotonic 校准（几千场） | 严重过拟合台阶（t=−4.74） | 用 Platt / temperature |
| 12 | conformal prediction 出区间 | 二分类下给标签集合，没用 | bootstrap / Venn-ABERS |

### B.4 工程坑

| # | 坑 | 症状 | 防御 |
|---|---|---|---|
| 1 | **loader 在 `features/` 包外** | api 用 2 年、train 用 5 年，**无任何报错** | loader 进包 + `/health` 暴露 `n_games` |
| 2 | **crosswalk 覆盖率做成启动 assert** | 转会期整站起不来 | gauge + 告警 + 单选手退化 |
| 3 | 用 `state` 字段做实时触发器 | state=inProgress 时 feed 可 204 达 20 分钟 | 204→200 跳变 |
| 4 | 单次 204 就判定无覆盖 | 前端永远转圈 / 过早放弃 | 15 分钟超时判定 |
| 5 | 不带 `startingTime` 调 window | 返回 20 分钟前的陈旧窗口但长得像当前 | 显式传 `now-70s` 对齐 10s |
| 6 | FastAPI 多 worker | artifact 每 worker 一份，内存线性翻倍 | `--workers 1`，横向扩加容器 |
| 7 | healthcheck `start_period` 太短 | 冷启动重启死循环 | 300s |
| 8 | 推理路径连 DuckDB | retrain 写锁时 API 连不上 | 推理只读 artifact |
| 9 | what-if 用 for 循环调 predict | 延迟从 5ms 涨到 2s | numpy 向量化 |
| 10 | 运行时打 CDN 取图标 | 断网不可用 | build 时落盘 173 张 |
| 11 | key 硬编码 / 进公开仓库 | 被限流 = 砍掉唯一实时源 | `.env` + `.gitignore` + bind 127.0.0.1 |
| 12 | CI 联网拉 OE | 配额耗尽时 CI 全红 | 用 `tests/fixtures/` |

## C. 术语表

| 术语 | 含义 |
|---|---|
| **BP** | Ban/Pick，英雄禁选阶段。本项目的预测时点 |
| **as-of** | 特征值只用某个时刻之前的信息计算，是 point-in-time correctness 的实现 |
| **point-in-time correctness** | 保证任一特征在预测时刻真实可得，不含未来信息 |
| **walk-forward / rolling-origin** | 时间序列交叉验证：训练窗口滚动前移，测试窗口始终在训练之后 |
| **expanding vs sliding window** | 训练窗口是累积增长还是固定长度滑动 |
| **embargo** | fold 边界前后留出的空白期，防止递归状态跨 fold 渗透 |
| **series_id** | Bo3/Bo5 系列赛的分组键。**本项目最重要的分组键** |
| **MDE** | Minimum Detectable Effect，给定样本量和功效能检出的最小差异。**本项目一律由 series 级 block bootstrap 的经验分布反解：`MDE = 2.8·sd(boot)`**（§6.3.1） |
| **ICC** | 组内相关（intra-class correlation）。这里指同一 series 内 per-game log loss 差值 `d_i` 的相关程度。**必须实测，不得假设**（§6.3.1 的 `effective_n()`） |
| **design effect (deff)** | `deff = 1 + (m̄−1)·ICC`，聚类抽样导致的方差膨胀因子。`m̄` = 每系列平均局数，**实测 2.14** |
| **n_eff** | 有效样本量 `= n_games / deff`。**v1.0 的 `n_games/2.5` 已废弃**——它等价于假设 ICC=1 且 m̄=2.5，两个数都不对 |
| **σ_d** | per-game log loss 差值的配对标准差。实测 0.089–0.126 |
| **block bootstrap** | 以 series 为单位重抽的 bootstrap，避免 CI 偏窄 |
| **Diebold-Mariano (DM)** | 预测精度差异的配对检验，需传 log loss 作为 cost function |
| **DeLong** | 两个 AUC 差异的检验 |
| **BH-FDR** | Benjamini-Hochberg 假发现率控制，比 Bonferroni 功效高 |
| **log loss** | `-[y log p + (1-y) log(1-p)]` 的均值。**主指标**，严格适当 |
| **Brier score** | `(p̂ - y)²` 的均值 |
| **Murphy 分解** | `BS = Reliability − Resolution + Uncertainty` |
| **Reliability** | 校准误差分量，应 ≈0。**漂移时它先涨** |
| **Resolution** | 区分力分量，是模型的真实增量（≈0.028） |
| **Uncertainty** | `p̄(1-p̄)`，降不下去的部分（≈0.249） |
| **ECE / MCE** | Expected / Maximum Calibration Error，用 15 个**等频** bin |
| **calibration slope** | 把 `logit(p̂)` 作唯一自变量重做 logistic 的斜率，应 ≈1。**<1 = 过度自信 = 过拟合指纹** |
| **strictly proper scoring rule** | 严格适当评分规则：只有报告真实概率才能最优 |
| **Platt scaling** | 用 sigmoid 做后处理校准，2 个参数，适合小校准集 |
| **temperature scaling** | 单参数校准，最省数据 |
| **isotonic regression** | 非参数校准，需大校准集（约 3.5 万），几千场会过拟合 |
| **Venn-ABERS** | 给概率区间且有有效性保证的校准方法 |
| **empirical Bayes (EB) 收缩** | `p̂ = (Σw·y + k·p_prior)/(Σw + k)`，`k = σ²_noise/σ²_effect` |
| **partial pooling / 层次收缩** | 把子层估计向父层收缩，收缩强度由观测数决定 |
| **split-half reliability** | 数据对半分各算一次统计量再求相关，Spearman-Brown 修正 `r_full = 2r/(1+r)` |
| **Bradley-Terry (BT)** | 成对比较模型：`logit P(A>B) = r_A − r_B` |
| **Elo / Glicko-2 / TrueSkill / OpenSkill** | 在线 rating 系统。Glicko-2 的 RD、TrueSkill 的 σ 提供不确定度 |
| **zero-init α** | 组件系数初始化为 0，保证最差退化为 baseline，且 `α̂` 的 CI 就是该组件的价值区间 |
| **反对称双线性** | `M = −Mᵀ`，用来表达 counter 的方向性，d=8 时仅 28 个自由参数 |
| **低秩分解 / embedding** | 用 `173 × d` 个参数替代 `173²` 个 pairwise 参数 |
| **FM (Factorization Machine)** | 二阶交互的低秩模型。本项目实测过拟合 |
| **ordered target statistics** | CatBoost 防 target leakage 的机制，需设 `has_time=True` |
| **OOD (out-of-distribution)** | 分布外。what-if 里用户手搓的罕见阵容属于这一类 |
| **train-serve skew** | 训练与线上推理的特征不一致。**不报错，只让模型悄悄变差** |
| **golden test** | 固定输入 → 固定输出的回归测试 |
| **negative control** | 负控制：label shuffle / 时间反转 / leak canary |
| **leak canary** | 故意注入泄漏列验证检测机制灵敏度 |
| **datacompleteness** | OE 字段，complete/partial。partial **只缺时点统计，BP 信息完整** |
| **platformGameId** | Riot 的比赛 ID。LPL 没有（走腾讯源） |
| **esportsPlayerId** | lolesports 的数字雪花 ID，与 OE 的 `oe:player:<hash>` 无关 |
| **crosswalk** | 跨源 ID 对照表。本项目最关键的工程产物之一 |
| **livestats window feed** | `feed.lolesports.com/livestats/v1/window/{gameId}`，实时阵容来源 |
| **artifact** | 推理用的预计算包（rating 表、θ 表、embedding、系数），几 MB |
| **hive partitioning** | 按目录名编码分区键的 parquet 布局，DuckDB 原生支持 pushdown |
| **flex pick** | 可在多个位置使用的英雄。实测真正的 flex 只有 27 个 |
| **自适应均衡** | draft 的内生性：队伍只选练过且版本强势的英雄，导致观测胜率被推回 50% |

---

## 结语：这个项目真正能交付什么

诚实地说，**模型准确率不会是亮点**——上限就在 63–66%，而且这已经与预测市场持平。

真正有价值、别人没做过、且你能做扎实的是这**五个带置信区间的数字**：

1. **`α̂`（draft 项系数）及其 95% CI** —— "BP 到底值多少胜率"的一个带区间的答案
2. **最优 `H_meta` / `H_train` 的实证值** —— "版本变化让老数据贬值多快"的定量回答，直接回应你"侧重近期数据"的直觉是对是错
3. **各层收缩强度 `k` 的实测标定**（champion≈476 vs player≈14）—— "英雄层几乎全是噪声、选手层才是真信号"的量化证据
4. **split-half reliability 表** —— "哪些特征在职业赛数据量下根本不可测量"的可证伪结论
5. **完整的 baseline 阶梯 + 双向消融矩阵**，每一行都带 n、CI 和 MDE

这五个数字配上一个校准良好的本地网站和一个 `make reproduce` 的复现包，比一个 65% 的模型有价值得多。

**而且，如果结论是"你设想的三个阵容特征都不显著"——那不是失败，那就是结果。** 预注册文件里已经写好了这句话。
