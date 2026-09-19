# 预注册 v1

> 冻结日期：2026-09-18
> 冻结时的 git sha：`599bfd6fd7baf2fbca9f08321836fd640765ab46`（本文件提交后回填）
> **在本文件被 commit 之前，不得对 holdout 做任何评估。**
> 依据：docs/PLAN.md §2.5

## 0. 冻结的口径

| 项 | 值 | 文件 |
|---|---|---|
| holdout 起始日 | `2026-06-01`（UTC） | `eval/HOLDOUT_START` |
| holdout 成员快照 | 1,071 场 | `eval/holdout_gameids_20260918.txt` |
| 评测赛区 | tier-1 + 国际赛 | `eval/tier1_leagues.yaml` |
| 训练赛区 | 全部（训宽评窄） | 同上 |
| 训练数据 | OE 2014–2026，102,084 场 | `data/manifest.jsonl`（13 条 sha256） |

**排除机制是 loader 里的日期截止线，不是这个成员名单文件。** 名单只是冻结时刻的快照；
2026 世界赛尚未开打，其比赛会在赛后自动落入 holdout 窗口。任何 `date >= HOLDOUT_START`
的比赛一律不得进入训练集或超参搜索。

## 1. Primary comparison（唯一一个）

```
M_main(全特征)  vs  L3(side_prior + team rating + player rating)
```

- 指标：**log loss**
- 样本：tier-1 + 国际赛子集的 walk-forward OOS
- 检验：series 级 block bootstrap + Diebold-Mariano（Harvey 小样本修正）
- 其余所有比较一律标记 **exploratory**，用 Benjamini-Hochberg FDR 控制

## 2. 成功判据

- (a) walk-forward Brier ≤ 0.228 且 calibration slope ∈ [0.93, 1.07]
- (b) draft 特征组相对无 draft 基线的配对 ΔlogLoss ≥ 0.004，
  且在 series 级聚类下 |t| > 2.5，且**连续两个留出期同号**

## 3. 失败判据（红线，CI 自动 fail）

```
任一 fold AUC > 0.78   或   全样本 accuracy > 67%   或   log loss < 0.60
```

触及任一条 → 强制走完 PLAN.md §6.9 的泄漏自查清单，结论写进报告，不得跳过。

参照实测：随机划分 + 整赛季聚合胜率会把 AUC 从诚实的 0.650 推到 0.794。
红线抓的就是这种。

## 4. 待定事项（不得事后选择对自己有利的那个）

**ICC 尚未测量。** PLAN.md v1.1 已删掉未经推导的 `n_eff = n_games / 2.5`。
真实的 design effect `deff = 1 + (m̄ − 1)·ICC` 要在阶段 3 用 `eval/power.py::effective_n()`
实测得出。在测出之前：

- `mastery_diff` 是否进主力模型 = **【裁决待定】**
- 裁决规则在此刻就写死：ICC 实测值出来后，按 §6.3 的门禁一次性判定，不得反复试

## 5. 提前声明

> 若 draft 特征不显著，我们将如实报告 α̂ ≈ 0 及其 95% CI，
> 并把「BP 阶段在职业赛数据量下贡献不可测量」作为本项目的主要结论。
> **这不是失败，这是结果。**

已知的先验证据（均为实测，见 PLAN.md §1.2 / §9）：

- 队伍 Elo 单项吃掉 log loss 0.0613，占全部可解释信号 0.070 的 87.6%
- Elo 之上加 171 维英雄 one-hot：0.6336 → 0.6342（变差）
- 显式 pairwise 协同 + 克制（29,420 参数）：0.7756，比抛硬币的 0.6912 还差

因此本项目的预期结论就是 α̂ ≈ 0。提前把它写下来，是为了防止后期因为
「做了两个月才涨 0.4pp」而去「修」出一个有泄漏的高分模型。

---

# 修订记录

预注册允许修订，但必须满足三个条件：**在任何结果数据被查看之前**、**公开记录**、
**说明理由**。以下每条都留了证据。

## A1 · 修复 tier-1 赛区集合的两处事实错误（2026-09-19）

**合法性前提（修订时的仓库状态，可用 git 复核）**

| 检查项 | 状态 |
|---|---|
| `models/` 目录 | 不存在 |
| `preds/` 目录 | 不存在 |
| `reports/` 目录 | 不存在 |
| git 全历史中是否出现过上述任一目录 | 否 |
| 源码中是否存在任何拟合/评估代码（`fit(` / `LogisticRegression` / `log_loss` / `roc_auc`） | 否 |
| 修订时 HEAD | `f7dfddde5e033fae999e02dcdef42c5bda4055c7` |

即：**尚无任何模型被训练，尚无任何 holdout 评估被执行**。本次修订不可能受结果影响。

**错误 1：世界赛 code 大小写写错**

v1 写的是 `WLDS`，OE 里的实际 code 是 `WLDs`（小写 s，1,349 场，2014-09-18 至今）。
后果：`eval/holdout_gameids_20260918.txt` 漏掉了 2026 世界赛已开打的 8 场
（2026-09-17、09-18），且随着世界赛进行会漏掉越来越多。
**国际赛恰恰是这个模型最该被检验的场景**（跨赛区强度校准是 §4.6 的核心难点），
漏掉它等于把最难的子集从评测里剔除了。

**错误 2：IC 被误列为国际赛**

实测 `IC` = Iberian Cup，参赛队是西班牙 LVP SL 与葡萄牙 LPLOL 的二级/青训队
（Barça eSports、BISONS ECLUB、UCAM 等），244 场，2022-10-17 ~ 2025-11-22，
属 tier 0。v1 的注释"Intel Extreme / 国际邀请类"是臆测。
对 holdout 窗口零影响（2026-06-01 后无 IC 比赛），但会污染训练期的 tier 标注。

**顺带补全**：v1 完全没有历史一线赛区。实测 13 年共 123 个 raw league code，
v1 与手册 §3.11.5 合计只覆盖 40 个。补入 NA LCS / EU LCS / OGN / LMS / OPL /
LCO / LCL / LLA / LLN / CLS / GPL 等，改名关系全部用交接年的队伍集合重叠度验证
（NA LCS→LCS 重叠 10/10，OPL→LCO 7/8，OGN→LCK 7/13）。这些赛区**全部已停办**
（最后一场 2024-08-24），对 holdout 零影响，只影响训练期的 tier 标注。

**修订结果**

| | v1 | v2 |
|---|---|---|
| holdout 场次 | 1,071 | **1,079** |
| 新增 | — | 世界赛 8 场 |
| 移除 | — | **0 场** |

**纯增量，未移除任何样本**，因此不存在"挑掉不利样本"的可能。
文件：`eval/tier1_leagues.yaml`（v1 保留为 `eval/tier1_leagues.v1.yaml`）、
`eval/holdout_gameids_20260919_v2.txt`。

**未修改的一项（需使用者决定）**

PCS 与 VCS 在 2025/2026 已无队伍进入世界赛或 MSI，阵容变为青训队，
按客观判据应降为 tier 0，影响 holdout 内 90 场（PCS 53 + VCS 37）。
**这是口径判断而非事实错误，故保持 v1 的 tier 1 不变。**
若要改，必须同样在任何 holdout 评估之前进行并记入本文件。

## A2 · PCS / VCS 自 2025 起降为 tier 0（2026-09-19，使用者裁定）

**这是本预注册中第一次【移除】holdout 样本的修订，因此证据标准更高。**

**合法性前提（与 A1 同样的检查，修订时 HEAD = `f0707d45b834f68c0f9d3bc86718465631fc399a`）**

`models/` `preds/` `reports/` 仍全部不存在，git 全历史中从未出现，
源码中仍无任何拟合或评估代码。**仍是零模型训练、零 holdout 评估。**

**判据与证据**

判据在看到任何模型结果之前就已确定：*该赛区当年是否有队伍出现在世界赛或 MSI*，
并以阵容构成佐证。实测（队名精确匹配）：

| 年份 | PCS 进国际赛队数 | VCS 进国际赛队数 |
|---|---|---|
| 2019 | 0 | 3 |
| 2020 | 2 | 0 |
| 2021 | 2 | 0 |
| 2022 | 3 | 2 |
| 2023 | 2 | 2 |
| 2024 | 2 | 2 |
| **2025** | **0** | **0** |
| **2026** | **0** | **0** |

阵容佐证（2026 实际参赛队名）：

- PCS：CTBC Flying Oyster **Academy**、Ground Zero **Academy**、Frank Esports、
  Xu Ji Basalt、wangting …
- VCS：MVK Esports **Academy**、1TAP Dino、Ngựa Hí Esports …

原一线队伍 GAM、Team Secret Whales、PSG Talon 等已转入 LCP。

**⚠️ 必须分段，不能一刀切**

VCS 在 2020、2021 同样是 0 队，但那是疫情期间未出赛，**不是降级**。
因此判据不能机械套用"零队即降级"，而要结合阵容构成。
本次只降 **2025-01-01 起**，PCS 2020-2024、VCS 2019/2022-2024 仍为 tier 1。
一刀切会错误地抹掉这些年份的合法一线数据。

实现：`eval/tier1_leagues.yaml` 新增 `tier1_until` 段，PCS 与 VCS 的
`valid_until` 均为 `2024-12-31`。

**修订结果**

| | v2 | v3 |
|---|---|---|
| holdout 场次 | 1,079 | **989** |
| 移除 | — | PCS 53 + VCS 37 = **90 场** |
| 新增 | — | 0 |

**为什么移除样本在此处是可接受的**：判据与结果无关（基于赛区建制而非模型表现），
在任何模型存在之前作出，且移除的是**整个赛区**而非挑选的个别比赛——
不存在按模型表现挑样本的可能。

文件：`eval/holdout_gameids_20260919_v3.txt`。v1/v2 的名单保留在 git 历史中以供审计。
