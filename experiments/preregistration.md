# 预注册 v1

> 冻结日期：2026-09-18
> 冻结时的 git sha：`<COMMIT_SHA>`（本文件提交后回填）
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
