# 工作流手册

面向：在这个仓库里继续做事的人（很可能是几个月后的你自己）。

`docs/PLAN.md` 是 4,487 行的设计手册，讲「为什么这么设计」。
本文只讲「日常怎么干」，尽量短。

---

## 0. 三十秒定位

```bash
make help        # 所有命令
make test        # 359 项，跑一遍确认环境没坏
make app         # 起站点看看东西长什么样
```

生产模型是 **BT4_LR**，五项 L2 logistic：

```
logit = 0.1600
      + 0.8398 · z(bt_player_diff)     分层 BT 选手潜变量差   ← 绝对主导
      + 0.1214 · z(champ_meta_diff)    英雄版本强度差
      + 0.0202 · z(bt_chem_diff)       队伍配合残差
      + 0.0123 · z(bt_league_diff)     赛区强度差
      + 0.0140 · z(side_prior)         按赛区收缩的蓝方先验
```

**外层朴素，复杂度全在特征怎么算出来。** 这是实测结果不是偷懒：
CatBoost 显著更差，DeepSet 加不动，分组 L2 与普通 L2 无差别，集成过不了门禁。

---

## 1. 日常循环

```bash
make data        # 拉 OE 年度 CSV + 五重校验（幂等，无更新自动跳过）
make bronze      # CSV → Parquet，物理丢弃 135 个赛后统计列
make silver      # 12 行折叠成 1 行 + 赛区归一 + series_id + patch + crosswalk
make artifact    # 训练打包（约 40 分钟，分层 BT 要全量拟合）
make app         # 起站点
```

改完任何东西，**固定三步**：

```bash
make test        # 含四个负控制
make lint
make gate        # 注入退化模型，门禁必须拒绝它
```

### 各层职责

| 层 | 产物 | 关键约束 |
|---|---|---|
| raw | `data/raw/*.csv` 约 700 MB | 五重校验，配额错误页会伪装成 HTTP 200 |
| bronze | `data/bronze/*.parquet` 31 MB | **物理丢弃赛后列**，想泄漏都拿不到 |
| silver | `data/silver/games.parquet` 10 MB | 一场一行，`game_id.nunique() == len(df)` 是硬断言 |
| 特征 | 三个状态机的产出 | 全部 as-of：先读状态写特征，再用赛果更新 |
| artifact | `models/prod.json` 10.7 MB | 自描述，服务启动时断言与代码版本一致 |

---

## 2. 要不要加一个新特征：走协议，不凭感觉

这是本仓库最重要的一段。**六步，任何一步不过就停。**

### 第一步 · 时间正确性

这个量在 **BP 结束、开赛前**真的已知吗？

正面例子：`game_in_series`（只取决于此前已打完的局）、`series_score_diff`。
反面例子：`series_length`（打到第 5 局意味着知道了前 4 局是 2-2，
而那在第 1 局的 BP 时点未知）。

判不准就写一个未来删除不变性检查：删掉某时点之后的所有行重算特征，
抽样对比必须 bit-identical。

### 第二步 · 便宜的可靠性闸门

```python
from lolwp.eval.reliability import run_gate
run_gate(dev, seed=s)   # 按 series 随机对半，Spearman-Brown 修正，门槛 r_full ≥ 0.20
```

**用 8 个随机种子，不要只跑一次。** 单次划分会撞运气——
单英雄胜率在 tier-1 上的 r_full 标准差有 0.10。

⚠️ **可靠性是必要条件不是充分条件。** 选手×英雄胜率的可靠性是 0.44（最高），
而它对模型的增量是 0.0001（几乎最低）。它可靠只是因为强选手用什么英雄都赢，
而那已经被选手潜变量吃掉了。闸门只能用来省钱，不能用来预测收益。

### 第三步 · 建模跑 walk-forward

```python
from lolwp.eval.baselines import LADDER, run_ladder
LADDER["CAND"] = [...base..., "新特征"]
preds = run_ladder(dev, feats, folds, names=["BASE", "CAND"])
```

90 折 rolling-origin，series 级分组，7 天 embargo。

### 第四步 · 该比较自己的 MDE

```python
from lolwp.eval.power import paired_power, footer
pw = paired_power(per_game_loss_diff, series_ids)
print(footer(pw))
```

**MDE 是每个比较各自的**，不是数据集常数。实测同一份数据上
不同配对的组内相关能从 0.008 变到 0.067。

`|Δ| < MDE` 一律标注【低于本样本量的可检出下限】，
**禁止写成「改进」或「变差」**。

### 第五步 · 连续两个留出期

把 OOS 按时间对半，两段的 Δ 必须**同号**，且各自 `|t| > 2.5`、`|Δ| > MDE`。

这一条最能挡住「整体勉强显著但其实是某一段的偶然」。
BO 内比分差就死在这里：前半段 t=1.01，后半段 t=2.51。

### 第六步 · 没过就登记，不是丢掉

```python
# src/lolwp/ops/backlog.py 的 REGISTRY 里加一条
Hypothesis(key=..., title=..., gate=..., threshold=..., why_parked=..., run=...)
```

**「现在测不出来」和「已被证伪」是两回事。** 克制的可靠性学习曲线还在爬
（12k→0.087, 96k→0.180），外推需约 15 万场，现有 10.2 万，几年后会自己越线。

登记后每年自动重测（已装 launchd）。一旦越线，
`test_nothing_that_passed_the_gate_is_still_parked` 会让测试变红逼你回来评估。

---

## 2.5 实时盯盘与赛后对账

```bash
make install-watch   # 挂 launchd，崩溃自动重启
make reconcile       # 随时对账
make uninstall-watch
```

盯盘做的事：轮询 `getLive` → 等 livestats 的 204→200 跳变 → 解析阵容 →
出 **BP 后的正式预测** → 落盘 `reports/live_watch.jsonl`（append-only）。

**一个反直觉的发现**：赛前因为「队伍不在历史数据里」而拒绝预测的场次，
**赛后往往能预测**。队伍名陌生不代表选手陌生——阵容一出来，
选手通过 crosswalk 解析成功，再由选手多数表决反推队伍。
2026-09-23 抓到的 4 局里有 3 局属于这种情况。

**对账的一个 API 陷阱**：`getEventDetails` 的 `games[].teams[]`
**只有 `id` 和 `side`，没有逐局胜负**。逐局结果只能从系列比分反推：

- Bo1：系列比分就是本局胜负，无歧义
- 多局系列：单靠 `gameWins` 推不出「第 k 局谁赢」，**必须留空而不是猜**

第一版假设 API 有逐局结果，写出来四条记录全是空值且不报错。

## 3. 定期任务

```bash
make monitor          # 滚动漂移：三条曲线，阈值由抽样分布推出
make reproduce        # 核对 7 项归档数字
make retest           # 重测被搁置的假设，追加进台账
make retest-history   # 只看趋势
```

**监控看校准斜率，不看 ECE。** 斜率对样本量无偏；
ECE 有系统性正偏且随样本缩小而增大（全样本 0.0105 的模型，
在 989 场窗口上期望就是 0.0455）。

---

## 4. holdout 的纪律

`load_holdout()` 默认抛异常。全项目**只揭盲一次**，已于 2026-09-20 执行
（git `437d1cb`，989 场，log loss 0.61653）。

此后任何基于 holdout 结果的模型调整都会污染这个估计。
若确有必要，必须在 `experiments/preregistration.md` 记为新修订并说明，
且明确标注「该估计已不再是无偏的样本外估计」。

预注册已有三条修订（A1/A2/A3），每条都留了「作出时尚无模型训练 / 无 holdout 评估」的证据。
**修订本身不可耻，藏着改才可耻。**

---

## 5. 五条会反复咬人的教训

### 5.1 不报错不等于对

这个项目里全部不报错、只是行为悄悄变了的坑：

| 坑 | 表现 |
|---|---|
| Drive 配额耗尽 | 返回 HTTP **200 + HTML**，`csv.DictReader` 读它得 0 行不抛异常，而 `0 % 12 == 0` 为真会让常见校验放行 |
| `year` 列 | 是**赛季年不是日历年**，2,709 场双向不一致，按它切时间会静默泄漏 |
| `datacompleteness=='complete'` 过滤 | 会删掉 LPL 2022–2025 **全部四年**（100% partial） |
| feed 与 OE 的位置名 | `jungle/bottom/support` vs `jng/bot/sup`，照搬会让三个位置错位 |
| `uv run` | 默认重新同步且**带 dev 组**，抵消前一步的 `uv sync --no-dev` |
| YAML 折叠标量 | 缩进更深的行保留换行，把 shell 命令切断，uvicorn 静默绑到默认网卡 |
| cron | 机器睡眠时**不补跑**，年度任务错过要等一年 |

应对只有一个：**每发现一个就钉成测试**。

### 5.2 阈值必须按实际评测样本量定

三次撞上同一个坑：

| 判据 | 误报率 | 处理 |
|---|---|---|
| 红线「任一 fold AUC > 0.78」 | 零信号下 **63%** | 修订 A3 改用合并 AUC |
| 预注册「ECE ≤ 0.03」 | 989 场上 **98% 过不了** | 揭盲前已记录，报告中说明 |
| 监控「reliability > 3×基线」 | **23/23 窗口全报警** | 阈值改由抽样分布推出 |

**一条 38% 时间在报警的告警器比没有告警器更糟**，因为它会让人养成忽略告警的习惯。

现在所有阈值都从抽样分布推出：从模型自身预测里抽同样大小的窗口若干次，
取 q 分位，健康模型的误报率就等于 1−q。

### 5.3 原始信号强 ≠ 有增量

| 量 | 原始信号 | 实际增量 |
|---|---|---|
| BO 内比分差 | 跨越 **24 个百分点**，完全单调 | +0.00073，低于 MDE |
| 选手×英雄胜率 | 可靠性 0.44，四项里最高 | +0.00009，几乎为零 |
| 英雄协同 pair | 看起来有规律 | 显式建模 log loss 0.7756，**比抛硬币还差** |

原因都一样：**已有特征已经把那部分信息吃掉了**。

### 5.4 符号一致性不能单独作为裁决依据

朴素两阶段拟合下，协同和克制的 α̂ 在 90 折里 **100% 同号**、系数大 8 倍，
而样本外 log loss 从 0.631 涨到 **0.997**。

是**校准斜率塌到 0.091** 当场拆穿的。斜率比 log loss 更早暴露过拟合。

### 5.5 先有裁判再有选手

第 3 周建评估基础设施，第 4 周才建第一个基线。四个负控制进 CI：

| 负控制 | 判据 | 实测 |
|---|---|---|
| 标签打乱（含状态机重跑） | log loss 回到 0.693 | 0.6921 ✓ |
| 未来删除不变性 | bit-identical | max\|diff\| = 0 ✓ |
| 泄漏探针（注入赛后列） | AUC 跳升 > 0.10 | +0.298 ✓ |
| 同版本重跑 | bit-identical | max\|diff\| = 0 ✓ |

**标签打乱必须连状态机一起重跑。** 只打乱评估时的标签而复用原特征是没有意义的，
那只能证明「随机标签不可预测」。

---

## 6. 常见任务速查

| 我想… | 怎么做 |
|---|---|
| 看某场真实比赛的预测 | `make app` → 阵容模拟 |
| 换个英雄看概率变化 | `make app` → what-if（173 个英雄 3 ms） |
| 试一个新特征 | 走第 2 节的六步协议 |
| 换生产模型 | 改 `models/artifact.py::PROD_FEATURES` → `make artifact` → `make gate` |
| 查模型现在什么状态 | `curl localhost:8000/health`（`make api` 先起） |
| 回滚 | `models/leaderboard.jsonl` 查履历，换 artifact 重启，`/health` 的 `run_id` 会变 |
| 确认数据没被改过 | `make reproduce`，核对 7 项归档数字 |
| 加一个被搁置的假设 | `ops/backlog.py::REGISTRY` 加一条，必须写 `why_parked` |

---

## 7. 现实预期

| 口径 | log loss | Brier | accuracy |
|---|---|---|---|
| walk-forward 样本外（23,060 场） | 0.61917 | 0.2154 | 65.3% |
| **final holdout（989 场）** | **0.61653** | 0.2146 | 65.3% |
| Polymarket 市场（同期 1,302 场） | 0.62748 | 0.2188 | 66.0% |

模型只把 Brier 从常数预测的 0.249 拉到 0.215，**「解释掉」的方差约 14%**。
剩下 86% 是同水平队伍对战的内在随机性，不是模型不够好。

**红线**：accuracy > 67% 或 log loss < 0.60 或合并 AUC > 0.78 → 几乎必有泄漏。
对照：随机划分加整赛季聚合胜率会把 AUC 从诚实的 0.650 推到 0.794。
