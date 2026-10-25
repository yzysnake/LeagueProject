# 工作流手册

面向：在这个仓库里继续做事的人（很可能是几个月后的你自己）。

`docs/PLAN.md` 是 4,487 行的设计手册，讲「为什么这么设计」。
本文只讲「日常怎么干」，尽量短。

---

## 0. 三十秒定位

```bash
make help        # 所有命令
make test        # 419 项，跑一遍确认环境没坏
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
make data        # 拉 OE 年度 CSV + 五重校验（当年文件每次重下，内容没变不替换）
make bronze      # CSV → Parquet，物理丢弃 135 个赛后统计列
make silver      # 12 行折叠成 1 行 + 赛区归一 + series_id + patch + crosswalk；另建赛后经济差表
make artifact    # 训练打包（约 40 分钟，分层 BT 要全量拟合）
make app         # 起站点
```

`make data` 撞上 Drive 配额时（HTTP 200 + 配额错误页，存成 `*.quota`）会保留旧文件、整体失败，
隔几个小时再跑即可。**2026-10-24 之前它其实从不刷新**：脚本「已存在就跳过」，而 OE 每天更新的是
Drive 上同一个文件，数据就一直停在 9/18。

**实力快照不能等重训**：外层系数长期稳定，但推理用的分层 BT 快照每老一周 log loss 约差 0.004、
老一个月差 0.012（`reports/exp_model_improvements.md` §1.2）。所以快照单独重解：盯盘每 30 分钟对账一次，
有新赛果就 `artifact.refresh()`（silver + 台账里 OE 还没收录的局，几秒，外层系数与 `trained_on` 不动），
站点、API 与盯盘按模型文件的修改时间自动换上。手动：`make refresh`。详见 2.5 节。

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
| 经济差表 | `data/silver/margins.parquet` | 终局双方总经济与时长，**只当历史局的 BT 标签**（修订 A6），从不并进特征 |
| 特征 | 三个状态机的产出 | 全部 as-of：先读状态写特征，再用赛果更新 |
| artifact | `models/prod.json` 10.7 MB | 自描述，服务启动时断言与代码版本一致；`meta.rating_target` 记快照用的标签 |

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
**只有 `id` 和 `side`，没有逐局胜负**。逐局结果用两路独立来源：

- **livestats 终局帧**：`gameState=finished` 那一帧里恰好一方 `inhibitors>0`
  → 那一方赢（推水晶必须先推水晶塔）。双方都>0 或都为 0（投降）→ 判不出
- **系列比分**：系列最后一局 = 系列胜方；零封时每局都是胜方；其余局推不出

两路都有答案却矛盾 → 留空。都判不出 → **留空而不是猜**，下次对账会重试
（ledger 仍 append-only，汇总取每局最后一条结果）。

第一版假设 API 有逐局结果，写出来四条记录全是空值且不报错。第二版只认 Bo1，
导致 10/6–10/7 的 17 局多局系列全部空着；2026-10-18 加入终局帧后全部补齐，
其中 8 局与系列比分交叉核对一致。

**第二个 API 陷阱：`side` 不可信**。`getEventDetails` 的 `games[].teams[].side`
经常和 livestats 对不上：LPL 约一半、DCGI 淘汰赛一部分、Worlds 2025 T1–AL 五局全反。
预测的 `p_blue` 是 **feed 的** `blueTeamMetadata` 那一方的胜率，所以蓝红方一律以 feed
的 `esportsTeamId` 为准；拿 `side` 去套系列比分，结果会整个翻过来（命中、log loss、
Brier 一起翻），两路明明一致也会被误判成矛盾。对账把 feed 的蓝红方和开局 / 终局时间
一起写进 result（`blue_team` / `red_team` / `started_at` / `ended_at`），早期的结果
在下一次对账时补齐这些字段、胜负不动。

**盯盘的一个 launchd 陷阱**：脚本每 `--hours` 窗口结束会 exit 0。
`KeepAlive={SuccessfulExit: false}` 只在崩溃时重启，所以 10/8 盯完一轮就停了，
漏掉了 Worlds 正赛。现在是 `KeepAlive=true`：每轮结束对账一次再开下一轮。

**合盖 = 盯盘停摆**：盯盘跑在这台 Mac 上，合盖就进 Clamshell Sleep，launchd 任务跟着冻结。
醒来后会补抓还在 feed 上的局，其中有的已经打完——这类局在台账里标「赛后补录」
（落盘晚于终局）。模型只用阵容，概率不受影响，但不算严格的实时预测；
10/6–10/7 的 21 局里有 5 局是这样。想不漏比赛，就得让机器在赛时保持唤醒（接电源不合盖）。

**Worlds 2026 漏局的复盘（2026-10-24）**：到 10/24 已开打 32 局（含当时正在打的 MKOI–CFO），台账里只有 6 局。两个原因，对得上每一局：

| 时段 | 漏了什么 | 原因 |
|---|---|---|
| 10/15–10/18 入围赛 | 6 个 BO5 共 22 局，全漏 | 盯盘 10/8 09:37 到 10/18 21:20 没在跑（上面那个 `SuccessfulExit` 陷阱，10/18 已修） |
| 10/23 瑞士轮首日 | 8 局里漏 3 局（KC–AL、LOS–T1、LYON–TSW），另有 2 局赛后补录 | 机器合盖睡眠（`pmset -g log`），每 15–17 分钟 DarkWake 一次、每次 2–61 秒 |

第二条还有一层代码原因：轮询器的节流用 `time.monotonic()`，macOS 上它是 `mach_absolute_time()`，
**睡眠时不走**。DarkWake 里节流器按「醒着的时间」算，常常认为离上次 getLive 还不到 60 秒，
不刷新赛程，新开打的局进不了 watchlist；而早就在 watchlist 里的局倒能在 DarkWake 里被补抓到——
所以 10/23 是「有的漏、有的赛后补录」。验证：漏掉的三局 feed 上的阵容至今完整，
同一时段每一局都落在睡眠区间里；醒着的时段（16:06–19:08、22:11–23:05 UTC）一局没漏。

修法三件：

- **轮询器改用墙钟**（`Poller(now=time.time)`）：所有时限都说的是比赛的真实时间，DarkWake 里第一个 tick 就会刷新赛程。
- **盯盘按 1 秒切片睡**（`_nap`）：醒来 1 秒内接着干活，而不是把剩下的半截 `sleep(30)` 睡完；
  一轮间隔超过 2 分钟就判定刚醒，下一圈先实时轮询、再补录睡着时打完的局。
- **赛后补录**（`make backfill`，盯盘启动与每次醒来也会自动跑）：从 `getSchedule` 找出打完了、台账里没有的局，
  读 feed 的 BP 后阵容，用当前模型、**站在开局时刻**补出预测，记录带 `source: "backfill"`。
  开局早于实力快照截止（`ratings_cutoff`：快照里最晚一局的终局，没重解过就是训练截止）的局一律不补——
  补出来的预测会用上它开打之后的赛果。所以盯盘醒来先补录、再对账重解。10/24 已补齐 Worlds 25 局。

```bash
make backfill                      # 最近 14 天、全部赛事
make backfill DAYS=30 LEAGUE=Worlds
```

补录只是让台账完整，不改变「严格实时」的口径：「近期战绩」页把它们归入「赛后补录（整局错过）」，
「排除赛后补录」一键去掉。**真想一局不漏地实时抓，还是得让机器在赛时醒着。**

**实力快照随赛果重解（2026-10-25 起）**：对账时顺手把这局 BP 后阵容解析成 OE 的队伍 / 选手 id、
连同终局双方总经济写进 result（`blue_oe_team` / `blue_oe_players` / `blue_gold` …；老结果在下一次对账时补上，
胜负不动）。`ledger.live_games` 把其中 OE 还没收录的局拼成和 silver 同形的行：两队对照可信
（primary；contested 的要至少 3 名选手 OE 里最近一年最后效力的就是这支队——2026 世界赛的 MVK 就是这种），
胜负判得出，开局晚于 silver 里该赛区最后一场 + 30 分钟、且 silver 里没有同一对队伍前后 30 分钟内开的局。
`artifact.refresh()` 在 silver + 这些局上重解快照（10/25 首次：61 场、1.3 秒，与从零重解差 2e-5），
标签与训练时一致（`meta.rating_target`；生产是赛后经济差软标签，台账的局用对账记下的 feed 终局总经济、
时长 = 终局 − 首帧，实测与 OE 同一量级：胜方软标签中位数 0.85 vs 0.84，时长中位数 31.4 vs 31.5 分钟），
写回 `models/prod.json`；外层系数、其余状态、`trained_on` 都不动，meta 里记 `ratings_as_of` /
`ratings_cutoff` / `n_live_games`，台账每条预测记 `ratings`（当时快照的时刻）。

近期状态那一层（修订 A7，`models/form.py`）的台账部分也在这一步重建：残差用开赛前落盘的 `p_base`
（加这一层之前的概率）。它一半的增益来自本系列已打完的局，所以有比赛在盯时对账改成每 3 分钟一次，
第 1 局的赛果赶在第 2 局 BP 结束、出预测之前进去（局间约 10 分钟）。

两条代价，都是刻意的：首次重解之前没补的局（10/25 时其他赛区约 530 局，EMEA Masters、WSCI、DCGI 居多）
不能再补成预测；它们的赛果对快照仍有用，留作后续（只记赛果、不出预测）。
状态机（side 先验、英雄版本强度）不跟着台账重解：它们要 patch 与选手英雄计数，变化也慢，跟 `make artifact` 走。

**看成绩**：网站「近期战绩」页（或 `make reconcile` 的汇总）。按比赛日 / 赛事筛，
可以排除赛后补录；「结果」筛选只筛表格，不影响上面的指标。

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

**揭盲之后（修订 A4，2026-10-24）**：生产模型改用全量数据重训——模型规格不动，只把训练截止
从 5/31 延到最新。分工如下，别混：

| 用途 | 数据入口 | 说明 |
|---|---|---|
| 生产模型与它的推理端 | `load_all()` | 揭盲记录不在时拒绝；`make artifact` 默认就是它 |
| 一切评测（walk-forward、门禁、重测） | `load_dev()` | 只看 holdout 之前，数字与揭盲前可比 |
| 复现揭盲时评测的模型 | `python -m lolwp.models.artifact --scope dev` | |

holdout 的 0.61653 只属于 dev 口径（训练截至 5/31）的模型。**全量模型在 holdout 期是样本内的**，
它的样本外成绩只看盯盘台账里训练截止日之后的预测（每条记录的 `model` 字段）。

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
| OE 的 `WLDs` | 混着各赛区的**世界赛资格赛**（LPL 每年、LCK 2022–2024，共 98 场），选「全球总决赛」会看到 9 月的 TES vs IG；silver 按「主赛事之前、两队同赛区」归回所属赛区 |

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
| 看某场真实比赛的预测 | `make app` → 阵容模拟（点头像选英雄、每个位置可换替补） |
| 补录盯盘错过的局 | `make backfill`（可选 `DAYS=` `LEAGUE=`），之后自动对账 |
| 换个英雄看概率变化 | `make app` → what-if（173 个英雄 3 ms） |
| 试一个新特征 | 走第 2 节的六步协议 |
| 换生产模型 | 改 `models/artifact.py::PROD_FEATURES` → `make artifact` → `make gate` |
| 用最新数据重训生产模型 | `make data bronze silver artifact`（约 40 分钟）；盯盘、站点、API 按文件修改时间自动换上，不用重启 |
| 让实力快照立刻跟上最新赛果 | `make reconcile refresh`（盯盘每 30 分钟会自动做） |
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
