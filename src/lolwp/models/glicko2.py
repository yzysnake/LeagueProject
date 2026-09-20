"""Glicko-2 双轨（队伍层 + 选手层）——Elo 的对照实现，主产出是 RD 而不是 log loss。

为什么要单独写一套而不是接着调 Elo（PLAN.md §4.2 G2 / §4.6.3 / §4.6.4）：
Elo 只给点估计，一支只打过 2 场的新队和一支打过 300 场的老队，rating 都是一个数，
模型无从分辨"1520 分"哪个可信。Glicko-2 多带一个 RD（rating deviation），它有两个
Elo 给不了的用处——
  (a) **冷启动不再虚假自信**：新队 RD=350，g(RD) 会自动把它的 rating 差往 0 压，
      预测不敢偏离 0.5 太远；Elo 做不到这件事，只能靠外面加 `has_new_team` 补丁。
  (b) **UI 的预测区间有了尺子**：§4.6.4 要求"国际赛的预测区间必须明显放宽"。
      RD 就是那个宽度的来源，`win_prob_band()` 把它直接算成概率区间。

--------------------------------------------------------------------------
§6.1.2 红线 3 归类：**在线递归类，豁免逐 fold 重估。**

理由（必须逐条成立，否则这个豁免就是错的）：
  1. 本模块的所有超参（τ=0.5、RD₀=350、σ₀=0.06、SCALE=173.7178）都是 Glickman
     论文里的常数，**没有一个是在本数据上标定的**，因此不存在"用未来数据定超参"。
  2. 特征的因果性由 rating period 的语义保证：一个 period 内的所有比赛，一律读
     period **起点**的快照；period 内的赛果攒在 pending 里，period 结束才批量写回。
     所以第 t 场的特征只依赖严格早于本 period 起点的比赛，与"训练段切到哪一天"无关。
  3. 唯一一个看起来像全局量的东西是 side 先验——但它不在本模块（在 priors.py）。
全局跑一遍状态机、所有 fold 共用是正确的；逐 fold 重估只会得到逐位相同的结果。
（这与 §4.1 分类表把 "Glicko RD" 明确放在"fold 内可复用"那一行一致。）

--------------------------------------------------------------------------
rating period 怎么切：代价与理由

Glicko-2 **不是逐场算法**，Glickman 建议一个 period 里每个选手有 10–15 场。
本数据实测（data/silver/games.parquet，98,592 场 dev 段）：
  * 每队每周 3.26 场（中位数 3，p90 6）→ 按周切只有 3 场/period，离 10–15 差得远；
  * 要凑到 10–15 场得按 3–4 周切，那 rating 就会陈旧到一个赛段只更新 3 次。
三种切法的代价：
  | 切法        | 优点                                  | 代价 |
  |---|---|---|
  | `"game"`   | 信息最新：Bo3 第 2 局能看到第 1 局的结果 | 退化成 Glicko，σ 的批量估计几乎没有样本支撑；且同一 series 内的强相关被当独立 |
  | `"week"`   | 真正的批量，σ 有 3 场左右的样本；与赛程天然对齐 | 同周内的比赛互相看不见，特征最陈旧可达 7 天 |
  | `"month"`  | σ 样本够（~13 场），最接近论文建议     | rating 陈旧到一个赛段只动 3 次，赛季初几乎不更新 |
**默认取 `"week"`**：它是唯一同时满足"是个真批量"和"陈旧度不超过一个赛程周"的切法。
三种都实现了，`benchmark()` 会把三种的 log loss 并排跑出来——差别是实测的，不是猜的。

休赛期 RD 膨胀用连续时间版本（论文的 step 6 是"每空一个 period 加一个 σ²"，
这里推广成"每空 `inflation_period_days` 天加一个 σ²"）。这样三种切法的 RD 才可比：
按周切与按场切，同样过一周注入的方差量相同。实测最大空档：全局 34 天，
单队 3,286 天，单选手 4,102 天——所以 `max_rd` 的截断是必需的，不是装饰。

--------------------------------------------------------------------------
实测 **[2026-09-19，load_dev() 98,592 场建状态；评测 tier≥1 且 2019 起，
90 折 walk-forward，OOS 23,060 场 / 12,274 series；口径与阶段 4 基线逐字相同，
因为直接复用了 eval.baselines.run_ladder。下表全部经过第二次独立复跑核对，
`benchmark(periods=("week","game","month"))` 22s 内可复现]**

① 替换 Elo 之后的 log loss（`benchmark()` 可复现）

  | 模型 | 特征 | period=week | period=game | period=month |
  |---|---|---|---|---|
  | L2  | side + team_elo               | 0.63391 | 0.63391 | 0.63391 |
  | L3  | side + team_elo + player_elo  | 0.63230 | 0.63230 | 0.63230 |
  | G2  | side + team_glicko            | 0.63655 | 0.63321 | 0.64283 |
  | G3  | side + team_glicko + player_glicko | **0.62403** | **0.62081** | 0.62916 |
  | G3rd| G3 + 三个 RD 列               | 0.62350 | 0.62038 | 0.62858 |
  | L3rd| **L3 + 三个 RD 列**（Elo 点估计 + Glicko 不确定度） | 0.63064 | 0.63047 | 0.63084 |
  | L3G3| L3 与 G3 的特征全都给         | 0.62414 | 0.62103 | 0.62691 |

  **和"大概率差别很小"的预期相反：G3 比 L3 好 0.00828（week）/ 0.01149（game）**，
  远超实测 MDE 0.00284 / 0.00264（口径 = §6.3.1 做法 A：2.8·sd(series 级 block
  bootstrap)；做法 B 的 n_eff 版给 0.00279 / 0.00257，两种口径结论一致，
  下文所有 MDE 都是做法 A），DM t=8.23 / 12.41，p=2.0e-16 / 3.6e-35，
  series 级 block bootstrap 95% CI [+0.0062,+0.0103] / [+0.0096,+0.0133]，
  ΔAUC +0.0081 / +0.0130（DeLong p=1.9e-8 / 8.9e-26）。
  但**这 0.0115 里只有一半是 Glicko 的功劳**，见②。
  **反过来，period=month 就什么都不剩**：ΔLL +0.00314 低于 MDE 0.00332，
  ΔAUC +0.0002（DeLong p=0.92）——切得太粗时 Glicko 相对 Elo 的优势判定为
  **不可检出**。所以"Glicko 比 Elo 好"这句话只在 period ≤ 一周时成立。

  两条反向结论，同样是实测：
  * **队伍层单独换成 Glicko 没有好处**：G2 vs L2 在 week 下 **−0.0026（更差）**，
    在 game 下 +0.0007（低于 MDE 0.0028）。增益全部来自选手层。
  * **RD 加在 Glicko 自己头上没用，加在 Elo 头上有用**：
    G3rd vs G3 只有 +0.00052（MDE 0.00065）/ +0.00044（MDE 0.00059），两个都低于
    MDE ——Glicko 的点估计里已经含着同一份不确定度信息，再喂一遍是重复的。
    但 **L3rd vs L3 = +0.00166（week）/ +0.00183（game），MDE 0.00094，
    DM t=4.98 / 5.45，95% CI [+0.0010,+0.0023] / [+0.0012,+0.0025]，超 MDE ✓**。
    也就是说：如果生产模型出于任何理由要继续用 Elo 的点估计，
    **只把 Glicko 的三个 RD 列挂上去就是一个可检出的增益**，代价只是多跑一遍状态机。
    这是 RD 在 log loss 口径上唯一一处过门槛的地方，别把它和"RD 没用"混为一谈。

  这个差不是某几折撑起来的：**90 折里 G3 赢 66 折（week）/ 74 折（game）**，
  折级 ΔLL 中位数 +0.0079 / +0.0102；按年拆也全部为正
  （2019–2026 逐年 ΔLL：week +0.0070/+0.0040/+0.0140/+0.0108/+0.0090/+0.0078/
  +0.0024/+0.0108，game 全部 ≥ +0.0049）。最弱的一年是 2025。

  period 的敏感性曲线（G3，全部实测，单调随"特征陈旧度"变差）：
  game 0.62081 < day 0.62271 < week 0.62403 < biweek 0.62611 < month 0.62916。
  相邻两档都超 MDE（day vs week +0.00132/MDE 0.00092；biweek vs week
  −0.00208/MDE 0.00106），所以"period 切多长"是一个真的要做的选择，不是口味问题。

② 增益归因（把 G3(game) 的 +0.01149 拆开，每一段都过 MDE）

  | 台阶 | log loss | ΔLL | MDE | t |
  |---|---|---|---|---|
  | L3（现行 Elo，休赛期 μ 衰减 ρ=0.65）      | 0.63230 | —        | —       | — |
  | ├ 只关掉休赛期 μ 衰减（仍是 Elo）         | 0.62667 | +0.00563 | 0.00203 | 7.70 |
  | ├ 再换成 Glicko-2 / period=week           | 0.62403 | +0.00264 | 0.00239 | 3.07 |
  | └ 再把 period 收到 game                   | 0.62081 | +0.00321 | 0.00123 | 7.19 |

  **一半的"Glicko 增益"其实是 priors.py 的休赛期 μ 衰减在拖后腿。**
  `r ← 1500 + 0.65(r−1500)`（45 天空档触发）每年在赛季间把实力信息扔掉 35%；
  Glicko-2 的做法（μ 不动、只把 RD 放大）明显更对。把 ρ=0.65 直接搬进 Glicko-2
  反而毁掉它：0.62081 → 0.62896（game）、0.62403 → 0.63370（week，比 L3 还差）。
  ⚠ "关掉衰减更好"这一条是在 OOS 上发现的，**属于 exploratory，要动 priors.py
  之前必须单独预注册再测一次**，不能拿这里的数字当裁决。

  被数据否定的两个假设：
  * 任务书假设"5 人共用队伍期望 = 没做功劳分配"是 player Elo 不达标的主因。
    **不是。** 把 Elo 的选手期望改成个体期望（e_i 用选手自己的 rating 对阵对方均值），
    log loss 0.63230 → 0.63224，**几乎纹丝不动**。
  * 把一场 5v5 拆成 25 对 1v1（pairwise）也不是主因：pairwise 0.62081 与
    "聚合对手"0.62097 基本相同。真正起作用的是 Glicko-2 的 **RD 自适应步长**
    （新面孔 RD=350 学得快、老面孔 RD≈55 学得稳），Elo 的固定 K 做不到。
    顺带一条陷阱：pairwise 若按权重 1 计（假装 5 次独立观测），
    log loss 掉到 0.63409，比 L3 还差——所以默认权重是 1/5。

③ RD 分布（period=week，tier≥1 且 2019 起，23,702 场）

  | 队伍出场数 | n | 平均 RD | p10 | p90 |
  |---|---|---|---|---|
  | 0（首秀） | 299 | **239.6** | 204.5 | 350.0 |
  | 1–5 | 317 | 141.3 | 118.9 | 163.9 |
  | 6–20 | 1,335 | 95.7 | 77.4 | 117.0 |
  | 21–100 | 5,806 | 65.9 | 55.4 | 76.6 |
  | 101–500 | 12,813 | 57.4 | 49.5 | 66.7 |
  | >500 | 3,132 | **52.8** | 46.7 | 59.9 |

  选手层同向（按 `player_n_games_min` 分箱）：首秀 727 场比赛里 RD_max 均值
  **231.9**，1–5 场 153.4，6–20 场 102.6，21–100 场 70.6，101–500 场 60.1，
  >500 场（仅 113 行）51.6。**冷启动与老手差 4.5 倍**，
  这正是 Elo 给不出来的那一列。

  赛季初 vs 赛季末（按月）：1–2 月 team RD **75.2**、3–5 月 62.0、6–8 月 60.6、
  9–11 月 63.6。赛季初比赛季中宽 **24%**。但要说清楚：这个抬升主要来自
  换队/新队，不是来自 drift 本身——σ=0.06 的 drift 很弱，
  RD 60 的队空 34 天（本数据全局最大空档）只涨到 **64.2**，空 90 天到 70.7，
  空一年到 96.3。想让休赛期膨胀更凶要调 `inflation_period_days`，而那是超参。

  **一条负面结论，直接影响 §4.6.4**：国际赛的 RD 反而**更小**
  （tier2 team RD 53.5 vs 区内 65.3），因为能打国际赛的都是场次最多的队。
  所以"国际赛区间要更宽"**不能**靠 team/player RD 实现——国际赛的多出来的
  不确定性是**跨赛区锚定**的不确定性，单池 Glicko 根本看不见它。
  这件事得靠分层 BT 的 `L_region` 后验方差，或者对 tier2 单独做一个方差项。
  唯一的好消息是校准：国际赛子集上 L3 的 calibration slope 只有 **0.700**
  （严重过度自信），G3 是 **1.111（week）/ 1.135（game）**，
  落在 §6 要求的 [0.85, 1.15] 内；同一子集（n=1,698）的 log loss
  L3 0.66932 → G3 0.64959 / 0.64594，基准率 0.69105。

  冷启动子集（§6 阶段 6 验收③"不劣于该子集的蓝方基准率"）：
  | 子集 | n | 基准率 LL | L2 | L3 | G3(week) | G3(game) |
  |---|---|---|---|---|---|---|
  | team_n_games_min ≤ 5   | 547 / 462   | 0.6909 | 0.6651 | 0.6530 | **0.6382** | **0.6311** |
  | player_n_games_min ≤ 5 | 1,590 / 1,380 | 0.6908 | 0.6118 | 0.6063 | **0.5823** | **0.5639** |
  （两列 n 分别是 week / game 两种 period 下各自 `*_n_games_min ≤ 5` 圈出来的子集，
  所以 G3(game) 那一列不是在同一批场次上算的。把子集固定成 week 口径再看 game，
  两个数是 0.62507 / 0.57152，结论不变。）
  两个子集 G3 都明显优于基准率，验收③过。**但要如实说一句负面的**：
  冷启动队伍子集上 G3 的 calibration slope 只有 0.84（week）/ 0.80（game），
  掉出了 [0.85, 1.15] —— RD 把点估计往 0.5 压得还不够狠，
  在"两队都没打过几场"的情形下仍然偏自信。

④ 与参考实现对数：对 PyPI `glicko2 2.1.0` 跑 200 组随机输入，
  max|Δrating| = 5.5e-3、max|ΔRD| = 2.0e-3、max|Δσ| = 9.9e-5。
  残差不为 0 是因为参考实现的 `_f()` 把论文的 φ² 写成了内部尺度的 μ²；
  本模块按论文写。论文第 3 页算例上两者都给 1464.05/151.52/0.05999。

⑤ 成本（M 芯片单核，wall clock）：全量 98,592 场跑一遍状态机
  week 2.9–3.6s / day 4.0s / game 4.8s / biweek 3.2s / month 3.1s，
  对照 priors.py 的 Elo 2.8s —— 同一量级，不是瓶颈。
  由于是在线递归类（见上），90 折只跑一次，不随 fold 数放大。
  推理侧：快照 JSON 1.59 MB（2,070 队 + 10,863 选手），
  `features_from_glicko_state()` 单场 **16 µs**（2,000 次均值），
  离 §4.6.4 的实时预算（35s 端到端）差四个数量级，可以放心每次请求现算。

⑥ 泄漏控制（因为 +0.0115 这个幅度必须先自证清白）
  * **T2 档 A，未来删除不变性**：删掉 t 之后的所有比赛重跑状态机，第 t 场的
    18 列特征 **max|Δ| = 0.0**（真实 dev 数据上 period ∈ {game, week} 各 20 个
    切点，取自第 40,308–97,926 场；合成数据上 4 种 period × 12 个切点）。
    在线递归类必须 bit-identical，这条做到了。
  * **N1 标签打乱**（§6.9，含状态机用打乱标签重跑，两个 seed）：G3 的 OOS log loss
    seed0 game **0.69181 / AUC 0.5036**、week 0.69185 / 0.5026；
    seed1 game 0.69187 / 0.4995、week 0.69183 / 0.5019；对照 L3 0.69196 / 0.69190。
    8 个数全部落在 [0.685, 0.700] 内，AUC 全部贴着 0.50。

⑦ 一条必须写下来的数据质量问题（不在本模块，但会污染本模块的对照）：
  dev 段有 **2,494 个 team 槽位（blue 1,255 + red 1,239，占 98,592 场的 2.53%）
  的 team_id 是空的**，涉及 **2,131 场（2.16%）**（VCS/PCS 为主，2019–2021 居多）。
  `str(None)` 之后它们全部落进同一个 `"nan"` 键，等于把上百支不同的队伍
  当成一支"超级队"在算 rating。priors.py 的 Elo 用的是同一行写法，
  所以**两条轨道受污染的程度相同、对照仍然公平**，但两边的 team rating 都因此
  偏了一点。落到评测窗口里只有 12 行（0.05%），对上面的 log loss 影响可以忽略；
  要彻底修得在 silver 层给无 id 的队伍生成稳定代理键，那是 store/ 的活。

⑧ 一处不会报错、但线上一定会踩的 training/serving skew（**2026-09-19 新测**）
  快照总是在"当前 period 还没走完"的时刻导出的，而 `run_glicko_state_machine`
  收尾时会把 pending 的那半个 period **flush 掉**。于是线上读到的特征比训练时
  **更新鲜**：训练用的是 period 起点的状态，线上用的是"截至此刻所有已结束比赛"的状态。
  量化（同一套 90 折；用 period=game 的特征当"最新鲜"的上界）：

  | 训练特征 | 服务特征 | log loss | slope | ECE |
  |---|---|---|---|---|
  | week | week（离线口径）   | 0.62403 | 1.023 | 0.0073 |
  | week | game（线上口径的上界） | 0.62091 | 1.047 | 0.0110 |
  | game | game               | 0.62081 | 1.023 | 0.0103 |

  ΔLL(week训/game服 − week训/week服) = **+0.00311，MDE 0.00121，t=7.19**。
  也就是说这个 skew 在 log loss 上是**好事**（更新鲜的 rating 确实更准），
  代价是 ECE 抬高 0.0037、slope 从 1.023 推到 1.047。
  要两边严丝合缝只有一个办法：**服务端用 `period="game"`** —— 那时"period 起点"
  就是这一场自己，快照里永远没有未 flush 的残余，训练/推理逐位一致
  （`test_features_from_state_matches_state_machine` 锁的就是这件事）。
  这条写在文档里由调用方决定，而不是让函数偷偷改语义：谁导出快照、导在哪个时刻，
  函数是看不见的。

⑨ RD 到底有没有兑现"冷启动不虚假自信"（**2026-09-19 新测**，period=game，
  按 `team_rd_joint` 五分位切，每桶 4,612 场）

  | 分位 | RD_joint 均值 | LL(L3/Elo) | LL(G3) | slope L3 | slope G3 | G3 的平均偏离 0.5 |
  |---|---|---|---|---|---|---|
  | Q1 | 67.9  | 0.6426 | 0.6359 | **0.785** | 0.975 | 0.137 |
  | Q2 | 74.5  | 0.6397 | 0.6289 | **0.780** | 0.941 | 0.147 |
  | Q3 | 80.9  | 0.6397 | 0.6305 | 0.907 | 1.011 | 0.139 |
  | Q4 | 89.2  | 0.6235 | 0.6161 | 1.122 | 1.082 | 0.146 |
  | Q5 | 128.3 | 0.6161 | 0.5926 | **1.380** | 1.095 | 0.164 |

  **Elo 的 calibration slope 在 RD 的两端整个失控（0.78 → 1.38，跨度 0.60），
  Glicko-2 全程 0.94–1.10（跨度 0.15）**，而且逐桶 log loss 全赢。
  这才是 RD 不可替代的地方：它不是让平均更准，而是让**不同确定度的场次各自都准**。
  （注意 Q5 的 log loss 反而最低——高 RD 常常意味着"新队 vs 老队"，本来就好判。）

  UI 区间（`win_prob_band`，蓝方比红方高 100 分时的 95% 区间宽度）：
  双方老队 RD=55 → **37.7 pp**；典型 RD=65 → 40.7 pp；一方冷启动 RD=200 → **73.3 pp**；
  全新队 RD=350 → 86.1 pp。§4.6.4 要的"区间明显放宽"在冷启动上做到了（×1.9 宽），
  在国际赛上做不到（原因见③那条负面结论）。

⚠ 默认 `period="week"` 是**看数据之前**按"既是真批量、陈旧度又不超一个赛程周"
  定下的；实测 `period="game"` 更好（0.62081 vs 0.62403）。要把默认改成 "game"，
  按 §6.7 的规矩得先预注册——所以这里保留 week，把两个数并排报出来。
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "ANCHOR",
    "SCALE",
    "Glicko2Config",
    "Rating",
    "drift",
    "expected_score",
    "features_from_glicko_state",
    "g_factor",
    "rd_report",
    "run_glicko_state_machine",
    "solve_volatility",
    "update_rating",
    "win_prob_band",
]

# Glickman 定义的 Elo↔内部尺度换算。改这两个常数就不是 Glicko-2 了。
SCALE = 173.7178
ANCHOR = 1500.0

PeriodKind = Literal["game", "day", "week", "biweek", "month"]
PlayerMode = Literal["pairwise", "team_mean", "off"]

# (mu_j, phi_j, score, weight)：一条对局记录，内部尺度。
Opponent = tuple[float, float, float, float]


# ==========================================================================
# 1. 纯函数层：Glickman 论文的算法，逐步对应，可单测
# ==========================================================================


@dataclass(frozen=True)
class Rating:
    """一个实体的 Glicko-2 状态。mu/phi 是内部尺度，sigma 是波动率。

    存内部尺度而不是 Elo 尺度，是因为除了展示以外所有公式都在内部尺度上；
    每次读写都换算一遍既慢又容易漏。展示用 `.rating` / `.rd`。
    """

    mu: float = 0.0
    phi: float = 350.0 / SCALE
    sigma: float = 0.06

    @property
    def rating(self) -> float:
        """Elo 尺度的 rating，可以直接和 priors.py 的 team_elo 摆在一起看。"""
        return ANCHOR + SCALE * self.mu

    @property
    def rd(self) -> float:
        """Elo 尺度的 RD。350 = 一无所知，30–60 = 打了很多场。"""
        return SCALE * self.phi


@dataclass(frozen=True)
class Glicko2Config:
    """全部取 Glickman 论文的默认值。**任何一个都不许在 OOS 上调**（见模块 docstring）。"""

    init_rating: float = ANCHOR
    init_rd: float = 350.0
    init_vol: float = 0.06
    # τ 约束波动率的变化速度。论文建议 0.3–1.2，越小越不允许 rating 突变。
    tau: float = 0.5
    max_rd: float = 350.0
    # RD 下限。0.0 = 纯论文版（不设限）。设了下限相当于承认"再多场次也留一点疑问"。
    min_rd: float = 0.0
    period: PeriodKind = "week"
    # RD 膨胀的时间单位：每过这么多天注入一个 σ²。与 period 解耦，见模块 docstring。
    inflation_period_days: float = 7.0
    player_mode: PlayerMode = "pairwise"
    # 5v5 拆成 25 对 1v1 时每对的权重。None = 自动取 1/n_对手，
    # 这样"一场比赛"仍然只提供一场比赛的信息量（5 个结果完全相关，不是 5 次独立观测）。
    pairwise_weight: float | None = None
    # 休赛期 μ 回归。1.0 = 关闭（纯 Glicko-2：不确定性涨回来就够了，不动点估计）。
    season_decay: float = 1.0
    season_gap_days: float = 45.0
    eps: float = 1e-6
    max_iter: int = 100

    def initial(self) -> Rating:
        return Rating(
            (self.init_rating - ANCHOR) / SCALE, self.init_rd / SCALE, self.init_vol
        )


def g_factor(phi: float) -> float:
    """g(φ)：对手越不确定，这一局对我的 rating 影响越小。论文 step 3。"""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def expected_score(mu: float, mu_j: float, phi_j: float) -> float:
    """E(μ, μ_j, φ_j)：注意它被对手的 φ_j **压向 0.5**，这正是 Elo 没有的性质。"""
    return 1.0 / (1.0 + math.exp(-g_factor(phi_j) * (mu - mu_j)))


def solve_volatility(
    sigma: float,
    delta: float,
    phi: float,
    v: float,
    *,
    tau: float = 0.5,
    eps: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """论文 step 5：用 Illinois 算法解 f(x)=0，返回新的 σ'。

    f(x) = e^x(Δ²−φ²−v−e^x) / (2(φ²+v+e^x)²) − (x−a)/τ²，a = ln σ²。
    这里刻意不用 scipy.optimize：Illinois 是论文指定的求根法，逐字实现才能和
    参考实现对数；换个求根器在 Δ² 远大于 φ²+v 时会落到不同的根上。
    """
    a = math.log(sigma * sigma)
    d2 = delta * delta
    p2 = phi * phi
    tau2 = tau * tau

    def f(x: float) -> float:
        ex = math.exp(x)
        num = ex * (d2 - p2 - v - ex)
        den = 2.0 * (p2 + v + ex) ** 2
        return num / den - (x - a) / tau2

    big = d2 - p2 - v
    if big > 0.0:
        b = math.log(big)
    else:
        k = 1
        while f(a - k * tau) < 0.0:
            k += 1
            if k > max_iter:  # 兜底：理论上不会发生，发生了也不能死循环
                break
        b = a - k * tau

    fa, fb = f(a), f(b)
    lo = a
    it = 0
    while abs(b - lo) > eps and it < max_iter:
        c = lo + (lo - b) * fa / (fb - fa)
        fc = f(c)
        if fc * fb <= 0.0:
            lo, fa = b, fb
        else:
            fa = fa / 2.0
        b, fb = c, fc
        it += 1
    return math.exp(lo / 2.0)


def drift(r: Rating, elapsed: float, max_phi: float) -> Rating:
    """休赛期膨胀：φ ← min(√(φ² + elapsed·σ²), φ_max)。elapsed 的单位是 period。

    论文 step 6 的连续时间推广。elapsed=1 时逐字等于论文的"空过一个 period"。
    截断是必需的：本数据里单队最长空档 3,286 天，不截断 φ 会飞到无意义的量级。
    """
    if elapsed <= 0.0:
        return r
    phi = math.sqrt(r.phi * r.phi + elapsed * r.sigma * r.sigma)
    return Rating(r.mu, min(phi, max_phi), r.sigma)


def update_rating(
    r: Rating,
    opponents: Sequence[Opponent],
    *,
    elapsed: float = 1.0,
    cfg: Glicko2Config | None = None,
) -> Rating:
    """一个 rating period 的批量更新。opponents 为空 = 没打，只膨胀 RD（step 6）。

    elapsed=1.0 且所有 weight=1.0 时，本函数逐字等于 Glickman 论文的 step 3–8，
    tests/test_glicko2.py 用论文第 3 页的算例（1500/200/0.06 打 3 场 → 1464.06/151.52/0.05999）
    锁死这一点。weight 是本项目加的：5v5 拆成 25 对 1v1 时，那 25 个结果完全相关，
    按 1 计权会让 RD 收缩快 5 倍（假信息量），所以默认每对计 1/5。
    """
    cfg = cfg or Glicko2Config()
    max_phi = cfg.max_rd / SCALE
    min_phi = cfg.min_rd / SCALE
    if not opponents:
        return drift(r, elapsed, max_phi)

    inv_v = 0.0
    s = 0.0
    for mu_j, phi_j, score, w in opponents:
        gj = g_factor(phi_j)
        e = 1.0 / (1.0 + math.exp(-gj * (r.mu - mu_j)))
        inv_v += w * gj * gj * e * (1.0 - e)
        s += w * gj * (score - e)
    if inv_v <= 0.0:  # 数值退化（E 完全饱和），当作没打
        return drift(r, elapsed, max_phi)

    v = 1.0 / inv_v
    delta = v * s
    sigma_new = solve_volatility(
        r.sigma, delta, r.phi, v, tau=cfg.tau, eps=cfg.eps, max_iter=cfg.max_iter
    )
    phi_star = min(
        math.sqrt(r.phi * r.phi + elapsed * sigma_new * sigma_new), max_phi
    )
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + inv_v)
    phi_new = min(max(phi_new, min_phi), max_phi)
    return Rating(r.mu + phi_new * phi_new * s, phi_new, sigma_new)


def win_prob_band(
    blue: Rating, red: Rating, *, z: float = 1.96
) -> tuple[float, float, float]:
    """Glicko 自己的胜率 + 由 RD 直接给出的区间。§4.6.4 的 UI 宽度就用这个。

    点估计用 g(√(φ_b²+φ_r²)) 压过的 logit——两边越不确定，预测越靠近 0.5。
    区间把 μ 差的后验 sd = √(φ_b²+φ_r²) 传播过去。国际赛/新队伍的区间会自动变宽，
    不需要在 UI 层写任何"如果是国际赛就放宽"的硬编码分支。
    """
    sd = math.sqrt(blue.phi * blue.phi + red.phi * red.phi)
    g = g_factor(sd)
    d = blue.mu - red.mu
    lo, hi = d - z * sd, d + z * sd
    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-g * x))

    return sig(d), sig(lo), sig(hi)


# ==========================================================================
# 2. rating period 的切分
# ==========================================================================


def _period_start_days(ts: pd.Series, kind: PeriodKind) -> np.ndarray:
    """返回每一场所属 rating period 的**起点**（自 epoch 的天数，float）。

    用 period 起点而不是比赛时刻，是为了保证同一 period 内每一场读到的状态完全相同
    ——这正是"period 内互相看不见"这条语义的实现。
    """
    days = ts.to_numpy(dtype="datetime64[ns]").astype("int64") / 86_400e9
    if kind == "game":
        return days
    if kind == "month":
        ym = (ts.dt.year * 12 + ts.dt.month - 1).to_numpy()
        starts = pd.to_datetime(
            {"year": ym // 12, "month": ym % 12 + 1, "day": 1}, utc=True
        )
        return starts.to_numpy(dtype="datetime64[ns]").astype("int64") / 86_400e9
    width = {"day": 1.0, "week": 7.0, "biweek": 14.0}[kind]
    return np.floor(days / width) * width


def _period_start_scalar(ts: pd.Timestamp, kind: PeriodKind) -> float:
    """单个时刻的 period 起点（天）。推理路径必须和训练路径落在同一个网格上，
    否则 `elapsed` 会差最多一个 period —— 又一处典型的 training/serving skew。"""
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    days = t.value / 86_400e9
    if kind == "game":
        return float(days)
    if kind == "month":
        start = pd.Timestamp(year=t.year, month=t.month, day=1, tz="UTC")
        return float(start.value / 86_400e9)
    width = {"day": 1.0, "week": 7.0, "biweek": 14.0}[kind]
    return float(math.floor(days / width) * width)


# ==========================================================================
# 3. 状态机（priors.py 的 return_state 模式）
# ==========================================================================


class _Track:
    """一条 rating 轨道（队伍层或选手层）。批量更新 + 惰性膨胀。

    惰性膨胀（读到才算）而不是每个 period 给所有实体算一遍，是因为本数据有 2,069 支
    队伍和 10,863 名选手，但一个 period 只有几十个实体在打；全量膨胀是 O(实体×period)，
    惰性是 O(出场次数)。两者数值上完全等价（膨胀只依赖时间差）。
    """

    __slots__ = (
        "_init", "_pending", "_pending_n", "_prior",
        "cfg", "last_t", "max_phi", "n", "r",
    )

    def __init__(self, cfg: Glicko2Config) -> None:
        self.cfg = cfg
        self.r: dict[str, Rating] = {}
        self.last_t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.max_phi = cfg.max_rd / SCALE
        self._init = cfg.initial()
        # 本 period 的缓存：key -> (用于更新的先验 Rating, elapsed, 用于出特征的 Rating)
        self._prior: dict[str, tuple[Rating, float, Rating]] = {}
        self._pending: dict[str, list[Opponent]] = {}
        # 场次计数与 opponent 行数不是一回事：pairwise 模式下一场比赛会产生 5 行。
        # 特征里那个 *_n_games_min 必须是**场次**，否则冷启动分箱整体偏 5 倍。
        self._pending_n: dict[str, int] = {}

    def view(self, key: str, t: float) -> Rating:
        """读 period 起点 t 的 as-of 状态（含休赛期膨胀与可选的赛季间回归）。"""
        hit = self._prior.get(key)
        if hit is not None:
            return hit[2]
        base = self.r.get(key)
        if base is None:
            base = self._init
            self.r[key] = base
            self.last_t[key] = t
        # df 已断言升序，max 只防 period 定义被改错时出负数
        elapsed = max(t - self.last_t[key], 0.0)
        if (
            self.cfg.season_decay != 1.0
            and elapsed * self.cfg.inflation_period_days >= self.cfg.season_gap_days
        ):
            mu0 = (self.cfg.init_rating - ANCHOR) / SCALE
            base = Rating(mu0 + self.cfg.season_decay * (base.mu - mu0), base.phi, base.sigma)
        shown = drift(base, elapsed, self.max_phi)
        self._prior[key] = (base, elapsed, shown)
        return shown

    def record(self, key: str, opps: Iterable[Opponent]) -> None:
        """登记 key 在本 period 的一场比赛（opps 是这一场拆出来的若干条对局记录）。"""
        self._pending.setdefault(key, []).extend(opps)
        self._pending_n[key] = self._pending_n.get(key, 0) + 1

    def flush(self, t: float) -> None:
        """period 结束：把攒下的对局一次性写回。**这里才允许改 self.r。**"""
        for key, opps in self._pending.items():
            base, elapsed, _ = self._prior[key]
            self.r[key] = update_rating(base, opps, elapsed=elapsed, cfg=self.cfg)
            self.last_t[key] = t
            self.n[key] = self.n.get(key, 0) + self._pending_n[key]
        self._pending.clear()
        self._pending_n.clear()
        self._prior.clear()

    def snapshot(self) -> dict[str, list[float]]:
        return {
            k: [v.mu, v.phi, v.sigma, self.last_t.get(k, 0.0), float(self.n.get(k, 0))]
            for k, v in self.r.items()
        }


def _clean_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    return [str(p) for p in raw if p is not None and not pd.isna(p)]


_TEAM_COLS = (
    "team_glicko_diff",
    "blue_team_glicko",
    "red_team_glicko",
    "team_rd_blue",
    "team_rd_red",
    "team_rd_joint",
    "team_glicko_g_diff",
    "team_glicko_prob",
    "team_vol_blue",
    "team_vol_red",
    "team_n_games_min",
)
_PLAYER_COLS = (
    "player_glicko_diff",
    "player_rd_blue",
    "player_rd_red",
    "player_rd_max_blue",
    "player_rd_max_red",
    "player_glicko_g_diff",
    "player_n_games_min",
)
FEATURE_COLS = _TEAM_COLS + _PLAYER_COLS


def run_glicko_state_machine(
    df: pd.DataFrame,
    *,
    cfg: Glicko2Config | None = None,
    return_state: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """单遍产出 Glicko-2 的 as-of 特征。df 必须按 kickoff_utc 升序、一场一行。

    与 priors.py 的区别只有一处，但它是本模块的全部要点：**更新是按 rating period
    批量的**。同一 period 内的比赛读同一份快照，period 结束才写回。
    因此 `period="week"` 下，同一周的第 3 场看不到第 1 场的结果——这是 Glicko-2 的
    定义，不是 bug；代价（特征陈旧最多 7 天）在模块 docstring 里量化过。
    """
    cfg = cfg or Glicko2Config()
    if not df.kickoff_utc.is_monotonic_increasing:
        raise ValueError("df 必须按 kickoff_utc 升序")

    n = len(df)
    pstart = _period_start_days(df.kickoff_utc, cfg.period) / cfg.inflation_period_days
    team = _Track(cfg)
    player = _Track(cfg)
    out = {c: np.empty(n) for c in FEATURE_COLS}

    cur = math.nan
    for i, row in enumerate(df.itertuples(index=False)):
        t = float(pstart[i])
        if t != cur:
            if not math.isnan(cur):
                team.flush(cur)
                player.flush(cur)
            cur = t

        y = float(row.y_blue_win)
        bt, rt = str(row.blue_team_id), str(row.red_team_id)
        rb, rr = team.view(bt, t), team.view(rt, t)

        # ---------- 先读状态，写特征 ----------
        sd = math.sqrt(rb.phi * rb.phi + rr.phi * rr.phi)
        gj = g_factor(sd)
        out["team_glicko_diff"][i] = rb.rating - rr.rating
        out["blue_team_glicko"][i] = rb.rating
        out["red_team_glicko"][i] = rr.rating
        out["team_rd_blue"][i] = rb.rd
        out["team_rd_red"][i] = rr.rd
        out["team_rd_joint"][i] = SCALE * sd
        out["team_glicko_g_diff"][i] = gj * (rb.rating - rr.rating)
        out["team_glicko_prob"][i] = 1.0 / (1.0 + math.exp(-gj * (rb.mu - rr.mu)))
        out["team_vol_blue"][i] = rb.sigma
        out["team_vol_red"][i] = rr.sigma
        out["team_n_games_min"][i] = min(team.n.get(bt, 0), team.n.get(rt, 0))

        bp, rp = _clean_ids(row.blue_player_ids), _clean_ids(row.red_player_ids)
        bv = [player.view(p, t) for p in bp]
        rv = [player.view(p, t) for p in rp]
        if bv and rv:
            mb = sum(x.mu for x in bv) / len(bv)
            mr = sum(x.mu for x in rv) / len(rv)
            pb = math.sqrt(sum(x.phi * x.phi for x in bv) / len(bv))
            pr = math.sqrt(sum(x.phi * x.phi for x in rv) / len(rv))
            out["player_glicko_diff"][i] = SCALE * (mb - mr)
            out["player_rd_blue"][i] = SCALE * pb
            out["player_rd_red"][i] = SCALE * pr
            out["player_rd_max_blue"][i] = SCALE * max(x.phi for x in bv)
            out["player_rd_max_red"][i] = SCALE * max(x.phi for x in rv)
            out["player_glicko_g_diff"][i] = (
                g_factor(math.sqrt(pb * pb + pr * pr)) * SCALE * (mb - mr)
            )
            out["player_n_games_min"][i] = min(player.n.get(p, 0) for p in bp + rp)
        else:
            # 选手名单缺失：给中性值，并让 RD 顶到上限（"我什么都不知道"）
            out["player_glicko_diff"][i] = 0.0
            out["player_rd_blue"][i] = cfg.max_rd
            out["player_rd_red"][i] = cfg.max_rd
            out["player_rd_max_blue"][i] = cfg.max_rd
            out["player_rd_max_red"][i] = cfg.max_rd
            out["player_glicko_g_diff"][i] = 0.0
            out["player_n_games_min"][i] = 0

        # ---------- 再把赛果攒进 pending（period 结束才写回） ----------
        team.record(bt, [(rr.mu, rr.phi, y, 1.0)])
        team.record(rt, [(rb.mu, rb.phi, 1.0 - y, 1.0)])

        if cfg.player_mode == "pairwise" and bv and rv:
            wb = cfg.pairwise_weight if cfg.pairwise_weight is not None else 1.0 / len(rv)
            wr = cfg.pairwise_weight if cfg.pairwise_weight is not None else 1.0 / len(bv)
            red_opp = [(x.mu, x.phi, y, wb) for x in rv]
            blue_opp = [(x.mu, x.phi, 1.0 - y, wr) for x in bv]
            for p in bp:
                player.record(p, red_opp)
            for p in rp:
                player.record(p, blue_opp)
        elif cfg.player_mode == "team_mean" and bv and rv:
            # Elo 现行做法的 Glicko 版：5 人共用一个"聚合对手"。
            # 聚合对手的 φ 取 RMS 而不是 /√5——5 个人的表现不独立，别假装信息量有 5 份。
            opp_r = (sum(x.mu for x in rv) / len(rv),
                     math.sqrt(sum(x.phi**2 for x in rv) / len(rv)), y, 1.0)
            opp_b = (sum(x.mu for x in bv) / len(bv),
                     math.sqrt(sum(x.phi**2 for x in bv) / len(bv)), 1.0 - y, 1.0)
            for p in bp:
                player.record(p, [opp_r])
            for p in rp:
                player.record(p, [opp_b])

    if not math.isnan(cur):
        team.flush(cur)
        player.flush(cur)

    feats = pd.DataFrame(out, index=df.index)
    if not return_state:
        return feats
    state = {
        "team": team.snapshot(),
        "player": player.snapshot(),
        "cfg": {
            "init_rating": cfg.init_rating, "init_rd": cfg.init_rd,
            "init_vol": cfg.init_vol, "tau": cfg.tau, "max_rd": cfg.max_rd,
            "min_rd": cfg.min_rd, "period": cfg.period,
            "inflation_period_days": cfg.inflation_period_days,
            "player_mode": cfg.player_mode, "pairwise_weight": cfg.pairwise_weight,
            "season_decay": cfg.season_decay, "season_gap_days": cfg.season_gap_days,
        },
        "as_of_days": float(pstart[-1]) * cfg.inflation_period_days if n else 0.0,
        "as_of": str(df.kickoff_utc.max()) if n else None,
        "n_games": n,
    }
    return feats, state


def _state_view(state: dict[str, Any], track: str, key: str, t: float) -> Rating:
    c = state["cfg"]
    row = state[track].get(str(key))
    if row is None:
        return Rating((c["init_rating"] - ANCHOR) / SCALE, c["init_rd"] / SCALE, c["init_vol"])
    mu, phi, sigma, last_t, _ = row
    base = Rating(mu, phi, sigma)
    elapsed = max(0.0, t - last_t)
    if c["season_decay"] != 1.0 and elapsed * c["inflation_period_days"] >= c["season_gap_days"]:
        mu0 = (c["init_rating"] - ANCHOR) / SCALE
        base = Rating(mu0 + c["season_decay"] * (base.mu - mu0), base.phi, base.sigma)
    return drift(base, elapsed, c["max_rd"] / SCALE)


def features_from_glicko_state(
    state: dict[str, Any],
    *,
    blue_team_id: str,
    red_team_id: str,
    blue_player_ids: Sequence[str],
    red_player_ids: Sequence[str],
    now: pd.Timestamp | None = None,
) -> dict[str, float]:
    """从快照算单场特征。**必须与 run_glicko_state_machine 的公式逐字一致。**

    training/serving skew 是这类系统最常见的静默故障（PLAN.md §7.4）：两条路径
    各写一遍公式，差一点点，线上比离线差却查不出原因。tests/test_glicko2.py 用
    golden test 把两条路径锁死。`now` 给了就会按休赛期膨胀 RD——线上推理时一定要给，
    否则一支半年没打的队会顶着半年前那个窄 RD 出现在界面上。
    """
    c = state["cfg"]
    if now is None:
        t = float(state.get("as_of_days", 0.0)) / c["inflation_period_days"]
    else:
        t = _period_start_scalar(now, c["period"]) / c["inflation_period_days"]

    rb = _state_view(state, "team", str(blue_team_id), t)
    rr = _state_view(state, "team", str(red_team_id), t)
    bp = _clean_ids(list(blue_player_ids))
    rp = _clean_ids(list(red_player_ids))
    bv = [_state_view(state, "player", p, t) for p in bp]
    rv = [_state_view(state, "player", p, t) for p in rp]

    sd = math.sqrt(rb.phi**2 + rr.phi**2)
    gj = g_factor(sd)
    tn = state["team"]
    pn = state["player"]
    feats = {
        "team_glicko_diff": rb.rating - rr.rating,
        "blue_team_glicko": rb.rating,
        "red_team_glicko": rr.rating,
        "team_rd_blue": rb.rd,
        "team_rd_red": rr.rd,
        "team_rd_joint": SCALE * sd,
        "team_glicko_g_diff": gj * (rb.rating - rr.rating),
        "team_glicko_prob": 1.0 / (1.0 + math.exp(-gj * (rb.mu - rr.mu))),
        "team_vol_blue": rb.sigma,
        "team_vol_red": rr.sigma,
        "team_n_games_min": float(min(
            tn.get(str(blue_team_id), [0, 0, 0, 0, 0])[4],
            tn.get(str(red_team_id), [0, 0, 0, 0, 0])[4],
        )),
    }
    if bv and rv:
        mb = sum(x.mu for x in bv) / len(bv)
        mr = sum(x.mu for x in rv) / len(rv)
        pb = math.sqrt(sum(x.phi**2 for x in bv) / len(bv))
        pr = math.sqrt(sum(x.phi**2 for x in rv) / len(rv))
        feats |= {
            "player_glicko_diff": SCALE * (mb - mr),
            "player_rd_blue": SCALE * pb,
            "player_rd_red": SCALE * pr,
            "player_rd_max_blue": SCALE * max(x.phi for x in bv),
            "player_rd_max_red": SCALE * max(x.phi for x in rv),
            "player_glicko_g_diff": g_factor(math.sqrt(pb * pb + pr * pr)) * SCALE * (mb - mr),
            "player_n_games_min": float(min(pn.get(p, [0, 0, 0, 0, 0])[4] for p in bp + rp)),
        }
    else:
        feats |= {
            "player_glicko_diff": 0.0,
            "player_rd_blue": c["max_rd"], "player_rd_red": c["max_rd"],
            "player_rd_max_blue": c["max_rd"], "player_rd_max_red": c["max_rd"],
            "player_glicko_g_diff": 0.0, "player_n_games_min": 0.0,
        }
    return feats


# ==========================================================================
# 4. RD 的画像与 Elo 对照实验（可复现：python -m lolwp.models.glicko2）
# ==========================================================================


def rd_report(df: pd.DataFrame, feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """RD 的分布画像：冷启动 vs 老手、赛季初 vs 赛季末。

    这是本模块真正的产出——log loss 的差别大概率很小，RD 是 Elo 完全给不出来的东西。
    """
    d = pd.concat([df[["kickoff_utc", "league_tier", "league_canon"]], feats], axis=1)
    d = d.assign(
        month=d.kickoff_utc.dt.month,
        year=d.kickoff_utc.dt.year,
        player_rd=(d.player_rd_blue + d.player_rd_red) / 2,
        team_rd=(d.team_rd_blue + d.team_rd_red) / 2,
    )
    exp_bins = [-1, 0, 5, 20, 100, 500, 10**9]
    exp_lab = ["0 场（首秀）", "1-5", "6-20", "21-100", "101-500", ">500"]
    by_exp_t = d.groupby(
        pd.cut(d.team_n_games_min, exp_bins, labels=exp_lab), observed=True
    ).agg(n=("team_rd", "size"), rd_mean=("team_rd", "mean"),
          rd_p10=("team_rd", lambda s: s.quantile(.10)),
          rd_p90=("team_rd", lambda s: s.quantile(.90)))
    by_exp_p = d.groupby(
        pd.cut(d.player_n_games_min, exp_bins, labels=exp_lab), observed=True
    ).agg(n=("player_rd_max_blue", "size"),
          rd_mean=("player_rd", "mean"),
          rd_max_mean=("player_rd_max_blue", "mean"))
    # 赛季初 vs 赛季末：LoL 赛历 1-2 月开赛、10-11 月世界赛收尾
    seg = pd.cut(d.month, [0, 2, 5, 8, 11, 12],
                 labels=["1-2 赛季初", "3-5", "6-8", "9-11 赛季末", "12 休赛"])
    by_seg = d.groupby(seg, observed=True).agg(
        n=("team_rd", "size"), team_rd=("team_rd", "mean"),
        player_rd=("player_rd", "mean"), team_rd_joint=("team_rd_joint", "mean"))
    intl = d.assign(intl=d.league_tier >= 2).groupby("intl", observed=True).agg(
        n=("team_rd", "size"), team_rd=("team_rd", "mean"),
        player_rd=("player_rd", "mean"), team_rd_joint=("team_rd_joint", "mean"))
    return {
        "by_team_experience": by_exp_t,
        "by_player_experience": by_exp_p,
        "by_season_segment": by_seg,
        "by_international": intl,
    }


# G-阶梯：把 L2/L3 的 Elo 特征原地换成 Glicko，其余口径（标准化、C、fold）完全不动。
G_LADDER: dict[str, list[str]] = {
    "G2": ["side_prior", "team_glicko_diff"],
    "G3": ["side_prior", "team_glicko_diff", "player_glicko_diff"],
    "G3g": ["side_prior", "team_glicko_g_diff", "player_glicko_g_diff"],
    "G3rd": ["side_prior", "team_glicko_diff", "player_glicko_diff",
             "team_rd_joint", "player_rd_max_blue", "player_rd_max_red"],
    "L3rd": ["side_prior", "team_elo_diff", "player_elo_diff",
             "team_rd_joint", "player_rd_max_blue", "player_rd_max_red"],
    "L3G3": ["side_prior", "team_elo_diff", "player_elo_diff",
             "team_glicko_diff", "player_glicko_diff"],
}


def benchmark(
    *,
    periods: Sequence[PeriodKind] = ("week",),
    since: str = "2019-01-01",
    min_tier: int = 1,
) -> dict[str, Any]:
    """与现行 Elo 的同口径对照（PLAN.md §6.1 walk-forward）。

    口径必须与阶段 4 的基线逐字相同，否则 log loss 没法比。做法是**复用**
    `eval.baselines.run_ladder`（同一个标准化器、同一个 C、同一组 fold），
    只是临时往它的 LADDER 里塞几行 Glicko 版本的特征清单，跑完就撤掉。
    这样"差异来自特征"这个结论才站得住——代码路径是同一条。
    """
    from lolwp.eval import baselines, metrics
    from lolwp.eval.walkforward import make_folds, summary
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine

    full = load_dev()
    elo_feats = run_state_machine(full)
    frames: dict[str, pd.DataFrame] = {}
    for per in periods:
        frames[per] = run_glicko_state_machine(full, cfg=Glicko2Config(period=per))

    mask = full.league_tier >= min_tier
    if since:
        mask &= full.kickoff_utc >= pd.Timestamp(since, tz="UTC")
    ev = full[mask].reset_index(drop=True)
    folds = make_folds(ev)

    results: dict[str, Any] = {"folds": summary(ev, folds), "rows": []}
    preds: dict[str, pd.DataFrame] = {}
    saved = dict(baselines.LADDER)
    try:
        for tag, gf in frames.items():
            feats = pd.concat([elo_feats, gf], axis=1)[mask].reset_index(drop=True)
            baselines.LADDER.update(G_LADDER)
            names = ["L2", "L3", *G_LADDER]
            p = baselines.run_ladder(ev, feats, folds, names=names)
            preds[tag] = p
            for nm, g in p.groupby("model"):
                row: dict[str, Any] = dict(metrics.evaluate(g.y.to_numpy(), g.p.to_numpy()))
                row["period"] = tag
                row["model"] = nm
                results["rows"].append(row)
    finally:
        baselines.LADDER.clear()
        baselines.LADDER.update(saved)
    results["preds"] = preds
    results["frames"] = frames
    results["eval_df"] = ev
    return results


def main() -> None:  # pragma: no cover - 手动跑的实验入口
    res = benchmark(periods=("week", "game", "month"))
    tab = pd.DataFrame(res["rows"])[
        ["period", "model", "n", "log_loss", "brier", "auc", "ece", "calib_slope"]
    ]
    print(res["folds"])
    print(tab.round(5).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
