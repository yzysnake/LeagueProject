# LoL 职业赛 BP 后胜率预测

BP（ban/pick）结束、开赛前，给出蓝方获胜概率。覆盖 tier-1 全部赛区与国际赛事。
完全本地部署，离线可用。

## 效果

| | log loss | Brier | AUC | accuracy |
|---|---|---|---|---|
| walk-forward 样本外（23,060 场） | 0.61917 | 0.21543 | 0.7107 | 0.6525 |
| **final holdout（989 场，一次性揭盲）** | **0.61653** | 0.21457 | 0.6992 | 0.6532 |
| 对照：Polymarket 市场（1,302 场交集） | 0.62748 | 0.21875 | 0.6992 | 0.6598 |

两点值得注意：

- **holdout 与开发期几乎一致**（差 0.0026，门槛 0.02），说明模型选择过程本身没有过拟合。
- **与真金白银的预测市场统计上无法区分**（ΔBrier +0.00076，远低于 MDE）。

## 主要结论是一个负结果

预注册第一天就写好并允许为负的那条，现已兑现：

> **BP 阶段本身（协同、克制、选手熟练度）在职业赛的数据量下贡献不可测量。**
> 可预测的部分几乎全部来自队伍与选手实力。

| 加入的东西 | ΔlogLoss | 超过 MDE |
|---|---|---|
| 队伍实力 | +0.0577 | ✓ |
| 分层 BT 选手潜变量（替代 Elo） | +0.0143 | ✓ |
| 英雄版本强度 | +0.0015 | ✓（边缘） |
| 选手熟练度 | +0.0004 | ✗ |
| 协同 + 克制 | −0.0004 | ✗ |

## 快速开始

零凭证即可跑通主链路（数据源全部匿名可得）。

```bash
make setup      # uv sync --frozen，Python 3.11
make data       # 下载 OE 2014-2026 共 13 个 CSV（约 700 MB）+ 五重校验
make bronze     # → Parquet，物理丢弃 135 个赛后统计列
make silver     # → 一场一行的建模宽表
make icons      # 173 张英雄图标，供离线 UI 使用
make artifact   # 训练生产模型（约 40 分钟，分层 BT 要全量拟合）
make app        # 启动本地站点
```

其余命令：`make test` / `make lint` / `make gate` / `make monitor` / `make reproduce`。

### 前后端拆分部署

```bash
make api          # 只起 FastAPI（127.0.0.1:8000，单 worker）
make up           # docker compose：api + web，**纯离线即可用**
make up-online    # 额外起每日拉数与实时轮询
```

七个端点：`/health` `/champions` `/players` `/predict` `/predict/batch`
`/explain` `/live`。实测单次 `/predict` 中位 **0.5 ms**（p99 0.7 ms），
`/predict/batch` 遍历 173 个英雄 3 ms。

三条写进 compose 的硬约束：

- **`--workers 1`**：artifact 含状态快照（2,070 队 / 10,863 选手 /
  113,058 条选手英雄计数），每 worker 一份，多 worker 内存线性翻倍。
  横向扩要加容器实例，不是加 worker。
- **只绑 `127.0.0.1`**：不对外暴露。
- **`profiles: ["online"]`**：让「离线降级」成为默认状态，
  而不是一条需要额外实现和测试的路径。

**可选**（不影响主链路）：注册免费 Fandom 账号可用 Leaguepedia 补充源；
加入 OE Discord 可获知 schema 变更预警。详见 `docs/PLAN.md` §7.11。

## 网站四个 tab

| tab | 做什么 |
|---|---|
| 阵容模拟 | 手动选队伍、选手、英雄，出胜率与「实力 / 阵容 / 赛况」三层分解 |
| what-if | 换任一位置的英雄，看概率怎么变（173 个英雄 3 ms 出结果） |
| **实时比赛** | 自动发现正在进行的职业比赛，读 BP 后的阵容并给出胜率 |
| 历史回看 | 拿真实比赛复盘，对照预测与结果 |

实时页的设计要点：

- **唯一权威的开赛信号是 livestats 的 204→200 跳变**，不是赛程接口的 `state`。
  实测赛程标 `inProgress` 时比赛未必已开打——它标的是「局间休息 + BP 开始」。
- 六种对局状态各有明确文案，**接口挂掉只降级本页**，手动摆阵容完全不受影响。
- 实测：从 livestats 首次返回 200 到页面出概率 **0.221 秒**。

## 模型

**BT4_LR** = 按赛区收缩的 side 先验 + 分层 Bradley-Terry 三项（选手潜变量 /
chemistry / 赛区校准）+ 两层经验贝叶斯的英雄版本强度，外层普通 L2 logistic。

标准化系数：选手潜变量 0.840，英雄版本强度 0.121，chemistry 0.020，
赛区 0.012，side 先验 0.014。

阶段 8 跑了 7 个候选（含 CatBoost、zero-init DeepSet、两种集成），
**无一在 MDE 之上胜过它**。

## 这个项目真正的技术含量在评估协议

网上同类项目大多败在这里。本项目做了：

- **series 级分组切分 + 7 天 embargo**，90 折 rolling-origin walk-forward。
- **四个负控制进 CI**：标签打乱回到 0.6921 / 未来删除不变性 bit-identical /
  泄漏探针 AUC +0.298 / 同版本两次 bit-identical。
- **每个比较各自测组内相关**，MDE 主判据用 series 级 block bootstrap。
- **预注册 + 三条公开修订**（A1/A2/A3），holdout 一次性揭盲。

### 三次撞上同一个坑，值得单独写下来

阈值按全样本尺度定、却拿去判小样本，会得到一个永远在报警的告警器：

| 判据 | 误报率 | 处理 |
|---|---|---|
| 预注册红线「任一 fold AUC > 0.78」 | 零信号下 63% | 修订 A3 换成合并 AUC |
| 预注册「ECE ≤ 0.03」 | 989 场上 98% 过不了 | 揭盲前已记录，报告中说明 |
| 监控「reliability > 3×基线」 | 23/23 窗口全报警 | 阈值改由抽样分布推出 |

现在所有阈值都从抽样分布推出，健康模型的误报率等于 1−q。

## 文档

| 文件 | 内容 |
|---|---|
| `docs/PLAN.md` | 开发手册 v1.2，4,487 行 |
| `reports/final_technical_report.md` | 最终技术报告，含揭盲结果 |
| `experiments/preregistration.md` | 预注册、三条修订、揭盲记录 |
| `reports/stage3_power_and_icc.md` | ICC 与功效分析（含一处自我更正） |
| `reports/stage4_baseline_ladder.md` | 基线阶梯 L0–L5 |
| `reports/stage6_variance_components.md` | 方差成分定标 |
| `reports/stage7_falsification.md` | 协同/克制的证伪 |
| `reports/stage8_comparison.md` | 对照模型、消融、外部对标 |

## 免责

本项目并非由 Riot Games 开发，也未获得 Riot Games 认可。
Riot Games 及其所有相关财产均为 Riot Games, Inc. 的商标或注册商标。
比赛数据来自 [Oracle's Elixir](https://oracleselixir.com)，英雄数据来自 Riot Data Dragon。
