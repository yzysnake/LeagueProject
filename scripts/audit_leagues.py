#!/usr/bin/env python
"""阶段 2 · 赛区归一映射：扫描全量 OE 数据，重新生成 configs/league_canonical.yaml。

为什么需要这个脚本（见 docs/PLAN.md §3.11.5 / §9.5 / §3.2.6）
----------------------------------------------------------------
OE 的 `league` 原始 code 词表逐年漂移，同一个实体会换 code（2025 `LCS`→`LTA N`、
`CBLOL`→`LTA S`，2026 又改回 `LCS`/`CBLOL`）。不做归一化会把一个赛区切成多个互不
相干的实体，照抄某一年的 code 定义 tier-1 又会整年丢数据。所以必须有一张
**双向、年份无关**的显式映射表，并且每次数据刷新都要重新审计。

告警不阻塞（§3.11.4）
----------------------
本脚本发现未登记的新 code 时以退出码 1 报警，但 `canon_league()` 永远不抛异常，
未知 code 一律落 UNKNOWN 桶。管道（silver.py 等）应当 import `canon_league` /
`league_tier`，绝不能 import 本脚本的 main()。

用法
----
    .venv/bin/python scripts/audit_leagues.py            # 审计 + 重写 yaml
    .venv/bin/python scripts/audit_leagues.py --check    # 只审计不写盘（CI 用）
退出码：0 = 全部 code 已登记；1 = 发现未登记 code（告警，不代表管道该停）。
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import logging
import os
from collections import defaultdict
from typing import Any

log = logging.getLogger("audit_leagues")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO, "data", "raw")
OUT_YAML = os.path.join(REPO, "configs", "league_canonical.yaml")

csv.field_size_limit(10_000_000)

UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# 1. 登记表：raw code -> (canon, note)
#
# canon 命名规则【实测依据见 note】：取该实体**在数据快照里最后一个赛季实际使用的
# code**。理由：(a) 与手册 §3.11.5 给的 2026 版表一致（LTA N→LCS、LTA S→CBLOL）；
# (b) 新增数据永远用当前 code，恒等映射是热路径；(c) 已停办的实体自然保留末代名字。
#
# 合并判据（统一口径，避免拍脑袋造实体连续性）：
#   仅当 A、B 时间上不重叠且构成 1:1 接续，且（交接年队伍重叠率 ≥0.50 且共享 ≥5 支
#   队）或（归一化后共享 ≥8 个俱乐部）时才合并。重叠率 = |A∩B| / min(|A|,|B|)。
#   归一化 = 去掉 Academy/Challengers/Youth/Esports/Gaming 之类后缀，因为
#   「X Academy → X Challengers」这种改名会让原始名重叠率严重低估。
#   N:1 的**合并（merger）不是改名**，一律不并（例：BL+DL→ESLOL、LLN+CLS→LLA）。
# ---------------------------------------------------------------------------
ALIASES: dict[str, tuple[str, str]] = {
    # ── 一线赛区及其历史别名 ────────────────────────────────────────────────
    "NA LCS": ("LCS", "2014-2018 北美一线；NA LCS@2018 vs LCS@2019 队伍重叠 10/10=1.00"),
    "LCS": ("LCS", "2015-2018 仅 8-14 场/年（北美区域决赛，队伍全是 NA LCS 队）；2019 起为主 code"),
    "LTA N": ("LCS", "2025 改名；LTA N@2025 vs LCS@2026 重叠 7/8=0.88，vs LCS@2024 6/8=0.75"),
    "EU LCS": ("LEC", "2014-2018 欧洲一线；EU LCS@2018 vs LEC@2019 重叠 6/10=0.60，LEC@2018 的 4 队全在 EU LCS@2018 内"),
    "LEC": ("LEC", "2015-2018 仅 11-14 场/年（欧洲区域决赛）；2019 起为主 code"),
    "OGN": ("LCK", "2015-01~09 的韩国一线（OGN Champions）；OGN@2015 vs LCK@2016 归一化重叠 7/13=0.54，LCK code 2015-09-11 才出现"),
    "LCK": ("LCK", "韩国一线，2015-09 起"),
    "LPL": ("LPL", "中国一线"),
    "CBLOL": ("CBLOL", "巴西一线"),
    "LTA S": ("CBLOL", "2025 改名；LTA S@2025 vs CBLOL@2026 重叠 7/8=0.88（另吸收了 LLA 的 Isurus/Leviatan）"),
    "LTA": ("LTA(2025-only)", "2025 独立存在的 40 场 LTA 跨分区赛（11 队 = LTA N + LTA S 混合），不是任一分区的别名；canon 串沿用手册 §3.11.5 写法"),
    "LCP": ("LCP", "2025 新建的亚太一线，吸收了 PCS/VCS/LJL 的头部队伍"),
    "PCS": ("PCS", "2020-2024 是太平洋一线；2025 起头部队伍转入 LCP，PCS 变成台港澳二级（见 tier_by_period）"),
    "VCS": ("VCS", "2018-2024 越南一线；2025 起头部队伍转入 LCP，VCS 变成越南二级（见 tier_by_period）"),
    "LMS": ("LMS", "2015-2019 台港澳一线，已停办"),
    "OPL": ("LCO", "2015-2020 大洋洲一线；OPL@2020 vs LCO@2021 重叠 7/8=0.88"),
    "LCO": ("LCO", "2021-2024 大洋洲一线，2023 起失去世界赛席位，2024 停办"),
    "LCL": ("LCL", "2016-2022 独联体一线，已停办"),
    "LLA": ("LLA", "2019-2024 拉美一线，2025 并入 LTA S；与 LTA S 队伍重叠仅 2/7=0.29，不做别名合并"),
    "LLN": ("LLN", "2016-2018 拉美北部一线；与 LLA@2019 重叠 3/9=0.33，未达合并阈值"),
    "CLS": ("CLS", "2016-2018 拉美南部一线；与 LLA@2019 重叠 3/10=0.30，未达合并阈值"),
    "GPL": ("GPL", "2016-2018 东南亚一线，已停办"),
    "TCL": ("TCL", "土耳其一线；2015-2021 有世界赛席位，2022 起并入 EMEA 体系（见 tier_by_period）"),
    "LJL": ("LJL", "日本一线；2016-2024 有世界赛席位，2025 起头部转入 LCP（见 tier_by_period）"),
    # ── 国际赛 ──────────────────────────────────────────────────────────────
    "WLDs": ("WLDs", "世界赛。注意原始 code 大小写是 WLDs，不是 WLDS"),
    "MSI": ("MSI", "季中邀请赛"),
    "FST": ("FST", "First Stand，2025 新增（手册已证伪 'FS' 这个写法）"),
    "EWC": ("EWC", "Esports World Cup，2024 起"),
    "IEM": ("IEM", "2015-2017 IEM 国际赛事，队伍来自 19 个其它 code"),
    "IWCI": ("IWCI", "2016 International Wildcard Invitational"),
    "MSC": ("MSC", "2020 Mid-Season Cup（LPL×LCK）"),
    "ASI": ("ASI", "2025 亚洲邀请赛，8 队来自 LCK/LPL/LCP"),
    "Riot": ("Riot", "2020 的 20 场跨区赛（GAM/Machi/PSG Talon/Team Flash）"),
    # ── 二级 / 区域 / 青训 / 杯赛：有别名合并的 ────────────────────────────
    "SLO": ("LES", "西班牙：SLO(2018-2020)→LVP SL；SLO@2020 vs LVP SL@2021 重叠 10/11=0.91"),
    "LVP SL": ("LES", "西班牙：LVP SL(2021-2025)→LES(2026)；LVP SL@2025 vs LES@2026 重叠 6/8=0.75"),
    "LES": ("LES", "西班牙二级，2026 当前 code"),
    "ESLOL": ("ROL", "比荷卢：ESLOL(2022-2024)→ROL(2025-)；ESLOL@2024 vs ROL@2025 重叠 8/9=0.89"),
    "ROL": ("ROL", "比荷卢二级，2025 起"),
    "UL": ("RL", "波兰：Ultraliga(2018-2024)→RL(2025-)；UL@2024 vs RL@2025 重叠 6/12=0.50"),
    "RL": ("RL", "波兰二级，2025 起"),
    "PGN": ("LIT", "意大利：PG Nationals(2018-2023)→LIT(2024-)；PGN@2023 vs LIT@2024 重叠 7/11=0.64"),
    "LIT": ("LIT", "意大利二级，2024 起"),
    "LGL": ("HLL", "希腊：LGL(2019, 8 场)与 GLL@2019 队伍 4/4=1.00，是同一批开赛周"),
    "GLL": ("HLL", "希腊：GLL(2019-2024)→HLL(2025-)；GLL@2024 vs HLL@2025 归一化重叠 9/12=0.75"),
    "HLL": ("HLL", "希腊二级，2025 起"),
    "KeSPA": ("KeSPA Cup", "韩国杯赛：KeSPA(2015-2025)→KeSPA Cup(2026)；重叠 10/10=1.00"),
    "KeSPA Cup": ("KeSPA Cup", "韩国杯赛 2026 当前 code；参赛队是 LCK 一线队，但属国内杯赛不是国际赛"),
    "LCSA": ("NACL", "北美二级：LCS Academy(2018-2022)→NACL(2023-)；原始名重叠仅 5/24=0.20，但去掉 Academy/Challengers 后缀后 14/24=0.58"),
    "NACL": ("NACL", "北美二级，2023 起"),
    "LSPL": ("LDL", "中国二级：LSPL(2016-2017)→LDL(2018-)；归一化共享 8 个俱乐部（0.42）"),
    "LDL": ("LDL", "中国二级，2018-2024（2025 起无数据）"),
    "EUM": ("EM", "EU Masters(2018-2022)→EMEA Masters(2023-)；归一化共享 15 个俱乐部（0.38）"),
    "EM": ("EM", "EMEA Masters，2023 起"),
    # ── 二级 / 区域 / 青训 / 杯赛：不合并的 ────────────────────────────────
    "AC": ("AC", "美洲跨分区杯赛；2024 版是学院级(6 队)，2026 版是一线级(C9/FURIA/RED/Sentinels, 19 场)。冻结的 eval 口径未收录，故保持 tier 0，见 tier_open_questions"),
    "AL": ("AL", "Arabian League，中东地区赛"),
    "AOL": ("AOL", "2021 北美学院邀请赛"),
    "ASCI": ("ASCI", "亚洲青训跨区赛（LCKC/LDL/PCL 的二队）"),
    "Asia Master": ("Asia Master", "亚洲青训跨区赛，2025 起"),
    "BIG": ("BIG", "北美业余/大学循环赛"),
    "BL": ("BL", "比利时联赛 2020-2021，2022 与 DL 一同并入 ESLOL；N:1 合并不做别名"),
    "BM": ("BM", "波罗的海 Baltic Masters"),
    "BRCC": ("BRCC", "巴西 Circuito Challenger 2017-2020"),
    "CBLOLA": ("CBLOLA", "CBLOL Academy 2021-2024；与 CD@2025 归一化重叠仅 6/16=0.38，未达阈值，不合并"),
    "CCWS": ("CCWS", "2026 德语区跨赛区杯赛（PRM/PRMP 队伍）"),
    "CD": ("CD", "巴西二级 2025-（Circuito Desafiante 系）；与 CBLOLA 关系见其 note"),
    "CDF": ("CDF", "法国杯 Coupe de France 2022-2024"),
    "CISC": ("CISC", "独联体次级 2017-2018"),
    "CK": ("CK", "Challengers Korea 2016-2020；与 LCKC@2021 队伍重叠 1/14=0.10（CK 是独立俱乐部，LCKC 是一线队青训），不合并"),
    "CT": ("CT", "西班牙 Circuito Tormenta 2019-2025"),
    "CU": ("CU", "北美大学/业余 2020-2021"),
    "DCup": ("DCup", "德玛西亚杯，中国国内杯赛（参赛队是 LPL 一线队，但不是国际赛）"),
    "DDH": ("DDH", "墨西哥次级 2019-2023"),
    "DL": ("DL", "荷兰联赛 2020-2021，2022 并入 ESLOL；N:1 合并不做别名"),
    "EBL": ("EBL", "巴尔干 Esports Balkan League"),
    "EBLPA": ("EBLPA", "EBL 次级/预选"),
    "EGL": ("EGL", "2021 北美业余"),
    "EL": ("EL", "中美洲 Elite League 2022-2023"),
    "EPL": ("EPL", "2023-2024 跨 ERL 邀请赛"),
    "EU CS": ("EU CS", "欧洲 Challenger Series 2014-2017（二级）"),
    "GL": ("GL", "拉美 Golden League 2021-2023"),
    "GLLPA": ("GLLPA", "希腊次级/预选，与 GLL 同期并行，不合并"),
    "GSG": ("GSG", "2021 北美业余"),
    "HC": ("HC", "捷克/斯洛伐克次级（队伍多为 HM 队的 Academy）"),
    "HM": ("HM", "捷克/斯洛伐克一级国内联赛 Hitpoint Masters"),
    "HW": ("HW", "捷克/斯洛伐克，与 HM 同期并行的另一赛事，不合并"),
    "IC": ("IC", "**Iberian Cup**（西班牙 LVP SL + 葡萄牙 LPLOL 的跨国杯赛）。实测 2024/2025 参赛队全是西葡二级队，不是国际赛；eval/tier1_leagues.yaml 把它列进 international 是错的"),
    "IGNIS": ("IGNIS", "2024 巴西青训"),
    "LAS": ("LAS", "**LCK Academy Series**（韩国一线队青训队），不是拉美赛区"),
    "LCKC": ("LCKC", "LCK Challengers League，2021 起"),
    "LFL": ("LFL", "法国一级国内联赛"),
    "LFL2": ("LFL2", "法国二级 2020-2025；与 NL@2026 重叠 4/10=0.40，未达阈值，不合并"),
    "LHE": ("LHE", "智利/秘鲁次级 2019-2023"),
    "LJLA": ("LJLA", "LJL Academy 2020-2023"),
    "LJLCS": ("LJLCS", "LJL Challenger Series 2016-2018"),
    "LMF": ("LMF", "阿根廷 Liga Master Flow 2019-2023"),
    "LPLOL": ("LPLOL", "葡萄牙一级国内联赛"),
    "LRN": ("LRN", "拉美北部区域联赛，2023 起"),
    "LRS": ("LRS", "拉美南部区域联赛，2023 起"),
    "NA CS": ("NA CS", "北美 Challenger Series 2014-2017（二级）"),
    "NASG": ("NASG", "北美 Scouting Grounds"),
    "NEST": ("NEST", "2020 中国全国电子竞技大赛（参赛队全是 LPL 队，国内赛事）"),
    "NERD": ("NERD", "2021 北美业余"),
    "NEXO": ("NEXO", "西班牙次级 2020-2025"),
    "NL": ("NL", "2026 法国赛事（10 队）；与 LFL2@2025 重叠 4/10=0.40，疑似 LFL2 后继但证据不足，不合并"),
    "NLC": ("NLC", "北欧/英国 Northern League of Legends Championship"),
    "NLC Aurora Open": ("NLC Aurora Open", "NLC 公开赛/次级"),
    "OCS": ("OCS", "大洋洲 Challenger Series 2016-2020"),
    "OTBLX": ("OTBLX", "比荷卢业余 2020-2021"),
    "PCL": ("PCL", "Pacific Challenger League（亚太青训）2023-2024"),
    "PGC": ("PGC", "2022 北美业余大循环（74 队）"),
    "PRM": ("PRM", "德语区一级国内联赛 Prime League"),
    "PRMP": ("PRMP", "Prime League 次级/杯赛，与 PRM 并行"),
    "RCL": ("RCL", "2020-2021 北美业余"),
    "SL (LATAM)": ("SL (LATAM)", "秘鲁次级 2021-2023"),
    "TAL": ("TAL", "土耳其学院联赛 2019-2022"),
    "TCS": ("TCS", "土耳其次级 2018"),
    "TPL": ("TPL", "土耳其次级 2015-2017；与 TCS@2018 重叠 4/9=0.44，未达阈值"),
    "TSC": ("TSC", "2024 土耳其杯赛"),
    "UGP": ("UGP", "2021 北美业余"),
    "UKLC": ("UKLC", "英国/爱尔兰 2019-2021"),
    "UPL": ("UPL", "北美大学/业余 2020-2022"),
    "USP": ("USP", "波兰次级 2022-2024"),
    "VL": ("VL", "厄瓜多尔/哥伦比亚次级 2021-2023"),
}
# ---------------------------------------------------------------------------
# 2. tier：canonical -> 0/1/2
#
# tier 2 = 国际赛（参赛队来自 ≥3 个不同常规赛区的跨区赛事）
# tier 1 = 一线赛区（某赛季把队伍送上 WLDs/MSI/IWCI/MSC 的顶级地区联赛）
# tier 0 = 其余（二级、青训、业余、地区联赛、以及国内杯赛）
#
# 口径来源：eval/tier1_leagues.yaml（**已冻结的预注册文件，优先**）。本表在它之外
# 补了历史上确实存在过、但该文件没收的一线赛区（LMS/LCO(OPL)/LCL/LLA/LLN/CLS/GPL），
# 它们全部已停办（最后一场 2024-08-24），因此对 2026-06-01 起的 holdout 评测口径
# **零影响**。详见 yaml 里的 eval_scope_delta_vs_frozen。
# ---------------------------------------------------------------------------
TIER: dict[str, int] = {
    # tier 2 国际赛
    "MSI": 2, "WLDs": 2, "FST": 2, "EWC": 2,
    "IEM": 2, "IWCI": 2, "MSC": 2, "ASI": 2, "Riot": 2,
    # tier 1 一线赛区（现役）
    "LPL": 1, "LCK": 1, "LEC": 1, "LCS": 1, "CBLOL": 1, "LCP": 1, "PCS": 1, "VCS": 1,
    "LTA(2025-only)": 1,
    # tier 1 一线赛区（已停办，冻结文件未收，本表补入）
    "LMS": 1, "LCO": 1, "LCL": 1, "LLA": 1, "LLN": 1, "CLS": 1, "GPL": 1,
}

# tier 随时间变化的实体：[canon, 起, 止(含), tier]。**默认不生效**，只作记录。
# 依据：该赛季是否有队伍出现在 WLDs/MSI（原始队名精确匹配，不做 Academy 归一化）。
TIER_BY_PERIOD: list[dict[str, Any]] = [
    {"canon": "PCS", "from": "2020-01-01", "to": "2024-12-31", "tier": 1,
     "why": "2020-2024 每年 2-3 队进世界赛"},
    {"canon": "PCS", "from": "2025-01-01", "to": None, "tier": 0,
     "why": "2025/2026 零队进世界赛，2025 阵容变成 CFO Academy/PSG Talon Academy 等二队；"
            "但 eval/tier1_leagues.yaml 已冻结且把 PCS 列为 tier-1，默认 tier 仍取 1"},
    {"canon": "VCS", "from": "2018-01-01", "to": "2024-12-31", "tier": 1,
     "why": "2018/2019/2022/2023/2024 有队进世界赛（2020/2021 疫情缺席）"},
    {"canon": "VCS", "from": "2025-01-01", "to": None, "tier": 0,
     "why": "2025/2026 零队进世界赛，GAM/Team Secret Whales 已转入 LCP；同上，默认 tier 仍取 1"},
    {"canon": "LJL", "from": "2016-01-01", "to": "2024-12-31", "tier": 1,
     "why": "2016-2024 每年 1-2 队进世界赛；2025 起头部转入 LCP，默认 tier 取 0"},
    {"canon": "TCL", "from": "2015-01-01", "to": "2021-12-31", "tier": 1,
     "why": "2015-2021 每年 1-2 队进世界赛；2022 起并入 EMEA 体系，默认 tier 取 0"},
    {"canon": "LCO", "from": "2016-01-01", "to": "2022-12-31", "tier": 1,
     "why": "OPL 2016-2020 + LCO 2021-2022 有世界赛席位；2023-2024 无，默认 tier 取 1"},
]

# 冻结文件 eval/tier1_leagues.yaml 里被本表判定为错的条目（只报告，不改那个文件）
FROZEN_EVAL_ISSUES = [
    {"item": "international 含 'WLDS'",
     "problem": "原始 code 实际是 'WLDs'（小写 s），全量 1,349 场。**这个 bug 已经生效**："
                "eval/holdout_gameids_20260918.txt 是 1,071 场，而 date>=2026-06-01 且 tier∈{1,2} 的"
                "实测是 1,079 场，差的正好是 2026 世界赛那 8 场（8 场 gameid 无一在 holdout 名单里）。"
                "冻结名单不能改，但下游必须知道 holdout 里没有世界赛样本"},
    {"item": "international 含 'IC'",
     "problem": "IC = Iberian Cup（西葡二级队的跨国杯赛，2022-2025 共 244 场），不是国际赛。"
                "本表判为 tier 0"},
    {"item": "tier1_2026 含 'PCS' / 'VCS'",
     "problem": "实测 2025 起两者都零队进世界赛、阵容变成二队；2026 分别 86 / 126 场。"
                "因该文件已冻结，本表仍给 tier 1，仅在 tier_by_period 记录"},
    {"item": "缺历史一线赛区",
     "problem": "NA LCS / EU LCS / OGN 由别名归一解决；LMS / OPL+LCO / LCL / LLA / LLN / CLS / GPL "
                "该文件完全没提，本表补为 tier 1（全部已停办，对 holdout 零影响）"},
]


def _read_holdout_start() -> str:
    """读 eval/HOLDOUT_START。只用于**描述性**统计（报告 holdout 窗口有多少场），不参与任何选择。"""
    path = os.path.join(REPO, "eval", "HOLDOUT_START")
    try:
        return open(path, encoding="utf-8").read().strip()
    except OSError:
        log.warning("读不到 %s，跳过 holdout 窗口统计", path)
        return ""


def _read_frozen_eval_codes() -> set[str]:
    """读已冻结的 eval/tier1_leagues.yaml，取它**字面**列出的 raw code 集合。

    只读不写。目的是量化「照字面用冻结文件过滤」与「用本表 canonical tier 过滤」的差异，
    让人能看见 WLDS 大小写、IC 误归国际赛这类问题到底影响多少场。
    """
    path = os.path.join(REPO, "eval", "tier1_leagues.yaml")
    try:
        import yaml  # 只在审计时需要，silver 侧不依赖
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001 - 读不到就降级，不让审计脚本崩
        log.warning("读不到/解析不了 %s（%s），跳过冻结口径对比", path, exc)
        return set()
    out: set[str] = set()
    for key in ("tier1_2026", "tier1_2025_aliases", "international"):
        out |= {str(x) for x in (d.get(key) or [])}
    return out


def canon_league(raw: str) -> str:
    """raw league code -> canonical。未登记的 code 告警后落 UNKNOWN 桶，**绝不抛异常**。"""
    hit = ALIASES.get(raw)
    if hit is not None:
        return hit[0]
    log.warning("未知 league code: %r，归入 %s 桶", raw, UNKNOWN)
    return UNKNOWN


def league_tier(canon: str) -> int:
    """canonical -> tier。未登记一律 0（二级兜底，不会把未知赛区误当一线）。"""
    return TIER.get(canon, 0)


# ---------------------------------------------------------------------------
# 3. 扫描
# ---------------------------------------------------------------------------
def scan_raw() -> dict[str, Any]:
    """流式扫 13 个年度 CSV。不用 pandas：只要 5 列，csv 模块比 read_csv 省内存也够快。"""
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*_LoL_esports_match_data_from_OraclesElixir.csv")))
    if not paths:
        raise SystemExit(f"FAIL: {RAW_DIR} 下没有年度 CSV")
    hold_start = _read_holdout_start()
    games: dict[str, set[str]] = defaultdict(set)
    hold: dict[str, set[str]] = defaultdict(set)
    per_year: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    dmin: dict[str, str] = {}
    dmax: dict[str, str] = {}
    teams: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lg = row["league"]
                # 关键：按 date 的日历年分桶，不用 `year` 列。实测 `year` 列是赛季标签，
                # 与 date 不一致的行有 2 万余行（例：NA LCS 2014-09-09 的行 year=2015，
                # LTA N 2025-10-04 的行 year=2026）。用 year 列会把改名年份算错。
                d = row["date"][:10]
                games[lg].add(row["gameid"])
                per_year[lg][d[:4]].add(row["gameid"])
                if hold_start and d >= hold_start:
                    hold[lg].add(row["gameid"])
                if lg not in dmin or d < dmin[lg]:
                    dmin[lg] = d
                if lg not in dmax or d > dmax[lg]:
                    dmax[lg] = d
                if row["position"] == "team" and row["teamname"].strip():
                    teams[lg].add(row["teamname"].strip())
    return {
        "paths": paths,
        "games": {k: len(v) for k, v in games.items()},
        "per_year": {k: {y: len(g) for y, g in sorted(d.items())} for k, d in per_year.items()},
        "first": dmin,
        "last": dmax,
        "n_teams": {k: len(v) for k, v in teams.items()},
        "holdout_start": hold_start,
        "holdout_games": {k: len(v) for k, v in hold.items()},
        "total_games": len({g for s in games.values() for g in s}),
    }


# ---------------------------------------------------------------------------
# 4. 生成 yaml
# ---------------------------------------------------------------------------
def _q(s: str) -> str:
    """YAML 双引号标量：内部双引号和反斜杠需转义。"""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_yaml(stats: dict[str, Any], missing: list[str]) -> str:
    codes = sorted(stats["games"])
    canon_of = {c: (ALIASES[c][0] if c in ALIASES else UNKNOWN) for c in codes}
    # canonical 聚合
    agg: dict[str, dict[str, Any]] = {}
    for c in codes:
        k = canon_of[c]
        a = agg.setdefault(k, {"aliases": [], "games": 0, "first": "9999", "last": "0000",
                               "years": defaultdict(int)})
        a["aliases"].append(c)
        a["games"] += stats["games"][c]
        a["first"] = min(a["first"], stats["first"][c])
        a["last"] = max(a["last"], stats["last"][c])
        for y, n in stats["per_year"][c].items():
            a["years"][y] += n
    tier_games: dict[int, int] = defaultdict(int)
    tier_canon: dict[int, list[str]] = defaultdict(list)
    for k, a in agg.items():
        t = league_tier(k)
        tier_games[t] += a["games"]
        tier_canon[t].append(k)
    # 手册 §3.2.6 断言②：canonical 后每年 tier-1 场次不得比上一年掉 >30%
    years = [str(y) for y in range(2014, 2027)]
    t1_by_year = {y: sum(a["years"].get(y, 0) for k, a in agg.items() if league_tier(k) == 1)
                  for y in years}

    L: list[str] = []
    ap = L.append
    ap("# ============================================================================")
    ap("# 赛区 code 归一映射表（阶段 2 产出）")
    ap("#")
    ap("# 本文件由 scripts/audit_leagues.py 从 data/raw/ 全量重算生成，不要手改：")
    ap("#   .venv/bin/python scripts/audit_leagues.py")
    ap("# 改映射请改脚本里的 ALIASES / TIER，再重新生成。")
    ap("#")
    ap("# 依据 docs/PLAN.md §3.11.5（canonical 表 + UNKNOWN 兜底）、§9.5（赛区）、")
    ap("# §3.2.6（赛区集合与样本量）、§3.11.4（告警不阻塞）。")
    ap("# ============================================================================")
    ap("")
    ap("meta:")
    ap("  generated_by: scripts/audit_leagues.py")
    ap(f"  generated_at: {_q(_dt.datetime.now(_dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ'))}")
    ap(f"  source_files: {len(stats['paths'])}")
    ap(f"  data_first_date: {_q(min(stats['first'].values()))}")
    ap(f"  data_last_date: {_q(max(stats['last'].values()))}")
    ap(f"  total_games: {stats['total_games']}")
    ap(f"  n_raw_codes: {len(codes)}")
    ap(f"  n_canonical: {len(agg)}")
    ap(f"  n_codes_merged_away: {len(codes) - len(agg)}")
    ap("  note: >-")
    ap("    场次按 gameid 去重统计；年份按 date 的日历年分桶，**不用 `year` 列**——实测 `year`")
    ap("    是赛季标签，与 date 不同年的行有 2 万余行（NA LCS 2014-09-09 的行 year=2015，")
    ap("    LTA N 2025-10-04 的行 year=2026），用它会把改名发生的年份算错。")
    ap("")
    ap("# ---------------------------------------------------------------------------")
    ap("# UNKNOWN 兜底策略（§3.11.5 要求显式写出；§3.11.4 要求告警不阻塞）")
    ap("# ---------------------------------------------------------------------------")
    ap("unknown_policy:")
    ap(f"  bucket: {UNKNOWN}")
    ap("  tier: 0")
    ap("  on_unregistered_code:")
    ap("    - 1. canon_league() 记 WARNING 并返回 UNKNOWN，**不抛异常**：新赛区上线不得让管道崩。")
    ap("    - 2. 该场比赛照常进 silver/训练，league_canon=UNKNOWN、league_tier=0。")
    ap("    - 3. UNKNOWN 桶是一个真实的 categorical level，不是 NaN；模型里它就是「没见过的小赛区」。")
    ap("    - 4. scripts/audit_leagues.py 以退出码 1 告警（CI 红），人工登记后重新生成本文件。")
    ap("    - 5. 绝不允许把 UNKNOWN 并进任何一线赛区，也绝不允许用 get_dummies 现场生成赛区特征。")
    ap("  forbidden:")
    _l25 = stats["per_year"].get("LTA N", {}).get("2025", 0) + stats["per_year"].get("LTA S", {}).get("2025", 0)
    _l26 = stats["per_year"].get("LCS", {}).get("2026", 0) + stats["per_year"].get("CBLOL", {}).get("2026", 0)
    ap(f"    - 任何地方都不得用原始 code 定义 tier-1：照抄 2024 的 code 过滤，2025 会丢 LTA N+LTA S 共 {_l25} 场；")
    ap(f"      照抄 2025 的 code 过滤，2026 会丢 LCS+CBLOL 共 {_l26} 场。两个方向都会静默丢一整年北美+巴西比赛。")
    ap("    - 不得对 UNKNOWN 做 assert/raise（§3.11.4：覆盖率监控绝不能做成启动 assert）。")
    ap("")
    ap("tier_definition:")
    ap('  "2": 国际赛 —— 参赛队来自 ≥3 个不同常规赛区的跨区赛事')
    ap('  "1": 一线赛区 —— 该赛季把队伍送上 WLDs/MSI/IWCI/MSC 的顶级地区联赛')
    ap('  "0": 二级 / 青训 / 业余 / 地区联赛 / 国内杯赛（含未知）')
    ap("  authority: >-")
    ap("    现役一线口径以已冻结的 eval/tier1_leagues.yaml 为准；该文件没覆盖到的历史赛区，")
    ap("    用「该赛季是否有队伍出现在 WLDs/MSI（原始队名精确匹配）」客观判定。")
    ap("    国内杯赛（DCup/KeSPA Cup/NEST）的参赛队虽然也进世界赛，但它们不是地区联赛，判 tier 0。")
    ap("")
    ap("# ---------------------------------------------------------------------------")
    ap("# 主表：raw league code -> {canon, tier, note}")
    ap("# tier 是 canon 的属性，这里冗余展开是为了让 silver 侧一次查表拿全。")
    ap("# ---------------------------------------------------------------------------")
    ap("leagues:")
    for c in codes:
        k = canon_of[c]
        note = ALIASES[c][1] if c in ALIASES else "未登记，落 UNKNOWN 桶"
        py = stats["per_year"][c]
        ap(f"  {_q(c)}:")
        ap(f"    canon: {_q(k)}")
        ap(f"    tier: {league_tier(k)}")
        ap(f"    games: {stats['games'][c]}")
        ap(f"    first_date: {_q(stats['first'][c])}")
        ap(f"    last_date: {_q(stats['last'][c])}")
        ap(f"    n_teams: {stats['n_teams'].get(c, 0)}")
        ap("    years: {" + ", ".join(f'"{y}": {n}' for y, n in py.items()) + "}")
        ap(f"    note: {_q(note)}")
    ap("")
    ap("# ---------------------------------------------------------------------------")
    ap("# canonical 实体汇总")
    ap("# ---------------------------------------------------------------------------")
    ap("canonical:")
    for k in sorted(agg, key=lambda x: (-agg[x]["games"], x)):
        a = agg[k]
        ap(f"  {_q(k)}:")
        ap(f"    tier: {league_tier(k)}")
        ap(f"    games: {a['games']}")
        ap(f"    first_date: {_q(a['first'])}")
        ap(f"    last_date: {_q(a['last'])}")
        ap(f"    raw_codes: [{', '.join(_q(x) for x in sorted(a['aliases']))}]")
    ap("")
    ap("# 便捷集合（下游请用这两个，别再手写字符串字面量）")
    ap(f"tier1_canon: [{', '.join(_q(x) for x in sorted(tier_canon[1]))}]")
    ap(f"intl_canon:  [{', '.join(_q(x) for x in sorted(tier_canon[2]))}]")
    ap("")
    ap("tier_totals:")
    for t in (2, 1, 0):
        ap(f"  \"{t}\": {{n_canonical: {len(tier_canon[t])}, games: {tier_games[t]}}}")
    ap("")
    ap("tier1_games_by_year:")
    for y in years:
        ap(f"  \"{y}\": {t1_by_year[y]}")
    ap("")
    ap("# ---------------------------------------------------------------------------")
    ap("# tier 随时间变化的实体（**默认不生效**，只作记录；要用请在 experiments/ 下显式开）")
    ap("# ---------------------------------------------------------------------------")
    ap("tier_by_period:")
    for r in TIER_BY_PERIOD:
        ap(f"  - canon: {_q(r['canon'])}")
        ap(f"    from: {_q(r['from'])}")
        ap(f"    to: {_q(r['to']) if r['to'] else 'null'}")
        ap(f"    tier: {r['tier']}")
        ap(f"    why: {_q(r['why'])}")
    ap("")
    ap("# ---------------------------------------------------------------------------")
    ap("# 「照字面用冻结文件的 raw code 过滤」 vs 「用本表 canonical tier∈{1,2} 过滤」")
    ap("# 的逐 code 差异。holdout_games 只是描述性计数，不用于任何调参或模型选择。")
    ap("# ---------------------------------------------------------------------------")
    frozen = _read_frozen_eval_codes()
    hs = stats.get("holdout_start") or ""
    ap("eval_scope_delta_vs_frozen:")
    ap("  frozen_file: eval/tier1_leagues.yaml")
    ap(f"  frozen_literal_codes: [{', '.join(_q(x) for x in sorted(frozen))}]")
    ap(f"  holdout_start: {_q(hs)}")
    only_frozen, only_mine = [], []
    for c in codes:
        inf = c in frozen
        inm = league_tier(canon_of[c]) in (1, 2)
        if inf and not inm:
            only_frozen.append(c)
        elif inm and not inf:
            only_mine.append(c)
    ap("  in_frozen_but_tier0_here:   # 冻结文件收了、本表判 tier 0")
    if not only_frozen:
        ap("    []")
    for c in only_frozen:
        ap(f"    - {{code: {_q(c)}, games: {stats['games'][c]}, "
           f"holdout_games: {stats['holdout_games'].get(c, 0)}, reason: {_q(ALIASES.get(c, ('', ''))[1][:120])}}}")
    ap("  in_scope_here_but_not_in_frozen:   # 本表判 tier 1/2、冻结文件字面没有")
    if not only_mine:
        ap("    []")
    for c in only_mine:
        ap(f"    - {{code: {_q(c)}, tier: {league_tier(canon_of[c])}, games: {stats['games'][c]}, "
           f"holdout_games: {stats['holdout_games'].get(c, 0)}, last_date: {_q(stats['last'][c])}}}")
    add_h = sum(stats["holdout_games"].get(c, 0) for c in only_mine)
    drop_h = sum(stats["holdout_games"].get(c, 0) for c in only_frozen)
    ap(f"  holdout_games_added: {add_h}")
    ap(f"  holdout_games_dropped: {drop_h}")
    ap("")
    ap("# ---------------------------------------------------------------------------")
    ap("# 与已冻结的 eval/tier1_leagues.yaml 的差异（只报告，不修改那个文件）")
    ap("# ---------------------------------------------------------------------------")
    ap("frozen_eval_issues:")
    for r in FROZEN_EVAL_ISSUES:
        ap(f"  - item: {_q(r['item'])}")
        ap(f"    problem: {_q(r['problem'])}")
    ap("")
    if missing:
        ap("unregistered_codes:   # 本次扫描发现的未登记 code，请尽快登记")
        for c in missing:
            ap(f"  - {_q(c)}")
    else:
        ap("unregistered_codes: []   # 本次扫描：全部 raw code 均已登记")
    ap("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap_ = argparse.ArgumentParser(description="赛区 code 归一映射审计 / 生成器")
    ap_.add_argument("--check", action="store_true", help="只审计不写盘")
    args = ap_.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = scan_raw()
    codes = sorted(stats["games"])
    missing = [c for c in codes if c not in ALIASES]
    stale = [c for c in ALIASES if c not in stats["games"]]

    text = build_yaml(stats, missing)
    if not args.check:
        os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)
        with open(OUT_YAML, "w", encoding="utf-8") as fh:
            fh.write(text)
        log.info("已写出 %s（%d 行）", OUT_YAML, text.count("\n") + 1)

    n_canon = len({ALIASES[c][0] for c in codes if c in ALIASES} | ({UNKNOWN} if missing else set()))
    log.info("raw code %d 个 -> canonical %d 个（合并掉 %d 个）",
             len(codes), n_canon, len(codes) - n_canon)
    for t in (2, 1, 0):
        ks = {ALIASES[c][0] for c in codes if c in ALIASES and league_tier(ALIASES[c][0]) == t}
        g = sum(stats["games"][c] for c in codes if c in ALIASES and league_tier(ALIASES[c][0]) == t)
        log.info("  tier %d: %d 个 canonical, %d 场", t, len(ks), g)
    if stale:
        log.warning("登记表里有 %d 个 code 在数据中已不存在（保留即可，属历史别名）：%s",
                    len(stale), stale)
    if missing:
        # §3.11.4：告警，不阻塞管道。这里退非零只是让 CI 红，silver 侧照跑。
        log.error("发现 %d 个未登记 league code（已落 UNKNOWN 桶，管道不受影响）：%s",
                  len(missing), missing)
        log.error("处理办法：在 scripts/audit_leagues.py 的 ALIASES/TIER 里登记后重新生成本文件。")
        return 1
    log.info("OK: 全部 %d 个 raw league code 均已登记", len(codes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
