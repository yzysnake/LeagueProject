"""把 lolesports feed 的实体解析成模型认识的 ID（PLAN.md §3.11 / §7.7.2）。

feed 和模型讲的是两套完全不同的语言：

    feed  esportsPlayerId  数字雪花 110350224593953686
          esportsTeamId    数字雪花 103461966965149786
          championId       DDragon 字符串 id "KSante"
          role             "top/jungle/mid/bottom/support"
          patchVersion     "16.17.810.4348"
    模型  oe:player:<hash> / oe:team:<hash> / DDragon 数字 key 897 /
          OE position "top/jng/mid/bot/sup" / (major, minor)

**总原则（§3.11 开篇）：全部显式，绝对禁止 difflib / fuzzy 兜底。**
有实测先例：difflib 把 `Beijing JDG Esports` 静默映射成了 `LNG Esports`——
另一支队伍，无报错。本模块里**没有任何一处按名字相似度做决策**；
名字只在 `validate_team_crosswalk()` 里当**只读校验器**用，不参与建表。

──────────────────────────────────────────────────────────────────────
四张表的来源与可靠性
──────────────────────────────────────────────────────────────────────
1. 英雄 `data/ddragon/champion_map.json`：程序生成的双射，173 个，往返 100%
   （§3.11.4：唯一该做成 CI 断言的就是这张）。
2. 位置 `FEED_ROLE_TO_OE`：常量 5 条（§3.11.2）。
3. 选手 `data/crosswalk/player_crosswalk.parquet`：**只取 confidence=="primary"**。
   全表含 weak/alias 有 294 个一对多，用错会把某选手的历史查到别人头上。
4. 队伍 `data/crosswalk/team_crosswalk.parquet`：**本模块新建**，见下。

──────────────────────────────────────────────────────────────────────
队伍对照表：为什么不用 getTeams 的 code/name 做显式映射
──────────────────────────────────────────────────────────────────────
任务给的提示是「用 getTeams 的 code/name + silver 的 teamname 做显式映射」。
实测这条路不必走，而且更差：

  · 名字口径两边根本不一致。实测 489 支队里有 101 支（20.7%）feed 名与 OE 名
    逐字不等：`Gen.G Esports` vs `Gen.G`、`Shenzhen NINJAS IN PYJAMAS` vs
    `Ninjas in Pyjamas`、`kt Challengers` vs `KT Rolster Challengers`。
    要把这 20.7% 接上，除了 fuzzy 没有别的办法——而 fuzzy 是被明令禁止的。
  · 根本不需要。`data/crosswalk/_cache/observations.parquet`（219,558 行，
    阶段 2 用「同日 + 10 英雄多重集」硬 join 产出）里**每一行都同时带
    `esports_team_id` 与 `oe_teamid`**。也就是说 (feed 队 ↔ OE 队) 的确定性
    对照已经附带存在了，只是从来没人把它物化成表。本模块把它物化出来，
    **零 fuzzy、零名字参与**。

冲突消解规则（唯一需要判断的地方）：一个 esportsTeamId 会对上多个 oe_teamid，
实测 489 支里有 49 支（10.0%）。两个成因必须区别对待：

  (a) **真实改名换 id**。OE 在战队改名时会换 teamid。
      实测 103461966965149786（MKOI）：
        oe:team:6a0084da… "MAD Lions KOI"  84 局  2024-01-13 → 2024-10-07
        oe:team:4acc4065… "Movistar KOI"  194 局  2025-01-18 → 2026-09-18
      serving 要的是**当前在用的那个**，因为 team Elo 是按 oe_teamid 存的，
      改名后的新 id 才承载着近期战绩。
  (b) **join 噪声**。个别局被错链，表现为 n_games==1 的孤儿条目。

所以规则是：先按场次门槛滤掉噪声（n_games>=3 或占比>=50%），
再取 **last_day 最晚** 的那个作为 primary，同分用场次打破。

这条规则做过**时间留出验证**（把 2026-06-30 之后的观测完全藏起来建表，
再去预测藏起来的 5,288 个 (局, 队) 观测）：

    命中 4,776 / 5,288 = 90.32%
    错映     58 / 5,288 =  1.10%   ← 其中 54 个（93%）的正确 oe_teamid
                                     在截断日之前**从未出现过**（截断后才改的名），
                                     规则不可能知道；真正的规则错误 4 个 = 0.08%
    未登记  454 / 5,288 =  8.59%   ← 截断后才首次出现的新队，走降级路径
    对照：朴素「取场次最多」规则只有 88.0%，而且系统性地选中已废弃的旧 id

生产里对照表每天随 crosswalk 一起重建，陈旧度是 1 天而不是这里的 2.5 个月，
所以 8.59% 的「未登记」是本验证的上界，不是线上的真实值。

──────────────────────────────────────────────────────────────────────
整条解析链的实测精度（2026 dev 段、tier>=1、有 livestats 覆盖的 1,522 局）
──────────────────────────────────────────────────────────────────────
把 window 的 gameMetadata 重放一遍，和 silver 的同一场逐字段比
（复现脚本见本模块末尾 `replay_against_silver()`，实测 2026-09-19）：

    patch 解析          1522/1522 = 100.00%
    赛区                1522/1522 = 100.00%
    10 个英雄 key        1522/1522 = 100.00%
    2 个队伍 id          1519/1522 =  99.80%   （槽级 3044 个：错映 6 个 = 0.20%）
    10 个选手 id         1436/1522 =  94.35%   （槽级 15220 个：
                                                未命中→降级 100 个 = 0.66%
                                                张冠李戴        23 个 = 0.15%）
    进模型的 4 个特征     1448/1522 =  95.14%   逐位（1e-12）一致

**预测值的偏离幅度**（同一场，live 路径 vs silver 路径，两边都走 app.predict）：
74/1522 = 4.86% 的局有非零差异，其中 |Δp_blue| 中位 2.70 pp / p99 16.51 pp /
最大 31.26 pp；摊到全部 1522 局的平均是 0.16 pp。

**按成因拆开才看得懂这个分布**——三类错误的代价差一个数量级：

    成因              局数   |Δp| 均值   中位    最大
    全对              1435     0.00     0.00    0.00
    选手未命中→降级      80     2.28     2.56   10.15
    选手张冠李戴          4     3.09     0.65   11.06
    **队伍错/缺**         3    15.93     8.47   31.26   ← 单项最贵

也就是说：**一个选手解析失败值约 2-3 pp，一支队伍解析错值约 16 pp。**
这正是两者都必须在 UI 上标黄告警、而不是静默降级的原因；
也说明队伍对照表的 0.20% 槽级错误率比选手的 0.66% 更值得盯。
⚠️ 最坏的 5 局全是 LPL。LPL 的 `gameId` 对照本身就是模糊的（§3.3.5），
所以这几局里有多少是「队伍对照错」、有多少是「整场 join 错了」分不开，
这个 31.26 pp 应当读作**上界**。

⚠️ **这个口径只对「近期比赛」成立。** 拿今天的对照表去重放历史比赛，
队伍 id 整局一致率：2024 = 88.58% / 2025 = 90.48% / 2026 = 96.97%
（全 tier 口径，n = 6812 / 7755 / 4493）——因为对照表指向的是战队**当前**的
oe_teamid，而 2024 年那场比赛在 silver 里记的是当时的 id。
这不是 bug，是 as-of 语义：**serving 要当前 id，回放历史要用 `until=` 重建当时的表。**

──────────────────────────────────────────────────────────────────────
这些统计量要不要逐 fold 重估？（§6.1.2 红线 3）
──────────────────────────────────────────────────────────────────────
**不要，本模块全部豁免，理由不是「在线递归」而是「不是统计量」。**
红线 3 管的是"依赖训练段全局分布"的量（EB 收缩强度、标准化器、校准器），
它们把标签分布的信息带进特征。本模块产出的四张表全是**恒等映射
(identity resolution)**：它回答"feed 的这个 id 和 OE 的哪个 id 是同一个实体"，
**不携带任何胜负信息**，对任何一折的标签分布都不敏感。
把它逐 fold 重估既没有统计意义，也只会让小折的对照表更残缺。

但有一条**时间纪律仍然成立**，而且是另一回事：serving 时对照表必须用
「≤ 当前时刻」的观测重建，因为队伍会改名换 id（上面那个 1.10% 就是量化）。
这不是泄漏问题，是陈旧问题。

⚠️ **一个需要明说的判断**：`build_team_crosswalk()` 默认吃 observations 的全部
日期，**包含 holdout 段（2026-06-01 起）**。这是刻意的，理由有三条：
(1) observations 里没有任何胜负字段，它是纯身份对照；
(2) serving 要预测的就是"现在"这场比赛，那本来就在 holdout 之后，用不到当时的表
    反而是错的；
(3) 真要在 holdout 上回放 live 链路做评估，必须显式传 `until=<origin>`，
    这条有单测钉住（test_team_crosswalk_as_of_cut_changes_nothing_for_stable_teams）。
`build_league_feed_map()` / `validate_team_crosswalk()` 则**没有这个必要性**，
它们照常截在 holdout 之前。

──────────────────────────────────────────────────────────────────────
降级（§3.11.4 / §7.7.3）：覆盖率永远只做 gauge + 告警，绝不做启动 assert
──────────────────────────────────────────────────────────────────────
  选手未命中 → 退化到「该赛区 × 该位置的 player Elo 中位数」+ warning
  队伍未命中 → 退化到「该赛区 team Elo 中位数」+ warning
  英雄未命中 → champ_key=None，draft 层自动退回该位置基准 + warning
  赛区未登记 → UNKNOWN 桶 + warning
任何一条都不抛异常。转会期和新人首发**必然**把命中率打下来，
做成 assert 就是"转会期整站起不来"。
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

# ── 路径常量 ────────────────────────────────────────────────────────────
PLAYER_CROSSWALK = "data/crosswalk/player_crosswalk.parquet"
TEAM_CROSSWALK = "data/crosswalk/team_crosswalk.parquet"
OBSERVATIONS = "data/crosswalk/_cache/observations.parquet"
CHAMPION_MAP = "data/ddragon/champion_map.json"
LEAGUE_FEED_MAP = "data/crosswalk/league_feed_map.json"

# ── 常量表 ──────────────────────────────────────────────────────────────
# §3.11.2 表 2：feed 的 role → OE 的 position。常量 5 条，不从数据里学。
FEED_ROLE_TO_OE: dict[str, str] = {
    "top": "top",
    "jungle": "jng",
    "mid": "mid",
    "bottom": "bot",
    "support": "sup",
}
ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
UNKNOWN_LEAGUE = "UNKNOWN"

# 一线 + 国际赛的 feed 联赛名 → league_canon。**显式常量，不是 fuzzy 兜底。**
# 每一条都由 build_league_feed_map() 在全量数据上以 share=1.00（LCK 0.990 /
# LPL 0.987 除外，差额是 join 噪声）验证过；这里写死是因为 §6.1.3 要求
# 建表只用 holdout 之前的数据，而 EWC(7月) 与 KeSPA Cup(12月) 整个落在 holdout 段里，
# 数不出来。tier-1/2 的赛区归属绝不能因为"这几个月的样本被藏起来了"就掉进 UNKNOWN 桶。
FEED_LEAGUE_FALLBACK: dict[str, str] = {
    "LCK": "LCK", "LPL": "LPL", "LEC": "LEC", "LCS": "LCS",
    "LTA North": "LCS",          # 2025 = LTA N；2024/2026 = LCS（§3.11.5）
    "CBLOL": "CBLOL", "LTA South": "CBLOL",
    "LTA Cross-Conference": "LTA(2025-only)",
    "LCP": "LCP", "PCS": "PCS", "VCS": "VCS",
    "MSI": "MSI", "Worlds": "WLDs", "First Stand": "FST",
    "Esports World Cup": "EWC", "KeSPA Cup": "KeSPA Cup",
}

# 队伍对照表的冲突消解门槛，见模块 docstring
_TEAM_MIN_GAMES = 3
_TEAM_MIN_SHARE = 0.5
# 降级哨兵 id 的前缀。刻意用 OE 不可能产生的形状，一眼能在日志里认出来。
FALLBACK_TEAM_PREFIX = "fallback:team:league_median:"
FALLBACK_PLAYER_PREFIX = "fallback:player:league_pos_median:"

_PATCH_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.\d+)*$")


def _holdout_start() -> pd.Timestamp:
    """holdout 截止线。**serving 侧的任何建表也不许越过它**（§6.1.3）。"""
    from lolwp.features.loader import holdout_start

    return holdout_start()


# ════════════════════════════════════════════════════════════════════════
# 建表：队伍对照 + 赛区对照
# ════════════════════════════════════════════════════════════════════════
def load_player_map(path: str = PLAYER_CROSSWALK) -> dict[str, str]:
    """esportsPlayerId → oe_playerid，**只取 confidence=="primary"**。

    两个方向的性质完全不同，serving 用的是被忽略的那一个：

      OE → feed（silver 建表用的方向）：primary 过滤后是干净的一对一，
           3,413 → 3,413，`store/silver.py` 直接 assert 住。
      feed → OE（**本模块用的方向**）：**不是**函数，实测 3,413 条里有
           21 个 esportsPlayerId 对上两个 oe_playerid。成因不是映射错，
           是 **OE 自己给同一个人发过两个 playerid**——多半是大小写变体
           （`MihawK`/`Mihawk`、`FenRir`/`Fenrir`）或改 ID（`Chapters`→`KUMEW`）。

    serving 必须挑一个，规则与队伍表一致：**取 last_day 最晚的那个**，
    同分用 n_games 打破。理由是 player Elo 按 oe_playerid 存，
    近期那个 id 才承载着近期战绩；挑旧 id 等于把选手打回几年前的水平。
    """
    cw = pd.read_parquet(path)
    pri = cw[cw.confidence == "primary"].sort_values(
        ["esports_player_id", "last_day", "n_games"], ascending=[True, False, False]
    )
    win = pri.groupby("esports_player_id", as_index=False).head(1)
    return dict(zip(win.esports_player_id.astype(str), win.oe_playerid))


def build_team_crosswalk(
    obs_path: str = OBSERVATIONS,
    out_path: str | None = TEAM_CROSSWALK,
    *,
    until: str | None = None,
) -> pd.DataFrame:
    """从 observations 物化 esportsTeamId → oe_teamid 对照表。

    `until` 给定时只用 <= 该日期的观测——**时间留出验证与 as-of 重建都靠它**。
    不给则用全量（日常生产：每天重建一次，陈旧度 1 天）。

    输出列：esports_team_id / oe_teamid / n_games / share / n_candidates /
            first_day / last_day / confidence
    confidence == "primary" 表示唯一候选或占比 >= 50%；其余为 "contested"。
    **serving 两种都用**（contested 也比退化到赛区中位数强），但 contested 会告警。
    """
    obs = pd.read_parquet(
        obs_path, columns=["esports_team_id", "oe_teamid", "esports_game_id", "day"]
    )
    if until is not None:
        obs = obs[obs.day <= until]
    g = (
        obs.groupby(["esports_team_id", "oe_teamid"], as_index=False)
        .agg(
            n_games=("esports_game_id", "nunique"),
            first_day=("day", "min"),
            last_day=("day", "max"),
        )
    )
    tot = g.groupby("esports_team_id").n_games.sum().rename("_tot")
    g = g.join(tot, on="esports_team_id")
    g["share"] = g.n_games / g._tot
    # (b) 先滤掉 join 噪声：场次太少且占比也低的孤儿条目
    cand = g[(g.n_games >= _TEAM_MIN_GAMES) | (g.share >= _TEAM_MIN_SHARE)].copy()
    n_cand = cand.groupby("esports_team_id").size().rename("n_candidates")
    cand = cand.join(n_cand, on="esports_team_id")
    # (a) 再按「最近一次用的 oe_teamid」定胜负，同分用场次
    cand = cand.sort_values(
        ["esports_team_id", "last_day", "n_games"], ascending=[True, False, False]
    )
    win = cand.groupby("esports_team_id", as_index=False).head(1).copy()
    win["confidence"] = np.where(
        (win.n_candidates == 1) | (win.share >= _TEAM_MIN_SHARE), "primary", "contested"
    )
    win = win.drop(columns=["_tot"]).reset_index(drop=True)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        win.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
    return win


def build_league_feed_map(
    obs_path: str = OBSERVATIONS,
    silver_path: str = "data/silver/games.parquet",
    out_path: str | None = LEAGUE_FEED_MAP,
) -> dict[str, dict[str, Any]]:
    """feed 的 league name → silver 的 league_canon，同样从硬 join 的观测里数出来。

    不手写常量表的理由：feed 名与 OE code 的对应本身就在改
    （`LTA North` 2025 是 OE 的 `LTA N`、2026 又变回 `LCS`，§3.11.5），
    手写表每年都要跟着改一次，漏改就静默走进 UNKNOWN 桶。
    从数据里数则自动跟随。

    实测：一线赛区全部 >= 0.97 占比（LEC/LCS/CBLOL/PCS/VCS/LCP = 1.00，
    LCK 0.992，LPL 0.985，`LTA North`→LCS、`LTA South`→CBLOL 均 1.00）。
    只有两个二级联赛天然一名多 code（Circuito Desafiante 0.577 跨 CD/CBLOLA、
    Hitpoint Masters 0.713 跨 HM/HW），share<0.9 时解析仍给多数派但会告警。

    **只数 holdout 起始日之前的比赛。** 这张表不含任何胜负信息，理论上碰
    holdout 也不构成泄漏，但「唯一数据入口是 load_dev」这条纪律比省这点样本重要，
    而且 3 个半月的增量对一张联赛名对照表本来就没有影响。
    """
    obs = pd.read_parquet(obs_path, columns=["esports_game_id", "oe_gameid", "league_feed"])
    obs = obs.drop_duplicates()
    sv = pd.read_parquet(silver_path, columns=["game_id", "league_canon", "kickoff_utc"])
    sv = sv[sv.kickoff_utc < _holdout_start()].drop(columns=["kickoff_utc"])
    m = obs.merge(sv, left_on="oe_gameid", right_on="game_id")
    g = m.groupby(["league_feed", "league_canon"]).size().rename("n").reset_index()
    tot = g.groupby("league_feed").n.sum().rename("_tot")
    g = g.join(tot, on="league_feed")
    g["share"] = g.n / g._tot
    win = g.sort_values(["league_feed", "n"], ascending=[True, False]).groupby(
        "league_feed", as_index=False
    ).head(1)
    out = {
        str(r.league_feed): {"canon": str(r.league_canon), "share": float(r.share), "n": int(r.n)}
        for r in win.itertuples()
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, out_path)
    return out


def validate_team_crosswalk(
    xw: pd.DataFrame,
    teams_json: str = "data/crosswalk/_cache/teams.json",
    silver_path: str = "data/silver/games.parquet",
    player_crosswalk: str = PLAYER_CROSSWALK,
) -> dict[str, Any]:
    """**只读校验器，不参与建表**：拿 getTeams 名册投票复核队伍对照。

    路径完全独立于建表用的「同日+英雄多重集」join：
        getTeams 名册的 esportsPlayerId → 选手 crosswalk → oe_playerid
        → silver 里该选手最近一场所属的 oe_teamid → 投票
    多数票与对照表不一致的会被列出来人工看。

    ⚠️ 这个校验器**本身是有噪声的**：getTeams 的名册把二队/青训选手一起列在
    一队名下（实测 Gen.G 名册里 6 个人最近一场打的是 Gen.G Challengers、
    5 个人打的是 Gen.G，多数票就投给了二队）。所以它的不一致率
    （实测 46/343 = 13.4%）是**校验器的上界噪声**，不是对照表的错误率。
    对照表的真实错误率以模块 docstring 里的时间留出验证为准（0.08%）。
    """
    with open(teams_json) as fh:
        teams = json.load(fh)
    cw = pd.read_parquet(player_crosswalk)
    pri = cw[cw.confidence == "primary"]
    pmap = dict(zip(pri.esports_player_id.astype(str), pri.oe_playerid))
    sv = pd.read_parquet(
        silver_path,
        columns=["blue_team_id", "red_team_id", "blue_player_ids", "red_player_ids", "kickoff_utc"],
    )
    sv = sv[sv.kickoff_utc < _holdout_start()]
    rows: list[tuple[str, str, Any]] = []
    for side in ("blue", "red"):
        for tid, pids, ts in zip(sv[f"{side}_team_id"], sv[f"{side}_player_ids"], sv.kickoff_utc):
            rows.extend((p, tid, ts) for p in pids if p)
    pr = pd.DataFrame(rows, columns=["pid", "tid", "ts"]).sort_values("ts")
    last_team = pr.groupby("pid").tid.last().to_dict()

    tmap = dict(zip(xw.esports_team_id, xw.oe_teamid))
    agree = disagree = novote = 0
    bad: list[dict[str, Any]] = []
    for t in teams:
        eid = str(t.get("id"))
        if eid not in tmap:
            continue
        votes: dict[str, int] = {}
        for p in t.get("players") or []:
            oe = pmap.get(str(p.get("id")))
            tid = last_team.get(oe) if oe else None
            if tid:
                votes[tid] = votes.get(tid, 0) + 1
        if not votes:
            novote += 1
            continue
        top = max(votes.items(), key=lambda kv: kv[1])[0]
        if top == tmap[eid]:
            agree += 1
        else:
            disagree += 1
            bad.append({"esports_team_id": eid, "name": t.get("name"),
                        "mapped": tmap[eid], "roster_vote": top, "votes": votes})
    return {"agree": agree, "disagree": disagree, "no_vote": novote,
            "agree_rate": agree / max(agree + disagree, 1), "examples": bad[:20]}


# ════════════════════════════════════════════════════════════════════════
# 解析结果的数据结构
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ResolvedSide:
    """一方的解析结果。列表一律按 ROLES=(top,jng,mid,bot,sup) 排好序。

    排序这件事非做不可：feed 的 participantMetadata 是按 participantId 1..5 给的，
    silver 的数组是按 OE position 排的，两者**看起来**都是"上中野下辅"但没有
    任何东西保证它们一致（摇摆位、feed 侧偶发乱序）。
    train-serve skew golden test 里这是第一个会炸的地方。
    """
    esports_team_id: str | None
    oe_team_id: str | None
    team_confidence: str
    player_ids: list[str | None]
    champ_keys: list[int | None]
    roles: list[str]
    feed_player_ids: list[str | None]
    feed_champ_ids: list[str | None]
    summoner_names: list[str | None]


@dataclass
class ResolvedGame:
    esports_game_id: str | None
    esports_match_id: str | None
    league_canon: str
    patch: tuple[int, int] | None
    patch_raw: str | None
    blue: ResolvedSide
    red: ResolvedSide
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_players_resolved(self) -> int:
        return sum(
            1 for s in (self.blue, self.red) for p in s.player_ids
            if p and not p.startswith(FALLBACK_PLAYER_PREFIX)
        )

    @property
    def n_champs_resolved(self) -> int:
        return sum(1 for s in (self.blue, self.red) for c in s.champ_keys if c is not None)


# ════════════════════════════════════════════════════════════════════════
# Resolver
# ════════════════════════════════════════════════════════════════════════
class Resolver:
    """一次加载、反复使用。**不做任何网络 I/O**，纯查表。"""

    def __init__(
        self,
        player_map: dict[str, str],
        team_map: dict[str, tuple[str, str]],
        champ_map: dict[str, int],
        league_map: dict[str, dict[str, Any]],
        *,
        champ_key_to_id: dict[int, str] | None = None,
        ddragon_version: str = "",
    ) -> None:
        self.player_map = player_map
        self.team_map = team_map
        self.champ_map = champ_map
        self.league_map = league_map
        self.champ_key_to_id = champ_key_to_id or {}
        self.ddragon_version = ddragon_version

    # ── 构造 ────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        *,
        player_crosswalk: str = PLAYER_CROSSWALK,
        team_crosswalk: str = TEAM_CROSSWALK,
        champion_map: str = CHAMPION_MAP,
        league_feed_map: str = LEAGUE_FEED_MAP,
        build_missing: bool = True,
    ) -> Resolver:
        pmap = load_player_map(player_crosswalk)

        if not os.path.exists(team_crosswalk) and build_missing:
            build_team_crosswalk(out_path=team_crosswalk)
        txw = pd.read_parquet(team_crosswalk)
        tmap = {
            str(r.esports_team_id): (str(r.oe_teamid), str(r.confidence))
            for r in txw.itertuples()
        }

        with open(champion_map) as fh:
            cm = json.load(fh)
        cmap = {c["dd_id"]: int(c["dd_key"]) for c in cm["champions"]}
        ckey = {int(c["dd_key"]): c["dd_id"] for c in cm["champions"]}

        if not os.path.exists(league_feed_map) and build_missing:
            build_league_feed_map(out_path=league_feed_map)
        lmap: dict[str, dict[str, Any]] = {}
        if os.path.exists(league_feed_map):
            with open(league_feed_map) as fh:
                lmap = json.load(fh)
        return cls(pmap, tmap, cmap, lmap,
                   champ_key_to_id=ckey, ddragon_version=str(cm.get("version", "")))

    # ── 单项解析 ────────────────────────────────────────────────────────
    def player(self, esports_player_id: str | None) -> str | None:
        if not esports_player_id:
            return None
        return self.player_map.get(str(esports_player_id))

    def team(self, esports_team_id: str | None) -> tuple[str | None, str]:
        """返回 (oe_teamid, confidence)。未命中返回 (None, "unmapped")。"""
        if not esports_team_id:
            return None, "unmapped"
        hit = self.team_map.get(str(esports_team_id))
        return hit if hit else (None, "unmapped")

    def champion(self, champion_id: str | None) -> int | None:
        """DDragon 字符串 id → 数字 key。`"KSante"` → 897。

        feed 的 championId 就是 dd_id，不需要任何规范化；新英雄上线时
        champion_map 还没更新就会返回 None，由 draft 层退回该位置基准。
        """
        if not champion_id:
            return None
        return self.champ_map.get(str(champion_id))

    def role(self, feed_role: str | None) -> str | None:
        if not feed_role:
            return None
        return FEED_ROLE_TO_OE.get(str(feed_role).lower())

    def league(self, feed_league_name: str | None) -> tuple[str, float]:
        """feed 的联赛名 → silver 的 league_canon，附带该映射的占比（可信度）。"""
        if not feed_league_name:
            return UNKNOWN_LEAGUE, 0.0
        name = str(feed_league_name)
        hit = self.league_map.get(name)
        if not hit:
            # 数据里没数出来（例如整个赛事都落在 holdout 段）→ 查显式常量表
            canon = FEED_LEAGUE_FALLBACK.get(name)
            return (canon, 1.0) if canon else (UNKNOWN_LEAGUE, 0.0)
        return str(hit["canon"]), float(hit.get("share", 1.0))

    # ── 主入口 ──────────────────────────────────────────────────────────
    def resolve_window(
        self, payload: dict[str, Any], *, feed_league_name: str | None = None
    ) -> ResolvedGame:
        """从 window 的 payload 解析出模型输入。**只读 gameMetadata，绝不碰 frames。**

        frames 是赛中数据（金币/击杀/时间），进 pre-game 模型就是泄漏（§3.3.2）。
        """
        gm = payload.get("gameMetadata") or {}
        patch_raw = gm.get("patchVersion")
        warns: list[str] = []

        canon, share = self.league(feed_league_name)
        if feed_league_name and canon == UNKNOWN_LEAGUE:
            warns.append(f"赛区「{feed_league_name}」未登记，已归入 UNKNOWN 桶，蓝方先验退化为全局值")
        elif share and share < 0.9:
            warns.append(
                f"赛区「{feed_league_name}」在历史数据里对应多个 OE code"
                f"（多数派 {canon} 占 {share:.0%}），可能归错赛区"
            )

        sides: dict[str, ResolvedSide] = {}
        for feed_key, side in (("blueTeamMetadata", "blue"), ("redTeamMetadata", "red")):
            sides[side] = self._resolve_side(gm.get(feed_key) or {}, side, warns)

        patch = norm_feed_patch(patch_raw)
        if patch_raw and patch is None:
            warns.append(f"patch 版本号「{patch_raw}」无法解析")

        rg = ResolvedGame(
            esports_game_id=_s(payload.get("esportsGameId")),
            esports_match_id=_s(payload.get("esportsMatchId")),
            league_canon=canon,
            patch=patch,
            patch_raw=_s(patch_raw),
            blue=sides["blue"],
            red=sides["red"],
            warnings=warns,
        )
        rg.stats = {
            "n_players_resolved": rg.n_players_resolved,
            "n_players_total": 10,
            "player_resolve_rate": rg.n_players_resolved / 10.0,
            "n_champs_resolved": rg.n_champs_resolved,
            "n_teams_resolved": sum(
                1 for s in (rg.blue, rg.red) if s.oe_team_id
                and not s.oe_team_id.startswith(FALLBACK_TEAM_PREFIX)
            ),
            "league_canon": canon,
            "league_share": share,
        }
        return rg

    def _resolve_side(
        self, meta: dict[str, Any], side: str, warns: list[str]
    ) -> ResolvedSide:
        cn = "蓝方" if side == "blue" else "红方"
        eteam = _s(meta.get("esportsTeamId"))
        oe_team, conf = self.team(eteam)
        if oe_team is None:
            warns.append(f"{cn}战队 esportsTeamId={eteam} 不在对照表中，实力退化为该赛区中位数")
        elif conf == "contested":
            warns.append(f"{cn}战队 {eteam} → {oe_team} 的对照有多个候选，可能选错")

        parts = list(meta.get("participantMetadata") or [])
        # 按 OE position 排序 —— 不是按 participantId。见 ResolvedSide 的 docstring。
        by_pos: dict[str, dict[str, Any]] = {}
        for p in parts:
            pos = self.role(p.get("role"))
            if pos is None:
                warns.append(f"{cn}出现未知 role「{p.get('role')}」，该选手被跳过")
                continue
            if pos in by_pos:
                warns.append(f"{cn}位置 {pos} 出现两次，后一个被丢弃")
                continue
            by_pos[pos] = p

        pids: list[str | None] = []
        keys: list[int | None] = []
        feed_pids: list[str | None] = []
        feed_cids: list[str | None] = []
        names: list[str | None] = []
        for pos in ROLES:
            p = by_pos.get(pos)
            if p is None:
                warns.append(f"{cn}缺少 {pos} 位选手")
                pids.append(None)
                keys.append(None)
                feed_pids.append(None)
                feed_cids.append(None)
                names.append(None)
                continue
            epid = _s(p.get("esportsPlayerId"))
            oe_pid = self.player(epid)
            if oe_pid is None:
                warns.append(
                    f"{cn} {pos}「{p.get('summonerName')}」(id={epid}) 不在 primary "
                    "对照表中，实力退化为该赛区该位置中位数"
                )
            cid = _s(p.get("championId"))
            ck = self.champion(cid)
            if cid and ck is None:
                warns.append(f"{cn} {pos} 的英雄「{cid}」不在 champion_map 中（新英雄？）")
            pids.append(oe_pid)
            keys.append(ck)
            feed_pids.append(epid)
            feed_cids.append(cid)
            names.append(_s(p.get("summonerName")))

        return ResolvedSide(
            esports_team_id=eteam, oe_team_id=oe_team, team_confidence=conf,
            player_ids=pids, champ_keys=keys, roles=list(ROLES),
            feed_player_ids=feed_pids, feed_champ_ids=feed_cids, summoner_names=names,
        )


# ════════════════════════════════════════════════════════════════════════
# 降级：把「赛区中位数」物化成哨兵实体，注入 artifact 状态
# ════════════════════════════════════════════════════════════════════════
def install_fallback_ratings(
    prior_state: dict[str, Any],
    games: pd.DataFrame,
    *,
    lookback_days: int = 365,
) -> dict[str, Any]:
    """在 artifact 的状态快照里**就地**补上降级用的哨兵评分（§3.11.4 第 3 条）。

    为什么用哨兵 id 而不是改 features_from_state：
    `features/priors.py` 是训练侧的公式，train-serve 必须逐字同一套。
    在推理侧给它塞一个"名字叫『LEC 中位队』、评分等于 LEC 队伍 Elo 中位数"的
    虚拟实体，比在公式里加 if-else 安全得多——公式一个字没动。

    为什么是中位数不是均值：§3.11.4 明写「退化到该赛区 × 该位置的 rating 中位数」。
    中位数对赛区里个别暴涨/暴跌的队伍稳健。

    `games` 必须来自 `features.loader.load_dev()`——**绝不能直接读 silver 全量**，
    那会把 holdout 段的阵容信息带进 serving 路径。
    """
    te: dict[str, float] = prior_state["team_elo"]
    pe: dict[str, float] = prior_state["player_elo"]
    init = float(prior_state["elo_cfg"]["init"])
    if len(games):
        cutoff = games.kickoff_utc.max() - pd.Timedelta(days=lookback_days)
        recent = games[games.kickoff_utc >= cutoff]
    else:
        recent = games

    # 队伍：赛区内出现过的队伍的当前 Elo 中位数
    by_lg_team: dict[str, list[float]] = {}
    for lg, bt, rt in zip(recent.league_canon, recent.blue_team_id, recent.red_team_id):
        for t in (bt, rt):
            r = te.get(str(t))
            if r is not None:
                by_lg_team.setdefault(str(lg), []).append(r)
    for lg, vals in by_lg_team.items():
        te[f"{FALLBACK_TEAM_PREFIX}{lg}"] = float(np.median(vals))
    te[f"{FALLBACK_TEAM_PREFIX}{UNKNOWN_LEAGUE}"] = init

    # 选手：赛区 × 位置的当前 Elo 中位数
    by_lg_pos: dict[tuple[str, str], list[float]] = {}
    for lg, bp, rp in zip(recent.league_canon, recent.blue_player_ids, recent.red_player_ids):
        for arr in (bp, rp):
            for pos, p in zip(ROLES, list(arr)):
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                r = pe.get(str(p))
                if r is not None:
                    by_lg_pos.setdefault((str(lg), pos), []).append(r)
    for (lg, pos), vals in by_lg_pos.items():
        pe[f"{FALLBACK_PLAYER_PREFIX}{lg}:{pos}"] = float(np.median(vals))
    for pos in ROLES:
        pe.setdefault(f"{FALLBACK_PLAYER_PREFIX}{UNKNOWN_LEAGUE}:{pos}", init)

    prior_state.setdefault("_fallbacks_installed", True)
    return prior_state


def _fallback_team(league_canon: str, prior_state: dict[str, Any]) -> str:
    key = f"{FALLBACK_TEAM_PREFIX}{league_canon}"
    if key in prior_state.get("team_elo", {}):
        return key
    return f"{FALLBACK_TEAM_PREFIX}{UNKNOWN_LEAGUE}"


def _fallback_player(league_canon: str, pos: str, prior_state: dict[str, Any]) -> str:
    key = f"{FALLBACK_PLAYER_PREFIX}{league_canon}:{pos}"
    if key in prior_state.get("player_elo", {}):
        return key
    return f"{FALLBACK_PLAYER_PREFIX}{UNKNOWN_LEAGUE}:{pos}"


def to_draft(
    rg: ResolvedGame,
    prior_state: dict[str, Any] | None = None,
    *,
    now: pd.Timestamp | None = None,
):
    """ResolvedGame → app.predict.Draft。未命中的实体在这里换成哨兵 id。

    prior_state 给定（且已 install_fallback_ratings）时走赛区中位数降级；
    不给则保留 None，让 features_from_state 退回全局 init（1500）。
    """
    from lolwp.app.predict import Draft

    def ids(side: ResolvedSide) -> tuple[str, list[str | None]]:
        tid = side.oe_team_id
        if tid is None:
            tid = (
                _fallback_team(rg.league_canon, prior_state)
                if prior_state is not None
                else f"{FALLBACK_TEAM_PREFIX}{rg.league_canon}"
            )
        pl: list[str | None] = []
        for pos, p in zip(ROLES, side.player_ids):
            if p is None and prior_state is not None:
                p = _fallback_player(rg.league_canon, pos, prior_state)
            pl.append(p)
        return tid, pl

    bt, bp = ids(rg.blue)
    rt, rp = ids(rg.red)
    # Draft 的标注是 list[str] / list[int]，但**实际契约允许 None**：
    # silver 自己就存着 None 的选手/英雄槽（2026 约 675+380 场缺 id），
    # build_features / draft_features_from_state 都显式处理了 None。
    # 这里 cast 而不是填一个假值——填假值会静默算错，None 会走已验证的降级分支。
    return Draft(
        league=rg.league_canon,
        blue_team_id=bt,
        red_team_id=rt,
        blue_player_ids=cast("list[str]", bp),
        red_player_ids=cast("list[str]", rp),
        blue_champ_keys=cast("list[int]", list(rg.blue.champ_keys)),
        red_champ_keys=cast("list[int]", list(rg.red.champ_keys)),
        now=now,
    )


# ════════════════════════════════════════════════════════════════════════
# 小工具
# ════════════════════════════════════════════════════════════════════════
def norm_feed_patch(patch_version: str | None) -> tuple[int, int] | None:
    """feed 的四段式 patchVersion → (major, minor)。`"16.17.810.4348"` → (16, 17)。

    **入参必须是字符串。** store/patches.py 已经证过这个坑：JSON number 会把
    `16.10` 解析成 float 16.1，再 str() 回来就成了 patch 16.01，比真实值早 9 个版本。
    feed 的 patchVersion 本身就是字符串，这里只做防御性断言。
    """
    if patch_version is None:
        return None
    if isinstance(patch_version, float):
        raise TypeError("patchVersion 必须是字符串；float 会丢掉零填充（16.10 → 16.1）")
    m = _PATCH_RE.match(str(patch_version).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _s(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x)
    return s if s else None


# ════════════════════════════════════════════════════════════════════════
# 复现：本模块 docstring 里那组精度数字的生成脚本
# ════════════════════════════════════════════════════════════════════════
def window_payload_from_cache(rec: dict[str, Any]) -> dict[str, Any]:
    """把 `_cache/windows.jsonl` 的一条压缩记录还原成 window 端点的 payload 形状。

    缓存是阶段 2 建 crosswalk 时顺手落的，字段已经摊平；这里还原成
    `gameMetadata` 嵌套结构，**为的是让重放走与线上完全同一个 resolve_window**。
    自己另写一套解析去对照，对照的就不是生产代码了。
    """
    sides: dict[str, list[dict[str, Any]]] = {"blue": [], "red": []}
    for p in rec["participants"]:
        sides[p["side"]].append({
            "participantId": p["pid"], "esportsPlayerId": p["player_id"],
            "summonerName": p["name"], "championId": p["champ"], "role": p["role"],
        })
    return {
        "esportsGameId": rec["game_id"],
        "esportsMatchId": rec.get("match_id"),
        "gameMetadata": {
            "patchVersion": rec["patch"],
            "blueTeamMetadata": {"esportsTeamId": rec["blue_team_id"],
                                 "participantMetadata": sides["blue"]},
            "redTeamMetadata": {"esportsTeamId": rec["red_team_id"],
                                "participantMetadata": sides["red"]},
        },
    }


def replay_against_silver(
    games: pd.DataFrame,
    *,
    windows_path: str = "data/crosswalk/_cache/windows.jsonl",
    game_link_path: str = "data/crosswalk/_cache/game_link.parquet",
    resolver: Resolver | None = None,
) -> pd.DataFrame:
    """把历史 window 重放一遍，与 silver 同一场逐字段比，返回逐局明细。

    这是 **train-serve skew 的全量版本**：tests/test_live.py 里那三条 golden test
    只钉住 3 场（LEC / LPL / LCK 各一），这里跑全部有 livestats 覆盖的场次，
    用来产出本模块 docstring 里那张精度表。放在生产模块而不是 scratchpad 里，
    是因为**那组数字一旦写进 docstring 就必须可复现**。

    `games` 必须是 `load_dev()` 的输出（或它的子集）——**绝不能传 silver 全量**，
    否则会把 holdout 段的比赛拉进来重放。

    返回列：oe_gameid / esports_game_id / year / league_canon /
            patch_ok / league_ok / champ_ok / team_ok / player_ok /
            n_champ_ok / n_team_ok / n_player_ok / n_player_none / n_player_wrong
    """
    res = resolver or Resolver.load()
    link = pd.read_parquet(game_link_path, columns=["esports_game_id", "oe_gameid"])
    l2o = dict(zip(link.esports_game_id.astype(str), link.oe_gameid))
    sv = games.drop_duplicates("game_id").set_index("game_id")

    def _norm(x: Any) -> Any:
        """silver 的缺失可能是 None 也可能是 NaN，统一成 None 再比。"""
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return x

    rows: list[dict[str, Any]] = []
    with open(windows_path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            oe = l2o.get(str(rec["game_id"]))
            if oe is None or oe not in sv.index:
                continue
            s = sv.loc[oe]
            rg = res.resolve_window(window_payload_from_cache(rec),
                                    feed_league_name=rec.get("league_name"))
            sv_p = [_norm(x) for x in list(s.blue_player_ids) + list(s.red_player_ids)]
            lv_p = list(rg.blue.player_ids) + list(rg.red.player_ids)
            sv_c = [None if _norm(x) is None else int(x)
                    for x in list(s.blue_champ_keys) + list(s.red_champ_keys)]
            lv_c = list(rg.blue.champ_keys) + list(rg.red.champ_keys)
            n_p = sum(1 for a, b in zip(sv_p, lv_p) if a == b)
            n_c = sum(1 for a, b in zip(sv_c, lv_c) if a == b)
            n_t = (int(s.blue_team_id == rg.blue.oe_team_id)
                   + int(s.red_team_id == rg.red.oe_team_id))
            rows.append({
                "oe_gameid": oe, "esports_game_id": str(rec["game_id"]),
                "year": int(pd.Timestamp(s.kickoff_utc).year),
                "league_canon": s.league_canon,
                "patch_ok": (int(s.patch_major), int(s.patch_minor)) == (rg.patch or (-1, -1)),
                "league_ok": bool(s.league_canon == rg.league_canon),
                "champ_ok": n_c == 10, "team_ok": n_t == 2, "player_ok": n_p == 10,
                "n_champ_ok": n_c, "n_team_ok": n_t, "n_player_ok": n_p,
                "n_player_none": sum(1 for b in lv_p if b is None),
                "n_player_wrong": sum(1 for a, b in zip(sv_p, lv_p) if b is not None and a != b),
            })
    return pd.DataFrame(rows)
