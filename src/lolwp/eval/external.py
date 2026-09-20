"""外部基准对标：Polymarket 市场共识概率 vs 本项目模型（PLAN.md §3.7 / §6.4 验收④）。

**为什么要做这一项**：内部阶梯（L0…BT4）只能回答"加这个特征有没有用"，
回答不了"这个模型到底算不算好"。市场共识是唯一一个**外生的、带真金白银的**参照：
它聚合了公开信息 + 私有信息（伤病、替补、训练赛口碑），是实践上的信息效率上界。
若模型显著赢过市场 0.015 Brier 以上，**第一反应必须是查泄漏**（手册 §3.7 判读规则），
因为 arXiv 2609.08060 在 928 张地图上的结论是市场仅微弱领先（ΔBrier=+0.0093）。

**为什么不能拿 Polymarket 当全样本 baseline**：它的 LoL 市场最早只到 2025-09，
而本项目评测窗口从 2019 起。所以它只是"近期子集上的参照上界"，
在它上面比较时必须把所有模型都在同一交集子集重算一遍。

三层结构（每层都能独立落盘、独立复现）：

1. `fetch_events` / `parse_markets`：Gamma API → 事件 → 逐张地图的市场行；
2. `match_markets`：市场行 ↔ silver `game_id` 对齐（队名归一 + 日期窗口 + 局序）；
3. `pregame_probs` / `devig`：CLOB 历史成交价 → 开赛前最后一笔 → 去 vig。

**无鉴权**，但必须带 User-Agent（裸 urllib 默认 UA 会 403，§3.7 [实测]）。
"""
from __future__ import annotations

import http.client
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Polymarket 的 LoL 市场覆盖起点（§3.7 实测：最早 endDate 2025-09-15）
PM_FIRST_DAY = "2025-09-01"


def _get_json(url: str, *, timeout: int = 60, retries: int = 4) -> Any:
    """带 UA 的 GET。gamma 偶发 5xx/429，退避重试；422 视为"参数越界"直接抛。"""
    last: Exception | None = None
    for k in range(retries):
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 422:
                raise
            last = exc
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            last = exc
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"GET 失败 {url}: {last}")


# ----------------------------------------------------------------------------
# 1. 抓取
# ----------------------------------------------------------------------------
def fetch_events(
    start: str = PM_FIRST_DAY,
    end: str = "2026-06-15",
    out_path: str = "data/external/pm_events.jsonl",
    *,
    tag: str = "league-of-legends",
    window_days: int = 7,
) -> int:
    """按 endDate 的周窗口分页抓取 LoL 相关 event，逐行写 jsonl。

    **为什么按周窗口而不是 offset 翻页**：gamma 的 offset 在 ~2100 处返回 422，
    直接 offset 翻页会静默丢掉最密集的那几个月（实测 2026-03 只翻到 187 个事件，
    而相邻的 2026-04 有 532 个）——这种截断不报错，会伪装成"那个月没比赛"。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen: set[str] = set()
    n = 0
    bounds = pd.date_range(start, end, freq=f"{window_days}D", tz="UTC")
    with open(out_path, "w") as fh:
        for lo, hi in itertools.pairwise(bounds):
            for closed in ("true", "false"):
                off = 0
                while True:
                    url = (
                        f"{GAMMA}/events?tag_slug={tag}&closed={closed}"
                        f"&limit=100&offset={off}"
                        f"&end_date_min={lo.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                        f"&end_date_max={hi.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    )
                    page = _get_json(url)
                    if not page:
                        break
                    for ev in page:
                        if ev["id"] in seen:
                            continue
                        seen.add(ev["id"])
                        fh.write(json.dumps(ev) + "\n")
                        n += 1
                    fh.flush()
                    if len(page) < 100:
                        break
                    off += 100
            print(f"[fetch_events] {lo.date()}..{hi.date()} 累计 {n}", flush=True)
    return n


# ----------------------------------------------------------------------------
# 2. 解析
# ----------------------------------------------------------------------------
_GAME_RE = re.compile(r"-game(\d+)$")


def _as_list(x: Any) -> list:
    """gamma 把 outcomes / prices / clobTokenIds 编码成 JSON 字符串，不是数组。"""
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return []
    return []


def parse_markets(jsonl_path: str = "data/external/pm_events.jsonl") -> pd.DataFrame:
    """把 event jsonl 摊平成"一张市场一行"，只保留二元（两队）市场。

    `game_no=0` 表示整轮 BO 的赛果市场（系列赛胜者），`>=1` 是逐张地图。
    """
    rows: list[dict[str, Any]] = []
    with open(jsonl_path) as fh:
        for line in fh:
            ev = json.loads(line)
            if not str(ev.get("slug", "")).startswith("lol-"):
                continue
            for mk in ev.get("markets", []) or []:
                outcomes = _as_list(mk.get("outcomes"))
                prices = _as_list(mk.get("outcomePrices"))
                tokens = _as_list(mk.get("clobTokenIds"))
                if len(outcomes) != 2 or len(tokens) != 2:
                    continue
                slug = mk.get("slug") or ""
                m = _GAME_RE.search(slug)
                gst = mk.get("gameStartTime") or mk.get("startDate") or ev.get("startDate")
                rows.append(
                    {
                        "event_slug": ev.get("slug"),
                        "market_slug": slug,
                        "question": mk.get("question"),
                        "game_no": int(m.group(1)) if m else 0,
                        "team_a": outcomes[0],
                        "team_b": outcomes[1],
                        "price_a_final": float(prices[0]) if len(prices) == 2 else np.nan,
                        "token_a": tokens[0],
                        "token_b": tokens[1],
                        "game_start": pd.to_datetime(gst, utc=True, errors="coerce"),
                        "end_date": pd.to_datetime(ev.get("endDate"), utc=True, errors="coerce"),
                        "volume": float(mk.get("volumeNum") or 0.0),
                        "liquidity": float(mk.get("liquidityNum") or 0.0),
                        "closed": bool(mk.get("closed")),
                        "resolution": mk.get("umaResolutionStatus"),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. 队名归一（Polymarket 用广播简称，OE 用注册全名）
# ----------------------------------------------------------------------------
_SUFFIX = re.compile(
    r"\b(esports?|e-sports?|gaming|game|club|team|academy|pro|kr|cn|eu|na)\b", re.IGNORECASE
)
_NONWORD = re.compile(r"[^a-z0-9]+")

# 只列 Polymarket 缩写 → OE 全名的确定性映射。**宁可漏配也不要错配**：
# 一条错配会把别人的比赛当成自己的，静默污染整张对比表。
ALIAS: dict[str, str] = {
    "t1": "t1", "geng": "geng", "gen g": "geng", "hle": "hanwha life esports",
    "kt": "kt rolster", "drx": "drx", "ns": "nongshim redforce",
    "dk": "dplus kia", "dplus": "dplus kia", "bfx": "bnk fearx", "bnk": "bnk fearx",
    "dnf": "dn freecs", "dnfreecs": "dn freecs",
    "blg": "bilibili gaming", "tes": "top esports", "jdg": "jd gaming",
    "lng": "lng esports", "wbg": "weibo gaming", "ig": "invictus gaming",
    "al": "anyones legend", "fpx": "funplus phoenix", "edg": "edward gaming",
    "omg": "oh my god", "we": "team we", "tt": "thunder talk gaming",
    "rng": "royal never give up", "lgd": "lgd gaming", "ra": "rare atom",
    "ttg": "thunder talk gaming", "up": "ultra prime", "nip": "ninjas in pyjamas",
    "g2": "g2 esports", "fnc": "fnatic", "fnatic": "fnatic", "mkoi": "movistar koi",
    "koi": "movistar koi", "th": "team heretics", "vit": "team vitality",
    "sk": "sk gaming", "bds": "team bds", "kc": "karmine corp", "gx": "giantx",
    "rge": "rogue", "tl": "team liquid", "c9": "cloud9", "flyquest": "flyquest",
    "fly": "flyquest", "100t": "100 thieves", "nrg": "nrg", "dig": "dignitas",
    "sr": "shopify rebellion", "lyon": "lyon", "dsg": "disguised",
    "psg": "psg talon", "ctbc": "ctbc flying oyster", "cfo": "ctbc flying oyster",
    "gam": "gam esports", "vks": "vivo keyd stars", "loud": "loud",
    "pain": "pain gaming", "png": "pain gaming", "furia": "furia",
    "isg": "isurus", "est": "estral esports", "r7": "movistar r7",
    "ses": "secret whales", "tsw": "team secret whales",
    "hw": "hanwha life esports",
}


def norm_team(name: str) -> str:
    """队名归一：小写 → 去后缀词 → 去非字母数字 → 查别名表。

    先做一次"缩写直查"再做去后缀，否则 `SK`（SK Gaming）会被后缀正则吃成空串。
    """
    raw = (name or "").strip().lower()
    key = _NONWORD.sub(" ", raw).strip()
    if key in ALIAS:
        return _NONWORD.sub("", ALIAS[key])
    stripped = _NONWORD.sub(" ", _SUFFIX.sub(" ", key)).strip()
    if stripped in ALIAS:
        return _NONWORD.sub("", ALIAS[stripped])
    out = _NONWORD.sub("", stripped)
    return out or _NONWORD.sub("", key)


# ----------------------------------------------------------------------------
# 4. 与 silver 对齐
# ----------------------------------------------------------------------------
@dataclass
class MatchConfig:
    """对齐规则。`day_slack` 放宽是因为市场用当地开赛日、OE 用 UTC 结束时刻。"""

    day_slack: int = 1
    min_volume: float = 0.0
    kinds: tuple[int, ...] = field(default_factory=lambda: (1, 2, 3, 4, 5))


def match_markets(
    mk: pd.DataFrame, dev: pd.DataFrame, cfg: MatchConfig | None = None
) -> pd.DataFrame:
    """把逐地图市场配到 silver 的 `game_id` 上。

    配对键 = （归一后的无序队名对, 日期±slack, 局序 game_no）。
    **一对多必须丢弃**：同一天同两队打两轮 BO（极少见但存在），
    无法判断哪张地图对应哪个市场时宁可不要这条样本。
    """
    cfg = cfg or MatchConfig()
    d = dev.copy()
    d["na"] = d["blue_team_name"].map(norm_team)
    d["nb"] = d["red_team_name"].map(norm_team)
    d["pair"] = [tuple(sorted(x)) for x in zip(d.na, d.nb, strict=False)]
    d["day"] = d["kickoff_utc"].dt.floor("D")

    m = mk[mk.game_no.isin(cfg.kinds)].copy()
    m = m[m.volume >= cfg.min_volume]
    m["na"] = m["team_a"].map(norm_team)
    m["nb"] = m["team_b"].map(norm_team)
    m["pair"] = [tuple(sorted(x)) for x in zip(m.na, m.nb, strict=False)]
    m["day"] = m["game_start"].dt.floor("D")
    m = m[m.na != m.nb]

    out = []
    for off in range(-cfg.day_slack, cfg.day_slack + 1):
        mm = m.copy()
        mm["day"] = mm["day"] + pd.Timedelta(days=off)
        j = d.merge(
            mm,
            left_on=["pair", "day", "game_in_series"],
            right_on=["pair", "day", "game_no"],
            how="inner",
            suffixes=("", "_pm"),
        )
        j["day_offset"] = off
        out.append(j)
    if not out:
        return pd.DataFrame()
    j = pd.concat(out, ignore_index=True)
    # 同一 game_id 可能通过多个 offset 命中同一市场；保留 |offset| 最小的那条
    j["absoff"] = j["day_offset"].abs()
    j = j.sort_values(["game_id", "absoff"]).drop_duplicates("game_id", keep="first")
    # 一个市场只能对应一个 game_id；否则整组丢弃
    dup = j.market_slug.value_counts()
    j = j[j.market_slug.map(dup).eq(1)]
    return j.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 5. 开赛前价格 + 去 vig
# ----------------------------------------------------------------------------
def fetch_price_history(
    token: str, start_ts: int, end_ts: int, *, fidelity: int = 1
) -> list[dict[str, float]]:
    """CLOB 成交价历史，fidelity 单位是分钟。

    **必须显式传 startTs/endTs**：手册 §3.7 写的 `interval=max` 对已结算的历史市场
    一律返回 `{"history": []}` 且 HTTP 200（实测 2026-09-19）——
    这是个**静默失败**，照抄手册会得到"Polymarket 没有历史价格"的错误结论。
    """
    url = (
        f"{CLOB}/prices-history?market={token}"
        f"&startTs={int(start_ts)}&endTs={int(end_ts)}&fidelity={fidelity}"
    )
    js = _get_json(url)
    return js.get("history", []) if isinstance(js, dict) else []


def last_price_before(hist: list[dict[str, float]], ts: pd.Timestamp) -> tuple[float, int]:
    """开赛前最后一笔价格；返回 (price, 距开赛的分钟数)。没有则 (nan, -1)。

    **必须严格 `<` 开赛时刻**：等号会把开赛后第一笔吃进来，
    那时候已经有比赛内信息了，属于教科书级的前视泄漏。
    """
    cut = ts.timestamp()
    best_t, best_p = None, np.nan
    for pt in hist:
        t = float(pt["t"])
        if t < cut and (best_t is None or t > best_t):
            best_t, best_p = t, float(pt["p"])
    if best_t is None:
        return float("nan"), -1
    return best_p, int((cut - best_t) / 60)


def devig(p_a: float, p_b: float) -> float:
    """两边隐含概率归一化。Polymarket 的两个 token 价格不保证和为 1。

    价差来自做市价差与手续费（vig）。不归一化的话，
    `sum > 1` 会让 Brier 系统性偏高，看上去像"市场更差"。
    """
    if not np.isfinite(p_a):
        return float("nan")
    if not np.isfinite(p_b):
        return p_a
    s = p_a + p_b
    if s <= 0:
        return float("nan")
    return p_a / s


# CLOB 的 prices-history 对**时间跨度**有硬上限：实测 7 天 + fidelity=1 可以，
# 21 天无论 fidelity 多粗都 400。所以窗口只能开 7 天。
PRICE_WINDOW_DAYS = 7
# 屏蔽 kickoff 所在的那一分钟桶：fidelity=1 的桶以分钟起点打标，
# 桶内可能含 kickoff 之后几十秒的成交。宁可少 1 分钟信息也不碰前视。
GUARD_SEC = 60


def pregame_probs(
    matched: pd.DataFrame,
    out_path: str = "data/external/pm_prices.parquet",
    *,
    guard_sec: int = GUARD_SEC,
    flush_every: int = 50,
) -> pd.DataFrame:
    """对每个已配对的 game 抓两个 token 的分钟级历史，取开赛前最后一笔并去 vig。

    每 `flush_every` 行落一次盘：这一步要几千个 HTTP 请求，
    中途断网就丢全部进度是不可接受的。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done: set[str] = set()
    if os.path.exists(out_path):
        prev = pd.read_parquet(out_path)
        done = set(prev.game_id)
    else:
        prev = pd.DataFrame()

    rows: list[dict[str, Any]] = []
    todo = matched[~matched.game_id.isin(done)]
    for i, r in enumerate(todo.itertuples(index=False), 1):
        k = r.kickoff_utc
        lo = int((k - pd.Timedelta(days=PRICE_WINDOW_DAYS)).timestamp())
        hi = int((k + pd.Timedelta(hours=2)).timestamp())
        cut = k - pd.Timedelta(seconds=guard_sec)
        rec: dict[str, Any] = {"game_id": r.game_id, "market_slug": r.market_slug}
        for side, tok in (("a", r.token_a), ("b", r.token_b)):
            try:
                hist = fetch_price_history(str(tok), lo, hi)
            except (RuntimeError, urllib.error.HTTPError, http.client.HTTPException):
                hist = []
            p, age = last_price_before(hist, cut)
            rec[f"p_{side}"] = p
            rec[f"age_{side}_min"] = age
            rec[f"n_pts_{side}"] = len(hist)
            # 系列赛开赛前（不含本系列已打完的地图信息）的对照口径
            p0, age0 = last_price_before(hist, r.game_start - pd.Timedelta(seconds=guard_sec))
            rec[f"p0_{side}"] = p0
            rec[f"age0_{side}_min"] = age0
        rows.append(rec)
        if i % flush_every == 0:
            cur = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
            cur.to_parquet(out_path, index=False)
            print(f"[pregame_probs] {i}/{len(todo)} → {out_path}", flush=True)
    cur = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
    cur.to_parquet(out_path, index=False)
    print(f"[pregame_probs] 完成 {len(cur)} 行", flush=True)
    return cur
