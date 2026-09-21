"""被搁置假设的重测台账（annual re-test backlog）。

**为什么需要它**：本项目有一批假设是「现在测不出来」而不是「已被证伪」。
两者的处理方式完全不同——前者应当随数据积累定期复查，后者不必再碰。

实测的例子：
  · 英雄克制的 split-half 可靠性学习曲线**还在爬**（12k→0.087, 24k→0.119,
    48k→0.151, 96k→0.180），对数外推需约 15 万场职业比赛才越过 0.20 的门槛，
    现有 10.2 万场。**再等几年数据它自己就会越线。**
  · BO 内比分差在后半段已经是 ΔLL +0.00100 / MDE 0.00112 / t=2.51，
    余量 0.89×，趋势在往显著走。

没有这个机制，这些假设就靠人凭感觉想起来才复查一次——而人通常想不起来。

台账是 **append-only** 的：每次重测追加一条，于是「这个量随时间怎么变」
本身成为一条可读的曲线，而不是只有最新一次的是非判断。
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

LEDGER = "reports/retest_ledger.jsonl"


@dataclass
class Hypothesis:
    key: str
    title: str
    gate: str                      # 门禁的一句话描述
    threshold: float
    why_parked: str                # 当初为什么搁置
    run: Callable[[], dict[str, Any]] = field(repr=False, default=None)  # type: ignore[assignment]


@dataclass
class Result:
    key: str
    run_date: str
    n_games: int
    statistic: float
    threshold: float
    passed: bool
    margin: float                  # statistic / threshold，>1 即通过
    detail: dict[str, Any] = field(default_factory=dict)
    unchanged: bool = False        # 与上次逐位相同 → 不写台账，但算一次确定性复核


# ─────────────────────────── 各假设的重测实现 ───────────────────────────

def _reliability(kind: str) -> dict[str, Any]:
    """协同 / 克制的 split-half 可靠性。多种子取均值，避免单次划分撞运气。"""
    import numpy as np

    from lolwp.eval.reliability import GATE, _pair_stats, reliability
    from lolwp.features.loader import load_dev

    d = load_dev(min_tier=1, since="2019-01-01")
    same = kind == "synergy"
    vals = [reliability(d, lambda x: _pair_stats(x, same_team=same), kind,
                        min_obs=10, seed=s).r_full for s in range(8)]
    a = np.array(vals, dtype=float)
    return {"n_games": len(d), "statistic": float(np.nanmean(a)),
            "threshold": GATE,
            "detail": {"sd": float(np.nanstd(a)), "min": float(np.nanmin(a)),
                       "max": float(np.nanmax(a)), "n_seeds": len(a),
                       "all_pass": bool((a >= GATE).all()),
                       "all_fail": bool((a < GATE).all())}}


def _mde_gate(extra_cols: list[str], label: str) -> dict[str, Any]:
    """在 BT4 之上加一组特征，看 ΔlogLoss 是否超过该比较自测的 MDE。"""
    from lolwp.eval.baselines import LADDER, run_ladder
    from lolwp.eval.metrics import per_game_log_loss
    from lolwp.eval.power import paired_power
    from lolwp.eval.walkforward import WFConfig, make_folds
    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine
    from lolwp.features.series_ctx import build as build_series
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.hier_bt import HierBTConfig

    full = load_dev()
    F = pd.concat([run_state_machine(full), run_draft_state_machine(full),
                   rolling_ratings_cached(full, HierBTConfig(refit_every_days=3)),
                   build_series(full)], axis=1)
    m = (full.league_tier >= 1) & (full.kickoff_utc >= "2019-01-01")
    dev = full[m].reset_index(drop=True)
    FF = F.loc[full[m].index].reset_index(drop=True)
    folds = make_folds(dev, WFConfig())
    base = ["side_prior", "bt_player_diff", "bt_chem_diff", "bt_league_diff",
            "champ_meta_diff"]
    LADDER["_BASE"] = base
    LADDER["_CAND"] = base + extra_cols
    p = run_ladder(dev, FF, folds, names=["_BASE", "_CAND"])
    w = p.pivot_table(index=["game_id", "series_id"], columns="model",
                      values="p", aggfunc="first").reset_index()
    y = p.drop_duplicates("game_id").set_index("game_id").y
    w["y"] = w.game_id.map(y)
    yy = w.y.to_numpy()
    d = (per_game_log_loss(yy, w["_BASE"].to_numpy())
         - per_game_log_loss(yy, w["_CAND"].to_numpy()))
    pw = paired_power(d, w.series_id.to_numpy())
    for k in ("_BASE", "_CAND"):
        LADDER.pop(k, None)
    return {"n_games": int(pw["n_games"]), "statistic": float(pw["mean_delta"]),
            "threshold": float(pw["mde"]),
            "detail": {"label": label, "t_clustered": float(pw["t_clustered"]),
                       "rho": float(pw["icc"]), "n_series": int(pw["n_series"])}}


REGISTRY: list[Hypothesis] = [
    Hypothesis(
        key="counter_reliability",
        title="英雄克制的 split-half 可靠性",
        gate="r_full ≥ 0.20（8 个随机划分的均值）",
        threshold=0.20,
        why_parked="2026-09 实测 0.1566，8 次划分全不过。但学习曲线仍在爬，"
                   "对数外推需约 15 万场（现有 10.2 万），再等几年会自己越线。",
        run=lambda: _reliability("counter"),
    ),
    Hypothesis(
        key="synergy_reliability",
        title="英雄协同的 split-half 可靠性",
        gate="r_full ≥ 0.20",
        threshold=0.20,
        why_parked="2026-09 实测 0.1754，均值低于门槛且随种子翻面。",
        run=lambda: _reliability("synergy"),
    ),
    Hypothesis(
        key="series_score_diff",
        title="BO 内比分差",
        gate="ΔlogLoss > 该比较自测的 MDE",
        threshold=0.0,
        why_parked="2026-09 实测全样本 +0.00073（MDE 0.00084）、"
                   "后半段 +0.00100（MDE 0.00112, t=2.51），余量 0.89×，"
                   "**全项目最接近通过的一次近失**。",
        run=lambda: _mde_gate(["series_score_diff"], "series_score_diff"),
    ),
    Hypothesis(
        key="mastery_diff",
        title="选手熟练度（decayed 场次）",
        gate="ΔlogLoss > MDE",
        threshold=0.0,
        why_parked="2026-09 实测 +0.00040，在全部 ICC 假设下都低于 MDE。",
        run=lambda: _mde_gate(["mastery_diff"], "mastery_diff"),
    ),
]


def _last_per_key(ledger: str) -> dict[str, dict]:
    if not os.path.exists(ledger):
        return {}
    out: dict[str, dict] = {}
    with open(ledger) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def run_all(keys: list[str] | None = None, ledger: str = LEDGER) -> pd.DataFrame:
    """跑重测并**只在结果有变化时**追加。

    为什么要去重：同一份数据重跑会产生逐位相同的条目，灌进台账只会淹没趋势。
    但重跑本身有价值——它是一次免费的确定性检查，所以结果仍然返回，
    只是标记 `appended=False`。
    """
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    prev = _last_per_key(ledger)
    rows: list[Result] = []
    for h in REGISTRY:
        if keys and h.key not in keys:
            continue
        try:
            r = h.run()
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {h.key}: {type(e).__name__}: {e}")
            continue
        thr = r["threshold"] if h.threshold == 0.0 else h.threshold
        stat = r["statistic"]
        passed = stat > thr if h.threshold == 0.0 else stat >= thr
        res = Result(key=h.key, run_date=today, n_games=r["n_games"],
                     statistic=stat, threshold=thr, passed=bool(passed),
                     margin=float(stat / thr) if thr else float("nan"),
                     detail=r.get("detail", {}))
        old = prev.get(h.key)
        res.unchanged = bool(
            old and old["n_games"] == res.n_games
            and abs(old["statistic"] - res.statistic) < 1e-12
        )
        rows.append(res)

    fresh = [r for r in rows if not r.unchanged]
    os.makedirs(os.path.dirname(ledger) or ".", exist_ok=True)
    if fresh:
        with open(ledger, "a") as fh:
            for r in fresh:
                d = asdict(r)
                d.pop("unchanged", None)
                fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    return pd.DataFrame([asdict(r) for r in rows])


def history(ledger: str = LEDGER) -> pd.DataFrame:
    """读全部历史。**趋势比单次结论重要**——这正是台账 append-only 的理由。"""
    if not os.path.exists(ledger):
        return pd.DataFrame()
    with open(ledger) as fh:
        return pd.DataFrame([json.loads(x) for x in fh if x.strip()])
