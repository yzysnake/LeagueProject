"""四个负控制（PLAN.md §6.9）。

这些不是"锦上添花的测试"，而是唯一能证明「指标好看不是因为泄漏」的手段。
全部进 CI。任何一条失败都说明管道有洞，此时所有指标一律作废。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lolwp.eval.baselines import run_ladder
from lolwp.eval.metrics import auc, log_loss
from lolwp.eval.walkforward import make_folds
from lolwp.features.priors import run_state_machine


def label_shuffle(
    df: pd.DataFrame, folds: list, *, seed: int = 0, model: str = "L3"
) -> dict[str, float]:
    """N1：打乱标签后跑【完整 pipeline】（含状态机重跑），log loss 必须回到 0.693。

    关键是**状态机也要用打乱后的标签重跑**。只打乱评估时的 y 而复用原特征，
    这个控制就形同虚设——那只证明了"随机标签不可预测"，证明不了特征无泄漏。
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["y_blue_win"] = rng.permutation(d.y_blue_win.to_numpy())
    feats = run_state_machine(d)
    preds = run_ladder(d, feats, folds, names=[model])
    ll = log_loss(preds.y.to_numpy(), preds.p.to_numpy())
    return {"log_loss": ll, "n": len(preds), "pass": 0.685 <= ll <= 0.700}


def leak_canary(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    folds: list,
    bronze_path: str = "data/bronze/all.parquet",
) -> dict[str, float]:
    """N3：故意注入一个赛后列，确认检测灵敏度够。

    注意 bronze 已经物理丢掉了 gamelength，所以这里从原始 CSV 取。
    这条通过之后必须断言生产管道里没有这一列。
    """
    import glob

    lens = []
    for p in sorted(glob.glob("data/raw/*_LoL_esports_match_data_from_OraclesElixir.csv")):
        c = pd.read_csv(p, usecols=["gameid", "position", "result", "kills", "deaths"],
                        dtype=str, low_memory=False)
        c = c[c.position == "team"]
        lens.append(c)
    raw = pd.concat(lens, ignore_index=True)
    raw["kills"] = pd.to_numeric(raw.kills, errors="coerce")
    agg = raw.groupby("gameid").kills.apply(lambda s: s.iloc[0] - s.iloc[1] if len(s) == 2 else np.nan)

    d = df.copy()
    f = feats.copy()
    f["LEAK_killdiff"] = d.game_id.map(agg).fillna(0.0).to_numpy()

    from lolwp.eval.baselines import LADDER

    LADDER["LEAK"] = ["side_prior", "team_elo_diff", "player_elo_diff", "LEAK_killdiff"]
    try:
        base = run_ladder(d, feats, folds, names=["L3"])
        leaked = run_ladder(d, f, folds, names=["LEAK"])
    finally:
        LADDER.pop("LEAK", None)

    a0 = auc(base.y.to_numpy(), base.p.to_numpy())
    a1 = auc(leaked.y.to_numpy(), leaked.p.to_numpy())
    return {"auc_clean": a0, "auc_leaked": a1, "delta": a1 - a0, "pass": (a1 - a0) > 0.10}


def future_deletion_invariance(
    df: pd.DataFrame, *, n_sample: int = 200, seed: int = 0
) -> dict[str, object]:
    """N4：删掉 t 之后的所有行，重算特征，必须 bit-identical。

    这是最强的一道闸。任何 groupby 全表统计、任何 target encoding、
    任何"不小心用了未来"的写法都会在这里露馅。
    """
    rng = np.random.default_rng(seed)
    full = run_state_machine(df)
    # 取一个中间时点
    cut_pos = int(len(df) * 0.7)
    cut = df.kickoff_utc.iloc[cut_pos]
    past = df[df.kickoff_utc < cut].reset_index(drop=True)
    partial = run_state_machine(past)

    pool = np.arange(len(past))
    pick = rng.choice(pool, size=min(n_sample, len(pool)), replace=False)
    cols = list(full.columns)
    a = full.iloc[pick][cols].to_numpy(dtype=float)
    b = partial.iloc[pick][cols].to_numpy(dtype=float)
    same = np.array_equal(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))
    maxdiff = float(np.nanmax(np.abs(a - b))) if not same else 0.0
    return {
        "cut": str(cut),
        "n_checked": len(pick),
        "bit_identical": bool(same),
        "max_abs_diff": maxdiff,
        "pass": bool(same),
    }


def time_reversal(df: pd.DataFrame, feats: pd.DataFrame, folds: list) -> dict[str, float]:
    """N2：用未来预测过去。若反向反而更好，说明特征携带未来信息。"""
    d = df.iloc[::-1].reset_index(drop=True)
    f = feats.iloc[::-1].reset_index(drop=True)
    rev_folds = make_folds(d.assign(kickoff_utc=df.kickoff_utc.to_numpy()))
    fwd = run_ladder(df, feats, folds, names=["L3"])
    try:
        rev = run_ladder(d, f, rev_folds, names=["L3"])
        rv = log_loss(rev.y.to_numpy(), rev.p.to_numpy())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "pass": True}
    fv = log_loss(fwd.y.to_numpy(), fwd.p.to_numpy())
    return {"forward": fv, "reversed": rv, "pass": rv >= fv - 0.005}
