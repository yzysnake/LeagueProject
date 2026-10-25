#!/usr/bin/env python3
"""P3：冻结规格在 OE 截止（2026-09-18）之后的比赛上的一次性前瞻评估（预注册见 experiments/preregistration.md「P3」）。

三个规格都只用截止之前的数据训练，步骤与生产训练（models.artifact.train）同一口径：
  S0  BT4_LR            胜负标签的实力快照（修订 A4 的规格）
  S1  BT4_LR+MOV        赛后经济差软标签（修订 A6）
  S2  BT4_LR+MOV+FORM   S1 + 近期状态第二层（修订 A7，生产）
评估局 = 截止之后、tier≥1 的 OE 比赛。特征一律 as-of（分层 BT 3 天网格滚动、两个状态机单遍）；
外层系数与近期状态的 β 冻结在截止时刻；近期状态的残差：截止前用样本内预测、截止后用 S1 自己的预测。

    python scripts/p3_prospective.py                  # 正式评估：一次性，记录已存在就拒绝
    python scripts/p3_prospective.py --rehearse 2026-03-31   # 彩排：开发期数据、截止挪到该日，不写记录
    python scripts/p3_prospective.py --when-ready     # 每日任务（launchd）：到了预注册的触发条件才跑正式评估
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2026-09-18 23:20:29", tz="UTC")   # 冻结规格的训练截止 = 生产 artifact 的 meta.trained_on
TRIGGER = pd.Timestamp("2026-11-16", tz="UTC")           # 预注册：这天及以后第一次拿到截止之后的 OE 数据就评估
RECORD = "reports/p3_prospective_result.json"
RAW_2026 = "data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv"
MIN_GAMES = 30                                           # 截止之后 tier≥1 不足这么多场就不评（别把一次性评估花在空数据上）
SPECS = {"S0": "BT4_LR", "S1": "BT4_LR+MOV", "S2": "BT4_LR+MOV+FORM"}


def evaluate_specs(full: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """三个规格在 (cutoff, 数据末尾] 的 tier≥1 局上的成绩与两两配对比较。full 按开赛升序、silver 口径。"""
    from sklearn.linear_model import LogisticRegression

    from lolwp.eval.gate import paired_gate
    from lolwp.eval.metrics import evaluate
    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.priors import run_state_machine
    from lolwp.models.artifact import FORM_SINCE, PROD_FEATURES
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.form import DEFAULT, fit_beta, form_diff
    from lolwp.models.hier_bt import HierBTConfig
    from lolwp.models.margin import TARGET, WIN, with_margin_target

    full = with_margin_target(full)
    sm = pd.concat([run_state_machine(full), run_draft_state_machine(full)], axis=1)
    cfg = HierBTConfig(refit_every_days=3)
    t1 = (full.league_tier >= 1).to_numpy()
    tr = t1 & (full.kickoff_utc <= cutoff).to_numpy()
    y = full.y_blue_win.to_numpy(dtype=float)
    z = {}
    for name, target in (("S0", WIN), ("S1", TARGET)):
        X = pd.concat([sm, rolling_ratings_cached(full, cfg, target=target)], axis=1)[PROD_FEATURES].to_numpy(float)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        z[name] = lr.decision_function((X - mu) / sd)
    # S2：近期状态。tier≥1 序列上算 form（截止前的残差用样本内 S1、截止后用 S1 的样本外预测），β 冻结在截止时刻
    g = full[t1].reset_index(drop=True)
    z1 = z["S1"][t1]
    f = form_diff(g, 1.0 / (1.0 + np.exp(-z1)), DEFAULT)
    g_tr = (g.kickoff_utc <= cutoff).to_numpy()
    fit = g_tr & (g.kickoff_utc >= pd.Timestamp(FORM_SINCE, tz="UTC")).to_numpy()
    beta = fit_beta(z1[fit], f[fit], g.y_blue_win.to_numpy(dtype=float)[fit])
    z2 = np.full(len(full), np.nan)
    z2[t1] = z1 + beta * f
    z["S2"] = z2

    ev = t1 & (full.kickoff_utc > cutoff).to_numpy()
    p = {k: 1.0 / (1.0 + np.exp(-v[ev])) for k, v in z.items()}
    games = full[ev]
    ye, ser = y[ev], games.series_id.to_numpy()
    out = {"cutoff": str(cutoff), "data_end": str(full.kickoff_utc.max()), "n_games": int(ev.sum()),
           "n_series": int(pd.Series(ser).nunique()), "form_beta": float(beta),
           "leagues": games.league_canon.value_counts().to_dict(), "specs": {}, "compare": {}}
    if ev.sum() == 0:
        return out
    for k, name in SPECS.items():
        e = evaluate(ye, p[k])
        out["specs"][k] = {"name": name, **{m: float(e[m]) for m in
                                                ("log_loss", "brier", "auc", "ece", "calib_slope", "accuracy")}}
    for a, b in (("S0", "S1"), ("S1", "S2"), ("S0", "S2")):
        out["compare"][f"{b} vs {a}"] = paired_gate(ye, p[a], p[b], ser)
    return out


def _print(res: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"截止 {res['cutoff'][:16]} → 数据末尾 {res['data_end'][:16]}：tier≥1 {res['n_games']} 场 / "
          f"{res['n_series']} 个系列；近期状态 β = {res['form_beta']:.3f}")
    print("  " + "、".join(f"{k} {v}" for k, v in list(res["leagues"].items())[:8]))
    if not res["specs"]:
        return
    print(f"\n{'规格':<20}{'log loss':>10}{'Brier':>9}{'AUC':>8}{'ECE':>8}{'斜率':>7}{'acc':>8}")
    for k, s in res["specs"].items():
        print(f"{k + ' ' + s['name']:<20}{s['log_loss']:>10.5f}{s['brier']:>9.5f}{s['auc']:>8.4f}"
              f"{s['ece']:>8.4f}{s['calib_slope']:>7.3f}{s['accuracy']:>8.4f}")
    print(f"\n{'比较':<12}{'Δ':>10}{'MDE':>9}{'余量':>8}{'t':>7}{'前半':>10}{'后半':>10}")
    for k, c in res["compare"].items():
        print(f"{k:<12}{c['delta']:>+10.5f}{c['mde']:>9.5f}{c['delta'] / c['mde']:>+7.2f}×{c['t_clustered']:>7.2f}"
              f"{c['half_1']:>+10.5f}{c['half_2']:>+10.5f}")


def run_real(record: str = RECORD) -> int:
    """正式评估：一次性。记录已存在就拒绝（要重做得设 LOLWP_P3_RERUN=yes，并在预注册里记一笔）。"""
    if os.path.exists(record) and os.environ.get("LOLWP_P3_RERUN") != "yes":
        print(f"拒绝：P3 已做过（{record}）。这是一次性的前瞻评估。", file=sys.stderr)
        return 2
    from lolwp.features.loader import load_all

    full = load_all()
    n_after = int(((full.kickoff_utc > CUTOFF) & (full.league_tier >= 1)).sum())
    if n_after < MIN_GAMES:
        print(f"截止之后 tier≥1 只有 {n_after} 场（< {MIN_GAMES}），不评——一次性评估别花在空数据上", file=sys.stderr)
        return 3
    t0 = time.time()
    res = evaluate_specs(full, CUTOFF)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    res.update(protocol="P3（experiments/preregistration.md）", git_sha=sha,
               run_at=str(pd.Timestamp.now(tz="UTC").floor("s")), seconds=round(time.time() - t0))
    _print(res, "P3 前瞻评估")
    os.makedirs(os.path.dirname(record) or ".", exist_ok=True)
    with open(record, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\n→ {record}")
    return 0


def raw_has_games_after(cutoff: pd.Timestamp, path: str = RAW_2026) -> bool:
    if not os.path.exists(path):
        return False
    d = pd.to_datetime(pd.read_csv(path, usecols=["date"]).date, utc=True, errors="coerce")
    return bool((d > cutoff).any())


def _notify(msg: str) -> None:
    """本机桌面通知（只在本机弹，不发往任何外部服务）。"""
    try:
        subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "LoL 胜率模型 · P3"'],
                       timeout=10, check=False)
    except Exception as e:  # noqa: BLE001  通知失败不影响评估本身
        print(f"桌面通知失败：{type(e).__name__}: {e}")


def when_ready(today: pd.Timestamp | None = None, *, run=subprocess.run) -> int:
    """每日任务：做过了就什么都不做；没到预注册的触发日不动 OE；到了就试着拉 OE，
    拿到截止之后的比赛就重建 bronze / silver / 经济差表，跑一次正式评估并弹通知。"""
    if os.path.exists(RECORD):
        return 0
    today = today or pd.Timestamp.now(tz="UTC")
    if today < TRIGGER:
        print(f"[{today:%Y-%m-%d}] 未到预注册的触发日 {TRIGGER:%Y-%m-%d}")
        return 0
    if run(["bash", "scripts/day0_fetch.sh"]).returncode != 0:
        print(f"[{today:%Y-%m-%d}] OE 仍不可用（Drive 配额？），明天再试")
        return 0
    py = [sys.executable]
    if run(py + ["scripts/day0_validate.py"]).returncode != 0 or not raw_has_games_after(CUTOFF):
        print(f"[{today:%Y-%m-%d}] OE 文件里还没有截止之后的比赛，明天再试")
        return 0
    for mod in ("lolwp.store.bronze", "lolwp.store.silver", "lolwp.store.margins"):
        if run(py + ["-m", mod]).returncode != 0:
            _notify(f"重建 {mod} 失败，P3 没跑，见 logs/p3.log")
            return 1
    rc = run_real()
    if rc == 0:
        with open(RECORD) as fh:
            r = json.load(fh)
        s2 = r["specs"]["S2"]["log_loss"]
        _notify(f"P3 完成：{r['n_games']} 场，生产规格 log loss {s2:.4f}。报告 {RECORD}")
    elif rc == 3:
        print("截止之后的比赛太少，明天再试")
        return 0
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rehearse", metavar="DATE", help="彩排：开发期数据、截止挪到 DATE，不写记录")
    g.add_argument("--when-ready", action="store_true", help="每日任务：到了触发条件才跑正式评估")
    a = ap.parse_args()
    if a.when_ready:
        return when_ready()
    if a.rehearse:
        from lolwp.features.loader import load_dev

        res = evaluate_specs(load_dev(), pd.Timestamp(a.rehearse, tz="UTC"))
        _print(res, f"P3 彩排（开发期数据，截止挪到 {a.rehearse}；不是评估，不写记录）")
        return 0
    return run_real()


if __name__ == "__main__":
    sys.exit(main())
