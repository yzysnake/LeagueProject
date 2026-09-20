"""测量 α̂：BP 阶段到底值多少胜率（PLAN.md §4.5.4）。

手册称 α̂ 的区间是"整个项目最有科研价值的单个数字"。

做法（每一条都是防泄漏的必要条件）：
1. base 模型固定（L4 或 BT4），只在它之上加 zero-init 的 synergy / counter 项；
2. embedding **在每个 fold 的训练段内重新拟合**，绝不用全量预训练
   （§4.5.5：用全量预训练再打 OOS 等于把未来 meta 喂进特征，不报错只让指标好看）；
3. α̂ 就是外层 logistic 上 synergy / counter 的系数；
4. 增量用 series 级 block bootstrap 给区间，并与该比较自测的 MDE 对照。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lolwp.eval.metrics import evaluate, per_game_log_loss
from lolwp.eval.power import footer, paired_power
from lolwp.eval.walkforward import WFConfig, make_folds
from lolwp.features.draft_priors import run_draft_state_machine
from lolwp.features.loader import load_dev
from lolwp.features.priors import run_state_machine
from lolwp.models.draft_lowrank import DraftLowRank, LowRankConfig

BASE_L4 = ["side_prior", "team_elo_diff", "player_elo_diff", "champ_meta_diff"]
BASE_BT = ["side_prior", "bt_player_diff", "bt_chem_diff", "bt_league_diff", "champ_meta_diff"]


def _fit_predict(Xtr, ytr, Xva):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)
    m.fit((Xtr - mu) / sd, ytr)
    return m.predict_proba((Xva - mu) / sd)[:, 1], m.coef_[0], sd


def run(base_cols: list[str], dev: pd.DataFrame, F: pd.DataFrame, folds, *,
        dim: int = 8, l2: float = 3.0, use_counter: bool = True,
        pretrain: pd.DataFrame | None = None) -> dict:
    """pretrain 给定时走【冻结迁移】：embedding 在不相交语料上预训练后冻结，
    只在 tier-1 上学 α（§4.5.5）。

    **为什么必须这样**：朴素两阶段（embedding 与外层 logistic 都在同一批训练行上拟合）
    是构造性的目标泄漏——外层看到的是样本内的 embedding 预测，会给出巨大的 α，
    而 OOS 立刻崩。实测 L4 上 log loss 从 0.63076 涨到 **0.99658**、
    校准斜率塌到 0.0911。这个失败本身就是"为什么要冻结迁移"的最好证据。
    """
    y = dev.y_blue_win.to_numpy()
    bg = list(dev.blue_champ_keys)
    rg = list(dev.red_champ_keys)
    rec_base, rec_full, alphas = [], [], []
    pt_time = pretrain.kickoff_utc if pretrain is not None else None
    for f in folds:
        tr, va = f.train_idx, f.valid_idx
        lr = DraftLowRank(LowRankConfig(dim=dim, l2=l2, use_counter=use_counter))
        if pretrain is not None:
            # 预训练语料只取该 fold 训练段截止时刻之前的**非评测赛区**比赛，
            # 与 tier-1 的拟合行零重叠，且不含任何未来信息
            sub = pretrain[pt_time < f.train_end]
            if len(sub) < 2000:
                continue
            lr.fit(list(sub.blue_champ_keys), list(sub.red_champ_keys),
                   sub.y_blue_win.to_numpy())
        else:
            lr.fit([bg[i] for i in tr], [rg[i] for i in tr], y[tr])
        Ztr = lr.transform([bg[i] for i in tr], [rg[i] for i in tr])
        Zva = lr.transform([bg[i] for i in va], [rg[i] for i in va])

        Xb_tr = F.loc[tr, base_cols].to_numpy(float)
        Xb_va = F.loc[va, base_cols].to_numpy(float)
        p_base, _, _ = _fit_predict(Xb_tr, y[tr], Xb_va)
        p_full, coef, _ = _fit_predict(np.hstack([Xb_tr, Ztr]), y[tr], np.hstack([Xb_va, Zva]))

        k = len(base_cols)
        alphas.append({"fold": f.idx, "alpha_syn": float(coef[k]),
                       "alpha_cnt": float(coef[k + 1]) if use_counter else 0.0})
        for name, p, rec in (("base", p_base, rec_base), ("full", p_full, rec_full)):
            rec.append(pd.DataFrame({"game_id": dev.loc[va, "game_id"].to_numpy(),
                                     "series_id": dev.loc[va, "series_id"].to_numpy(),
                                     "y": y[va], "p": p}))
    B = pd.concat(rec_base, ignore_index=True)
    Fu = pd.concat(rec_full, ignore_index=True)
    d = per_game_log_loss(B.y.to_numpy(), B.p.to_numpy()) - per_game_log_loss(Fu.y.to_numpy(), Fu.p.to_numpy())
    pw = paired_power(d, B.series_id.to_numpy())
    A = pd.DataFrame(alphas)
    return {"base": evaluate(B.y.to_numpy(), B.p.to_numpy()),
            "full": evaluate(Fu.y.to_numpy(), Fu.p.to_numpy()),
            "power": pw, "alphas": A, "preds_base": B, "preds_full": Fu}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["L4", "BT4"], default="L4")
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--bt-grid", type=int, default=14)
    ap.add_argument("--pretrain", choices=["none", "tier0"], default="tier0",
                    help="tier0=在非评测赛区上预训练后冻结（§4.5.5 的冻结迁移）")
    a = ap.parse_args()

    full = load_dev()
    F_all = pd.concat([run_state_machine(full), run_draft_state_machine(full)], axis=1)
    cols = BASE_L4
    if a.base == "BT4":
        from lolwp.models.bt_cache import rolling_ratings_cached
        from lolwp.models.hier_bt import HierBTConfig
        t0 = time.time()
        F_all = pd.concat([F_all, rolling_ratings_cached(full, HierBTConfig(refit_every_days=a.bt_grid))], axis=1)
        print(f"BT 特征 {time.time()-t0:.0f}s（命中缓存则接近 0）", flush=True)
        cols = BASE_BT

    m = (full.league_tier >= 1) & (full.kickoff_utc >= "2019-01-01")
    dev = full[m].reset_index(drop=True)
    F = F_all.loc[full[m].index].reset_index(drop=True)
    folds = make_folds(dev, WFConfig())
    print(f"base={a.base}  {len(dev):,} 场  {len(folds)} 折  d={a.dim}  l2={a.l2}", flush=True)

    pre = None
    if a.pretrain == "tier0":
        pre = full[full.league_tier == 0].reset_index(drop=True)
        print(f"预训练语料：非评测赛区 {len(pre):,} 场（与 tier-1 拟合行零重叠）", flush=True)
    t0 = time.time()
    out = run(cols, dev, F, folds, dim=a.dim, l2=a.l2, pretrain=pre)
    print(f"完成 {time.time()-t0:.0f}s\n", flush=True)

    b, f_, pw, A = out["base"], out["full"], out["power"], out["alphas"]
    print(f"{'':<8}{'log loss':>10}{'Brier':>9}{'AUC':>8}{'ECE':>8}{'slope':>8}")
    for nm, e in (("base", b), ("+draft", f_)):
        print(f"{nm:<8}{e['log_loss']:>10.5f}{e['brier']:>9.5f}{e['auc']:>8.4f}{e['ece']:>8.4f}{e['calib_slope']:>8.4f}")
    print(f"\nΔlogLoss = {pw['mean_delta']:+.5f}")
    print(f"  {footer(pw)}")
    print(f"  裁决：{'超过 MDE ✓' if pw['exceeds_mde'] else '【低于本样本量的可检出下限】✗'}")
    print("\nα̂（各 fold 的外层系数，标准化尺度）")
    for c in ("alpha_syn", "alpha_cnt"):
        v = A[c].to_numpy()
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"  {c}: 中位数 {np.median(v):+.5f}  均值 {v.mean():+.5f}  "
              f"跨 fold 95% 区间 [{lo:+.5f}, {hi:+.5f}]  同号比例 {max((v>0).mean(),(v<0).mean()):.0%}")
    with open(f"reports/alpha_{a.base}_d{a.dim}_{a.pretrain}.json", "w") as fh:
        json.dump({"delta_ll": pw["mean_delta"], "mde": pw["mde"],
                   "exceeds": bool(pw["exceeds_mde"]),
                   "base_ll": b["log_loss"], "full_ll": f_["log_loss"],
                   "alpha_syn_median": float(A.alpha_syn.median()),
                   "alpha_cnt_median": float(A.alpha_cnt.median())}, fh, indent=2)


if __name__ == "__main__":
    main()
