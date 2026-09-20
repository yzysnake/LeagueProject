"""双向消融矩阵（PLAN.md §6.5）：减法（leave-one-out）与加法（正向增量）两张表。

**为什么一定要双向**：单向 LOO 会把互相共线的特征一起判死刑 —— 若 A 与 B 携带
同一份信息，去掉任何一个模型都不掉，两行都显示"无用"，照此办理就会把两个都砍掉，
而真相是"必须留一个"。正向增量是另一个极端：先进来的那个吃掉全部共享增益，
后进来的看起来一文不值。**两张表方向不一致的行就是共线的指纹**，
本模块把这件事做成可自动检出的判定（`disagreements()`），而不是留给人眼去看。

符号约定（两张表统一，报告里也统一）：

    Δ > 0  ⇔  这一组特征**带来增量**
      减法 A：Δ = LL(去掉它) − LL(全特征)        去掉后变差 → Δ > 0
      加法 B：Δ = LL(加它之前) − LL(加它之后)     加上后变好 → Δ > 0

裁决不看 Δ 的绝对大小，看 |Δ| 与**该行自己**的 MDE 的关系（§6.3、§6.6.4）：
`|Δ| < MDE` 一律写成【低于本样本量的可检出下限】，既不叫"改进"也不叫"变差"。
每一行都单独跑 `power.paired_power()`，因为 σ_d 与 ρ 逐行不同 —— 用全局 MDE
去卡每一行是错的（强特征的配对差方差比弱特征大一个量级）。

所有行都是 exploratory，走 `compare.benjamini_hochberg` 控 FDR；
预注册的唯一 primary comparison（M_full vs L3，experiments/preregistration.md §1）
单独报告，**不进 BH 池**（把预注册检验混进 FDR 池等于自己惩罚自己）。
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lolwp.eval import compare, metrics, power

# ---------------------------------------------------------------------------
# 1. 特征组定义
# ---------------------------------------------------------------------------
#: 消融的最小单位。**一组一起进、一起出**，不拆列 —— 拆列做出来的是"列消融"，
#: 回答不了"这个信息源值不值得维护"这个真正的工程问题。
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "side_prior": ("side_prior",),
    "team_elo": ("team_elo_diff",),
    "player_elo": ("player_elo_diff",),
    "bt_player": ("bt_player_diff",),
    "bt_chem": ("bt_chem_diff",),
    "bt_league": ("bt_league_diff",),
    "champ_meta": ("champ_meta_diff",),
    "mastery": ("mastery_diff",),
    # patch_age 的主效应对"蓝方是否获胜"是对称的、先验上应当 ≈ 0；
    # 它真正的假设（§1317：新版本 meta 不稳定 → 模型应降低对英雄项的信心）
    # 只能写成交互项。所以这一组 = 主效应 + 交互，缺一不可：
    # 只放主效应等于故意去测一个我们预期为 0 的东西。
    "patch_age": ("patch_age_log", "champ_meta_x_fresh"),
    # 缺失/覆盖指示器：模型对"本场我到底知道多少"的自我认知。
    # 注意这一组有 6 列而别的组只有 1-2 列，它的 Δ 天然占便宜（也更容易过拟合），
    # 报告里必须连列数一起写出来。
    "missing": (
        "player_missing_n",
        "player_missing_diff",
        "patch_age_na",
        "team_games_min_log",
        "player_games_min_log",
        "bt_known_players",
    ),
}

#: 正向增量的顺序。**顺序是这张表的一部分，必须预先写死并公开**，
#: 否则"先加谁"就成了可以事后挑选的自由度（§6.6.5 的 20 次尝试问题）。
#: 顺序按「信息源的获取成本从低到高」排，与 §5.2 的基线阶梯一致；
#: champ_meta 必须排在 patch_age 之前 —— patch_age 组里含 champ_meta 的交互项，
#: 反过来加会让一个没有英雄主效应的模型先看到英雄信息，那一步的 Δ 就没法解释了。
ADDITION_ORDER: tuple[str, ...] = (
    "side_prior",
    "team_elo",
    "player_elo",
    "bt_player",
    "bt_chem",
    "bt_league",
    "champ_meta",
    "mastery",
    "patch_age",
    "missing",
)

#: 组合消融：**共线的组必须一起拿掉再看一次**，否则 LOO 会给出"三个都没用"的假象。
#: 这三组就是本项目里已知互相替代的三团信息（Elo 两项 / BT 三项 / draft 三项）。
#: 后两项（champ_all / extras_all）是**看过第一轮结果之后补的**，用来结掉表 A/表 B
#: 的两处不一致，因此它们照样进 BH 池（§6.6.5：事后加的检验必须付 FDR 的账），
#: 报告里也明确标成"第二轮补测"，不得当成预先声明的结果读。
COMBOS: dict[str, tuple[str, ...]] = {
    "elo_all": ("team_elo", "player_elo"),
    "bt_all": ("bt_player", "bt_chem", "bt_league"),
    "draft_all": ("champ_meta", "mastery", "patch_age"),
    # champ_meta_diff 与 champ_meta_x_fresh 分居两组（后者在 patch_age 组里），
    # 单拿掉任何一个，另一个都能把英雄先验重建出来 —— 必须一起拿掉才测得到。
    "champ_all": ("champ_meta", "patch_age"),
    # 加法表最后两步都往回走（Δ<0）。逐步看都低于 MDE，合起来值不值得测，只能合测。
    "extras_all": ("patch_age", "missing"),
}

#: L3 = 预注册里 primary comparison 的对照组（side + team rating + player rating）。
L3_GROUPS: tuple[str, ...] = ("side_prior", "team_elo", "player_elo")

#: 刻意不进消融矩阵的列，以及原因。
EXCLUDED: dict[str, str] = {
    "bt_logit": "分层 BT 自己的合成预测值，与三个 bt_* 分量重复计数",
    "bt_team_level_diff": "bt_player_diff 的线性组合近似，留着只会让 LOO 互相抵消",
    "blue_team_elo": "绝对水位；模型只用得上 diff，留着等于给 side 一个额外截距",
    "red_team_elo": "同上",
    "champ_cover_min": "与 patch_age_na / bt_known_players 高度重合，避免缺失组再膨胀",
    "min_champ_games_diff": "mastery_diff 的另一种写法，同源",
    "n_first_time_champ_diff": "同上",
    "blue_champ_missing": "全库恒为 0，零方差",
    "red_champ_missing": "全库恒为 0，零方差",
}

#: §6.9 红线（experiments/preregistration.md §3）。触线即停，不得解释成"模型很强"。
RED_LINE_ACC = 0.67
RED_LINE_LL = 0.60
RED_LINE_FOLD_AUC = 0.78

#: 新版本窗口的半衰期（天）。14 天 ≈ 一个补丁周期，与 §6.5 的分桶口径同量级。
FRESH_HALF_LIFE_DAYS = 14.0


def group_columns(groups: Sequence[str]) -> list[str]:
    """把组名展开成列名，保持 FEATURE_GROUPS 的声明顺序（可复现）。"""
    out: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError(f"未知特征组 {g!r}；已知：{sorted(FEATURE_GROUPS)}")
        out.extend(FEATURE_GROUPS[g])
    return out


def assemble_features(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """把 priors / draft_priors / 分层 BT 的输出拼成消融用的设计矩阵。

    `df` 是 silver 行（提供 patch_age_days 与缺失计数），`raw` 是三个特征生成器
    concat 后的结果。两者必须**同索引同长度**。

    这里只做**不依赖标签、也不依赖未来**的逐行变换：
    - `patch_age_log = log1p(天数)`：age 的分布右偏（p50=12, max=148），
      线性项会被少数陈年补丁的长尾主导；
    - `champ_meta_x_fresh = champ_meta_diff · exp(−age/14)`：补丁越新，
      这一项越接近 champ_meta_diff 本身，logistic 学到的系数若为负，
      就是"新版本请打折使用英雄先验"的直接证据；
    - NaN 的 patch_age 一律填 0 并打 `patch_age_na=1`（0.7% 的行）。
      填常数而不是填中位数，是因为中位数要在每折训练段上重算才不泄漏，
      而指示器本来就能把这个常数偏移吸收掉。
    """
    if len(df) != len(raw) or not df.index.equals(raw.index):
        raise ValueError("df 与 raw 必须同索引同长度（消融矩阵按位置对齐）")

    age = pd.to_numeric(df["patch_age_days"], errors="coerce")
    na = age.isna()
    age_f = age.fillna(0.0).clip(lower=0.0)
    fresh = np.exp(-age_f.to_numpy(dtype=float) / FRESH_HALF_LIFE_DAYS)
    bpm = pd.to_numeric(df["blue_player_missing"], errors="coerce").fillna(0.0)
    rpm = pd.to_numeric(df["red_player_missing"], errors="coerce").fillna(0.0)

    out = pd.DataFrame(index=df.index)
    for col in ("side_prior", "team_elo_diff", "player_elo_diff", "bt_player_diff",
                "bt_chem_diff", "bt_league_diff", "champ_meta_diff", "mastery_diff",
                "bt_known_players"):
        out[col] = raw[col].to_numpy(dtype=float)
    out["patch_age_log"] = np.log1p(age_f.to_numpy(dtype=float))
    out["champ_meta_x_fresh"] = out["champ_meta_diff"].to_numpy() * fresh
    out["patch_age_na"] = na.to_numpy(dtype=float)
    out["player_missing_n"] = (bpm + rpm).to_numpy(dtype=float)
    out["player_missing_diff"] = (bpm - rpm).to_numpy(dtype=float)
    out["team_games_min_log"] = np.log1p(raw["team_games_min"].to_numpy(dtype=float))
    out["player_games_min_log"] = np.log1p(raw["player_games_min"].to_numpy(dtype=float))

    need = set(group_columns(list(FEATURE_GROUPS)))
    missing = need - set(out.columns)
    if missing:
        raise KeyError(f"设计矩阵缺列：{sorted(missing)}")
    return out


# ---------------------------------------------------------------------------
# 2. walk-forward 预测
# ---------------------------------------------------------------------------
def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, *, C: float) -> np.ndarray:
    """标准化器只在训练段拟合（§6.1.2 红线 3），常数列的 sd 置 1 避免除零。"""
    mu = X_tr.mean(0)
    sd = X_tr.std(0)
    sd[sd == 0] = 1.0
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=3000)
    m.fit((X_tr - mu) / sd, y_tr)
    return m.predict_proba((X_va - mu) / sd)[:, 1]


def oos_predictions(
    dev: pd.DataFrame,
    X: pd.DataFrame,
    folds: Sequence[Any],
    models: Mapping[str, Sequence[str]],
    *,
    C: float = 1.0,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """一次遍历 fold 把所有模型都训完，返回 (spine, 逐模型 OOS 概率, 逐折 AUC)。

    **一次遍历**不是为了省时间，是为了保证所有模型看到**逐行完全相同**的
    训练/验证切分。分多次跑再 join，任何一次的 fold 定义漂移都会静默地
    污染配对检验 —— 而配对检验正是本表的全部功力来源。

    `models` 的 value 是**列名**列表；空列表 = 只用训练段基准率的截距模型。
    """
    if not isinstance(dev.index, pd.RangeIndex) or dev.index.start != 0:
        raise ValueError("dev 必须 reset_index(drop=True) 后再传进来")
    y = dev.y_blue_win.to_numpy(dtype=float)
    M = X.to_numpy(dtype=float)
    ix = {c: i for i, c in enumerate(X.columns)}
    names = list(models)
    cols_ix = {n: np.array([ix[c] for c in models[n]], dtype=int) for n in names}

    parts: dict[str, list[np.ndarray]] = {n: [] for n in names}
    spines: list[pd.DataFrame] = []
    auc_rows: list[dict[str, float]] = []
    t0 = time.time()
    for k, f in enumerate(folds):
        tr = np.asarray(f.train_idx, dtype=int)
        va = np.asarray(f.valid_idx, dtype=int)
        spines.append(pd.DataFrame({
            "game_id": dev.loc[va, "game_id"].to_numpy(),
            "series_id": dev.loc[va, "series_id"].to_numpy(),
            "league_canon": dev.loc[va, "league_canon"].to_numpy(),
            "league_tier": dev.loc[va, "league_tier"].to_numpy(),
            "fold": f.idx,
            "y": y[va],
        }))
        row: dict[str, float] = {"fold": float(f.idx), "n_valid": float(len(va))}
        for n in names:
            ci = cols_ix[n]
            if len(ci) == 0:
                p = np.full(len(va), float(y[tr].mean()))
            else:
                p = _fit_predict(M[np.ix_(tr, ci)], y[tr], M[np.ix_(va, ci)], C=C)
            parts[n].append(p)
            row[n] = metrics.auc(y[va], p) if len(np.unique(y[va])) > 1 else float("nan")
        auc_rows.append(row)
        if verbose and (k + 1) % 10 == 0:
            print(f"  fold {k + 1}/{len(folds)}  {time.time() - t0:.0f}s", flush=True)
    spine = pd.concat(spines, ignore_index=True)
    preds = {n: np.concatenate(parts[n]) for n in names}
    return spine, preds, pd.DataFrame(auc_rows)


def red_line_check(
    y: np.ndarray, p: np.ndarray, fold_auc: pd.DataFrame, model: str = "FULL",
    *, min_n: int = 200,
) -> dict[str, Any]:
    """§6.9 / 预注册 §3 的三条红线。**返回而不是抛异常**，让报告必须写出结论。

    额外报 `max_fold_auc_big`（只看 n_valid ≥ min_n 的折）与触线折的规模：
    「任一 fold AUC > 0.78」这条线对**小折**几乎必然触发 —— 90 折里有 3 场的折，
    它的 AUC 只能取 {0, 0.5, 1}。把折规模摆出来，是为了让排查从"有没有泄漏"
    变成"这条线本身在这个折规模下还有没有判别力"（结论见 reports/stage8_ablation.md）。
    """
    ll = metrics.log_loss(y, p)
    acc = float(((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).mean())
    a = fold_auc[model].to_numpy(dtype=float)
    n_valid = fold_auc["n_valid"].to_numpy(dtype=float)
    mx = float(np.nanmax(a)) if len(a) else float("nan")
    j = int(np.nanargmax(a)) if len(a) else -1
    big = a[n_valid >= min_n]
    mx_big = float(np.nanmax(big)) if len(big) else float("nan")
    hits = []
    if acc > RED_LINE_ACC:
        hits.append(f"accuracy={acc:.4f} > {RED_LINE_ACC}")
    if ll < RED_LINE_LL:
        hits.append(f"log_loss={ll:.5f} < {RED_LINE_LL}")
    if mx > RED_LINE_FOLD_AUC:
        hits.append(f"max_fold_auc={mx:.4f} > {RED_LINE_FOLD_AUC}")
    return {"log_loss": ll, "accuracy": acc, "auc": metrics.auc(y, p),
            "max_fold_auc": mx, "argmax_fold": int(fold_auc["fold"].to_numpy()[j]),
            "n_valid_at_max": float(n_valid[j]),
            "max_fold_auc_big": mx_big, "min_n": min_n,
            "n_folds_over_line": int(np.nansum(a > RED_LINE_FOLD_AUC)),
            "n_folds_over_line_big": int(np.nansum(big > RED_LINE_FOLD_AUC)),
            "n_folds": len(a), "median_n_valid": float(np.median(n_valid)),
            "hits": hits, "tripped": bool(hits)}


def simulate_max_fold_auc(
    p: np.ndarray, fold: np.ndarray, *, n_sim: int = 200, seed: int = 0,
    min_n: int = 200,
) -> dict[str, float]:
    """红线「任一 fold AUC > 0.78」的**零泄漏参数化自举**。

    构造：把模型自己的 OOS 概率当成真相，重抽 `y* ~ Bernoulli(p)`。
    这样的世界里**按定义不存在任何泄漏**（标签只依赖模型已知的 p），
    于是"90 折里最大的那个 fold AUC"的分布就是纯粹的抽样噪声。
    如果实测的 max fold AUC 落在这个分布的中间，那条红线在本折规模下
    就不具备判别力 —— 这是对红线本身的检验，不是对红线的辩解：
    结论要么是"红线误报"，要么是"实测值连噪声上界都超了，确有问题"，
    两种都必须照单写进报告。
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    fold = np.asarray(fold)
    idx = [np.nonzero(fold == f)[0] for f in np.unique(fold)]
    big = np.array([len(i) >= min_n for i in idx])
    mx = np.empty(n_sim)
    mx_big = np.empty(n_sim)
    over = np.empty(n_sim)
    for s_ in range(n_sim):
        ys = (rng.random(len(p)) < p).astype(float)
        a = np.array([
            metrics.auc(ys[i], p[i]) if len(np.unique(ys[i])) > 1 else np.nan
            for i in idx
        ])
        mx[s_] = np.nanmax(a)
        mx_big[s_] = np.nanmax(a[big])
        over[s_] = np.nansum(a > RED_LINE_FOLD_AUC)
    return {"n_sim": float(n_sim),
            "p_max_over_line": float((mx > RED_LINE_FOLD_AUC).mean()),
            "p_max_big_over_line": float((mx_big > RED_LINE_FOLD_AUC).mean()),
            "max_fold_auc_p50": float(np.median(mx)),
            "max_fold_auc_p95": float(np.percentile(mx, 95)),
            "max_fold_auc_big_p50": float(np.median(mx_big)),
            "max_fold_auc_big_p95": float(np.percentile(mx_big, 95)),
            "n_over_line_mean": float(over.mean())}


# ---------------------------------------------------------------------------
# 3. 单行统计
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AblationRow:
    """一行 = 一次配对比较。字段名直接落进报告表头，别改顺序。"""

    direction: str
    group: str
    n_cols: int
    model: str
    log_loss: float
    ll_worse: float
    ll_better: float
    delta: float
    ci_lo: float
    ci_hi: float
    mde: float
    t_clustered: float
    sigma_d: float
    rho: float
    deff: float
    n_eff: float
    dm_p: float
    footer: str

    @property
    def detectable(self) -> bool:
        return abs(self.delta) > self.mde


def compare_pair(
    direction: str,
    group: str,
    n_cols: int,
    model: str,
    y: np.ndarray,
    p_worse: np.ndarray,
    p_better: np.ndarray,
    series_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> AblationRow:
    """`p_worse` 是"没有这组特征"的那一侧，`p_better` 是"有"的那一侧。

    Δ = LL(worse) − LL(better)，两张表因此共用同一个"Δ>0 = 有用"的读法。
    CI 与点估计都来自同一次 series 级 block bootstrap（共用重采样索引，§6.6.1）。
    """
    d = metrics.per_game_log_loss(y, p_worse) - metrics.per_game_log_loss(y, p_better)
    pw = power.paired_power(d, series_ids)
    boot = compare.paired_bootstrap_delta(
        y, p_worse, p_better, metrics.log_loss, series_ids, n_boot=n_boot, seed=seed
    )
    dm = compare.diebold_mariano(d, series_ids)
    ll_worse = float(metrics.log_loss(y, p_worse))
    ll_better = float(metrics.log_loss(y, p_better))
    return AblationRow(
        direction=direction,
        group=group,
        n_cols=n_cols,
        model=model,
        # 表里那一列要显示**行名所指的那个模型**的 log loss：
        # 减法行名是"−g"，指的是被削过的模型；加法行名是"加完 g 之后"的模型。
        log_loss=ll_worse if direction.startswith("A") else ll_better,
        ll_worse=ll_worse,
        ll_better=ll_better,
        delta=float(pw["mean_delta"]),
        ci_lo=float(boot.lo),
        ci_hi=float(boot.hi),
        mde=float(pw["mde"]),
        t_clustered=float(pw["t_clustered"]),
        sigma_d=float(pw["sigma_d"]),
        rho=float(pw["icc"]),
        deff=float(pw["deff"]),
        n_eff=float(pw["n_eff"]),
        dm_p=float(dm.p_value),
        footer=power.footer(pw),
    )


# ---------------------------------------------------------------------------
# 4. 两张表
# ---------------------------------------------------------------------------
def build_models(
    groups: Sequence[str] = ADDITION_ORDER,
) -> tuple[dict[str, list[str]], list[str], list[tuple[str, str, str]]]:
    """列出两张表需要的全部模型。返回 (models, 减法顺序, 加法步骤)。"""
    full = list(groups)
    models: dict[str, list[str]] = {"FULL": group_columns(full)}
    for g in full:
        models[f"−{g}"] = group_columns([x for x in full if x != g])
    steps: list[tuple[str, str, str]] = []
    prev = "BASE"
    models["BASE"] = []
    acc: list[str] = []
    for g in full:
        acc.append(g)
        name = "FULL" if len(acc) == len(full) else "+" + "+".join(acc)
        models.setdefault(name, group_columns(acc))
        steps.append((g, prev, name))
        prev = name
    for name, gs in COMBOS.items():
        models[f"−{name}"] = group_columns([x for x in full if x not in gs])
    models["L3"] = group_columns(list(L3_GROUPS))
    return models, full, steps


def ablation_tables(
    spine: pd.DataFrame,
    preds: Mapping[str, np.ndarray],
    order: Sequence[str],
    steps: Sequence[tuple[str, str, str]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    models: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, compare.BHResult]:
    """跑出 A（减法）/ A2（组合减法）/ B（加法），并在**合池**上做一次 BH。

    合池是有意的：这些行全是 exploratory，分开各控各的 FDR 等于偷偷放宽。
    primary comparison 不在这个池子里（它由 main() 单独算、单独报）。
    """
    y = spine.y.to_numpy()
    sid = spine.series_id.to_numpy()
    rows = [
        compare_pair(d, g, n, name, y, preds[w], preds[b], sid, n_boot=n_boot, seed=seed)
        for d, g, n, name, w, b in comparison_pairs(order, steps)
    ]

    # 加法表的最后一步与减法表的同名行**是同一个比较**（FULL 减去最后进来的那组
    # = 倒数第二个前缀模型）。同一个 p 值进两次 BH 池会虚增检验数、压低所有人的 q，
    # 所以按 (差模型, 好模型) 去重，只进池一次，重复行共享那个 q。
    def ident(name: str) -> tuple[str, ...]:
        # 同一套列可能有两个名字（"−missing" 与加法表倒数第二个前缀是同一个模型），
        # 所以身份用**列集合**判定，不用名字。
        return tuple(sorted(models[name])) if models else (name,)

    keys = [(ident(w), ident(b)) for _d, _g, _n, _m, w, b in comparison_pairs(order, steps)]
    uniq: dict[tuple[str, str], int] = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, i)
    pool = list(uniq.values())
    bh = compare.benjamini_hochberg([rows[i].dm_p for i in pool], alpha=alpha)
    q_by_key = {k: (float(bh.qvalues[j]), bool(bh.reject[j]))
                for j, k in enumerate(uniq)}
    name_of = {i: nm for i, (_d, _g, _n, nm, _w, _b) in enumerate(comparison_pairs(order, steps))}
    recs = []
    for i, r in enumerate(rows):
        q, rej = q_by_key[keys[i]]
        d = r.__dict__.copy()
        d["q_value"] = q
        d["bh_reject"] = rej
        d["duplicate_of"] = None if uniq[keys[i]] == i else name_of[uniq[keys[i]]]
        d["detectable"] = r.detectable
        d["verdict"] = verdict(r.delta, r.mde, rej)
        recs.append(d)
    tbl = pd.DataFrame(recs)
    part = {k: tbl[tbl.direction == k].reset_index(drop=True) for k in ("A", "A2", "B")}
    return part["A"], part["A2"], part["B"], bh


def comparison_pairs(
    order: Sequence[str], steps: Sequence[tuple[str, str, str]]
) -> list[tuple[str, str, int, str, str, str]]:
    """两张表的全部配对：(方向, 组, 列数, 行名, 差模型键, 好模型键)。

    抽出来是为了让统计表与逐折符号表**用同一份定义**，不会一处改了另一处忘改。
    """
    out: list[tuple[str, str, int, str, str, str]] = []
    for g in order:
        out.append(("A", g, len(FEATURE_GROUPS[g]), f"−{g}", f"−{g}", "FULL"))
    for name, gs in COMBOS.items():
        out.append(("A2", name, sum(len(FEATURE_GROUPS[g]) for g in gs),
                    f"−{name}", f"−{name}", "FULL"))
    for g, prev, cur in steps:
        out.append(("B", g, len(FEATURE_GROUPS[g]), cur, prev, cur))
    return out


def fold_sign_table(
    spine: pd.DataFrame,
    preds: Mapping[str, np.ndarray],
    order: Sequence[str],
    steps: Sequence[tuple[str, str, str]],
) -> pd.DataFrame:
    """逐折符号一致性 + 前后半程各自的 Δ（§6.7 门禁的 (a) 条）。

    为什么要看它：合并样本上的 Δ 可以由少数几折主导。§6.7 要求"连续两个留出期同号"，
    这里用**按时间切成前后两半**来近似那两期 —— 一个 Δ 只要有一半是反号的，
    哪怕合并后过了 MDE，也不该当成稳定增量。
    """
    y = spine.y.to_numpy()
    fold = spine.fold.to_numpy()
    half = fold >= np.median(fold)
    recs = []
    for d, g, _n, name, w, b in comparison_pairs(order, steps):
        dd = metrics.per_game_log_loss(y, preds[w]) - metrics.per_game_log_loss(y, preds[b])
        per_fold = pd.Series(dd).groupby(fold).mean()
        recs.append({
            "direction": d, "group": g, "model": name,
            "delta": float(dd.mean()),
            "frac_folds_positive": float((per_fold > 0).mean()),
            "n_folds": int(per_fold.size),
            "delta_first_half": float(dd[~half].mean()),
            "delta_second_half": float(dd[half].mean()),
            "same_sign_halves": bool(np.sign(dd[~half].mean()) == np.sign(dd[half].mean())),
        })
    return pd.DataFrame(recs)


def verdict(delta: float, mde: float, bh_reject: bool) -> str:
    """裁决的唯一出口。**低于 MDE 的行不许出现"改进"/"变差"字样。**"""
    if abs(delta) <= mde:
        return "【低于本样本量的可检出下限】"
    if not bh_reject:
        return "过 MDE 但 BH-FDR 未通过（探索性，不作结论）"
    return "有增量 ✓" if delta > 0 else "有害 ✗"


def disagreements(tbl_a: pd.DataFrame, tbl_b: pd.DataFrame) -> pd.DataFrame:
    """找出两表方向不一致的组 —— 共线的指纹，报告里必须逐条解释。

    不一致的判定**不用符号**（低于 MDE 的行符号是噪声），用三档：
    `+`（Δ>MDE）/ `0`（|Δ|≤MDE）/ `−`（Δ<−MDE），两表档位不同即为不一致。
    """
    def band(row: pd.Series) -> str:
        if abs(row.delta) <= row.mde:
            return "0"
        return "+" if row.delta > 0 else "−"

    a = tbl_a.set_index("group")
    b = tbl_b.set_index("group")
    recs = []
    for g in a.index:
        ba, bb = band(a.loc[g]), band(b.loc[g])
        recs.append({"group": g, "band_A": ba, "band_B": bb,
                     "delta_A": float(a.loc[g, "delta"]), "delta_B": float(b.loc[g, "delta"]),
                     "mde_A": float(a.loc[g, "mde"]), "mde_B": float(b.loc[g, "mde"]),
                     "agree": ba == bb})
    return pd.DataFrame(recs)


def group_collinearity(X: pd.DataFrame) -> pd.DataFrame:
    """每一列被**其余所有列**线性解释的 R²（= 1 − 1/VIF）。

    这是"两表为什么不一致"的直接证据：LOO 的 Δ 之所以塌成 0，
    是因为拿掉的那一列可以被剩下的列线性重建（R² → 1），
    模型换个系数就把它补回来了。报告里必须把这个数摆在不一致那一行旁边。
    """
    A = X.to_numpy(dtype=float)
    A = (A - A.mean(0)) / np.where(A.std(0) == 0, 1.0, A.std(0))
    n, k = A.shape
    recs = []
    for j in range(k):
        rest = np.delete(A, j, axis=1)
        rest = np.hstack([np.ones((n, 1)), rest])
        beta, *_ = np.linalg.lstsq(rest, A[:, j], rcond=None)
        resid = A[:, j] - rest @ beta
        sst = float((A[:, j] ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else float("nan")
        recs.append({"col": X.columns[j], "r2_on_rest": r2,
                     "vif": float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)})
    out = pd.DataFrame(recs)
    col2grp = {c: g for g, cs in FEATURE_GROUPS.items() for c in cs}
    out["group"] = out.col.map(col2grp)
    return out


# ---------------------------------------------------------------------------
# 5. 渲染
# ---------------------------------------------------------------------------
_HEAD = ("| 组 | 列数 | 模型 log loss | ΔLL | 95% CI (series bootstrap) | MDE | "
         "\\|t\\| | DM p | BH q | 裁决 |\n|---|---|---|---|---|---|---|---|---|---|")


def render_table(tbl: pd.DataFrame) -> str:
    lines = [_HEAD]
    for r in tbl.itertuples(index=False):
        lines.append(
            f"| `{r.group}` | {r.n_cols} | {r.log_loss:.5f} | {r.delta:+.5f} | "
            f"[{r.ci_lo:+.5f}, {r.ci_hi:+.5f}] | {r.mde:.5f} | {abs(r.t_clustered):.2f} | "
            f"{r.dm_p:.3g} | {r.q_value:.3g} | {r.verdict} |"
        )
    return "\n".join(lines)


def render_footers(tbl: pd.DataFrame) -> str:
    """每行自己的功效画像。表脚必须逐行打，全局一行会掩盖 σ_d 的量级差异。"""
    return "\n".join(f"- `{r.group}`：{r.footer}" for r in tbl.itertuples(index=False))


def render_report(
    tbl_a: pd.DataFrame,
    tbl_a2: pd.DataFrame,
    tbl_b: pd.DataFrame,
    dis: pd.DataFrame,
    meta: Mapping[str, Any],
) -> str:
    """把两张表 + 表脚 + 不一致清单拼成 reports/stage8_ablation.md 的正文骨架。"""
    lines = [
        "# 阶段 8 · 双向消融矩阵（验收②）",
        "",
        (f"口径 {meta['scope']} ｜ {meta['n_games']:,} 场 OOS / {meta['n_series']:,} series"
         f" ／ {meta['n_folds']} 折 walk-forward ｜ holdout 由 `load_dev()` 物理排除"),
        "",
        "## 表 A · 减法（leave-one-out，基准 = FULL）",
        "",
        render_table(tbl_a),
        "",
        "表脚（`eval.power.footer()`，逐行）：",
        "",
        render_footers(tbl_a),
        "",
        "### 表 A2 · 组合减法（共线的组一起拿掉）",
        "",
        render_table(tbl_a2),
        "",
        render_footers(tbl_a2),
        "",
        "## 表 B · 加法（正向增量，顺序预先写死）",
        "",
        render_table(tbl_b),
        "",
        "表脚（`eval.power.footer()`，逐行）：",
        "",
        render_footers(tbl_b),
        "",
        "## 两表不一致 = 共线指纹",
        "",
        "| 组 | A 档 | B 档 | ΔLL_A | ΔLL_B | 一致 |",
        "|---|---|---|---|---|---|",
    ]
    for r in dis.itertuples(index=False):
        lines.append(f"| `{r.group}` | {r.band_A} | {r.band_B} | {r.delta_A:+.5f} | "
                     f"{r.delta_B:+.5f} | {'是' if r.agree else '**否**'} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def main() -> None:
    from lolwp.eval.walkforward import WFConfig, make_folds
    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.hier_bt import HierBTConfig

    ap = argparse.ArgumentParser(description="阶段 8 双向消融矩阵")
    ap.add_argument("--min-tier", type=int, default=1)
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--bt-grid", type=int, default=14)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", default=",".join(ADDITION_ORDER),
                    help="加法表的进入顺序（逗号分隔）。换顺序是共线性的直接检验："
                         "若某组的 Δ 随顺序剧变，它携带的信息就不是它自己的")
    ap.add_argument("--n-sim", type=int, default=200,
                    help="红线触发时跑多少次零泄漏参数化自举；0 = 不跑")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--out-json", default="reports/stage8_ablation.json")
    ap.add_argument("--out-md", default=None, help="给了就把表骨架写出去")
    ap.add_argument("--feature-cache", default=None,
                    help="priors+draft_priors 的 parquet（与 load_dev() 全量同行序）")
    a = ap.parse_args()

    t0 = time.time()
    full = load_dev()
    if a.feature_cache:
        raw = pd.read_parquet(a.feature_cache)
        raw.index = full.index
    else:
        raw = pd.concat([run_state_machine(full), run_draft_state_machine(full)], axis=1)
    bt = rolling_ratings_cached(full, HierBTConfig(refit_every_days=a.bt_grid))
    # 缓存文件可能已经含 bt_* 列（别的阶段写的）。直接 concat 会产生**同名重复列**，
    # 之后 raw[col] 返回 DataFrame 而不是 Series，静默地把设计矩阵拼错。
    # 以**现算的 BT 为准**覆盖缓存里的同名列：BT 的网格由 --bt-grid 决定，
    # 缓存里那份未必是同一个网格。
    raw = pd.concat([raw.drop(columns=[c for c in bt.columns if c in raw.columns]), bt], axis=1)
    print(f"特征就绪 {time.time() - t0:.0f}s  {raw.shape}", flush=True)

    m = (full.league_tier >= a.min_tier) & (full.kickoff_utc >= a.since)
    dev = full[m].reset_index(drop=True)
    X = assemble_features(full[m], raw.loc[full[m].index]).reset_index(drop=True)
    folds = make_folds(dev, WFConfig())
    models, order, steps = build_models(tuple(a.order.split(",")))
    print(f"{len(dev):,} 场  {len(folds)} 折  {len(models)} 个模型", flush=True)

    spine, preds, fold_auc = oos_predictions(dev, X, folds, models, C=a.C, verbose=True)
    print(f"OOS {len(spine):,} 场  拟合耗时 {time.time() - t0:.0f}s", flush=True)

    rl = red_line_check(spine.y.to_numpy(), preds["FULL"], fold_auc)
    print(f"红线：{rl}", flush=True)
    if rl["tripped"] and a.n_sim:
        rl["null_sim"] = simulate_max_fold_auc(
            preds["FULL"], spine.fold.to_numpy(), n_sim=a.n_sim, seed=a.seed)
        print(f"零泄漏模拟：{rl['null_sim']}", flush=True)

    tbl_a, tbl_a2, tbl_b, bh = ablation_tables(spine, preds, order, steps,
                                               n_boot=a.n_boot, seed=a.seed, models=models)
    dis = disagreements(tbl_a, tbl_b)
    primary = compare_pair("primary", "FULL_vs_L3", len(models["FULL"]), "FULL",
                           spine.y.to_numpy(), preds["L3"], preds["FULL"],
                           spine.series_id.to_numpy(), n_boot=a.n_boot, seed=a.seed)

    meta = {"scope": f"tier≥{a.min_tier} / {a.since} 起", "order": list(order),
            "n_games": len(spine),
            "n_series": int(spine.series_id.nunique()), "n_folds": len(folds),
            "red_line": rl, "bh_n_tests": bh.n_tests, "bh_n_reject": bh.n_reject,
            "excluded": EXCLUDED}
    out = {"meta": meta, "primary": primary.__dict__,
           "table_a": tbl_a.to_dict("records"), "table_a2": tbl_a2.to_dict("records"),
           "table_b": tbl_b.to_dict("records"),
           "disagreements": dis.to_dict("records"),
           "collinearity": group_collinearity(X).to_dict("records"),
           "fold_delta_sign": fold_sign_table(spine, preds, order, steps).to_dict("records"),
           "per_model": {n: metrics.evaluate(spine.y.to_numpy(), preds[n]) for n in models},
           "fold_auc_max": {n: float(fold_auc[n].max()) for n in models}}
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=float)
    print(f"写出 {a.out_json}", flush=True)
    if a.out_md:
        with open(a.out_md, "w") as fh:
            fh.write(render_report(tbl_a, tbl_a2, tbl_b, dis, meta))
        print(f"写出 {a.out_md}", flush=True)
    print(render_table(tbl_a))
    print()
    print(render_table(tbl_a2))
    print()
    print(render_table(tbl_b))
    print(f"\nprimary FULL vs L3：Δ={primary.delta:+.5f} "
          f"[{primary.ci_lo:+.5f},{primary.ci_hi:+.5f}] MDE={primary.mde:.5f} "
          f"|t|={abs(primary.t_clustered):.2f} p={primary.dm_p:.3g}")
    print(f"  {primary.footer}")
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
