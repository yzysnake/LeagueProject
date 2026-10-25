"""生产模型 artifact：训练 + 打包 + 加载 + 每日重解实力快照。

artifact 是自描述的（PLAN.md §4.7.2）：里面记着训练数据范围、快照 sha、
特征列表、模型口径。服务启动时断言 artifact 与代码版本一致，不一致直接拒绝启动，
而不是打个警告然后用错误的特征跑。

**实力快照不能冻结。** 推理用的分层 BT 快照只吃快照时刻之前的比赛；它每老一周 log loss 约差 0.004、
老一个月差 0.012（reports/exp_model_improvements.md §1.2），而外层系数是稳定的。所以 `refresh()`
只重解快照（几秒），不重训外层；数据 = silver 全量 + 盯盘台账里 OE 还没收录的局（`ledger.live_games`）。

**快照的训练标签是赛后经济差的软标签**（全量口径，预注册修订 A6，`models.margin`）：
同一个 BT、同一组外层特征，似然里的 y 从胜负换成 σ(0.5·终局经济差 / 10 分钟)，验收段 +0.00390。
用的是哪个标签记在 `meta.rating_target`；refresh 照它重解，不会把软标签快照换回胜负快照。

**外层之上还有一层近期状态**（全量口径，`models.form`，reports/exp_model_improvements.md §3）：
logit 再加 β ·（两队最近几局「赛果 − 模型预期」之差），状态存在 `prior_state["form"]`——
放在已有的 dict 里而不是新加一个字段，正在跑的旧代码照样读得了新 artifact（只是不加这一层）。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ARTIFACT_DIR = "models"
LIVE_LEDGER = "reports/live_watch.jsonl"
# 快照要用的列（silver 与台账补充的局共有）；标签不是胜负时再加上那一列
RATING_COLUMNS = ["game_id", "kickoff_utc", "league_canon", "league_tier", "blue_team_id",
                  "red_team_id", "blue_player_ids", "red_player_ids", "y_blue_win"]
WIN_TARGET = "y_blue_win"
# OE 的局只有开赛时间：按 75 分钟估一个「这局肯定打完了」的时刻（职业赛超过 75 分钟的极少）
_GAME_SPAN = pd.Timedelta(minutes=75)
# 各规格在开发期 walk-forward 上的成绩（tier>=1、2019 起、23,060 场；只走 load_dev，holdout 没碰）
OOS_WIN = {"oos_log_loss": 0.61650,
           "oos_note": "walk-forward OOS, tier>=1, 2019 起, 23,060 场，3 天网格、胜负标签"
                       "（reports/exp_model_improvements.md §1.1；阶段 8 报告里的 0.61917 是 14 天网格）"}
OOS_MARGIN = {"oos_log_loss": 0.61259,
              "oos_note": "walk-forward OOS, tier>=1, 2019 起, 23,060 场，3 天网格、赛后经济差软标签"
                          "（scripts/exp_margin_bt.py，reports/exp_model_improvements.md §2）"}
OOS_FORM = {"oos_log_loss": 0.61167,
            "oos_note": "walk-forward OOS, tier>=1, 2019 起, 23,060 场，3 天网格、赛后经济差软标签 + 近期状态第二层"
                        "（scripts/exp_form.py，reports/exp_model_improvements.md §3）"}
# 近期状态的 β 只在这之后的 tier>=1 局上拟合：与 exp_form 验收段的走前向同一段（调参段的 β 明显更大，时代不同）
FORM_SINCE = "2019-01-01"
FORM_KEEP_DAYS = 730          # 两年没打过比赛的队不带进 artifact
# 产品模型 = BT4_LR（阶段 8 对比表：7 个候选里无一在 MDE 之上胜出）。
#   · 分层 BT 的选手潜变量取代了 team_elo / player_elo——阶段 8 消融实测，
#     有了 bt_player 之后拿掉那两列毫无影响（−0.00004 / −0.00016，均低于 MDE），
#     所以生产模型直接不带它们，省一遍状态机。
#   · L5 的 mastery_diff 在全部 ICC 假设下都低于 MDE，不进主力，
#     但仍算出来给 UI 展示（§4.3.3）。
#   · 协同/克制已在阶段 7 被证伪（ΔLL −0.00042），不进。
PROD_FEATURES = [
    "side_prior", "bt_player_diff", "bt_chem_diff", "bt_league_diff", "champ_meta_diff",
]
LAYER_OF = {
    "side_prior": "赛况",
    "bt_player_diff": "实力",
    "bt_chem_diff": "实力",
    "bt_league_diff": "实力",
    "champ_meta_diff": "阵容",
    "form_diff": "实力",            # 第二层：近期状态（models.form），不在外层五个特征里
}


@dataclass
class Artifact:
    coef: dict[str, float]
    intercept: float
    mu: dict[str, float]
    sigma: dict[str, float]
    prior_state: dict[str, Any]
    draft_state: dict[str, Any]
    bt_state: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.coef)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(
                {
                    "coef": self.coef, "intercept": self.intercept,
                    "mu": self.mu, "sigma": self.sigma,
                    "prior_state": self.prior_state, "draft_state": self.draft_state,
                    "bt_state": self.bt_state, "meta": self.meta,
                },
                fh,
            )
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> Artifact:
        with open(path) as fh:
            d = json.load(fh)
        return Artifact(**d)


def rating_games(full: pd.DataFrame, ledger: str | None = LIVE_LEDGER,
                 target: str = WIN_TARGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """实力快照吃的比赛：silver（训练口径的全量）+ 台账里判出胜负、OE 还没收录的局。

    返回 (按开赛时间升序的合并表, 其中来自台账的那部分)。ledger=None 时只用 silver。
    target 不是胜负时 full 必须已带那一列（`models.margin.with_margin_target`）；台账的局自带。
    """
    from lolwp.live.ledger import LIVE_COLUMNS, live_games

    cols = RATING_COLUMNS + ([target] if target != WIN_TARGET else [])
    live = live_games(full, ledger) if ledger else pd.DataFrame(columns=LIVE_COLUMNS)
    games = full[cols]
    if len(live):
        games = pd.concat([games, live[cols]], ignore_index=True)
    return games.sort_values("kickoff_utc", kind="mergesort").reset_index(drop=True), live


def _rating_snapshot(games: pd.DataFrame, live: pd.DataFrame, cfg, warm=None, target: str = WIN_TARGET):
    """在全部已知比赛上解一次快照。返回 (快照, 写进 meta 的新鲜度字段)。

    ratings_cutoff = 快照里最晚一局「肯定打完了」的时刻（台账的局用终局时间，OE 的局按开赛 + 75 分钟）：
    开局早于它的局再补录预测，就用上了它开打之后才有的赛果——补录据此跳过（scripts/live_watch.py）。
    """
    from lolwp.live.ledger import LIVE_PREFIX
    from lolwp.models.hier_bt import fit_snapshot

    as_of = games.kickoff_utc.max() + pd.Timedelta(seconds=1)
    snap = fit_snapshot(games, as_of, cfg, warm=warm, target=target)
    oe = games.kickoff_utc[~games.game_id.astype(str).str.startswith(LIVE_PREFIX)]
    ends = [oe.max() + _GAME_SPAN] if len(oe) else []
    if len(live):
        ends.append(live.ended_utc.fillna(live.kickoff_utc + _GAME_SPAN).max())
    meta = {"ratings_as_of": str(snap.as_of), "ratings_cutoff": str(max(ends)),
            "n_live_games": len(live)}
    return snap, meta


def _live_form(live: pd.DataFrame) -> dict[str, list[list]]:
    """台账那部分的近期状态条目：tier>=1、有开赛前落盘预测的局，残差用当时的预测（ledger.live_games 的 p_pre）。"""
    from lolwp.models.form import DEFAULT, form_entries

    lv = live[(live.league_tier >= 1) & live.p_pre.notna()] if len(live) else live
    return form_entries(lv, lv.p_pre.to_numpy(dtype=float), DEFAULT) if len(lv) else {}


def _form_state(games: pd.DataFrame, z: np.ndarray, live: pd.DataFrame) -> dict[str, Any]:
    """近期状态第二层（models.form，reports/exp_model_improvements.md §3）的 β 与推理状态。

    games 是外层的训练局（tier>=1，按开赛升序），z 是外层在它们上的 logit。OE 那部分的残差用外层的样本内预测：
    5 个参数拟 3.6 万场，样本内外差在 1e-4 量级；分层 BT 特征本身是 as-of 的，残差里没有赛果泄漏。
    台账那部分用开赛前落盘的预测（_live_form），每次 refresh 重建。
    """
    from lolwp.models.form import DEFAULT, fit_beta, form_diff, form_entries

    g = games.reset_index(drop=True)
    p = 1.0 / (1.0 + np.exp(-z))
    y = g.y_blue_win.to_numpy(dtype=float)
    recent = (g.kickoff_utc >= pd.Timestamp(FORM_SINCE, tz="UTC")).to_numpy()
    beta = fit_beta(z[recent], form_diff(g, p, DEFAULT)[recent], y[recent])
    since = g.kickoff_utc.max() - pd.Timedelta(days=FORM_KEEP_DAYS)
    return {"cfg": asdict(DEFAULT), "beta": float(beta), "n_fit": int(recent.sum()),
            "silver": form_entries(g, p, DEFAULT, since), "live": _live_form(live)}


def train(out: str = f"{ARTIFACT_DIR}/prod.json", *, scope: str = "all") -> Artifact:
    """训练并打包生产模型。

    scope="all"（默认）：holdout 揭盲之后的生产口径，用全部数据（预注册修订 A4），
    实力快照用赛后经济差软标签（修订 A6）；
    scope="dev"：只用 holdout 之前的数据、胜负标签，即揭盲时评测的那个模型（原样复现用）。
    两种口径的外层特征与超参完全一样。
    """
    from sklearn.linear_model import LogisticRegression

    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_all, load_dev
    from lolwp.features.priors import run_state_machine
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.hier_bt import HierBTConfig
    from lolwp.models.margin import MARGIN_KAPPA, with_margin_target
    from lolwp.models.margin import TARGET as MARGIN_TARGET

    if scope not in ("all", "dev"):
        raise ValueError(f"scope 只能是 all / dev，收到 {scope!r}")
    target = MARGIN_TARGET if scope == "all" else WIN_TARGET
    full = load_all() if scope == "all" else load_dev()
    if target == MARGIN_TARGET:
        full = with_margin_target(full)
    f1, st1 = run_state_machine(full, return_state=True)
    f2, st2 = run_draft_state_machine(full, return_state=True)
    cfg_bt = HierBTConfig(refit_every_days=3)
    fbt = rolling_ratings_cached(full, cfg_bt, target=target)
    feats = pd.concat([f1, f2, fbt], axis=1)
    # 推理用的最终快照：训练截止之前的全部比赛；全量口径再加上台账里 OE 还没收录的局（同 refresh）。
    # dev 口径不碰台账：台账的局全在 holdout 期之后
    games, live = rating_games(full, LIVE_LEDGER if scope == "all" else None, target)
    bt_snap, fresh = _rating_snapshot(games, live, cfg_bt, target=target)

    # 训宽评窄：拟合只用 tier>=1（同质池子），状态机跑全赛区
    m = full.league_tier >= 1
    X = feats.loc[m, PROD_FEATURES].to_numpy(dtype=float)
    y = full.loc[m, "y_blue_win"].to_numpy()
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit((X - mu) / sd, y)
    # 第二层：近期状态（只给全量口径；dev 口径原样复现揭盲时评测的模型）
    form = _form_state(full.loc[m], clf.decision_function((X - mu) / sd), live) if scope == "all" else None
    if form is not None:
        st1["form"] = form

    art = Artifact(
        coef=dict(zip(PROD_FEATURES, clf.coef_[0].tolist())),
        intercept=float(clf.intercept_[0]),
        mu=dict(zip(PROD_FEATURES, mu.tolist())),
        sigma=dict(zip(PROD_FEATURES, sd.tolist())),
        prior_state=st1,
        draft_state=st2,
        # 推理用的 BT 快照（之后由 refresh() 随赛果重解）。推理走 HierBTSnapshot.features()——
        # 与训练**同一个函数**，这是 §7.4 防 train-serve skew 的关键。
        bt_state=bt_snap.to_dict(),
        meta={
            # 规格名跟着快照标签与第二层走：台账按它区分新旧规格的预测（同一个训练截止日也分得开）
            "model": "BT4_LR+MOV+FORM" if form is not None else "BT4_LR+MOV" if target == MARGIN_TARGET
                     else "BT4_LR",
            "trained_on": str(full.kickoff_utc.max()),
            "n_train_games": int(m.sum()),
            "n_state_games": len(full),
            "holdout_start": str(full.attrs.get("holdout_start")),
            # walk-forward 是本规格（网格、标签）在开发期的成绩；holdout 属于揭盲时评测的 dev 口径模型
            **(OOS_FORM if form is not None else OOS_MARGIN if target == MARGIN_TARGET else OOS_WIN),
            "holdout_log_loss": 0.61653,
            "holdout_note": "final holdout 989 场，一次性揭盲于 git 437d1cb（reports/final_technical_report.md §7）；"
                            "评测对象是 dev 口径（胜负标签、训练截至 2026-05-31）的模型",
            # 训练口径。holdout 只对 dev 口径的模型成立；all 口径在 holdout 期是样本内
            "data_scope": scope,
            "rating_target": target,
            **({"margin_kappa": MARGIN_KAPPA} if target == MARGIN_TARGET else {}),
            **({"form_beta": form["beta"], "form_half_life_games": form["cfg"]["half_life_games"]}
               if form is not None else {}),
            # 实力快照的新鲜度：refresh() 之后这几项会变，外层系数与 trained_on 不变
            **fresh, "ratings_n_games": len(games),
            **({"refit_note": "揭盲后的生产重训（预注册修订 A4）：外层五个特征不变、全量数据；实力快照的训练标签"
                              "改为赛后经济差软标签（修订 A6）；外层之上加近期状态第二层（修订 A7）。walk-forward "
                              "是本规格在开发期的成绩，holdout 是揭盲时评测的 dev 口径模型的成绩；本模型自己的"
                              "样本外成绩看盯盘台账。"}
               if scope == "all" else {}),
        },
    )
    art.save(out)
    return art


def refresh(path: str = f"{ARTIFACT_DIR}/prod.json", *, ledger: str | None = LIVE_LEDGER,
            force: bool = False) -> Artifact | None:
    """只重解实力快照：silver 全量 + 台账里判出胜负、OE 还没收录的局。外层系数与其余状态一律不动。

    没有新比赛（快照时刻与场数都没变）时什么都不写、返回 None——盯盘每次对账完都会调一次。
    只给全量口径（揭盲后的生产模型，修订 A4）用：dev 口径的 artifact 是复现揭盲用的，冻结才对。
    写回前再看一眼文件：读进来之后被 `make artifact` 换过，就放弃这一次，免得把新训的模型盖回旧系数。
    """
    from lolwp.features.loader import load_all
    from lolwp.models.hier_bt import HierBTSnapshot
    from lolwp.models.margin import TARGET as MARGIN_TARGET
    from lolwp.models.margin import with_margin_target

    m0 = os.path.getmtime(path)
    art = Artifact.load(path)
    if art.meta.get("data_scope") != "all":
        raise RuntimeError("只有全量口径（data_scope=all）的 artifact 才重解实力快照：dev 口径是复现揭盲用的")
    # 照训练时的标签重解：外层系数是在这个标签的快照特征上训的，换标签就是换了一个没评测过的模型
    target = art.meta.get("rating_target", WIN_TARGET)
    full = load_all()
    if target == MARGIN_TARGET:
        full = with_margin_target(full)
    games, live = rating_games(full, ledger, target)
    old = HierBTSnapshot.from_dict(art.bt_state)
    as_of = games.kickoff_utc.max() + pd.Timedelta(seconds=1)
    if not force and as_of <= old.as_of and art.meta.get("ratings_n_games") == len(games):
        return None
    snap, fresh = _rating_snapshot(games, live, old.cfg, warm=old, target=target)
    art.bt_state = snap.to_dict()
    if "form" in art.prior_state:          # 近期状态：台账那部分跟着重建，β 与 OE 那部分不动
        art.prior_state["form"]["live"] = _live_form(live)
    art.meta.update(fresh, ratings_n_games=len(games),
                    ratings_refreshed_at=str(pd.Timestamp.now(tz="UTC").floor("s")))
    if os.path.getmtime(path) != m0:
        return None
    art.save(path)
    return art


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "dev"], default="all",
                    help="all = 揭盲后的生产口径（全量数据）；dev = 揭盲时评测的那个模型")
    ap.add_argument("--refresh", action="store_true",
                    help="不重训，只用 silver + 盯盘台账重解实力快照（几秒）")
    a = ap.parse_args()
    if a.refresh:
        r = refresh(force=True)
        print("实力快照未更新（artifact 在此期间被换过）" if r is None else
              f"实力快照 → {r.meta['ratings_as_of']}（含盯盘补充 {r.meta['n_live_games']} 场）")
        raise SystemExit(0)
    art = train(scope=a.scope)
    print(f"artifact → {ARTIFACT_DIR}/prod.json")
    print(json.dumps(art.meta, ensure_ascii=False, indent=2))
    print("系数:", {k: round(v, 4) for k, v in art.coef.items()}, "截距", round(art.intercept, 4))
