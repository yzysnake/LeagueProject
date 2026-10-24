"""生产模型 artifact：训练 + 打包 + 加载。

artifact 是自描述的（PLAN.md §4.7.2）：里面记着训练数据范围、快照 sha、
特征列表、模型口径。服务启动时断言 artifact 与代码版本一致，不一致直接拒绝启动，
而不是打个警告然后用错误的特征跑。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

ARTIFACT_DIR = "models"
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


def train(out: str = f"{ARTIFACT_DIR}/prod.json", *, scope: str = "all") -> Artifact:
    """训练并打包生产模型。

    scope="all"（默认）：holdout 揭盲之后的生产口径，用全部数据（预注册修订 A4）；
    scope="dev"：只用 holdout 之前的数据，即揭盲时评测的那个模型（复现用）。
    两种口径的特征、超参、外层模型完全一样，只差训练数据的截止时间。
    """
    from sklearn.linear_model import LogisticRegression

    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_all, load_dev
    from lolwp.features.priors import run_state_machine
    from lolwp.models.bt_cache import rolling_ratings_cached
    from lolwp.models.hier_bt import HierBTConfig

    if scope not in ("all", "dev"):
        raise ValueError(f"scope 只能是 all / dev，收到 {scope!r}")
    full = load_all() if scope == "all" else load_dev()
    f1, st1 = run_state_machine(full, return_state=True)
    f2, st2 = run_draft_state_machine(full, return_state=True)
    cfg_bt = HierBTConfig(refit_every_days=3)
    fbt = rolling_ratings_cached(full, cfg_bt)
    feats = pd.concat([f1, f2, fbt], axis=1)
    # 推理用的最终快照：只吃训练截止时刻之前的比赛
    from lolwp.models.hier_bt import fit_snapshot
    bt_snap = fit_snapshot(full, full.kickoff_utc.max() + pd.Timedelta(seconds=1), cfg_bt)

    # 训宽评窄：拟合只用 tier>=1（同质池子），状态机跑全赛区
    m = full.league_tier >= 1
    X = feats.loc[m, PROD_FEATURES].to_numpy(dtype=float)
    y = full.loc[m, "y_blue_win"].to_numpy()
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit((X - mu) / sd, y)

    art = Artifact(
        coef=dict(zip(PROD_FEATURES, clf.coef_[0].tolist())),
        intercept=float(clf.intercept_[0]),
        mu=dict(zip(PROD_FEATURES, mu.tolist())),
        sigma=dict(zip(PROD_FEATURES, sd.tolist())),
        prior_state=st1,
        draft_state=st2,
        # 训练截止时刻的 BT 快照。推理走 HierBTSnapshot.features()——
        # 与训练**同一个函数**，这是 §7.4 防 train-serve skew 的关键。
        bt_state=bt_snap.to_dict(),
        meta={
            "model": "BT4_LR",
            "trained_on": str(full.kickoff_utc.max()),
            "n_train_games": int(m.sum()),
            "n_state_games": len(full),
            "holdout_start": str(full.attrs.get("holdout_start")),
            "oos_log_loss": 0.61917,
            "oos_note": "walk-forward OOS, tier>=1, 2019 起, 23,060 场（reports/stage8_comparison.md）",
            "holdout_log_loss": 0.61653,
            "holdout_note": "final holdout 989 场，一次性揭盲于 git 437d1cb（reports/final_technical_report.md §7）",
            # 训练口径。oos / holdout 两个数字只对 dev 口径的模型成立；all 口径在 holdout 期是样本内
            "data_scope": scope,
            **({"refit_note": "揭盲后的生产重训（预注册修订 A4）：同一模型规格、全量数据。"
                              "上面两个 log loss 是 dev 口径模型的成绩，本模型的样本外成绩看盯盘台账。"}
               if scope == "all" else {}),
        },
    )
    art.save(out)
    return art


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "dev"], default="all",
                    help="all = 揭盲后的生产口径（全量数据）；dev = 揭盲时评测的那个模型")
    a = train(scope=ap.parse_args().scope)
    print(f"artifact → {ARTIFACT_DIR}/prod.json")
    print(json.dumps(a.meta, ensure_ascii=False, indent=2))
    print("系数:", {k: round(v, 4) for k, v in a.coef.items()}, "截距", round(a.intercept, 4))
