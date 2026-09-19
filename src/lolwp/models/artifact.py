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
# 产品模型 = L4。L5 的 mastery_diff 在全部 ICC 假设下都低于 MDE（阶段 4 报告），
# 故不进主力；但 mastery 仍然算出来给 UI 展示（§4.3.3）。
PROD_FEATURES = ["side_prior", "team_elo_diff", "player_elo_diff", "champ_meta_diff"]
LAYER_OF = {
    "side_prior": "赛况",
    "team_elo_diff": "实力",
    "player_elo_diff": "实力",
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
                    "meta": self.meta,
                },
                fh,
            )
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> Artifact:
        with open(path) as fh:
            d = json.load(fh)
        return Artifact(**d)


def train(out: str = f"{ARTIFACT_DIR}/prod.json") -> Artifact:
    from sklearn.linear_model import LogisticRegression

    from lolwp.features.draft_priors import run_draft_state_machine
    from lolwp.features.loader import load_dev
    from lolwp.features.priors import run_state_machine

    full = load_dev()
    f1, st1 = run_state_machine(full, return_state=True)
    f2, st2 = run_draft_state_machine(full, return_state=True)
    feats = pd.concat([f1, f2], axis=1)

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
        meta={
            "model": "L4",
            "trained_on": str(full.kickoff_utc.max()),
            "n_train_games": int(m.sum()),
            "n_state_games": len(full),
            "holdout_start": str(full.attrs.get("holdout_start")),
            "oos_log_loss": 0.6307,
            "oos_note": "walk-forward OOS, tier>=1, 2019 起, 23,060 场（见 reports/stage4_baseline_ladder.md）",
        },
    )
    art.save(out)
    return art


if __name__ == "__main__":
    a = train()
    print(f"artifact → {ARTIFACT_DIR}/prod.json")
    print(json.dumps(a.meta, ensure_ascii=False, indent=2))
    print("系数:", {k: round(v, 4) for k, v in a.coef.items()}, "截距", round(a.intercept, 4))
