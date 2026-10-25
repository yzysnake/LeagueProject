"""近期状态（form）：每队最近几局「赛果 − 模型预期」的 EWMA，作为外层之上的第二层（reports/exp_model_improvements.md §3）。

    form_T = Σ_k a^k · r_k / (Σ_k a^k + prior)，r = y − p（该队视角），k = 0 是最近一局，a = 0.5^(1/半衰期)
    logit q = logit p + β · (form_蓝 − form_红)

r 里的 p 是模型自己对那一局的预测：分层 BT 是长记忆（半衰期 270 天）的实力，form 补的是它追不上的
近几局的起落——包括**本系列已打完的局**（第 2 局开打前第 1 局的赛果已经知道）。
prior 是收缩：历史不足的队伍 form 往 0 收。

时间正确性：一局的残差只进开赛晚于它的局的 form；同一时刻开赛的几局互相看不见。
训练与推理共用 `team_form` 这一个函数（§7.4 防 train-serve skew）。

推理侧的状态（存在 artifact 的 `prior_state["form"]`）：每队最近 `window` 局的 (开赛时刻, 残差)，
OE 那部分训练时定死（`silver`），盯盘台账那部分每次重解实力快照时重建（`live`，残差用当时落盘的预测）。
取值时只用开赛早于「现在」的条目，补录站在开局时刻预测也不会用上之后的局。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FormConfig:
    half_life_games: float = 5.0
    prior_games: float = 3.0
    window: int = 60               # 只看最近这么多局：第 60 局的权重是 0.5^(59/5) ≈ 0.0003
    include_series: bool = True    # 本系列已打完的局算不算（False 只用于对照实验）

    @property
    def alpha(self) -> float:
        return 0.5 ** (1.0 / self.half_life_games)


DEFAULT = FormConfig()   # 调参段在 6 个设定里选出的（scripts/exp_form.py）：半衰期 5 局、含本系列已打完的局


def team_form(resid: Sequence[float], cfg: FormConfig = DEFAULT) -> float:
    """一支队的 form。resid 按时间从早到晚、只含开赛早于当前局的那些局（该队视角的 y − p）。"""
    num = den = 0.0
    w = 1.0
    for r in reversed(resid[-cfg.window:]):
        num += w * r
        den += w
        w *= cfg.alpha
    return num / (den + cfg.prior_games)


def form_diff(games: pd.DataFrame, p: np.ndarray, cfg: FormConfig = DEFAULT) -> np.ndarray:
    """逐场 form_蓝 − form_红（as-of）。games 按开赛升序，含 blue/red_team_id、series_id、kickoff_utc、
    y_blue_win；p 是同序的模型蓝方胜率。同一时刻开赛的一批局先全部读 form、再一起写残差。"""
    kick = games.kickoff_utc.to_numpy()
    if len(kick) and not (kick[1:] >= kick[:-1]).all():
        raise ValueError("games 必须按 kickoff_utc 升序")
    y = games.y_blue_win.to_numpy(dtype=float)
    p = np.asarray(p, dtype=float)
    bt = games.blue_team_id.astype(str).to_numpy()
    rt = games.red_team_id.astype(str).to_numpy()
    ser = games.series_id.astype(str).to_numpy()
    hist: dict[str, list[tuple[str, float]]] = {}
    out = np.zeros(len(games))
    i = 0
    while i < len(games):
        j = i
        while j < len(games) and kick[j] == kick[i]:
            j += 1
        for k in range(i, j):
            f = []
            for team in (bt[k], rt[k]):
                h = hist.get(team, ())
                f.append(team_form([r for s, r in h if cfg.include_series or s != ser[k]], cfg))
            out[k] = f[0] - f[1]
        for k in range(i, j):
            hist.setdefault(bt[k], []).append((ser[k], y[k] - p[k]))
            hist.setdefault(rt[k], []).append((ser[k], p[k] - y[k]))
            for team in (bt[k], rt[k]):
                if len(hist[team]) > 4 * cfg.window:      # 只留够用的尾巴
                    del hist[team][:-cfg.window]
        i = j
    return out


def fit_beta(logit_p: np.ndarray, f: np.ndarray, y: np.ndarray, iters: int = 30) -> float:
    """logit q = logit p + β·f 的极大似然（一维牛顿）；f 全 0 时 β = 0。"""
    z, f, y = (np.asarray(v, dtype=float) for v in (logit_p, f, y))
    if not np.any(f):
        return 0.0
    b = 0.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(z + b * f)))
        step = float(((q - y) * f).sum() / ((q * (1 - q)) * f * f).sum())
        b -= step
        if abs(step) < 1e-12:
            break
    return b


# ── 推理侧的状态 ────────────────────────────────────────────────────────────
def form_entries(games: pd.DataFrame, p: np.ndarray, cfg: FormConfig = DEFAULT,
                 since: pd.Timestamp | None = None) -> dict[str, list[list]]:
    """每队最近 cfg.window 局的 [开赛时刻 ISO, 残差]（该队视角，按时间升序）。
    since 给定时只留最后一局不早于它的队伍（退役 / 解散的队不必带进 artifact）。"""
    y = games.y_blue_win.to_numpy(dtype=float)
    p = np.asarray(p, dtype=float)
    out: dict[str, list[list]] = {}
    for t, b, r, yy, pp in zip(games.kickoff_utc, games.blue_team_id.astype(str),
                               games.red_team_id.astype(str), y, p):
        if not np.isfinite(pp):
            continue
        ts = pd.Timestamp(t).isoformat()
        out.setdefault(b, []).append([ts, float(yy - pp)])
        out.setdefault(r, []).append([ts, float(pp - yy)])
    keep = {}
    for team, h in out.items():
        h.sort(key=lambda e: e[0])
        if since is None or pd.Timestamp(h[-1][0]) >= since:
            keep[team] = h[-cfg.window:]
    return keep


def form_at(state: dict, team_id: str, now: pd.Timestamp | None = None) -> float:
    """推理时一支队的 form：OE 与台账两部分合并，只用开赛早于 now 的条目（now=None 用全部）。"""
    cfg = FormConfig(**state["cfg"])
    h = list(state.get("silver", {}).get(str(team_id), ())) + list(state.get("live", {}).get(str(team_id), ()))
    if now is not None:
        cut = pd.Timestamp(now)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut
        h = [e for e in h if pd.Timestamp(e[0]) < cut]
    h.sort(key=lambda e: e[0])
    return team_form([e[1] for e in h], cfg)
