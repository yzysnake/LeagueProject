"""历史回看：拿真实比赛复盘，对照模型预测与实际结果。

⚠️ 这里包含 holdout 期的比赛，仅供肉眼复盘；任何写进报告的指标都必须走 walk-forward。
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from lolwp.app import data, ui
from lolwp.app.catalog import league_label
from lolwp.app.predict import Draft, predict


@st.cache_data(show_spinner="正在复盘…")
def _replay(n: int, leagues: tuple[str, ...], team: str | None) -> pd.DataFrame:
    art, g = data.artifact(), data.games()
    if leagues:
        g = g[g["league_canon"].isin(leagues)]
    if team:
        g = g[(g["blue_team_id"] == team) | (g["red_team_id"] == team)]

    def keys(ks) -> list[int | None]:
        return [None if x is None or pd.isna(x) else int(x) for x in ks]   # 老比赛偶有缺英雄的槽

    rows = []
    for r in g.head(n).itertuples():
        d = Draft(r.league_canon, r.blue_team_id, r.red_team_id,
                  list(r.blue_player_ids), list(r.red_player_ids),
                  keys(r.blue_champ_keys), keys(r.red_champ_keys), now=r.kickoff_utc)
        p = predict(art, d).p_blue
        rows.append({"日期": r.kickoff_utc.strftime("%Y-%m-%d"), "赛事": league_label(r.league_canon),
                     "蓝方": r.blue_team_name, "红方": r.red_team_name, "预测蓝胜": round(p, 4),
                     "实际": "蓝胜" if r.y_blue_win else "红胜",
                     "命中": (p >= 0.5) == bool(r.y_blue_win), "y": int(r.y_blue_win)})
    return pd.DataFrame(rows)


def render() -> None:
    ui.page_header("History", "历史回看",
                   "拿已经打完的真实比赛复盘：把当时的阵容喂给模型，对照预测与实际结果。",
                   [ui.pill("含 holdout 期 · 仅供肉眼复盘", "amber")])
    g = data.games()
    t = data.teams()
    with st.container(key="lw_filters"):
        c = st.columns([2.6, 2.6, 2], vertical_alignment="bottom")
        lgs = c[0].multiselect("赛事", sorted(g["league_canon"].unique()), key="hist_leagues", persist_state="session",
                               format_func=league_label, placeholder="全部赛事")
        act = t[t["active"]].sort_values("name")
        team = c[1].selectbox("战队", [None, *act["team_id"]], key="hist_team", persist_state="session",
                              format_func=lambda x: "全部战队" if x is None else
                              f"{data.team_row(x)['name']}（{league_label(str(data.team_row(x)['league']))}）")
        n = c[2].slider("看最近多少场", 20, 500, 100, step=20, key="hist_n", persist_state="session")
    df = _replay(n, tuple(lgs), team)
    if df.empty:
        ui.empty("没有符合条件的比赛", "换个赛事或战队试试。")
        return
    p = df["预测蓝胜"].clip(1e-9, 1 - 1e-9)
    ll = float(-(df["y"] * p.map(math.log) + (1 - df["y"]) * (1 - p).map(math.log)).mean())
    m = st.columns(4)
    m[0].metric("命中率", f"{df['命中'].mean():.1%}")
    m[1].metric("样本", f"{len(df)} 场")
    m[2].metric("log loss", f"{ll:.3f}", help="越低越好；永远猜 50% 是 0.693")
    m[3].metric("时间跨度", f"{df['日期'].iloc[-1][5:]} → {df['日期'].iloc[0][5:]}",
                help=f"{df['日期'].iloc[-1]} 到 {df['日期'].iloc[0]}")
    st.html(ui.note("这里包含 <b>holdout 期</b>的比赛，仅供肉眼复盘。任何写进报告的指标都必须走 walk-forward，"
                    "绝不能用这一页的数字。", "warn", "⚠"))
    out = df.drop(columns=["y"]).assign(命中=df["命中"].map({True: "✅", False: "❌"}))
    st.dataframe(out, hide_index=True, width="stretch", height=min(700, 38 + 35 * len(out)),
                 column_config={"预测蓝胜": st.column_config.ProgressColumn(
                     format="percent", min_value=0.0, max_value=1.0)})


render()
