"""阵容模拟：选赛事和两支队，点头像选英雄、按位置筛、换替补，出 BP 后胜率与三层构成。

状态都在 st.session_state["sim"]（Python 是唯一的真值，见 app/board.py）：
  {"rev", "seq", "blue": {"team", "players"[5], "champs"[5]}, "red": {...}}
没选的英雄按该位置的平均版本强度计——所以一个英雄都不选时，看到的就是纯实力层的胜率。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lolwp.app import catalog, data, ui
from lolwp.app.board import draft_board
from lolwp.app.catalog import league_label
from lolwp.app.predict import FEATURE_LABELS, Draft, predict
from lolwp.models.artifact import LAYER_OF

_LEAGUE_ORDER = ["LCK", "LPL", "LEC", "LCS", "LCP", "CBLOL", "LTA(2025-only)", "VCS", "PCS", "LJL"]


def _strength(tid: str) -> float:
    """默认首发的分层 BT 实力之和：只用来给默认对阵与列表排序，不进预测。"""
    snap = data.bt_snapshot()
    ro = data.rosters().get(tid) or {}
    return sum(snap.player_rating(p, tid) for p in (ro.get("default") or [None] * 5))


def _side(tid: str) -> dict:
    ro = data.rosters().get(tid) or {}
    return {"team": tid, "players": list(ro.get("default") or [None] * 5), "champs": [None] * 5}


def _team_info(tid: str) -> dict:
    t = data.team_row(tid)
    return {**t, "league_label": league_label(str(t.get("league") or ""))}


def _init() -> dict:
    sim = st.session_state.get("sim")
    if sim is not None:
        return sim
    t = data.teams()
    act = t[t["active"]].copy()
    act["s"] = [_strength(x) for x in act["team_id"]]
    top = act.sort_values("s", ascending=False)["team_id"].tolist()
    blue, red = (top + list(t["team_id"]))[:2]
    sim = {"rev": 0, "seq": 0, "blue": _side(blue), "red": _side(red)}
    st.session_state["sim"] = sim
    st.session_state["sim_blue_team"], st.session_state["sim_red_team"] = blue, red
    st.session_state.setdefault("sim_event", "WLDs")
    return sim


def _on_team(side: str) -> None:
    sim = st.session_state["sim"]
    tid = st.session_state[f"sim_{side}_team"]
    if tid != sim[side]["team"]:
        sim[side] = {**_side(tid), "champs": sim[side]["champs"]}   # 换队保留已选英雄，选手换成新队的首发
        sim["rev"] += 1


def _swap() -> None:
    sim = st.session_state["sim"]
    sim["blue"], sim["red"] = sim["red"], sim["blue"]
    st.session_state["sim_blue_team"], st.session_state["sim_red_team"] = sim["blue"]["team"], sim["red"]["team"]
    sim["rev"] += 1


def _team_options(include_all: bool, keep: list[str]) -> list[str]:
    t = data.teams()
    pool = t if include_all else t[t["active"] | t["team_id"].isin(keep)]
    order = {lg: i for i, lg in enumerate(_LEAGUE_ORDER)}
    pool = pool.assign(o=pool["league"].map(order).fillna(len(order)), act=~pool["active"])
    return pool.sort_values(["act", "o", "league", "name"])["team_id"].tolist()


def _team_label(tid: str) -> str:
    t = data.team_row(tid)
    tail = "" if t.get("active") else f" · {str(t.get('last_game'))[:4]}"
    return f"{t['name']}（{league_label(str(t['league']))}{tail}）"


def _events() -> list[str]:
    g = data.games()
    recent = g[g["kickoff_utc"] >= g["kickoff_utc"].max() - pd.Timedelta(days=catalog.WINDOW_DAYS)]
    seen = recent["league_canon"].value_counts()
    intl = [c for c in catalog.INTL if c in seen.index or c == "WLDs"]
    regional = sorted((c for c in seen.index if c not in catalog.INTL),
                      key=lambda c: (_LEAGUE_ORDER.index(c) if c in _LEAGUE_ORDER else 99, c))
    return intl + regional


def render() -> None:
    art, champs = data.artifact(), data.champions()
    sim = _init()

    ui.page_header(
        "Draft Simulator", "阵容模拟",
        "点头像选英雄、按位置筛，每个位置都能换替补；胜率随每一次改动实时更新。",
        [ui.pill(f"模型 {art.meta['model']}", "gold"), ui.pill(f"英雄 {champs['version']}")])

    with st.container(key="lw_filters"):
        c = st.columns([2.1, 3, 0.55, 3, 1.6], vertical_alignment="bottom")
        include_all = c[4].toggle("历史队伍", key="sim_all_teams", persist_state="session",
                                  help="默认只列最近一年打过一线赛事的队伍；打开后可以摆任何年份的队")
        keep = [sim["blue"]["team"], sim["red"]["team"]]
        opts = _team_options(include_all, keep)
        events = _events()
        if st.session_state.get("sim_event") not in events:
            st.session_state["sim_event"] = events[0]
        # persist_state="session"：切到别的页再回来，控件值不丢（否则下拉框回到第一项，和阵容板对不上）
        c[0].selectbox("赛事", events, key="sim_event", format_func=league_label, persist_state="session",
                       help="只影响「蓝方基准」：各赛区蓝色方的历史胜率不同。跨赛区对阵（如 Worlds）选国际赛事即可。")
        c[1].selectbox("蓝色方", opts, key="sim_blue_team", format_func=_team_label, persist_state="session",
                       on_change=_on_team, args=("blue",))
        c[2].button("", icon=":material/swap_horiz:", help="交换蓝红方（连同选手和英雄）",
                    on_click=_swap, width="stretch", key="sim_swap")
        c[3].selectbox("红色方", opts, key="sim_red_team", format_func=_team_label, persist_state="session",
                       on_change=_on_team, args=("red",))

    hero = st.container()
    draft_board("sim", champs, data.rosters(), _team_info)

    if sim["blue"]["team"] == sim["red"]["team"]:
        hero.html(ui.note("蓝红两边选了同一支队伍，换一支再看。", "warn", "⚠"))
        return
    event = st.session_state["sim_event"]
    d = Draft(event, sim["blue"]["team"], sim["red"]["team"],
              list(sim["blue"]["players"]), list(sim["red"]["players"]),
              list(sim["blue"]["champs"]), list(sim["red"]["champs"]))
    pred = predict(art, d)
    st.session_state["draft"] = d          # what-if 页沿用这一套阵容

    n_empty = sum(k is None for s in ("blue", "red") for k in sim[s]["champs"])
    with hero:
        st.html(ui.matchup(_team_info(d.blue_team_id), _team_info(d.red_team_id), pred.p_blue,
                           event=league_label(event),
                           extra=f"{n_empty} 个位置待选" if n_empty else None))
        notes = []
        if cal := ui.calibration_note(pred.p_blue):
            notes.append(ui.note(cal, "gold", "📉"))
        if miss := data.uncovered(sim["blue"]["champs"], sim["red"]["champs"]):
            notes.append(ui.note("这些英雄在模型数据里缺少该位置的一线赛事出场：" + ui.esc("、".join(miss))
                                 + "。它们的版本强度按位置基准计，阵容层可信度下降。", "warn", "⚠"))
        for w in pred.warnings:
            if not w.startswith(data.COVERAGE_WARN):  # 上面已经按英雄逐个说明
                notes.append(ui.note(ui.esc(w), "warn", "⚠"))
        if notes:
            st.html("".join(notes))

    _breakdown(art, pred)


def _breakdown(art, pred) -> None:
    lay = pred.layers
    ui.section("胜率构成", "从 50% 出发，依次加上每一层对 logit 的贡献")
    c1, c2 = st.columns([1.45, 1], gap="large")
    with c1, st.container(key="lw_panel"):
        # 顺序 赛况→实力→阵容：阵容那一段就是「在实力之上，BP 还能把胜率动多少」
        st.html(ui.waterfall(art.intercept, [(f"{n}层", ui.LAYER_HINT[n], lay.get(n, 0.0))
                                             for n in ui.LAYER_ORDER]))
    with c2:
        # 竖着排：并排三张卡在窄屏上数字会被截断；数值与左边瀑布逐段一致
        steps = ui.layer_steps(art.intercept, lay)
        for name in ("实力", "阵容", "赛况"):
            st.metric(f"{name}层", f"{steps[name]:+.1f} pp",
                      help=f"{ui.LAYER_HINT[name]}。对 logit 的贡献 {lay.get(name, 0.0):+.3f}；"
                           "pp = 按 赛况→实力→阵容 逐层加上时，这一层让蓝方胜率变了多少")
        st.html(ui.note(
            "<b>实力层通常占绝大部分</b>——这不是模型偷懒，是职业赛的真实结构：队伍与选手实力约占可解释信号的 87%，"
            "整个 BP 阶段的可测量贡献只有 ΔlogLoss ≈ 0.002。换一个英雄通常只动 1–2 个百分点。", "info", "ⓘ"))
    with st.expander("逐项贡献（明细）"):
        df = pd.DataFrame([{"特征": FEATURE_LABELS.get(k, k), "层": LAYER_OF.get(k, "—"),
                            "贡献 (logit)": round(v, 4), "原始值": round(pred.features[k], 4)}
                           for k, v in sorted(pred.contributions.items(), key=lambda x: -abs(x[1]))])
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"Σ贡献 + 截距 {art.intercept:+.4f} = {pred.logit:+.6f} = logit(p)，闭合误差 0")


render()
