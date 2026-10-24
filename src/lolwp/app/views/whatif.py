"""What-if：把某一边某个位置的英雄换成别的，看胜率怎么变（173 个英雄 3 ms 出结果）。

Δ 一律站在**被换的那一边**说：换红方英雄时，「提升」指红方胜率提升。
候选里去掉本局已被选走的英雄（职业赛一个英雄一局只能出现一次）。
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from lolwp.app import data, ui
from lolwp.app.catalog import ROLE_CN, ROLES
from lolwp.app.predict import whatif_swap

_SIDE_CN = {"blue": "蓝色方", "red": "红色方"}


def _apply(side: str, slot: int, key: int) -> None:
    sim = st.session_state.get("sim")
    if sim is None:
        return
    for s in ("blue", "red"):                      # 同一英雄不能出现两次
        sim[s]["champs"] = [None if k == key else k for k in sim[s]["champs"]]
    sim[side]["champs"][slot] = key
    sim["rev"] += 1


def render() -> None:
    art, champs = data.artifact(), data.champions()
    by_key = champs["by_key"]
    ui.page_header("What-if", "换一个英雄会怎样",
                   "固定其余九个位置，把一个位置换成每一个候选英雄，看胜率往哪边动、动多少。")
    d = st.session_state.get("draft")
    if d is None:
        ui.empty("还没有阵容", "先到「阵容模拟」摆一套阵容，这里会沿用它。")
        st.page_link("views/simulate.py", label="去阵容模拟", icon=":material/arrow_forward:")
        return

    with st.container(key="lw_filters"):
        c = st.columns([1.3, 2.6, 1.6], vertical_alignment="bottom")
        side = c[0].segmented_control("换哪一边", ["blue", "red"], format_func=_SIDE_CN.get,
                                      default="blue", key="wi_side", required=True,
                                      persist_state="session")
        role = c[1].segmented_control("换哪个位置", list(ROLES), format_func=ROLE_CN.get,
                                      default="mid", key="wi_role", required=True,
                                      persist_state="session")
        pool_only = c[2].toggle("只看该位置常用英雄", value=True, key="wi_pool", persist_state="session",
                                help="最近一年职业赛里在该位置出场过的英雄；关掉则遍历全部英雄")
    slot = ROLES.index(role)

    names = {"blue": [], "red": []}
    ro = data.rosters()
    for s, tid, pids in (("blue", d.blue_team_id, d.blue_player_ids), ("red", d.red_team_id, d.red_player_ids)):
        lookup = {p["id"]: p["name"] for lst in (ro.get(tid) or {}).get("roles", {}).values() for p in lst}
        names[s] = [lookup.get(p, "—") for p in pids]
    with st.container(key="lw_panel"):
        cl, cr = st.columns(2, gap="large")
        for col, s, keys in ((cl, "blue", d.blue_champ_keys), (cr, "red", d.red_champ_keys)):
            t = data.team_row(d.blue_team_id if s == "blue" else d.red_team_id)
            col.html(f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;'
                     f'{"flex-direction:row-reverse" if s == "red" else ""}">{ui.team_badge(t, 32)}'
                     f'<b>{ui.esc(t["name"])}</b>{ui.pill(_SIDE_CN[s], s)}</div>'
                     + ui.lineup([by_key.get(k) for k in keys], names[s], s,
                                 roles=[ROLE_CN[r] for r in ROLES], highlight=slot if s == side else None))

    target = d.blue_champ_keys if side == "blue" else d.red_champ_keys
    taken = {k for k in list(d.blue_champ_keys) + list(d.red_champ_keys) if k is not None}
    cand = champs["pool"][role] if pool_only else [c["key"] for c in champs["list"]]
    cand = [k for k in cand if k not in taken]
    if not cand:
        ui.empty("没有可换的英雄", "该位置的常用英雄都已被选走，关掉「只看该位置常用英雄」试试。")
        return
    wi = whatif_swap(art, d, side, slot, cand)
    base_blue = float(wi.attrs["base_p"])
    sign = 1.0 if side == "blue" else -1.0
    wi["d"] = wi["delta_pp"] * sign
    wi["p"] = wi["p_blue"] if side == "blue" else 1 - wi["p_blue"]
    wi = wi.sort_values("d", ascending=False, ignore_index=True)
    base = base_blue if side == "blue" else 1 - base_blue
    cur = by_key.get(target[slot])

    m = st.columns(4)
    m[0].metric(f"当前{_SIDE_CN[side]}胜率", f"{base:.1%}")
    m[1].metric(f"当前{ROLE_CN[role]}", cur["zh"] if cur else "未选")
    m[2].metric("最多提升", f"{wi['d'].max():+.2f} pp")
    m[3].metric("最多下降", f"{wi['d'].min():+.2f} pp")

    rows = [(by_key[int(r.champ_key)], r.d, r.p) for r in wi.itertuples(index=False)]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section(f"对{_SIDE_CN[side]}最有利", "Δ 为百分点")
        st.html(ui.champ_cards(rows[:8]))
    with c2:
        ui.section(f"对{_SIDE_CN[side]}最不利", "Δ 为百分点")
        st.html(ui.champ_cards(rows[::-1][:8]))

    ui.section("全部候选", f"{len(wi)} 个英雄 · 按 Δ 排序")
    top = pd.DataFrame({"英雄": [f"{c['zh']} {c['en']}" for c, _, _ in rows], "Δ (pp)": wi["d"].round(3),
                        "方向": ["提升" if x >= 0 else "下降" for x in wi["d"]]})
    show = pd.concat([top.head(10), top.tail(10)]).drop_duplicates()
    pal = ui.palette()
    chart = (alt.Chart(show).mark_bar(cornerRadius=3, height=16)
             .encode(x=alt.X("Δ (pp):Q", title="胜率变化（百分点）"),
                     y=alt.Y("英雄:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)),
                     color=alt.Color("方向:N", scale=alt.Scale(domain=["提升", "下降"],
                                                              range=[pal["green"], pal["red"]]), legend=None),
                     tooltip=["英雄", "Δ (pp)"])
             .properties(height=max(220, 26 * len(show))))
    st.altair_chart(chart, width="stretch")

    with st.container(key="lw_panel_2"):
        c = st.columns([3, 1.2], vertical_alignment="bottom")
        pick = c[0].selectbox(f"把{_SIDE_CN[side]}的{ROLE_CN[role]}换成", [r[0]["key"] for r in rows],
                              format_func=lambda k: f"{by_key[k]['zh']} · {by_key[k]['en']}（"
                                                    f"{wi.loc[wi.champ_key == k, 'd'].iloc[0]:+.2f} pp）",
                              key="wi_pick")
        if c[1].button("应用到阵容模拟", icon=":material/check:", width="stretch",
                       disabled="sim" not in st.session_state):
            _apply(side, slot, int(pick))
            st.switch_page("views/simulate.py")

    st.html(ui.note(
        "这是<b>分布外的相关性外推，不是因果反事实</b>：队伍只会选自己练过、当前版本能配合的英雄，"
        "模型没见过的组合给出的概率可信度很低。多数替换的影响小于本模型的不确定度（约 ±4 pp），"
        "排序反映的是方向性倾向，不是可靠的差距。", "warn", "🧭"))


render()
