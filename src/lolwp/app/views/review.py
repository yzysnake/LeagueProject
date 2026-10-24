"""近期战绩：盯盘落盘的预测 vs 赛后结果——唯一干净的线上成绩。

盯盘读到 BP 后阵容时落盘的预测只用阵容和冻结的模型，**不读任何赛中数据**，
和「历史回看」的事后重算不同。台账是另一个进程写的外部输入：这一页再怎么坏，
也不能让页脚和其它页面空白（main.py 已兜底）。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lolwp.app import ui

_SPANS = {"全部": None, "最近 7 天": 7, "最近 30 天": 30, "自定义": None}
_STATUS = ["命中", "未中", "待定"]


def render() -> None:
    from lolwp.live.ledger import LEDGER, load_review, summarize

    ui.page_header(
        "Track Record", "近期战绩",
        "盯盘读到 BP 后阵容时自动落盘的预测：只用阵容和冻结的模型，<b>不读任何赛中数据</b>，所以是干净的样本外成绩。"
        "盯盘休眠或停摆时错过的局会在事后补录（标「赛后补录」：概率和开局时算的一样，只是不算严格的实时预测）。")
    df = load_review(LEDGER)
    if df.attrs.get("n_bad_lines"):
        st.warning(f"台账有 {df.attrs['n_bad_lines']} 行无法解析，已跳过。", icon="⚠️")
    if df.empty:
        ui.empty("还没有盯盘记录", "`make install-watch` 挂上盯盘后，有比赛开打就会出现在这里。")
        return

    # 筛选器都按「留空 = 全部」设计：带 key 的控件在一个会话里会记住首次渲染时的选项
    # （Streamlit 1.64 里 min/max/options 变了也不重置），台账新增的日期或赛事会被悄悄筛掉。
    gt = df["game_time"]
    with st.container(key="lw_filters"):
        c1, c2, c3 = st.columns([3, 3, 1.6], vertical_alignment="bottom")
        span = c1.radio("时间段（UTC，按比赛日）", list(_SPANS), horizontal=True, key="review_span", persist_state="session")
        lo = hi = None
        if _SPANS[span]:
            hi = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
            lo = hi - pd.Timedelta(days=_SPANS[span])
        elif span == "自定义":
            d0, d1 = gt.min().date(), gt.max().date()
            # 不给 key：控件身份随数据边界变，台账多出新的一天时自动回到全范围
            rng = c1.date_input("日期范围", value=(d0, d1), min_value=d0, max_value=d1)
            rng = tuple(rng) if isinstance(rng, (tuple, list)) else (rng,)
            a = rng[0] if rng else d0
            b = rng[1] if len(rng) > 1 else d1  # 只点了起始日 = 从那天到最新，和控件上显示的一致
            lo, hi = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
        leagues = sorted(df["league"].unique())
        pick = c2.multiselect("赛事", leagues, key="review_leagues", placeholder="全部赛事",
                              persist_state="session") or leagues
        n_late = int(df["late"].eq(True).sum())
        strict = c3.toggle("排除赛后补录", key="review_strict", persist_state="session",
                           help=f"只算对局结束前就落盘的预测（台账里共 {n_late} 局是赛后补录）")

    scope = df[df["league"].isin(pick)]
    if lo is not None:
        scope = scope[(scope["game_time"] >= lo) & (scope["game_time"] < hi)]
    if strict:
        scope = scope[~scope["late"].eq(True)]

    sm = summarize(scope)
    m = st.columns(5)
    m[0].metric("对局", sm["n"])
    m[1].metric("命中率", "—" if sm["hit_rate"] is None else f"{sm['hit_rate']:.1%}",
                help=f"只算已判出胜负的 {sm['n_done']} 局")
    m[2].metric("log loss", "—" if sm["log_loss"] is None else f"{sm['log_loss']:.3f}",
                help="越低越好；永远猜 50% 是 0.693")
    m[3].metric("Brier", "—" if sm["brier"] is None else f"{sm['brier']:.3f}",
                help="越低越好；永远猜 50% 是 0.25")
    m[4].metric("待定", sm["n_pending"])
    n_live = int((~scope["late"].eq(True)).sum())
    n_bf = int(scope["backfill"].eq(True).sum()) if "backfill" in scope else 0
    bits = [ui.pill(f"实时 {n_live}", "green", dot=True),
            ui.pill(f"赛后补录 {len(scope) - n_live}", "amber", dot=True)]
    if n_bf:
        bits.append(ui.pill(f"其中整局错过、事后补录 {n_bf}", ""))
    msg = ""
    if 0 < sm["n_done"] < 30:
        msg = f"只有 {sm['n_done']} 局，命中率的抽样误差约 ±{1.96 * (0.25 / sm['n_done']) ** 0.5:.0%}，别读太多。"
    st.html(f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:4px 0 6px">'
            f'{"".join(bits)}<span style="color:var(--lw-faint);font-size:.8rem;margin-left:6px">{ui.esc(msg)}</span></div>')

    if len(scope):
        by = (scope.assign(h=scope["hit"].astype("float"))
              .groupby("league").agg(局数=("game_id", "size"), 已判=("h", "count"),
                                     命中=("h", "sum"), 命中率=("h", "mean"))
              .sort_values("局数", ascending=False).reset_index().rename(columns={"league": "赛事"}))
        by["命中"] = by["命中"].astype(int)
        by["命中率"] = by["命中率"].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
        with st.expander("按赛事", expanded=len(pick) > 1):
            st.dataframe(by, hide_index=True, width="stretch")

    ui.section("逐局明细", "新的在前")
    # 「结果」只筛下面这张表：拿它去筛指标，只看命中时命中率恒为 100%，毫无意义
    status = st.multiselect("结果", _STATUS, key="review_status", persist_state="session",
                            placeholder="全部结果（只筛下表，不影响上面的指标）") or _STATUS
    view = scope[scope["status"].isin(status)]
    when = view["game_time"].dt.strftime("%m-%d %H:%M")
    note = view["late"].map({True: "赛后补录"}).fillna("")
    if "backfill" in view:
        note = note.where(~view["backfill"].eq(True), "赛后补录（整局错过）")
    out = pd.DataFrame({
        # 还没对账、也不知道开局时间的局只有落盘时间，标 * 区分
        "比赛时间(UTC)": when.where(view["started_at"].notna(), when + " *"),
        "赛事": view["league"], "蓝方": view["blue"], "红方": view["red"], "预测蓝胜": view["p_blue"],
        "实际": view["blue_won"].map({True: "蓝胜", False: "红胜"}).fillna("—"),
        "结果": view["status"].map({"命中": "✅ 命中", "未中": "❌ 未中", "待定": "⏳ 待定"}),
        "备注": note,
    })
    st.dataframe(out, hide_index=True, width="stretch", height=min(640, 38 + 35 * max(1, len(out))),
                 column_config={"预测蓝胜": st.column_config.ProgressColumn(
                     format="percent", min_value=0.0, max_value=1.0, step=0.001)})
    if view["started_at"].isna().any():
        st.caption("带 * 的是落盘时间：这局还没对账，开局时间未知。")


try:
    render()
except Exception as e:  # noqa: BLE001
    # 台账是另一个进程写的外部输入；这一页再怎么坏，也只坏在这一页（页脚由 main.py 兜底）
    st.exception(e)
