"""模型说明：这个模型怎么算、准到什么程度、哪里别全信。数字出处见 README 与 reports/。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lolwp.app import data, ui
from lolwp.app.predict import FEATURE_LABELS
from lolwp.models.artifact import LAYER_OF


def render() -> None:
    art, champs = data.artifact(), data.champions()
    meta = art.meta
    refit = meta.get("data_scope") == "all"
    mov = meta.get("rating_target") == "y_margin"      # 实力快照用赛后经济差软标签训练（修订 A6）
    ui.page_header("Model", "模型说明",
                   "BP（ban/pick）结束、开赛前，给出蓝色方的获胜概率。覆盖 tier-1 全部赛区与国际赛事，完全本地运行。",
                   [ui.pill(meta.get("model", "?"), "gold"),
                    ui.pill(f"训练至 {str(meta.get('trained_on'))[:10]}"),
                    *([ui.pill(ui.ratings_note(meta) + (f"（含盯盘 {meta['n_live_games']} 场）"
                                                       if meta.get("n_live_games") else ""), "green")]
                      if ui.ratings_note(meta) else []),
                    *([ui.pill("揭盲后全量重训", "green")] if refit else []),
                    *([ui.pill("经济差软标签", "green")] if mov else []),
                    ui.pill(f"{meta.get('n_train_games', 0):,} 场训练")])

    # holdout 与准确率是揭盲时评测的那个模型（训练截至 2026-05-31、胜负标签）的；全量重训的模型在 holdout 期是样本内的。
    # walk-forward 是当前规格在开发期（holdout 之前）的成绩，规格变了它跟着变（meta.oos_log_loss）
    whose = "评测对象是训练截至 2026-05-31 的模型（dev 口径）；当前的全量重训模型在 holdout 期是样本内的。" if refit else ""
    m = st.columns(4)
    m[0].metric("Holdout log loss", "0.6165", help=f"989 场，一次性揭盲（git 437d1cb）。{whose}")
    m[1].metric("Walk-forward log loss", f"{meta.get('oos_log_loss', 0.61917):.4f}",
                help=f"{meta.get('oos_note')}。当前规格在开发期（holdout 之前）的成绩。")
    m[2].metric("准确率", "65.3%", help=f"final holdout 989 场。{whose}")
    m[3].metric("对照：Polymarket", "0.6275", help="同期 1,302 场的市场价格；与模型统计上无法区分")
    if refit:
        st.html(ui.note(
            ("当前模型是 holdout 揭盲之后<b>用全量数据重训</b>的版本（预注册修订 A4），实力快照改用<b>赛后经济差的软标签</b>"
             "训练（修订 A6）：walk-forward 是这个规格在开发期的成绩；holdout 与准确率属于揭盲时评测的、训练截至 5/31 的"
             "胜负标签模型。" if mov else
             "当前模型是 holdout 揭盲之后、<b>同一规格用全量数据重训</b>的版本（预注册修订 A4）：上面几个成绩属于揭盲时"
             "评测的那个训练截至 5/31 的模型。")
            + "这个模型自己的样本外成绩，看「近期战绩」里它的训练截止日之后落盘的预测。",
            "info", "ⓘ"))

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        ui.section("怎么算的", "五项 L2 logistic，系数来自当前 artifact")
        rows = sorted(art.coef.items(), key=lambda x: -abs(x[1]))
        tot = sum(abs(v) for _, v in rows)
        st.html('<div class="lw-card">' + "".join(
            f'<div class="lw-wf-row" style="grid-template-columns:190px 1fr 70px;margin:6px 0">'
            f'<div class="lw-wf-name">{ui.esc(FEATURE_LABELS.get(k, k))}<small>{ui.esc(LAYER_OF.get(k, ""))}层 · {ui.esc(k)}</small></div>'
            f'<div class="lw-wf-track"><div class="seg up" style="left:0;width:{abs(v) / tot * 100:.1f}%"></div></div>'
            f'<div class="lw-wf-val">{v:.3f}</div></div>' for k, v in rows)
            + f'<div style="color:var(--lw-faint);font-size:.8rem;margin-top:10px">标准化系数；截距 {art.intercept:+.4f}。'
              "外层朴素，复杂度全在特征怎么算出来：分层 Bradley-Terry 选手潜变量 / 配合 / 赛区校准 + 两层经验贝叶斯的英雄版本强度。"
              + ("分层 BT 用历史局的终局经济差做软标签：险胜和碾压分开记，本局的赛后统计照旧不进模型。" if mov else "")
              + (f"外层之上还有一层近期状态：两队最近几局「赛果 − 模型预期」之差 × {form['beta']:.2f}"
                 "（含本系列已打完的局）。" if (form := art.prior_state.get("form")) else "")
              + "CatBoost、DeepSet、集成都试过，无一在 MDE 之上胜过它。</div></div>")
    with c2:
        ui.section("主要结论是一个负结果", "预注册时就允许为负")
        st.dataframe(pd.DataFrame([
            ("队伍实力", "+0.0577", "✓"), ("分层 BT 选手潜变量", "+0.0143", "✓"),
            ("英雄版本强度", "+0.0015", "✓（边缘）"), ("选手熟练度", "+0.0004", "✗"), ("协同 + 克制", "−0.0004", "✗"),
        ], columns=["加入的东西", "ΔlogLoss", "超过 MDE"]), hide_index=True, width="stretch")
        st.html(ui.note("<b>BP 阶段本身（协同、克制、选手熟练度）在职业赛的数据量下贡献不可测量</b>，"
                        "可预测的部分几乎全部来自队伍与选手实力。所以换一个英雄通常只动 1–2 个百分点——这不是工具坏了。",
                        "info", "ⓘ"))

    ui.section("哪里别全信", "")
    c = st.columns(3, gap="medium")
    (lo95, n95, _, a95), (lo90, n90, m90, a90) = ui.EXTREME_BUCKETS[:2]
    c[0].html(ui.note(f"<b>高自信段过度自信。</b>模型给出 ≥{lo90:.0%} 的 {n90} 场里，平均预测 {m90:.1%}，"
                      f"实际只有 {a90:.1%}；≥{lo95:.0%} 的 {n95} 场实际 {a95:.1%}。"
                      "页面上落在这一段的概率会单独提示。", "gold", "📉"))
    c[1].html(ui.note("<b>模型没见过的人和英雄会退化。</b>新选手按其队伍的平均水平计，"
                      "一线赛事里没在该位置出场过的英雄按位置基准计；页面会逐个点名。", "warn", "⚠"))
    c[2].html(ui.note("<b>What-if 是相关性外推，不是因果反事实。</b>队伍只会选自己练过、当前版本能配合的英雄，"
                      "多数替换的影响小于模型的不确定度（约 ±4 pp）。", "", "🧭"))

    ui.section("数据与运行", "")
    st.html(f'<div class="lw-card"><dl class="lw-kv">'
            f"<dt>比赛数据</dt><dd>Oracle's Elixir 2014–2026（训练截至 {ui.esc(str(meta.get('trained_on'))[:10])}，"
            f"状态机 {meta.get('n_state_games', 0):,} 场）</dd>"
            f"<dt>holdout</dt><dd>{ui.esc(str(meta.get('holdout_start'))[:10])} 起，{ui.esc(meta.get('holdout_note', ''))}"
            + ("；揭盲之后生产模型改用全量数据重训，这段比赛已在训练集里" if refit else "") + "</dd>"
            f"<dt>英雄</dt><dd>Riot Data Dragon {ui.esc(champs['version'])}，{len(champs['list'])} 个</dd>"
            f"<dt>实时阵容</dt><dd>lolesports livestats（非官方接口，无 SLA）；"
            f"盯盘错过的局事后从赛程补录，标「赛后补录」</dd>"
            f"<dt>推理</dt><dd>{'经 API' if data.api_alive() else '进程内'}，单次预测约 0.5 ms，运行时零外部请求</dd>"
            f"</dl></div>")


render()
