"""实时比赛：自动发现正在进行的职业比赛，读 BP 后的阵容并给出胜率。

数据来自 Riot 的非官方 livestats 接口，无 SLA，随时可能变；**接口不可用时本页降级，
不影响其它页面**。唯一权威的开赛信号是 livestats 的 204→200 跳变，不是赛程的 state。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit.errors import FragmentHandledException

from lolwp.app import data, ui
from lolwp.app.catalog import ROLE_CN, ROLES
from lolwp.live.ledger import _side_code

# 六种状态各有文案，不能只处理 READY：(标签, 颜色, 是否闪烁, 说明)
_STATUS = {
    "ready":       ("阵容已读取", "green", True, "阵容已读取，预测就绪"),
    "waiting":     ("BP 中 · 等开局", "amber", True, "BP 已开始但还没开局读条"),
    "no_coverage": ("无实时覆盖", "", False, "该赛事无 livestats 覆盖"),
    "forbidden":   ("鉴权失败", "red", False, "接口鉴权失败，已降级"),
    "error":       ("接口异常", "red", False, "接口异常，正在重试"),
    "finished":    ("本局已结束", "", False, "本局已结束"),
}
_ORDER = {"ready": 0, "waiting": 1, "error": 2, "no_coverage": 3, "forbidden": 4, "finished": 5}


def _side_team(pred: dict, side: str, codes: list[str]) -> dict:
    """蓝红方以 feed 为准：先认解析出的 OE 队伍，认不出再用选手名的队伍 tag 投票。"""
    tid = pred.get(f"{side}_team")
    if tid and not str(tid).startswith("fallback:"):
        t = data.team_row(str(tid))
        if t.get("code") != "?":
            return t
    code = _side_code(pred.get(f"{side}_names"), codes) or ("蓝色方" if side == "blue" else "红色方")
    return {"name": code, "code": code, "logo": None, "league": ""}


def _game(g: dict) -> None:
    champs = data.champions()["by_key"]
    label, kind, live, _ = _STATUS.get(g["status"], ("未知状态", "", False, ""))
    teams = " vs ".join(g.get("teams") or ["?", "?"])
    st.html(f'<div class="lw-live-head"><div><div class="lw-live-title">{ui.esc(teams)}</div>'
            f'<div class="lw-live-msg">{ui.esc(g.get("league") or "?")} · {ui.esc(g.get("message", ""))}</div></div>'
            f'{ui.pill(label, kind + (" live" if live else ""), dot=True)}</div>')
    pred = g.get("prediction")
    if g["status"] != "ready" or not pred:
        if g["status"] == "waiting":
            st.html(ui.note("赛程接口标 <code>inProgress</code> 时比赛<b>未必</b>已开打——它标的是「局间休息 + BP 开始」。"
                            "唯一权威的开赛信号是 livestats 的 204→200 跳变，本页等的就是它。", "", "⏳"))
        return
    p_blue = pred.get("p_blue")
    codes = [c for c in (g.get("teams") or []) if c]
    blue, red = _side_team(pred, "blue", codes), _side_team(pred, "red", codes)
    if p_blue is None:
        st.warning("阵容已读到但概率未产出，请看下方 warnings。", icon="⚠️")
    else:
        p_blue = float(p_blue)
        st.html(ui.matchup(blue, red, p_blue))
        lay = pred.get("layers") or {}
        if lay:
            steps = ui.layer_steps(data.artifact().intercept, lay)
            cc = st.columns(3)
            for col, k in zip(cc, ["实力", "阵容", "赛况"]):
                col.metric(f"{k}层", f"{steps[k]:+.1f} pp",
                           help=f"{ui.LAYER_HINT[k]}。对 logit 的贡献 {float(lay.get(k, 0.0)):+.3f}")
        if cal := ui.calibration_note(p_blue):
            st.html(ui.note(cal, "gold", "📉"))
    notes = []
    if miss := data.uncovered(pred.get("blue_champ_keys") or [], pred.get("red_champ_keys") or []):
        notes.append(ui.note("这些英雄在模型数据里缺少该位置的一线赛事出场：" + ui.esc("、".join(miss))
                             + "。它们的版本强度按位置基准计，阵容层可信度下降。", "warn", "⚠"))
    for w in g.get("warnings") or []:
        if not (miss and w.startswith(data.COVERAGE_WARN)):   # 已逐个点名，不再重复通用告警
            notes.append(ui.note(ui.esc(w), "warn", "⚠"))
    if notes:
        st.html("".join(notes))
    lu = []
    for side in ("blue", "red"):
        keys = pred.get(f"{side}_champ_keys") or []
        names = pred.get(f"{side}_names") or []
        cs = [champs.get(int(k)) if k is not None else None for k in keys][:5]
        lu.append(ui.lineup(cs + [None] * (5 - len(cs)), (list(names) + [None] * 5)[:5], side,
                            roles=[ROLE_CN[r] for r in ROLES], size=46))
    st.html(f'<div class="lw-lineups" style="margin-top:14px">{"".join(lu)}</div>')
    lat = g.get("latency_200_to_pred_s")
    if lat is not None:
        st.caption(f"从 livestats 首次 200 到本页出概率：{lat:.3f} 秒")


@st.fragment(run_every="30s")
def _panel() -> None:
    """每 30 秒自动刷新一次。轮询间隔由 Poller 自己控制（getLive 60s / window 30s），
    这里只是触发 tick，不会真的每 30 秒打一次远端。"""
    pol = data.poller()
    if isinstance(pol, dict):
        st.error(f"轮询器无法构建：{pol['error']}", icon="🚫")
        st.html(ui.note("实时功能不可用不影响「阵容模拟」——那一页完全离线。", "info", "💡"))
        return
    try:
        pol.tick()
    except Exception as e:  # noqa: BLE001
        st.warning(f"本次轮询失败（已保留上次结果）：{type(e).__name__}: {e}", icon="⚠️")

    snap, health = pol.snapshot(), pol.health()
    m = st.columns(4)
    m[0].metric("监视中的对局", health["n_watched"])
    m[1].metric("已出预测", health["by_status"].get("ready", 0))
    # 没有监视中的对局时 poller 给 None（无从计算），不能直接 :.0% 格式化
    rate = health["live_resolve_rate"]
    m[2].metric("实体解析率", "—" if rate is None else f"{rate:.0%}")
    m[3].metric("接口状态", "降级" if health["api_degraded"] else "正常")
    st.caption(f"最后刷新 {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} UTC　·　"
               f"getLive 轮询 {snap['n_live_polls']} 次，window {snap['n_window_polls']} 次　·　每 30 秒自动刷新")

    games = snap.get("games") or []
    if not games:
        ui.empty("当前没有正在进行的比赛", "有比赛开打时这里会自动出现：BP 结束、开局读条后约 1 分钟出概率。")
        return
    for g in sorted(games, key=lambda x: _ORDER.get(x.get("status"), 9)):
        with st.container(border=True):
            _game(g)


def render() -> None:
    ui.page_header("Live", "实时比赛",
                   "自动发现正在进行的职业比赛，读取 BP 结束后的阵容并给出胜率。数据来自 Riot 的非官方 livestats 接口，"
                   "无 SLA；<b>接口不可用时只有本页降级，其余页面不受影响。</b>",
                   [ui.pill("自动刷新", "green live", dot=True)])
    try:
        _panel()
    except FragmentHandledException:
        # 片段里的异常 Streamlit 已经就地显示过了；不接住的话它会终止整次脚本运行，页脚空白
        pass


render()
