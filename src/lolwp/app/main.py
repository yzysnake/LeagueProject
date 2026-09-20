"""本地 Streamlit 站点：摆阵容 → 出胜率 → 看 what-if。

三个 tab：
  1. 阵容模拟   手动选队伍/选手/英雄，出胜率与分层分解
  2. what-if   换任一位置的英雄，看概率怎么变
  3. 历史回看   拿真实比赛复盘，对照模型预测与实际结果

**完全离线可用**（阶段 5 验收①）：英雄图标已 build 时落到 frontend/public/dd/，
模型与状态全在 models/prod.json 里，运行时零外部请求。
实时读取当前比赛是阶段 6 的增强功能，不是这一层的依赖。
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from lolwp.app.predict import Draft, predict, whatif_swap
from lolwp.features.draft_priors import ROLES
from lolwp.models.artifact import Artifact

ART = "models/prod.json"
CMAP = "data/ddragon/champion_map.json"
SILVER = "data/silver/games.parquet"
ROLE_CN = {"top": "上单", "jng": "打野", "mid": "中单", "bot": "ADC", "sup": "辅助"}


@st.cache_resource
def _art() -> Artifact:
    return Artifact.load(ART)


@st.cache_resource
def _champs() -> dict:
    with open(CMAP) as fh:
        cm = json.load(fh)
    return {
        "version": cm["version"],
        "by_name": {c["name"]: c for c in cm["champions"]},
        "by_key": {c["dd_key"]: c for c in cm["champions"]},
        "names": sorted(c["name"] for c in cm["champions"]),
    }


@st.cache_data
def _games() -> pd.DataFrame:
    g = pd.read_parquet(SILVER)
    return g[g.league_tier >= 1].sort_values("kickoff_utc", ascending=False)


def _icon(dd_id: str, version: str) -> str | None:
    p = f"frontend/public/dd/{version}/{dd_id}.png"
    return p if os.path.exists(p) else None


# 极端段的实测校准（样本外 23,060 场，reports/stage8_comparison.md）。
# 模型在高概率端系统性过度自信，必须在 UI 上诚实表达（§5.6.4）。
_EXTREME_BUCKETS = [
    (0.95, 94, 0.9685, 0.9255),
    (0.90, 308, 0.9363, 0.9286),
    (0.85, 750, 0.8993, 0.8880),
]


def _prob_bar(p: float, warns: list[str]) -> None:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.progress(p, text=f"蓝方胜率 {p:.1%}　｜　红方 {1 - p:.1%}")
    with c2:
        st.metric("蓝方胜率", f"{p:.1%}")
    hi = max(p, 1 - p)
    for lo, n, mean_pred, actual in _EXTREME_BUCKETS:
        if hi >= lo:
            st.info(
                f"这个概率落在模型的高自信段。历史上模型给出 ≥{lo:.0%} 的 {n} 场比赛里，"
                f"平均预测 {mean_pred:.1%} 而**实际胜率只有 {actual:.1%}**——"
                f"高自信段是系统性过度自信的，请按 {actual:.0%} 左右理解。",
                icon="📉",
            )
            break
    for w in warns:
        st.warning(w, icon="⚠️")


def _layers(pred) -> None:
    st.caption(
        "把 logit 拆成三层分别看。**实力层通常占绝大部分**——这不是模型偷懒，"
        "是职业赛的真实结构：队伍与选手实力约占可解释信号的 87%，"
        "整个 BP 阶段的可测量贡献只有 ΔlogLoss ≈ 0.002（见 reports/stage4_baseline_ladder.md）。"
    )
    cols = st.columns(3)
    for col, layer in zip(cols, ["实力", "阵容", "赛况"]):
        v = pred.layers.get(layer, 0.0)
        col.metric(f"{layer}层", f"{v:+.3f}", help="对 logit 的贡献，正值利好蓝方")
    with st.expander("逐项贡献"):
        df = pd.DataFrame(
            [{"特征": k, "贡献(logit)": round(v, 4), "原始值": round(pred.features[k], 4)}
             for k, v in sorted(pred.contributions.items(), key=lambda x: -abs(x[1]))]
        )
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption(f"Σ贡献 + 截距 = {pred.logit:+.6f} = logit(p)，闭合误差 0")


def _pick_side(label: str, champs: dict, key: str, default: list[str] | None = None) -> list[int]:
    st.markdown(f"**{label}**")
    keys = []
    cols = st.columns(5)
    for i, (col, role) in enumerate(zip(cols, ROLES)):
        with col:
            d = default[i] if default and i < len(default) else champs["names"][0]
            idx = champs["names"].index(d) if d in champs["names"] else 0
            n = st.selectbox(ROLE_CN[role], champs["names"], index=idx, key=f"{key}_{role}")
            c = champs["by_name"][n]
            p = _icon(c["dd_id"], champs["version"])
            if p:
                st.image(p, width=48)
            keys.append(c["dd_key"])
    return keys


def tab_simulate(art: Artifact, champs: dict, games: pd.DataFrame) -> None:
    st.subheader("阵容模拟")
    leagues = sorted(games.league_canon.unique())
    lg = st.selectbox("赛区", leagues, index=leagues.index("LCK") if "LCK" in leagues else 0)
    sub = games[games.league_canon == lg]
    teams = sorted(set(sub.blue_team_name.dropna()) | set(sub.red_team_name.dropna()))
    if len(teams) < 2:
        st.error("该赛区队伍不足")
        return
    c1, c2 = st.columns(2)
    bt = c1.selectbox("蓝方战队", teams, index=0)
    rt = c2.selectbox("红方战队", teams, index=1)

    def roster(name: str) -> tuple[str, list[str]]:
        m = sub[(sub.blue_team_name == name)]
        if len(m):
            r = m.iloc[0]
            return r.blue_team_id, list(r.blue_player_ids)
        m = sub[(sub.red_team_name == name)]
        if len(m):
            r = m.iloc[0]
            return r.red_team_id, list(r.red_player_ids)
        return "", [""] * 5

    btid, bpl = roster(bt)
    rtid, rpl = roster(rt)
    st.caption("选手默认取该队最近一场的首发。")

    bk = _pick_side("蓝方阵容", champs, "sim_b")
    rk = _pick_side("红方阵容", champs, "sim_r")

    d = Draft(lg, btid, rtid, bpl, rpl, bk, rk)
    pred = predict(art, d)
    st.divider()
    _prob_bar(pred.p_blue, pred.warnings)
    _layers(pred)
    st.session_state["draft"] = d


def tab_whatif(art: Artifact, champs: dict) -> None:
    st.subheader("what-if：换一个英雄会怎样")
    d = st.session_state.get("draft")
    if d is None:
        st.info("先到「阵容模拟」页摆一套阵容。")
        return
    c1, c2 = st.columns(2)
    side = c1.radio("换哪一边", ["blue", "red"], format_func=lambda s: "蓝方" if s == "blue" else "红方",
                    horizontal=True)
    role = c2.selectbox("换哪个位置", ROLES, format_func=lambda r: ROLE_CN[r])
    slot = ROLES.index(role)

    wi = whatif_swap(art, d, side, slot, [c["dd_key"] for c in champs["by_key"].values()])
    wi["英雄"] = wi.champ_key.map(lambda k: champs["by_key"][int(k)]["name"])
    base = wi.attrs["base_p"]
    st.caption(f"当前阵容蓝方胜率 {base:.2%}。下表是把该位置换成各英雄后的变化。")
    st.warning(
        "这是**分布外的相关性外推，不是因果反事实**。队伍只会选自己练过且当前版本能配合的英雄，"
        "所以模型没见过的组合给出的概率可信度很低。换一个英雄的真实影响量级通常在 ±2pp 以内。",
        icon="🧭",
    )
    c1, c2 = st.columns(2)
    c1.markdown("**提升最大**")
    c1.dataframe(wi.head(8)[["英雄", "p_blue", "delta_pp"]].round(4), hide_index=True, width="stretch")
    c2.markdown("**下降最大**")
    c2.dataframe(wi.tail(8).iloc[::-1][["英雄", "p_blue", "delta_pp"]].round(4),
                 hide_index=True, width="stretch")
    st.bar_chart(wi.set_index("英雄").delta_pp.head(20))


def tab_history(art: Artifact, champs: dict, games: pd.DataFrame) -> None:
    st.subheader("历史回看")
    n = st.slider("看最近多少场", 20, 500, 100, step=20)
    recent = games.head(n)
    rows = []
    for r in recent.itertuples():
        d = Draft(r.league_canon, r.blue_team_id, r.red_team_id,
                  list(r.blue_player_ids), list(r.red_player_ids),
                  [int(x) for x in r.blue_champ_keys], [int(x) for x in r.red_champ_keys],
                  now=r.kickoff_utc)
        p = predict(art, d)
        rows.append({
            "日期": str(r.kickoff_utc)[:10], "赛区": r.league_canon,
            "蓝方": r.blue_team_name, "红方": r.red_team_name,
            "预测蓝胜": round(p.p_blue, 3), "实际": "蓝胜" if r.y_blue_win else "红胜",
            "命中": (p.p_blue >= 0.5) == bool(r.y_blue_win),
        })
    df = pd.DataFrame(rows)
    c1, c2 = st.columns(2)
    c1.metric("命中率", f"{df['命中'].mean():.1%}")
    c2.metric("样本", f"{len(df)} 场")
    st.caption(
        "⚠️ 这里包含 **holdout 期**的比赛，仅供肉眼复盘。"
        "任何写进报告的指标都必须走 walk-forward，绝不能用这一页的数字。"
    )
    st.dataframe(df, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="LoL 职业赛胜率预测", layout="wide")
    st.title("英雄联盟职业比赛胜率预测")
    art, champs, games = _art(), _champs(), _games()
    st.caption(
        f"模型 {art.meta['model']}　｜　训练至 {art.meta['trained_on'][:10]}　｜　"
        f"{art.meta['n_train_games']:,} 场　｜　walk-forward OOS log loss {art.meta['oos_log_loss']}"
        f"　｜　英雄数据 Data Dragon {champs['version']}"
    )
    t1, t2, t3 = st.tabs(["阵容模拟", "what-if", "历史回看"])
    with t1:
        tab_simulate(art, champs, games)
    with t2:
        tab_whatif(art, champs)
    with t3:
        tab_history(art, champs, games)

    st.divider()
    st.caption(
        "本项目并非由 Riot Games 开发，也未获得 Riot Games 认可。"
        "Riot Games 及其所有相关财产均为 Riot Games, Inc. 的商标或注册商标。　｜　"
        "比赛数据来自 [Oracle's Elixir](https://oracleselixir.com)，英雄数据来自 Riot Data Dragon。"
    )


if __name__ == "__main__":
    main()
