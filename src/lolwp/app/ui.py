"""设计系统：一份全局 CSS + 一组 HTML 小部件，所有页面共用，页面里不写样式。

- 颜色走 CSS 变量（--lw-*），按 `st.context.theme.type` 选深色 / 浅色两套；
  Streamlit 原生控件的配色在 .streamlit/config.toml 的 [theme] 里。
- 所有进 HTML 的数据（队名、选手名）都经 html.escape：名字来自外部数据源。
- 只用 <img>/<div>/<svg> 与 class / style，不写脚本——st.html 会清洗掉脚本，
  需要交互的地方（阵容板）走 components v2（app/board.py）。
"""
from __future__ import annotations

import hashlib
import html
import math
from collections.abc import Iterable, Sequence
from typing import Any

import streamlit as st

BRAND = "LoLWP"
TAGLINE = "职业赛 BP 胜率"

_DARK = {
    "bg": "#0B1019", "surface": "#111827", "surface-2": "#172033", "raise": "#1C263B",
    "border": "#232D40", "border-strong": "#33405A", "text": "#E6EAF2", "muted": "#8D98AE",
    "faint": "#5F6B82", "gold": "#D7B46A", "gold-soft": "rgba(215,180,106,.13)",
    "blue": "#4F8CFF", "blue-soft": "rgba(79,140,255,.14)", "red": "#FF5C61",
    "red-soft": "rgba(255,92,97,.14)", "green": "#34D399", "green-soft": "rgba(52,211,153,.13)",
    "amber": "#F2A33A", "amber-soft": "rgba(242,163,58,.13)", "shadow": "0 1px 0 rgba(255,255,255,.03) inset, 0 8px 24px rgba(0,0,0,.25)",
}
_LIGHT = {
    "bg": "#F6F7FB", "surface": "#FFFFFF", "surface-2": "#F1F4FA", "raise": "#E9EEF7",
    "border": "#E1E6EF", "border-strong": "#CBD3E1", "text": "#141A26", "muted": "#5B6578",
    "faint": "#8A93A6", "gold": "#A9801F", "gold-soft": "rgba(169,128,31,.10)",
    "blue": "#2563EB", "blue-soft": "rgba(37,99,235,.09)", "red": "#DC2F3A",
    "red-soft": "rgba(220,47,58,.08)", "green": "#0F9F6E", "green-soft": "rgba(15,159,110,.09)",
    "amber": "#C77A0A", "amber-soft": "rgba(199,122,10,.10)", "shadow": "0 1px 2px rgba(16,24,40,.06), 0 6px 18px rgba(16,24,40,.06)",
}

_CSS = """
:root { %VARS% }
/* 承载这份样式的容器本身不占位（见 inject()） */
[data-testid="stElementContainer"]:has(.lw-css-mark), .stElementContainer:has(.lw-css-mark) { display: none; }
/* 开场动画的脚本元素：遮罩画在 document.body 上，它自己的容器不占位 */
[data-testid="stElementContainer"]:has(.lw-intro-mark), .stElementContainer:has(.lw-intro-mark) { display: none; }
/* ── Streamlit 外壳 ─────────────────────────────────────────── */
.stApp { background: radial-gradient(1200px 600px at 12% -8%, var(--lw-blue-soft), transparent 60%),
                     radial-gradient(1000px 520px at 92% -12%, var(--lw-red-soft), transparent 60%),
                     var(--lw-bg); }
header[data-testid="stHeader"] { background: color-mix(in srgb, var(--lw-bg) 82%, transparent);
  backdrop-filter: blur(10px); border-bottom: 1px solid var(--lw-border); }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1320px; padding-top: 4.2rem;
  padding-bottom: 3rem; }
[data-testid="stMetric"] { background: var(--lw-surface); border: 1px solid var(--lw-border);
  border-radius: 14px; padding: 14px 16px 12px; box-shadow: var(--lw-shadow); }
[data-testid="stMetricLabel"] p { color: var(--lw-muted); font-size: .82rem; letter-spacing: .02em; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
[data-testid="stExpander"] details { background: var(--lw-surface); border-radius: 14px; }
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > .lw-card-mark) { padding: 0; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
/* 关键容器：st-key-{key} 是 Streamlit 给带 key 的容器加的稳定 class */
.st-key-lw_panel, .st-key-lw_panel_2, .st-key-lw_filters { background: var(--lw-surface);
  border: 1px solid var(--lw-border); border-radius: 16px; padding: 16px 18px; box-shadow: var(--lw-shadow); }

/* ── 页头 ─────────────────────────────────────────────────── */
.lw-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px;
  margin: 0 0 14px; flex-wrap: wrap; }
.lw-eyebrow { color: var(--lw-gold); font-size: .72rem; font-weight: 700; letter-spacing: .16em;
  text-transform: uppercase; margin-bottom: 6px; }
.lw-title { font-size: 1.9rem; font-weight: 750; letter-spacing: -.02em; color: var(--lw-text);
  margin: 0; line-height: 1.15; }
.lw-sub { color: var(--lw-muted); margin: 8px 0 0; font-size: .95rem; max-width: 760px; line-height: 1.55; }
.lw-head-meta { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }

/* ── 小标签 ───────────────────────────────────────────────── */
.lw-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
  font-size: .76rem; font-weight: 600; color: var(--lw-muted); background: var(--lw-surface-2);
  border: 1px solid var(--lw-border); white-space: nowrap; line-height: 1.3; }
.lw-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.lw-pill.gold { color: var(--lw-gold); background: var(--lw-gold-soft); border-color: transparent; }
.lw-pill.blue { color: var(--lw-blue); background: var(--lw-blue-soft); border-color: transparent; }
.lw-pill.red { color: var(--lw-red); background: var(--lw-red-soft); border-color: transparent; }
.lw-pill.green { color: var(--lw-green); background: var(--lw-green-soft); border-color: transparent; }
.lw-pill.amber { color: var(--lw-amber); background: var(--lw-amber-soft); border-color: transparent; }
.lw-pill.live .dot { animation: lw-pulse 1.6s ease-in-out infinite; }
@keyframes lw-pulse { 0%,100% { opacity: 1 } 50% { opacity: .25 } }

/* ── 卡片 / 小节 ───────────────────────────────────────────── */
.lw-card { background: var(--lw-surface); border: 1px solid var(--lw-border); border-radius: 16px;
  padding: 18px 20px; box-shadow: var(--lw-shadow); }
.lw-section { display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  margin: 26px 0 12px; }
.lw-section h3 { font-size: 1.05rem; font-weight: 700; margin: 0; color: var(--lw-text); letter-spacing: -.005em; }
.lw-section span { color: var(--lw-faint); font-size: .82rem; }

/* ── 队徽 / 英雄头像 ───────────────────────────────────────── */
.lw-badge { display: inline-flex; align-items: center; justify-content: center; flex: none;
  border-radius: 12px; background: var(--lw-surface-2); border: 1px solid var(--lw-border);
  overflow: hidden; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.lw-badge img { width: 78%; height: 78%; object-fit: contain; }
.lw-portrait { position: relative; display: inline-flex; flex: none; border-radius: 10px; overflow: hidden;
  background: var(--lw-surface-2); border: 1px solid var(--lw-border); }
.lw-portrait img { width: 100%; height: 100%; object-fit: cover; transform: scale(1.1); display: block; }
.lw-portrait.empty { display: flex; align-items: center; justify-content: center; color: var(--lw-faint);
  font-size: .7rem; border-style: dashed; }
.lw-portrait.blue { box-shadow: 0 0 0 2px var(--lw-blue-soft); }
.lw-portrait.red { box-shadow: 0 0 0 2px var(--lw-red-soft); }

/* ── 对阵卡（胜率主视觉） ───────────────────────────────────── */
.lw-matchup { display: grid; grid-template-columns: 1fr minmax(320px, 1.25fr) 1fr; align-items: center;
  gap: 18px; padding: 18px 26px; border-radius: 20px; border: 1px solid var(--lw-border);
  background: linear-gradient(100deg, var(--lw-blue-soft), transparent 34%, transparent 66%, var(--lw-red-soft)),
              var(--lw-surface); box-shadow: var(--lw-shadow); }
.lw-mu-team { display: flex; align-items: center; gap: 14px; min-width: 0; }
.lw-mu-team.red { flex-direction: row-reverse; text-align: right; }
.lw-mu-name { font-size: 1.35rem; font-weight: 750; color: var(--lw-text); letter-spacing: -.015em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lw-mu-meta { color: var(--lw-muted); font-size: .8rem; margin-top: 3px; }
.lw-mu-side { font-weight: 700; }
.lw-mu-team.blue .lw-mu-side { color: var(--lw-blue); }
.lw-mu-team.red .lw-mu-side { color: var(--lw-red); }
.lw-mu-center { text-align: center; }
.lw-mu-probs { display: flex; align-items: baseline; justify-content: space-between; gap: 10px;
  font-variant-numeric: tabular-nums; }
.lw-mu-probs .p { font-size: 2.6rem; font-weight: 800; letter-spacing: -.035em; line-height: 1; }
.lw-mu-probs .p.blue { color: var(--lw-blue); }
.lw-mu-probs .p.red { color: var(--lw-red); }
.lw-mu-probs .p.dim { opacity: .55; }
.lw-mu-label { color: var(--lw-faint); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; }
.lw-bar { position: relative; height: 10px; border-radius: 999px; overflow: hidden; margin: 12px 0 10px;
  background: var(--lw-red); }
.lw-bar .b { position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg, #3B74F0, var(--lw-blue)); }
.lw-bar-wrap { position: relative; margin: 12px 0 10px; }
.lw-bar-wrap .lw-bar { margin: 0; }
.lw-bar-wrap .tick { position: absolute; left: 50%; bottom: -6px; width: 1px; height: 4px; background: var(--lw-faint); }
.lw-mu-verdict { color: var(--lw-muted); font-size: .85rem; }
.lw-mu-verdict b { color: var(--lw-text); }
.lw-mu-extra { color: var(--lw-faint); }

/* ── 胜率构成（瀑布） ───────────────────────────────────────── */
.lw-wf { display: flex; flex-direction: column; gap: 9px; }
.lw-wf-row { display: grid; grid-template-columns: 150px 1fr 92px; gap: 14px; align-items: center; }
.lw-wf-name { color: var(--lw-text); font-weight: 600; font-size: .9rem; }
.lw-wf-name small { display: block; color: var(--lw-faint); font-weight: 500; font-size: .74rem; margin-top: 2px; }
.lw-wf-track { position: relative; height: 22px; border-radius: 6px; background: var(--lw-surface-2); }
.lw-wf-track .axis { position: absolute; top: -4px; bottom: -4px; width: 1px; background: var(--lw-border-strong); }
.lw-wf-scale { position: relative; height: 14px; }
.lw-wf-scale span { position: absolute; transform: translateX(-50%); font-size: .68rem; color: var(--lw-faint);
  font-variant-numeric: tabular-nums; }
.lw-wf-track .seg { position: absolute; top: 3px; bottom: 3px; border-radius: 4px; }
.lw-wf-track .seg.up { background: var(--lw-blue); }
.lw-wf-track .seg.down { background: var(--lw-red); }
.lw-wf-track .seg.total { background: linear-gradient(90deg, #3B74F0, var(--lw-blue)); }
.lw-wf-val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-size: .92rem; }
.lw-wf-val small { display: block; color: var(--lw-faint); font-weight: 500; font-size: .72rem; }
.lw-wf-val.up { color: var(--lw-blue); } .lw-wf-val.down { color: var(--lw-red); }
.lw-wf-sep { border-top: 1px dashed var(--lw-border); margin: 2px 0; }

/* ── 阵容条（只读） ─────────────────────────────────────────── */
.lw-lineups { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.lw-lineup { display: flex; gap: 10px; }
.lw-lineup.red { flex-direction: row-reverse; }
.lw-lu-slot { flex: 1; min-width: 0; text-align: center; }
.lw-lu-slot .lw-portrait { margin: 0 auto 6px; }
.lw-lu-champ { font-size: .78rem; font-weight: 650; color: var(--lw-text); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.lw-lu-player { font-size: .72rem; color: var(--lw-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lw-lu-slot.hl .lw-portrait { box-shadow: 0 0 0 2px var(--lw-gold); }

/* ── 英雄卡（what-if） ───────────────────────────────────────── */
.lw-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(185px, 1fr)); gap: 10px; }
.lw-ccard { display: flex; gap: 10px; align-items: center; padding: 10px 12px; border-radius: 12px;
  background: var(--lw-surface); border: 1px solid var(--lw-border); }
.lw-ccard .nm { font-weight: 650; font-size: .86rem; color: var(--lw-text); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }
.lw-ccard .en { font-size: .7rem; color: var(--lw-faint); }
.lw-ccard .dp { font-variant-numeric: tabular-nums; font-weight: 750; font-size: .95rem; }
.lw-ccard .dp.up { color: var(--lw-green); } .lw-ccard .dp.down { color: var(--lw-red); }
.lw-ccard .pp { font-size: .7rem; color: var(--lw-muted); }

/* ── 实时对局卡 ─────────────────────────────────────────────── */
.lw-live-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.lw-live-title { font-weight: 700; font-size: 1.02rem; color: var(--lw-text); }
.lw-live-msg { color: var(--lw-muted); font-size: .85rem; }

/* ── 提示条 / 空状态 / 页脚 ─────────────────────────────────── */
.lw-note { display: flex; gap: 10px; align-items: flex-start; padding: 11px 14px; border-radius: 12px;
  font-size: .86rem; line-height: 1.55; color: var(--lw-text); background: var(--lw-surface-2);
  border: 1px solid var(--lw-border); margin: 8px 0; }
.lw-note .ic { flex: none; width: 18px; text-align: center; }
.lw-note.warn { background: var(--lw-amber-soft); border-color: transparent; }
.lw-note.info { background: var(--lw-blue-soft); border-color: transparent; }
.lw-note.gold { background: var(--lw-gold-soft); border-color: transparent; }
.lw-empty { text-align: center; padding: 46px 20px; border: 1px dashed var(--lw-border-strong);
  border-radius: 16px; color: var(--lw-muted); }
.lw-empty .t { color: var(--lw-text); font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
.lw-footer { margin-top: 42px; padding-top: 18px; border-top: 1px solid var(--lw-border); color: var(--lw-faint);
  font-size: .76rem; line-height: 1.7; display: flex; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
.lw-footer a { color: var(--lw-muted); }
.lw-kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px; font-size: .88rem; }
.lw-kv dt { color: var(--lw-muted); } .lw-kv dd { margin: 0; color: var(--lw-text); font-variant-numeric: tabular-nums; }
@media (max-width: 900px) {
  .lw-matchup { grid-template-columns: 1fr; text-align: center; }
  .lw-mu-team, .lw-mu-team.red { justify-content: center; flex-direction: row; text-align: left; }
  .lw-lineups { grid-template-columns: 1fr; }
  .lw-wf-row { grid-template-columns: 110px 1fr 80px; }
}
"""


def is_light() -> bool:
    """当前是否浅色主题。默认主题是深色（.streamlit/config.toml）；拿不到主题信息（AppTest）时按深色。"""
    try:
        return st.context.theme.type == "light"
    except Exception:  # noqa: BLE001
        return False


def palette() -> dict[str, str]:
    """当前主题的配色：自定义 HTML 用的 --lw-* 变量。"""
    return _LIGHT if is_light() else _DARK


def inject() -> None:
    """全局样式。每次脚本运行调用一次（main.py 在 pg.run() 之前）。

    ⚠️ 三条实测踩过的路（都不报错）：
    - st.html 只传 <style> → 被送进 event 容器，在 st.navigation 的多页应用里根本没进 DOM；
    - st.html 默认走 DOMPurify（USE_PROFILES html），<style> 标签被整个剥掉；
    - st.markdown 的原始 HTML 块能保留 <style>，但 markdown 渲染器是懒加载的，样式比页面上
      其它 st.html 晚 80 ms 左右才生效，首屏先闪一下无样式的页面。
    所以走 st.html(unsafe_allow_javascript=True)：这条路径的白名单额外放行 <style>，
    而且和页面上其它 st.html 是同一个渲染器、同一批渲染。内容是写死的常量 CSS，没有脚本。
    ⚠️ CSS 正文（含注释）里不能出现「< 后跟字母、/ 或 !」：DOMPurify 的 mXSS 防护见到这种
    文本会把整个 <style> 删掉（实测：注释里写了一句 .st-key-<key>，全站样式就没了）。
    tests/test_app_catalog.py 守着这一条。
    再带一个标记元素，让这份 CSS 把自己所在的那个空容器隐藏掉，不占版面。
    """
    pal = palette()
    css = _CSS.replace("%VARS%", " ".join(f"--lw-{k}: {v};" for k, v in pal.items()))
    st.html(f'<style>{css}</style><span class="lw-css-mark"></span>', unsafe_allow_javascript=True)


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


# ── 原子部件 ─────────────────────────────────────────────────────────


def pill(text: str, kind: str = "", *, dot: bool = False) -> str:
    return f'<span class="lw-pill {kind}">{"<span class=dot></span>" if dot else ""}{esc(text)}</span>'


def _hue(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:4], 16) % 360


def team_badge(team: dict[str, Any], size: int = 44) -> str:
    """队标；没有队标时用队伍代码做一个按代码固定取色的徽章。"""
    logo, code = team.get("logo"), str(team.get("code") or "?")
    style = f"width:{size}px;height:{size}px;border-radius:{max(8, size // 4)}px;"
    if logo:
        return f'<span class="lw-badge" style="{style}"><img src="{esc(logo)}" alt="{esc(code)}"></span>'
    fs = max(10, int(size * (0.36 if len(code) <= 3 else 0.28)))
    bg = f"hsl({_hue(code)} 45% 38%)"
    return (f'<span class="lw-badge" style="{style}background:{bg};font-size:{fs}px">'
            f"{esc(code[:4])}</span>")


def champ_img(c: dict[str, Any] | None, size: int = 44, side: str = "") -> str:
    style = f"width:{size}px;height:{size}px;"
    if not c:
        return f'<span class="lw-portrait empty {side}" style="{style}">—</span>'
    if c.get("icon"):
        return (f'<span class="lw-portrait {side}" style="{style}">'
                f'<img src="{esc(c["icon"])}" alt="{esc(c["en"])}" title="{esc(c["zh"])} · {esc(c["en"])}"></span>')
    return (f'<span class="lw-portrait empty {side}" style="{style};font-weight:700">'
            f'{esc(c["en"][:2])}</span>')


def note(text_html: str, kind: str = "", icon: str = "ⓘ") -> str:
    """提示条。text_html 由调用方保证已转义（允许 <b> 之类的强调）。"""
    return f'<div class="lw-note {kind}"><span class="ic">{icon}</span><div>{text_html}</div></div>'


def section(title: str, hint: str = "") -> None:
    st.html(f'<div class="lw-section"><h3>{esc(title)}</h3><span>{esc(hint)}</span></div>')


def empty(title: str, body: str) -> None:
    st.html(f'<div class="lw-empty"><div class="t">{esc(title)}</div><div>{esc(body)}</div></div>')


def page_header(eyebrow: str, title: str, sub: str = "", pills: Iterable[str] = ()) -> None:
    meta = "".join(pills)
    st.html(f'<div class="lw-head"><div><div class="lw-eyebrow">{esc(eyebrow)}</div>'
            f'<h1 class="lw-title">{esc(title)}</h1>'
            + (f'<p class="lw-sub">{sub}</p>' if sub else "")
            + f'</div><div class="lw-head-meta">{meta}</div></div>')


# ── 组合部件 ─────────────────────────────────────────────────────────


def matchup(blue: dict[str, Any], red: dict[str, Any], p_blue: float, *,
            event: str | None = None, label: str = "胜率", extra: str | None = None) -> str:
    """胜率主视觉：两边队伍 + 大号胜率 + 分割条 + 一句结论（extra 追加在结论后面，弱化显示）。"""
    pb = max(0.0, min(1.0, float(p_blue)))
    fav, edge = (blue, pb - 0.5) if pb >= 0.5 else (red, 0.5 - pb)
    verdict = (f'模型看好 <b>{esc(fav.get("name"))}</b> · 领先 {edge * 200:.1f} 个百分点'
               if edge >= 0.005 else "势均力敌")
    if event:
        verdict += f" · {esc(event)}"
    if extra:
        verdict += f'<span class="lw-mu-extra"> · {esc(extra)}</span>'

    def side(t: dict[str, Any], cls: str, cn: str) -> str:
        return (f'<div class="lw-mu-team {cls}">{team_badge(t, 58)}<div style="min-width:0">'
                f'<div class="lw-mu-name">{esc(t.get("name"))}</div>'
                f'<div class="lw-mu-meta"><span class="lw-mu-side">{cn}</span> · {esc(t.get("league_label") or t.get("league") or "")}</div>'
                f"</div></div>")

    return (f'<div class="lw-matchup">{side(blue, "blue", "蓝色方")}'
            f'<div class="lw-mu-center"><div class="lw-mu-probs">'
            f'<span class="p blue {"" if pb >= 0.5 else "dim"}">{pb:.1%}</span>'
            f'<span class="lw-mu-label">{esc(label)}</span>'
            f'<span class="p red {"" if pb < 0.5 else "dim"}">{1 - pb:.1%}</span></div>'
            f'<div class="lw-bar-wrap"><div class="lw-bar"><div class="b" style="width:{pb * 100:.2f}%"></div></div>'
            f'<span class="tick"></span></div>'
            f'<div class="lw-mu-verdict">{verdict}</div></div>'
            f'{side(red, "red", "红色方")}</div>')


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


# 瀑布与各页「x 层」卡片共用的口径：从截距出发，按这个顺序逐层加上去
LAYER_ORDER = ("赛况", "实力", "阵容")
LAYER_HINT = {"赛况": "蓝方先验（按赛区收缩）", "实力": "选手实力 · 配合 · 赛区强度", "阵容": "五个英雄的版本强度"}


def layer_steps(intercept: float, layers: dict[str, float]) -> dict[str, float]:
    """按 截距→赛况→实力→阵容 逐层加上 logit 贡献，每一层让蓝方胜率变了多少个百分点。

    和瀑布图逐段一一对应，连同截距那一步加起来正好等于「最终胜率 − 50%」。
    不用「去掉这一层会怎样」：那样几层加不回总数，同一层在卡片和瀑布上会是两个数。
    """
    z, out = intercept, {}
    for name in LAYER_ORDER:
        c = float(layers.get(name, 0.0))
        out[name] = (_sig(z + c) - _sig(z)) * 100
        z += c
    return out


# 极端段的实测校准（样本外 23,060 场，reports/stage8_comparison.md）：(下限, 场数, 平均预测, 实际)。
# 模型在高概率端系统性过度自信，必须在 UI 上诚实表达（§5.6.4）。
EXTREME_BUCKETS = [(0.95, 94, 0.9685, 0.9255), (0.90, 308, 0.9363, 0.9286), (0.85, 750, 0.8993, 0.8880)]


def calibration_note(p: float) -> str | None:
    """高自信段的诚实提示（已转义的 HTML）；不在高自信段返回 None。"""
    hi = max(p, 1 - p)
    for lo, n, mean_pred, actual in EXTREME_BUCKETS:
        if hi >= lo:
            return (f"这个概率落在模型的高自信段。历史上模型给出 ≥{lo:.0%} 的 {n} 场比赛里，"
                    f"平均预测 {mean_pred:.1%} 而<b>实际胜率只有 {actual:.1%}</b>——"
                    f"高自信段是系统性过度自信的，请按 {actual:.0%} 左右理解。")
    return None


def waterfall(intercept: float, steps: Sequence[tuple[str, str, float]]) -> str:
    """从 50%（抛硬币）出发，依次加上截距与各层 logit 贡献，每一步画成一段。

    steps: [(名称, 说明, logit 贡献)]。每段长度 = 这一步让蓝方胜率变了多少百分点，
    所以各段首尾相接、加起来正好等于最终胜率（顺序会影响单段大小，总和不受影响）。
    坐标轴按途经的胜率缩放（至少 ±10 个百分点）：几个百分点的段在 0–100% 的轴上看不见。
    """
    all_steps = [("蓝方基准", "截距：各赛区蓝方的平均优势", intercept), *steps]
    path, z = [0.5], 0.0
    for _, _, c in all_steps:
        z += c
        path.append(_sig(z))
    lo = max(0.0, min(min(path) - 0.03, 0.40))
    hi = min(1.0, max(max(path) + 0.03, 0.60))
    lo, hi = min(lo, 1 - hi), max(hi, 1 - lo)          # 以 50% 为中心对称，轴才不会骗人

    def x(p: float) -> float:
        return (p - lo) / (hi - lo) * 100

    rows = []
    for (name, hint, c), p0, p1 in zip(all_steps, path, path[1:]):
        a, b = sorted((x(p0), x(p1)))
        kind = "up" if p1 >= p0 else "down"
        rows.append(
            f'<div class="lw-wf-row"><div class="lw-wf-name">{esc(name)}<small>{esc(hint)}</small></div>'
            f'<div class="lw-wf-track"><div class="axis" style="left:{x(0.5):.2f}%"></div><div class="seg {kind}" '
            f'style="left:{a:.2f}%;width:{max(0.6, b - a):.2f}%"></div></div>'
            f'<div class="lw-wf-val {kind}">{(p1 - p0) * 100:+.1f} pp<small>logit {c:+.3f}</small></div></div>')
    p = path[-1]
    a, b = sorted((x(0.5), x(p)))
    rows.append('<div class="lw-wf-sep"></div>')
    rows.append(
        f'<div class="lw-wf-row"><div class="lw-wf-name">蓝方胜率<small>从 50% 出发的合计</small></div>'
        f'<div class="lw-wf-track"><div class="axis" style="left:{x(0.5):.2f}%"></div><div class="seg {"total" if p >= 0.5 else "down"}" '
        f'style="left:{a:.2f}%;width:{max(0.6, b - a):.2f}%"></div></div>'
        f'<div class="lw-wf-val">{p:.1%}<small>logit {z:+.3f}</small></div></div>')
    ticks = "".join(f'<span style="left:{x(t):.2f}%">{t:.0%}</span>'
                    for t in (lo, (lo + 0.5) / 2, 0.5, (hi + 0.5) / 2, hi))
    rows.append(f'<div class="lw-wf-row"><div></div><div class="lw-wf-scale">{ticks}</div><div></div></div>')
    return f'<div class="lw-wf">{"".join(rows)}</div>'


def lineup(champs: Sequence[dict | None], names: Sequence[str | None], side: str, *,
           roles: Sequence[str] = (), highlight: int | None = None, size: int = 50) -> str:
    slots = []
    for i, (c, n) in enumerate(zip(champs, names)):
        role = f"{roles[i]} · " if i < len(roles) else ""
        slots.append(
            f'<div class="lw-lu-slot {"hl" if i == highlight else ""}">{champ_img(c, size, side)}'
            f'<div class="lw-lu-champ">{esc(c["zh"]) if c else "未选"}</div>'
            f'<div class="lw-lu-player">{esc(role)}{esc(n or "—")}</div></div>')
    return f'<div class="lw-lineup {side}">{"".join(slots)}</div>'


def champ_cards(rows: Iterable[tuple[dict, float, float]]) -> str:
    """what-if 的英雄卡：(英雄, Δpp, 换上后的蓝方胜率)。"""
    out = []
    for c, d, p in rows:
        kind = "up" if d >= 0 else "down"
        out.append(f'<div class="lw-ccard">{champ_img(c, 40)}<div style="min-width:0;flex:1">'
                   f'<div class="nm">{esc(c["zh"])}</div><div class="en">{esc(c["en"])}</div></div>'
                   f'<div style="text-align:right"><div class="dp {kind}">{d:+.2f}</div>'
                   f'<div class="pp">→ {p:.1%}</div></div></div>')
    return f'<div class="lw-cards">{"".join(out)}</div>'


def footer(meta: dict[str, Any], dd_version: str) -> None:
    st.html(
        '<div class="lw-footer"><div>'
        "本项目并非由 Riot Games 开发，也未获得 Riot Games 认可。Riot Games 及其所有相关财产均为 "
        "Riot Games, Inc. 的商标或注册商标。<br>比赛数据来自 "
        '<a href="https://oracleselixir.com" target="_blank">Oracle\'s Elixir</a>，'
        f"英雄数据来自 Riot Data Dragon {esc(dd_version)}，实时阵容来自 lolesports livestats。</div>"
        f'<div style="text-align:right">{esc(BRAND)} · 模型 {esc(meta.get("model"))} · '
        f'训练至 {esc(str(meta.get("trained_on", ""))[:10])}{esc(ratings_note(meta, " · "))}'
        "<br>完全本地运行，离线可用</div></div>")


def ratings_note(meta: dict[str, Any], sep: str = "") -> str:
    """「实力更新至 …」（纯文本）：实力快照随赛果重解（artifact.refresh），和训练截止不是一回事。
    快照不比训练截止新就不写。"""
    import pandas as pd

    t, trained = meta.get("ratings_as_of"), meta.get("trained_on")
    if not t or not trained or pd.Timestamp(t) <= pd.Timestamp(trained) + pd.Timedelta(minutes=1):
        return ""
    return f"{sep}实力更新至 {str(t)[:16]} UTC"
