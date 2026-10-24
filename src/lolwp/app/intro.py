"""开场动画：打开站点时的品牌欢迎页（英雄联盟海克斯科技风格），遮住首轮载入，再平滑交给页面。

体验上的四条约定——让它像产品的一部分，而不是突然弹出来的东西：
1. **每个浏览器会话只放一次**（sessionStorage）：切页、重跑、刷新都不会再放；12 小时内
   看过完整版的人只放约 1 秒的短版（localStorage）。后台标签页打开的不放。
2. **它就是加载页**：至少放完入场（完整版 1.75 秒），之后页面一就绪（页脚出现）立刻退场，
   最长 3.8 秒封顶；不为了动画多让人等。点击、滚轮、任意键随时跳过。
3. **退场是交接不是消失**：中间的徽记飞向左上角的字标并与之交叉淡化，背景淡出的同时
   页面内容轻轻上浮到位——眼睛跟着徽记走，不会觉得画面「跳了一下」。
4. **prefers-reduced-motion**：不画线、不飞、不浮，静态徽记一闪即淡出。

性能：零图片、零字体、零依赖，只有内联 SVG 与 CSS。动画只动 transform / opacity
（外加 112 px 徽记上的 stroke-dashoffset），不用全屏 blur / backdrop-filter / 动画阴影；
徽记飞行走 Web Animations API（合成层）。放完即从 DOM 里整个移除，不留常驻动画。
实测（无头 Chrome，1440×900）：开场全程 300 帧 p95 16.8 ms，零 long task。

挂载：一段内联脚本，经 st.html(unsafe_allow_javascript=True) 和全局样式一起、在第一批元素里渲染，
遮罩从第一帧起就在——[实测] 换成 components v2 组件时，它的 JS 模块晚 80–100 ms 才跑起来，
期间页面已经画出来，看上去是「页面 → 开场 → 页面」闪一下。遮罩挂在 document.body 上，
不在 Streamlit 的 React 树里，所以页面重跑不会打断或重放它。脚本是写死的常量，不含任何外部数据。

⚠️ 脚本正文要先编码成 JSON 字符串（「<」写成 \u003c）再用 new Function 执行：st.html 的 DOMPurify
见到 <script> 的文本里有「< 后跟字母」（这里满是 HTML 模板）会把整段脚本删掉，而且不报错。
"""
from __future__ import annotations

import json

import streamlit as st

from lolwp.app import ui

# 只在开场期间注入 <head>，放完连同遮罩一起删掉。配色取自英雄联盟客户端：
# 金 #C8AA6E / 暗金 #785A28 / 米白 #F0E6D2 / 灰金 #A09B8C / 海克斯青 #0AC8B9
_CSS = r"""
.lwi { position: fixed; inset: 0; z-index: 2147483000; display: flex; flex-direction: column; align-items: center;
  justify-content: center; color: #F0E6D2; user-select: none; -webkit-user-select: none; cursor: default;
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Segoe UI", sans-serif; }
.lwi-bg { position: absolute; inset: 0;
  background: radial-gradient(46% 40% at 50% 41%, rgba(200,170,110,.12), transparent 72%),
              radial-gradient(120% 95% at 50% 38%, #101C30 0%, #0B1019 56%, #070A11 100%); }
.lwi-bg::after { content: ""; position: absolute; inset: 0; opacity: .07;
  background-image: radial-gradient(#C8AA6E 1px, transparent 1.4px); background-size: 26px 26px;
  -webkit-mask-image: radial-gradient(circle at 50% 42%, #000 0, transparent 58%);
          mask-image: radial-gradient(circle at 50% 42%, #000 0, transparent 58%); }
.lwi-stage { position: relative; width: 300px; height: 300px; display: grid; place-items: center; }
.lwi-rings { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible;
  animation: lwi-ring-in .9s .05s cubic-bezier(.2,.7,.2,1) both; }
.lwi-rings .slow { animation: lwi-spin 26s linear infinite; }
.lwi-rings .rev { animation: lwi-spin 7s linear infinite reverse; }
.lwi-glow { position: absolute; width: 230px; height: 230px; border-radius: 50%;
  background: radial-gradient(circle, rgba(215,180,106,.34) 0, rgba(215,180,106,.08) 42%, transparent 66%);
  animation: lwi-glow 1.5s .45s ease-out both; }
.lwi-emblem { position: relative; width: 112px; height: 112px; overflow: visible; }
.lwi-emblem .edge { stroke-dasharray: 1; animation: lwi-draw .8s .15s cubic-bezier(.6,0,.3,1) both; }
.lwi-emblem .fill { transform-box: fill-box; transform-origin: center; animation: lwi-fill .5s .72s cubic-bezier(.2,.7,.2,1) both; }
.lwi-emblem .mark { stroke-dasharray: 1; animation: lwi-draw .5s .98s cubic-bezier(.5,0,.3,1) both; }
.lwi-spark { position: absolute; left: 50%; top: 60%; width: 3px; height: 3px; margin-left: -1.5px; border-radius: 50%;
  background: #F3D58F; box-shadow: 0 0 6px 1px rgba(243,213,143,.65); opacity: 0;
  animation: lwi-spark 2s var(--d) cubic-bezier(.2,.6,.4,1) infinite; }
.lwi-title { margin-top: 14px; font-size: 46px; font-weight: 800; letter-spacing: .14em; padding-left: .14em; line-height: 1;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
  animation: lwi-track .9s .72s cubic-bezier(.2,.7,.2,1) both; }
.lwi-title .g { color: #D7B46A; }
.lwi-divider { position: relative; width: 280px; height: 12px; margin: 18px 0 14px; }
.lwi-divider::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px;
  background: linear-gradient(90deg, transparent, #785A28 18%, #C8AA6E 50%, #785A28 82%, transparent);
  animation: lwi-line .7s .95s cubic-bezier(.2,.7,.2,1) both; }
.lwi-divider i { position: absolute; left: 50%; top: 50%; width: 8px; height: 8px; margin: -4px 0 0 -4px;
  border: 1px solid #C8AA6E; background: #0B1019; transform: rotate(45deg); animation: lwi-gem .5s 1.2s ease-out both; }
.lwi-tag { text-align: center; animation: lwi-up .6s 1.05s cubic-bezier(.2,.7,.2,1) both; }
.lwi-tag b { display: block; font-size: 15px; font-weight: 600; letter-spacing: .34em; padding-left: .34em; color: #F0E6D2; }
.lwi-tag small { display: block; margin-top: 9px; font-size: 10.5px; letter-spacing: .42em; padding-left: .42em; color: #A09B8C;
  font-family: "Beaufort for LOL", Cinzel, Didot, "Bodoni 72", Georgia, "Times New Roman", serif; text-transform: uppercase; }
.lwi-status { position: absolute; bottom: 70px; left: 0; right: 0; display: flex; flex-direction: column; align-items: center;
  gap: 11px; animation: lwi-in .5s .85s both; }
.lwi-bar { width: 220px; height: 2px; border-radius: 2px; background: rgba(200,170,110,.16); overflow: hidden; }
.lwi-bar i { display: block; height: 100%; background: linear-gradient(90deg, #785A28, #F3D58F); transform-origin: 0 50%;
  transform: scaleX(.04); animation: lwi-load var(--min) .2s cubic-bezier(.1,.6,.3,1) forwards; }
.lwi-status span { font-size: 11px; letter-spacing: .26em; padding-left: .26em; color: #A09B8C; }
.lwi-skip { position: absolute; bottom: 28px; font-size: 11px; letter-spacing: .22em; color: rgba(160,155,140,.55);
  animation: lwi-in .5s 1.6s both; }

/* 短版：12 小时内看过完整版 */
.lwi.quick .lwi-emblem .edge, .lwi.quick .lwi-emblem .mark, .lwi.quick .lwi-emblem .fill { animation: none; }
.lwi.quick .lwi-rings { animation-duration: .5s; animation-delay: 0s; }
.lwi.quick .lwi-glow { animation-duration: .8s; animation-delay: 0s; }
.lwi.quick .lwi-title { animation-duration: .55s; animation-delay: .08s; }
.lwi.quick .lwi-divider::before { animation-duration: .45s; animation-delay: .15s; }
.lwi.quick .lwi-divider i { animation-delay: .3s; }
.lwi.quick .lwi-tag { animation-duration: .45s; animation-delay: .22s; }
.lwi.quick .lwi-status { animation-delay: .25s; }
.lwi.quick .lwi-spark, .lwi.quick .lwi-skip { display: none; }

/* 退场（正常收尾；中途跳过走整体淡出，见脚本） */
.lwi.out .lwi-title, .lwi.out .lwi-divider, .lwi.out .lwi-tag, .lwi.out .lwi-status, .lwi.out .lwi-skip {
  animation: lwi-away .34s ease-in forwards; }
.lwi.out .lwi-rings, .lwi.out .lwi-glow, .lwi.out .lwi-spark { animation: lwi-out .3s ease-in forwards; }
.lwi.out .lwi-bg { animation: lwi-out .5s .14s ease-in forwards; }
html.lwi-on img[data-testid="stHeaderLogo"] { opacity: 0; }
img[data-testid="stHeaderLogo"] { transition: opacity .28s ease; }
html.lwi-reveal [data-testid="stMainBlockContainer"] { animation: lwi-rise .62s cubic-bezier(.2,.7,.2,1) both; }

.lwi.reduce *, .lwi.reduce { animation: none !important; }
.lwi.reduce .lwi-spark, .lwi.reduce .lwi-skip, .lwi.reduce .lwi-rings .rev { display: none; }
@media (prefers-reduced-motion: reduce) {
  .lwi *, .lwi { animation: none !important; } .lwi-spark { display: none; }
  html.lwi-reveal [data-testid="stMainBlockContainer"] { animation: none; }
}

@keyframes lwi-in { from { opacity: 0; } }
@keyframes lwi-out { to { opacity: 0; } }
@keyframes lwi-away { to { opacity: 0; transform: translateY(-8px); } }
@keyframes lwi-ring-in { from { opacity: 0; transform: scale(.9); } }
@keyframes lwi-spin { to { transform: rotate(360deg); } }
@keyframes lwi-glow { 0% { opacity: 0; transform: scale(.7); } 45% { opacity: 1; } 100% { opacity: .7; transform: scale(1); } }
@keyframes lwi-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
@keyframes lwi-fill { from { opacity: 0; transform: scale(.86); } }
@keyframes lwi-spark { 0% { opacity: 0; transform: translate(var(--x), 0) scale(.6); } 18% { opacity: 1; }
  100% { opacity: 0; transform: translate(calc(var(--x) * 1.7), -130px) scale(1); } }
@keyframes lwi-track { from { opacity: 0; letter-spacing: .62em; } }
@keyframes lwi-line { from { opacity: 0; transform: scaleX(.15); } }
@keyframes lwi-gem { from { opacity: 0; transform: rotate(-135deg) scale(.4); } }
@keyframes lwi-up { from { opacity: 0; transform: translateY(7px); } }
@keyframes lwi-load { to { transform: scaleX(.84); } }
@keyframes lwi-rise { from { opacity: 0; transform: translateY(12px); } }
"""

_HEX = "M18 2.5 31.5 10.25v15.5L18 33.5 4.5 25.75v-15.5Z"          # 与 assets/ 里的徽记同一条路径
_W = "M10.8 12.6 14.4 23.6 18 15.4l3.6 8.2 3.6-11"

_JS = r"""
(function () {
const SHOWN = "lwp:intro:shown", FULL = "lwp:intro:full";
const CSS = %CSS%;
const HEX = %HEX%, W = %W%;
const DATA = %DATA%;

const store = (k) => { try { return window[k]; } catch (e) { return null; } };
const get = (s, k) => { try { return s && s.getItem(k); } catch (e) { return null; } };
const put = (s, k, v) => { try { s && s.setItem(k, v); } catch (e) { /* 隐私模式：只在内存里记 */ } };
const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function markup(d) {
  // 符文环：刻度（虚线圆）+ 四个方位菱形 + 一圈细金线 + 反向旋转的海克斯青弧
  const gems = [0, 90, 180, 270].map((a) => `<rect x="-3.5" y="-151.5" width="7" height="7" transform="rotate(${a}) rotate(45 0 -148)" fill="#0B1019" stroke="#C8AA6E" stroke-width="1"/>`).join("");
  const sparks = [-46, -30, -16, -6, 6, 18, 32, 44, -24, 26].map((x, i) =>
    `<i class="lwi-spark" style="--x:${x}px;--d:${(0.55 + i * 0.17).toFixed(2)}s"></i>`).join("");
  return `<div class="lwi-bg"></div>
  <div class="lwi-stage">
    <svg class="lwi-rings" viewBox="-160 -160 320 320" aria-hidden="true">
      <g class="slow"><circle r="148" fill="none" stroke="#C8AA6E" stroke-opacity=".38" stroke-width="5" stroke-dasharray="1.2 14.3"/>${gems}</g>
      <circle r="134" fill="none" stroke="#C8AA6E" stroke-opacity=".28" stroke-width="1"/>
      <g class="rev"><circle r="120" fill="none" stroke="#0AC8B9" stroke-opacity=".55" stroke-width="1.6" stroke-linecap="round" stroke-dasharray="120 64 26 64 210 270"/></g>
      <circle r="104" fill="none" stroke="#785A28" stroke-opacity=".5" stroke-width="1"/>
    </svg>
    <div class="lwi-glow"></div>
    <svg class="lwi-emblem" viewBox="0 0 36 36" aria-hidden="true">
      <defs><linearGradient id="lwi-gold" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F3D58F"/><stop offset="1" stop-color="#B4863A"/></linearGradient></defs>
      <path class="fill" d="${HEX}" fill="url(#lwi-gold)"/>
      <path class="edge" d="${HEX}" pathLength="1" fill="none" stroke="#F3D58F" stroke-width=".8" stroke-linejoin="round"/>
      <path class="mark" d="${W}" pathLength="1" fill="none" stroke="#0B1019" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    ${sparks}
  </div>
  <div class="lwi-title">${esc(d.brand_a)}<span class="g">${esc(d.brand_b)}</span></div>
  <div class="lwi-divider"><i></i></div>
  <div class="lwi-tag"><b>${esc(d.tagline)}</b><small>${esc(d.sub)}</small></div>
  <div class="lwi-status"><div class="lwi-bar"><i></i></div><span>${esc(d.status)}</span></div>
  <div class="lwi-skip">点击或按任意键跳过</div>`;
}

function play(d, quick, reduce) {
  const doc = document, html = doc.documentElement;
  const style = doc.createElement("style");
  style.id = "lwi-style";
  style.textContent = CSS;
  doc.head.appendChild(style);
  const minT = reduce ? 0 : quick ? 900 : 1750, capT = reduce ? 1500 : quick ? 2600 : 3800;
  const root = doc.createElement("div");
  root.className = "lwi" + (quick ? " quick" : "") + (reduce ? " reduce" : "");
  root.setAttribute("aria-hidden", "true");
  root.style.setProperty("--min", `${minT || 1}ms`);
  root.innerHTML = markup(d);
  doc.body.appendChild(root);
  html.classList.add("lwi-on");

  const t0 = performance.now();
  let done = false, timer = 0;
  const ready = () => !!doc.querySelector(".lw-footer");   // 页脚 = 首轮渲染跑完
  const skip = () => finish(true);
  const opts = { capture: true };
  const off = () => {
    root.removeEventListener("pointerdown", skip, opts);
    window.removeEventListener("keydown", skip, opts);
    window.removeEventListener("wheel", skip, opts);
  };
  const tick = () => {
    if (done) return;
    const e = performance.now() - t0;
    if ((e >= minT && ready()) || e >= capT) finish(false);
    else timer = setTimeout(tick, 110);
  };

  function finish(fast) {
    if (done) return;
    done = true;
    clearTimeout(timer);
    off();
    const bar = root.querySelector(".lwi-bar i");
    if (bar) {   // 进度条补满：先停掉 CSS 动画，再从当前位置过渡到 100%
      const cur = getComputedStyle(bar).transform;
      bar.style.animation = "none";
      bar.style.transform = cur === "none" ? "scaleX(.84)" : cur;
      void bar.offsetWidth;
      bar.style.transition = "transform .2s ease-out";
      bar.style.transform = "scaleX(1)";
    }
    const D = reduce ? 200 : fast ? 420 : 720;
    if (reduce || fast) {
      root.animate([{ opacity: 1 }, { opacity: 0 }], { duration: reduce ? 200 : 300, easing: "ease-in", fill: "forwards" });
    } else {
      root.classList.add("out");
    }
    const em = root.querySelector(".lwi-emblem"), logo = doc.querySelector('img[data-testid="stHeaderLogo"]');
    if (!reduce && em && logo && logo.getBoundingClientRect().height > 0) {
      // 徽记飞进左上角字标：字标高 h，徽记占它最左边的 h×h 方块（同一个 viewBox 36×36）
      const a = em.getBoundingClientRect(), b = logo.getBoundingClientRect(), h = b.height;
      const dx = b.left + h / 2 - (a.left + a.width / 2), dy = b.top + h / 2 - (a.top + a.height / 2);
      em.animate([
        { transform: "translate(0, 0) scale(1)", opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) scale(${h / a.width})`, opacity: 1, offset: 0.86 },
        { transform: `translate(${dx}px, ${dy}px) scale(${h / a.width})`, opacity: 0 },
      ], { duration: D, easing: "cubic-bezier(.65,0,.25,1)", fill: "forwards" });
    }
    if (!reduce) setTimeout(() => html.classList.add("lwi-reveal"), fast ? 60 : 140);
    setTimeout(() => html.classList.remove("lwi-on"), reduce ? 0 : Math.round(D * 0.82));
    setTimeout(() => root.remove(), D + 240);
    setTimeout(() => { html.classList.remove("lwi-reveal"); style.remove(); }, D + 900);
  }

  root.addEventListener("pointerdown", skip, opts);
  window.addEventListener("keydown", skip, opts);
  window.addEventListener("wheel", skip, { ...opts, passive: true });
  timer = setTimeout(tick, Math.min(minT, 300));
}

const ss = store("sessionStorage"), ls = store("localStorage");
// 元素重新挂载时脚本会再跑一遍：放过（或正在放）就什么都不做
if (window.__lwpIntro || get(ss, SHOWN)) return;
window.__lwpIntro = true;
put(ss, SHOWN, "1");
if (document.hidden) return;                         // 后台标签页打开的：等用户切过来早就加载完了
const reduce = !!(window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
const quick = Date.now() - Number(get(ls, FULL) || 0) < 12 * 3600 * 1000;
if (!quick && !reduce) put(ls, FULL, String(Date.now()));
play(DATA, quick, reduce);
})();
"""


def _data() -> dict[str, str]:
    brand_a, brand_b = ui.BRAND[:-2], ui.BRAND[-2:]          # 「LoL」+ 金色的「WP」，与字标一致
    return {"brand_a": brand_a, "brand_b": brand_b, "tagline": ui.TAGLINE + "预测",
            "sub": "Pro draft · win probability", "status": "正在载入模型与战队数据"}


def script() -> str:
    """开场脚本全文（常量拼装；json.dumps 负责转义）。"""
    return (_JS.replace("%CSS%", json.dumps(_CSS)).replace("%HEX%", json.dumps(_HEX))
            .replace("%W%", json.dumps(_W)).replace("%DATA%", json.dumps(_data(), ensure_ascii=False)))


def element() -> str:
    """真正放进 st.html 的 HTML：脚本正文编码成 JSON 字符串、「<」一律写成 \\u003c，
    所以 <script> 里一个「<」都没有——DOMPurify 不会删它，也不可能提前闭合标签。"""
    src = json.dumps(script(), ensure_ascii=False).replace("<", "\\u003c")
    return f'<script>(new Function({src}))()</script><span class="lw-intro-mark"></span>'


def intro() -> None:
    """挂载开场动画。每次脚本运行都挂（容器被 CSS 隐藏、不占位）；放不放、放哪一版由浏览器端决定。"""
    st.html(element(), unsafe_allow_javascript=True)
