"""阵容板：点头像选英雄、按位置筛、每个位置换选手。Streamlit components v2，无需构建步骤。

交互全在浏览器里（选中的位置、位置筛选、搜索、键盘导航），只有「阵容真的变了」
才发一个事件回 Python。三条约定，偏离了会出现「点了没反应」或「选的人被改回去」：

1. **Python 是唯一的真值**（st.session_state["sim"]）。组件每次发的是**整份阵容**
   （10 个英雄 + 10 名选手），不是增量——连点两下时 Streamlit 会把同一轮里的多个事件
   合并成最后一个，增量事件会丢一次点击，整份阵容不会。
2. **乐观更新 + 序号对齐**：组件先在本地改好再发事件，事件带 seq；Python 应用后把 seq
   回传。组件只在 Python 追上自己发出的最后一个 seq 之后，才用 Python 回传的阵容覆盖本地，
   所以快速连点时不会闪回旧阵容。
3. **rev 防止旧事件覆盖新队伍**：换队、交换蓝红、清空都由 Python 发起，rev 加一。
   带着旧 rev 的事件（换队那一刻浏览器里还在点）只采纳英雄，不采纳选手——选手是跟队走的。

浏览器端的状态（当前位置、筛选、搜索词）挂在组件根节点上，Python 重跑不会把它清掉。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from lolwp.app import component
from lolwp.app.catalog import ROLE_CN, ROLES

_CSS = r"""
:host { display: block; }
.lw { font-family: var(--st-font); color: var(--st-text-color); --bl: var(--st-blue-color, #4F8CFF);
  --rd: var(--st-red-color, #FF5C61); --gd: var(--st-primary-color); --bd: var(--st-border-color, rgba(128,128,128,.25));
  --sf: var(--st-secondary-background-color); --bg: var(--st-background-color);
  --mt: color-mix(in srgb, var(--st-text-color) 60%, transparent); --ft: color-mix(in srgb, var(--st-text-color) 40%, transparent); }
.draft { display: grid; grid-template-columns: minmax(270px, 320px) minmax(0, 1fr) minmax(270px, 320px); gap: 14px; align-items: stretch; }
.side { border: 1px solid var(--bd); border-radius: 18px; padding: 12px 10px; display: flex; flex-direction: column;
  background: linear-gradient(180deg, color-mix(in srgb, var(--bl) 13%, transparent), transparent 140px), var(--bg); }
.side.red { background: linear-gradient(180deg, color-mix(in srgb, var(--rd) 13%, transparent), transparent 140px), var(--bg); }
.side-head { display: flex; align-items: center; gap: 11px; padding: 2px 6px 12px; }
.side.red .side-head { flex-direction: row-reverse; text-align: right; }
.badge { width: 40px; height: 40px; border-radius: 11px; display: inline-flex; align-items: center; justify-content: center;
  background: var(--sf); border: 1px solid var(--bd); overflow: hidden; font-weight: 800; font-size: 13px; color: #fff; flex: none; }
.badge img { width: 78%; height: 78%; object-fit: contain; }
.tname { font-weight: 750; font-size: 16px; letter-spacing: -.01em; line-height: 1.2; }
.tmeta { font-size: 11.5px; color: var(--mt); margin-top: 2px; }
.side.blue .tmeta b { color: var(--bl); } .side.red .tmeta b { color: var(--rd); }
.slots { display: flex; flex-direction: column; gap: 5px; flex: 1; }
.slot { display: grid; grid-template-columns: 52px minmax(0, 1fr) 22px; align-items: center; gap: 10px; padding: 6px 7px;
  border-radius: 13px; border: 1px solid transparent; cursor: pointer; transition: background .12s, border-color .12s; }
.side.red .slot { grid-template-columns: 22px minmax(0, 1fr) 52px; }
.side.red .slot > .pt { order: 3; } .side.red .slot > .info { order: 2; text-align: right; } .side.red .slot > .x { order: 1; }
.slot:hover { background: var(--sf); }
.slot.active { border-color: color-mix(in srgb, var(--gd) 70%, transparent); background: color-mix(in srgb, var(--gd) 8%, transparent); }
.pt { width: 52px; height: 52px; border-radius: 12px; overflow: hidden; border: 1px dashed var(--bd); background: var(--sf);
  padding: 0; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--ft); font-size: 20px; }
.slot.filled .pt { border-style: solid; }
.pt img { width: 100%; height: 100%; object-fit: cover; transform: scale(1.1); display: block; }
.side.blue .slot.filled .pt { box-shadow: 0 0 0 2px color-mix(in srgb, var(--bl) 50%, transparent); }
.side.red .slot.filled .pt { box-shadow: 0 0 0 2px color-mix(in srgb, var(--rd) 50%, transparent); }
.slot.active .pt { box-shadow: 0 0 0 2px var(--gd) !important; }
.info { min-width: 0; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-width: 0; }
.side.red .top { flex-direction: row-reverse; }
.role { flex: none; display: inline-flex; align-items: center; gap: 4px; font-size: 10.5px; font-weight: 700; color: var(--mt);
  letter-spacing: .06em; }
.side.red .role { flex-direction: row-reverse; }
.role svg { width: 13px; height: 13px; }
.cname { font-weight: 700; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.cname .en { font-weight: 500; font-size: 10.5px; color: var(--ft); margin: 0 4px; }
.cname.none { color: var(--ft); font-weight: 600; font-size: 13px; }
select.player { margin-top: 5px; width: 100%; background: var(--sf); color: var(--st-text-color); border: 1px solid var(--bd);
  border-radius: 8px; padding: 3px 6px; font: inherit; font-size: 12px; cursor: pointer; text-overflow: ellipsis; }
.side.red select.player { text-align: right; }
.x { width: 22px; height: 22px; border-radius: 7px; border: none; background: transparent; color: var(--ft);
  cursor: pointer; font-size: 15px; line-height: 1; visibility: hidden; padding: 0; }
.slot.filled:hover .x { visibility: visible; } .x:hover { background: var(--bg); color: var(--st-text-color); }
.picker { border: 1px solid var(--bd); border-radius: 18px; padding: 14px; background: var(--bg); display: flex;
  flex-direction: column; min-width: 0; }
.pk-head { display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
.pk-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.pk-target { font-size: 13.5px; color: var(--mt); } .pk-target b { color: var(--st-text-color); }
.pk-target b.blue { color: var(--bl); } .pk-target b.red { color: var(--rd); }
.seg { display: inline-flex; background: var(--sf); border: 1px solid var(--bd); border-radius: 10px; padding: 3px; gap: 2px; flex-wrap: wrap; }
.seg button { border: none; background: transparent; color: var(--mt); font: inherit; font-size: 12.5px; font-weight: 650;
  padding: 5px 10px; border-radius: 7px; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; }
.seg button svg { width: 13px; height: 13px; }
.seg button.on { background: var(--bg); color: var(--st-text-color); box-shadow: 0 1px 2px rgba(0,0,0,.25); }
.search { flex: 1; display: flex; align-items: center; gap: 6px; background: var(--sf); border: 1px solid var(--bd); border-radius: 10px;
  padding: 0 10px; height: 34px; min-width: 190px; }
.search:focus-within { border-color: var(--gd); }
.search input { border: none; outline: none; background: transparent; color: var(--st-text-color); font: inherit; font-size: 13px; width: 100%; }
.search svg { width: 15px; height: 15px; opacity: .6; flex: none; }
.ghost { border: 1px solid var(--bd); background: transparent; color: var(--mt); font: inherit; font-size: 12.5px;
  border-radius: 9px; padding: 0 11px; height: 34px; cursor: pointer; white-space: nowrap; }
.ghost:hover { color: var(--st-text-color); background: var(--sf); }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px; align-content: start;
  flex: 1 1 auto; height: 0; min-height: 300px; overflow-y: auto; padding: 2px 2px 18px; scrollbar-width: thin;
  -webkit-mask-image: linear-gradient(to bottom, #000 calc(100% - 28px), transparent);
          mask-image: linear-gradient(to bottom, #000 calc(100% - 28px), transparent); }
.tile { border: 1px solid transparent; background: transparent; padding: 5px 2px 6px; border-radius: 11px; cursor: pointer;
  color: var(--st-text-color); font: inherit; display: flex; flex-direction: column; align-items: center; gap: 5px; min-width: 0; }
.tile .im { width: 54px; height: 54px; border-radius: 12px; overflow: hidden; border: 1px solid var(--bd); background: var(--sf);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px; color: var(--mt); }
.tile img { width: 100%; height: 100%; object-fit: cover; transform: scale(1.1); display: block; }
.tile span { font-size: 11.5px; font-weight: 600; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tile:hover { background: var(--sf); } .tile:hover .im { border-color: var(--gd); }
.tile.hl { background: var(--sf); border-color: var(--gd); }
.tile.taken { cursor: not-allowed; opacity: .3; filter: grayscale(.9); }
.tile.taken.blue .im { box-shadow: 0 0 0 2px var(--bl); } .tile.taken.red .im { box-shadow: 0 0 0 2px var(--rd); }
.tile.cur .im { box-shadow: 0 0 0 2px var(--gd); }
.pk-foot { margin-top: 10px; font-size: 11.5px; color: var(--ft); display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
.none-found { grid-column: 1 / -1; text-align: center; padding: 40px 10px; color: var(--ft); font-size: 13px; }
@media (max-width: 1100px) { .draft { grid-template-columns: 1fr 1fr; } .picker { grid-column: 1 / -1; order: 3; } }
@media (max-width: 640px) { .draft { grid-template-columns: 1fr; } }
"""

_JS = r"""
const ROLE_SVG = {
  top: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 3h13l-4 4H7v5l-4 4z" opacity=".95"/><path d="M10 10h4v4h-4zM21 7v14H7l3-3h8V10z" opacity=".35"/></svg>',
  jng: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M5 3c3 3 4 7 4 11l3 7 3-7c0-4 1-8 4-11-3 2-5 5-6 8h-2C10 8 8 5 5 3z"/></svg>',
  mid: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M17 3h4v4L7 21H3v-4z"/><path d="M3 3h9L9 6H6v3L3 12zM21 21h-9l3-3h3v-3l3-3z" opacity=".35"/></svg>',
  bot: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M21 21H8l4-4h5v-5l4-4z" opacity=".95"/><path d="M10 10h4v4h-4zM3 17V3h14l-3 3H6v8z" opacity=".35"/></svg>',
  sup: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3l3 3-3 5-3-5zM2 8h7l1.5 3L8 14l-3-1zM22 8h-7l-1.5 3L16 14l3-1zM10 13h4l-1 8h-2z"/></svg>',
};
const ORDER = [["blue",0],["blue",1],["blue",2],["blue",3],["blue",4],["red",0],["red",1],["red",2],["red",3],["red",4]];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const clone = (x) => JSON.parse(JSON.stringify(x));

function img(c, cls) {
  if (c && c.img) return `<img class="${cls||""}" src="${esc(c.img)}" alt="${esc(c.en)}" loading="lazy" draggable="false">`;
  return c ? esc(c.en.slice(0, 2)) : "+";
}

function badge(t) {
  if (t.logo) return `<span class="badge"><img src="${esc(t.logo)}" alt=""></span>`;
  return `<span class="badge" style="background:hsl(${t.hue} 45% 38%)">${esc((t.code||"?").slice(0,4))}</span>`;
}

function taken(st) {
  const m = {};
  for (const [side, i] of ORDER) { const k = st.local[side].champs[i]; if (k != null) m[k] = [side, i]; }
  return m;
}

function emit(st) {
  st.sent += 1;
  st.send("draft", { blue: st.local.blue, red: st.local.red, seq: st.sent, rev: st.rev });
}

function nextEmpty(st, from) {
  const start = ORDER.findIndex(([s, i]) => s === from.side && i === from.i);
  for (let k = 1; k <= ORDER.length; k++) {
    const [s, i] = ORDER[(start + k) % ORDER.length];
    if (st.local[s].champs[i] == null) return { side: s, i };
  }
  return from;
}

function playerOptions(list, sel) {
  const opts = list.map((p) => {
    const tier = p.pct == null ? "新人" : `前${Math.max(1, 100 - p.pct)}%`;
    return `<option value="${esc(p.id)}" ${p.id === sel ? "selected" : ""}>${esc(p.name)} · ${tier} · ${p.games}场</option>`;
  });
  if (!list.some((p) => p.id === sel)) opts.unshift(`<option value="" selected>（无选手数据）</option>`);
  return opts.join("");
}

function renderSides(root, st) {
  const d = st.data;
  for (const side of ["blue", "red"]) {
    const S = d.sides[side], L = st.local[side];
    const rows = d.roles.map(([role, cn], i) => {
      const k = L.champs[i], c = k == null ? null : d.byKey[k];
      const act = st.active.side === side && st.active.i === i;
      const name = c ? (side === "blue" ? `${esc(c.zh)}<span class="en">${esc(c.en)}</span>`
                                         : `<span class="en">${esc(c.en)}</span>${esc(c.zh)}`) : "选择英雄";
      return `<div class="slot ${act ? "active" : ""} ${c ? "filled" : ""}" data-side="${side}" data-i="${i}">
        <button class="pt" title="${c ? "更换英雄" : "选择英雄"}">${img(c)}</button>
        <div class="info"><div class="top"><div class="cname ${c ? "" : "none"}">${name}</div>
          <div class="role">${ROLE_SVG[role] || ""}<span>${esc(cn)}</span></div></div>
          <select class="player" data-side="${side}" data-i="${i}" title="换选手：该队最近一年在这个位置上场过的选手。「前 x%」是分层 BT 实力在全部已知选手里的排位，「新人」= 模型没见过">${playerOptions(S.players[i], L.players[i])}</select></div>
        <button class="x" title="清除这个英雄">×</button></div>`;
    }).join("");
    root.querySelector(`section.side.${side}`).innerHTML = `<header class="side-head">${badge(S.team)}<div style="min-width:0">
      <div class="tname">${esc(S.team.name)}</div><div class="tmeta"><b>${side === "blue" ? "蓝色方" : "红色方"}</b> · ${esc(S.team.league)}</div>
      </div></header><div class="slots">${rows}</div>`;
  }
}

function visible(st) {
  const d = st.data, q = st.q.trim().toLowerCase();
  if (q) {
    const score = (c) => {
      const f = [c.zh, c.en, c.ep, c.id].map((s) => String(s || "").toLowerCase());
      if (f.some((s) => s === q)) return 0;
      if (f.some((s) => s.startsWith(q))) return 1;
      if (f.some((s) => s.includes(q))) return 2;
      return 9;
    };
    return d.champs.map((c) => [score(c), c]).filter(([s]) => s < 9).sort((a, b) => a[0] - b[0]).map(([, c]) => c);
  }
  if (st.filter && st.filter !== "all") return (d.pool[st.filter] || []).map((k) => d.byKey[k]);
  return d.champs;
}

function renderPicker(root, st) {
  const d = st.data, side = st.active.side, i = st.active.i;
  const team = d.sides[side].team, roleCn = d.roles[i][1];
  root.querySelector(".pk-target").innerHTML =
    `为 <b class="${side}">${esc(team.name)} · ${esc(roleCn)}</b> 选择英雄`;
  root.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.f === (st.filter || "all")));
  const tk = taken(st), cur = st.local[side].champs[i];
  const list = visible(st);
  st.list = list;
  st.hl = Math.min(st.hl, Math.max(0, list.length - 1));
  const grid = root.querySelector(".grid"), keep = grid.scrollTop;
  grid.innerHTML = list.length ? list.map((c, idx) => {
    const t = tk[c.key], mine = t && t[0] === side && t[1] === i;
    const cls = ["tile", t && !mine ? `taken ${t[0]}` : "", mine ? "cur" : "", idx === st.hl && st.q ? "hl" : ""].join(" ");
    const n = st.filter && st.filter !== "all" ? ` · ${roleCn} ${c.n[st.filter] || 0} 场` : "";
    const tip = `${c.zh} · ${c.ep} · ${c.en}${n}${t && !mine ? "（已被选）" : ""}`;
    return `<button class="${cls}" data-k="${c.key}" title="${esc(tip)}" ${t && !mine ? "disabled" : ""}>
      <div class="im">${img(c)}</div><span>${esc(c.zh)}</span></button>`;
  }).join("") : `<div class="none-found">没有匹配「${esc(st.q)}」的英雄</div>`;
  grid.scrollTop = st.keepScroll ? keep : 0;
  st.keepScroll = true;
  const n = Object.keys(tk).length;
  root.querySelector(".pk-count").textContent =
    st.q ? `搜索到 ${list.length} 个英雄` : (st.filter && st.filter !== "all" ? `${roleCn}常用 ${list.length} 个 · 按最近一年职业赛出场排序` : `全部 ${list.length} 个英雄`);
  root.querySelector(".pk-picked").textContent = `已选 ${n}/10 · 变灰的英雄已被选走`;
}

function render(root, st) { renderSides(root, st); renderPicker(root, st); }

function activate(root, st, side, i) {
  st.active = { side, i };
  st.filter = st.data.roles[i][0];
  st.keepScroll = false;
  render(root, st);
}

function pick(root, st, key) {
  const { side, i } = st.active, tk = taken(st);
  if (tk[key] && !(tk[key][0] === side && tk[key][1] === i)) return;
  st.local[side].champs[i] = key;
  emit(st);
  const nx = nextEmpty(st, st.active);
  st.q = ""; root.querySelector(".search input").value = ""; st.hl = 0;
  activate(root, st, nx.side, nx.i);
}

function skeleton(root, st) {
  const roles = st.data.roles.map(([r, cn]) => `<button data-f="${r}">${ROLE_SVG[r] || ""}${esc(cn)}</button>`).join("");
  const wrap = document.createElement("div");
  wrap.className = "lw";
  wrap.innerHTML = `<div class="draft"><section class="side blue"></section>
    <div class="picker"><div class="pk-head">
      <div class="pk-row"><div class="pk-target"></div><button class="ghost clear">清空英雄</button></div>
      <div class="pk-row"><div class="seg"><button data-f="all">全部</button>${roles}</div>
        <label class="search"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
          <input type="search" placeholder="搜索：中文 / 英文 / 称号，回车选中" autocomplete="off" spellcheck="false"></label></div></div>
      <div class="grid"></div>
      <div class="pk-foot"><span class="pk-count"></span><span class="pk-picked"></span></div></div>
    <section class="side red"></section></div>`;
  root.appendChild(wrap);

  wrap.addEventListener("click", (e) => {
    const t = e.target;
    if (t.closest("select")) return;
    const x = t.closest(".x");
    if (x) {
      const s = x.closest(".slot");
      st.local[s.dataset.side].champs[+s.dataset.i] = null;
      emit(st); activate(root, st, s.dataset.side, +s.dataset.i); return;
    }
    const slot = t.closest(".slot");
    if (slot) { activate(root, st, slot.dataset.side, +slot.dataset.i); return; }
    const tile = t.closest(".tile");
    if (tile && !tile.disabled) { pick(root, st, +tile.dataset.k); return; }
    const f = t.closest(".seg button");
    if (f) { st.filter = f.dataset.f; st.q = ""; wrap.querySelector(".search input").value = ""; st.keepScroll = false; renderPicker(root, st); return; }
    if (t.closest(".clear")) {
      for (const side of ["blue", "red"]) st.local[side].champs = [null, null, null, null, null];
      emit(st); activate(root, st, "blue", 0);
    }
  });
  wrap.addEventListener("change", (e) => {
    const s = e.target.closest("select.player");
    if (!s || !s.value) return;
    st.local[s.dataset.side].players[+s.dataset.i] = s.value;
    emit(st);
  });
  const input = wrap.querySelector(".search input");
  input.addEventListener("input", () => { st.q = input.value; st.hl = 0; st.keepScroll = false; renderPicker(root, st); });
  input.addEventListener("keydown", (e) => {
    const cols = Math.max(1, Math.floor(wrap.querySelector(".grid").clientWidth / 76));
    const mv = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: cols, ArrowUp: -cols }[e.key];
    if (mv) { e.preventDefault(); st.hl = Math.max(0, Math.min((st.list || []).length - 1, st.hl + mv)); renderPicker(root, st); return; }
    if (e.key === "Enter") { e.preventDefault(); const c = (st.list || [])[st.hl]; if (c) pick(root, st, c.key); return; }
    if (e.key === "Escape") { input.value = ""; st.q = ""; renderPicker(root, st); }
  });
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement;
  if (!data) return;
  data.byKey = {};
  for (const c of data.champs) data.byKey[c.key] = c;
  let st = root.__lw;
  const fresh = !st;
  if (fresh) st = root.__lw = { sent: 0, rev: null, q: "", hl: 0, filter: null, active: null, keepScroll: false };
  // 规则 2/3：Python 发起的改动（rev 变了）或 Python 已追上本地最后一个事件，才用 Python 的阵容覆盖本地
  if (fresh || data.rev !== st.rev || data.seq >= st.sent) {
    st.local = clone(data.draft);
    st.rev = data.rev;
    if (data.seq > st.sent) st.sent = data.seq;
  }
  st.data = data;
  st.send = setTriggerValue;
  if (fresh) {
    skeleton(root, st);
    const first = ORDER.find(([s, i]) => st.local[s].champs[i] == null) || ORDER[0];
    st.active = { side: first[0], i: first[1] };
    st.filter = data.roles[st.active.i][0];
  }
  render(root, st);
}
"""

_NAME = "lolwp_draft_board"


def board_data(sim: dict[str, Any], champs: dict[str, Any], rosters: dict[str, dict],
               team_info: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    """组件需要的全部数据：英雄目录、两队名单、当前阵容。纯函数，便于测试。"""
    from lolwp.app.ui import _hue

    sides = {}
    for side in ("blue", "red"):
        t = team_info(sim[side]["team"])
        roles = (rosters.get(sim[side]["team"]) or {}).get("roles") or {}
        sides[side] = {
            "team": {"name": t.get("name"), "code": t.get("code"), "league": t.get("league_label") or t.get("league"),
                     "logo": t.get("logo"), "hue": _hue(str(t.get("code") or "?"))},
            "players": [roles.get(r, []) for r in ROLES],
        }
    return {
        "roles": [[r, ROLE_CN[r]] for r in ROLES],
        "champs": [{"key": c["key"], "id": c["id"], "en": c["en"], "zh": c["zh"], "ep": c["epithet"],
                    "img": c.get("icon"), "n": c["n"]} for c in champs["list"]],
        "pool": champs["pool"],
        "sides": sides,
        "draft": {s: {"champs": list(sim[s]["champs"]), "players": list(sim[s]["players"])}
                  for s in ("blue", "red")},
        "rev": sim["rev"], "seq": sim.get("seq", 0),
    }


def apply_event(sim: dict[str, Any], ev: dict[str, Any] | None, champ_keys: set[int]) -> bool:
    """把组件发来的整份阵容落到 sim 上。返回是否有改动。

    校验：英雄必须在目录里、同一英雄不能出现两次（后出现的作废）；
    rev 对不上的旧事件只采纳英雄（规则 3）。
    """
    if not isinstance(ev, dict):
        return False
    fresh = ev.get("rev") == sim["rev"]
    seen: set[int] = set()
    changed = False
    for side in ("blue", "red"):
        new = ev.get(side) or {}
        champs = list(new.get("champs") or [None] * 5)[:5]
        clean: list[int | None] = []
        for k in champs + [None] * (5 - len(champs)):
            k = int(k) if isinstance(k, (int, float)) and int(k) in champ_keys else None
            if k is not None and k in seen:
                k = None
            if k is not None:
                seen.add(k)
            clean.append(k)
        if clean != sim[side]["champs"]:
            sim[side]["champs"] = clean
            changed = True
        players = new.get("players")
        if fresh and isinstance(players, list) and len(players) == 5:
            players = [str(p) if p else None for p in players]
            if players != sim[side]["players"]:
                sim[side]["players"] = players
                changed = True
    seq = ev.get("seq")
    if isinstance(seq, int):
        sim["seq"] = max(int(sim.get("seq", 0)), seq)
    return changed


def draft_board(state_key: str, champs: dict[str, Any], rosters: dict[str, dict],
                team_info: Callable[[str], dict[str, Any]], *, key: str = "sim_board") -> None:
    """挂载阵容板。事件在回调里落到 st.session_state[state_key]（脚本主体运行之前），
    所以同一轮就能出新概率。回调里现取 session_state，不捕获上一轮的对象。"""
    keys = {c["key"] for c in champs["list"]}

    def _on_draft() -> None:
        ev = (st.session_state.get(key) or {}).get("draft")
        apply_event(st.session_state[state_key], ev, keys)

    component.get(_NAME, css=_CSS, js=_JS)(
        key=key, data=board_data(st.session_state[state_key], champs, rosters, team_info),
        on_draft_change=_on_draft, height="content")
