"""本地 Streamlit 站点入口：顶部导航 + 六个页面（各自一个文件，在 views/ 下）。

  阵容模拟   选赛事和两支队，点头像选英雄、按位置筛、换替补，出胜率与三层构成
  What-if    把任一位置换成别的英雄，看概率怎么变
  实时比赛   自动发现正在进行的比赛并出预测
  近期战绩   盯盘落盘的预测 vs 赛后结果，可按时间段 / 赛事筛，标出赛后补录的局
  历史回看   拿真实比赛复盘，对照模型预测与实际结果
  模型说明   模型卡：怎么算的、准到什么程度、哪里别全信

**完全离线可用**（阶段 5 验收①）：英雄图标、队标在 build 时落到 frontend/public/，
模型与状态全在 models/prod.json 里，运行时零外部请求。
实时读取当前比赛是增强功能，不是这一层的依赖：实时页挂了，其余页面照常。

页面之间互不依赖：每次交互只重跑当前页（旧版五个 tab 在同一个脚本里，
在任何一页点一下都会把五页全算一遍，包括历史回看的几百次预测）。
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lolwp.app import data, ui

HERE = Path(__file__).parent
ASSETS = HERE / "assets"

PAGES = [
    ("views/simulate.py", "阵容模拟", ":material/stadia_controller:"),
    ("views/whatif.py", "What-if", ":material/swap_horiz:"),
    ("views/live.py", "实时比赛", ":material/sensors:"),
    ("views/review.py", "近期战绩", ":material/insights:"),
    ("views/history.py", "历史回看", ":material/history:"),
    ("views/about.py", "模型说明", ":material/info:"),
]


def main() -> None:
    st.set_page_config(page_title=f"{ui.BRAND} · {ui.TAGLINE}", page_icon=str(ASSETS / "icon.svg"),
                       layout="wide", initial_sidebar_state="collapsed")
    ui.inject()
    # 不给 icon_image：没有侧栏时 Streamlit 会一直显示 icon_image，字标就没机会出现
    st.logo(str(ASSETS / ("logo_light.svg" if ui.is_light() else "logo_dark.svg")), size="large")
    pages = [st.Page(path, title=title, icon=icon, default=i == 0)
             for i, (path, title, icon) in enumerate(PAGES)]
    pg = st.navigation(pages, position="top")
    try:
        pg.run()
    finally:
        # 任何一页内部出错（Streamlit 已就地显示），页脚的免责声明也必须在
        ui.footer(data.artifact().meta, data.champions()["version"])


main()
