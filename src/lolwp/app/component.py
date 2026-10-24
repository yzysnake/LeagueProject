"""components v2 自定义组件的注册（阵容板、开场动画共用）。

组件注册在 Streamlit 的 Runtime 上，不是进程上：AppTest 每个用例起一个新 Runtime，
模块级注册只在第一个里有效（之后报「Component ... is not registered」）。所以每次挂载前
确认当前 Runtime 里有它，没有才注册——同一个 Runtime 里重复注册会刷警告。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

_mounts: dict[str, Callable[..., Any]] = {}


def get(name: str, *, css: str | None = None, js: str | None = None) -> Callable[..., Any]:
    """返回组件的挂载函数（必要时先在当前 Runtime 注册）。"""
    from streamlit.components.v2.get_bidi_component_manager import (
        get_bidi_component_manager,
    )

    if name not in _mounts or get_bidi_component_manager().get(name) is None:
        _mounts[name] = st.components.v2.component(name, css=css, js=js)
    return _mounts[name]
