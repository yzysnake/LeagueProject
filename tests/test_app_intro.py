"""开场动画：浏览器端行为由脚本自己守，这里守住「它在哪、长什么样、不会弄坏页面」。

逐帧效果、跳过、减少动态效果、不重放这些是在无头 Chrome 里实测的（见 app/intro.py 文档）；
这里能在 CI 里稳定测的是：每页都挂、只挂一次、脚本可解析、不会提前闭合 <script>、几条体验约定都还在。
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from lolwp.app import intro

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "src" / "lolwp" / "app" / "main.py")


_MXSS = re.compile(r"<[/\\w!]")   # DOMPurify：<style>/<script> 文本里见到它就整段删掉


def test_script_element_survives_dompurify_and_cannot_close_early():
    """st.html 的 DOMPurify 见到 <script> 文本里有「< 后跟字母」就删掉整段脚本（实测，不报错）；
    文本里有 </script> 还会提前闭合标签。所以 <script> 里一个「<」都不能有。"""
    el = intro.element()
    body = el[len("<script>"):el.index("</script>")]
    assert "<" not in body
    js = json.loads(body[len("(new Function("):-len("))()")])
    assert js == intro.script() and '"brand_a": "LoL"' in js and '"brand_b": "WP"' in js


def test_global_css_survives_dompurify():
    """同一条规则也管全站样式：注释里写一句 .st-key-<key>，整个 <style> 就被删、全站没样式。"""
    from lolwp.app import ui
    assert not _MXSS.search(ui._CSS), _MXSS.search(ui._CSS)


def test_experience_contracts_are_still_in_the_script():
    """四条体验约定：每会话一次、12 小时内放短版、后台标签页不放、尊重减少动态效果；
    以及页面就绪就退场、最长封顶、随时可跳过。删掉任何一条都该是有意的。"""
    js = intro.script()
    for needle in ("sessionStorage", "12 * 3600 * 1000", "document.hidden", "prefers-reduced-motion",
                   ".lw-footer", "capT", '"pointerdown"', '"keydown"', "stHeaderLogo", "root.remove()"):
        assert needle in js, f"开场脚本里找不到 {needle}"


@pytest.mark.skipif(shutil.which("node") is None, reason="没有 node，跳过语法检查")
def test_script_parses(tmp_path):
    f = tmp_path / "intro.js"
    f.write_text(intro.script())
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not os.path.exists(ROOT / "models" / "prod.json"), reason="需先跑 artifact.train()")
def test_every_page_mounts_the_intro_exactly_once_and_it_takes_no_space():
    from streamlit.testing.v1 import AppTest

    def _boom(*a, **k):
        raise RuntimeError("离线")
    with patch("lolwp.live.poller.build_default_poller", _boom):
        at = AppTest.from_file(APP, default_timeout=300)
        at.run()
        for page in (None, "views/review.py", "views/about.py"):
            if page:
                at.switch_page(page).run()
            bodies = [e.proto.body for e in at.get("html")]
            scripts = [b for b in bodies if b.startswith("<script>")]
            assert len(scripts) == 1 and 'class="lw-intro-mark"' in scripts[0]
            assert "new Function(" in scripts[0]
            css = next(b for b in bodies if b.startswith("<style>"))
            assert ":has(.lw-intro-mark)" in css and "display: none" in css   # 容器不占位
            assert len(at.exception) == 0
