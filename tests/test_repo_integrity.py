"""仓库完整性：防止"提交了测试却没提交它依赖的夹具"这类错误。

2026-09-20 的真实事故：阶段 6 提交了 tests/test_live.py，但它依赖的
tests/fixtures/window_live.json 从未被 git 跟踪；随后我又用 rm -f 把工作区
那份删掉，测试套件当场红 11 项。新克隆的人从第一天起就跑不通。

这条测试把"夹具必须被跟踪"变成硬约束。
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True)
    return set(out.stdout.split())


def test_all_fixture_files_are_tracked():
    tracked = _tracked()
    fx = ROOT / "tests" / "fixtures"
    if not fx.is_dir():
        pytest.skip("无 fixtures 目录")
    untracked = [p.name for p in fx.iterdir()
                 if p.is_file() and f"tests/fixtures/{p.name}" not in tracked]
    assert not untracked, f"夹具未进版本库，新克隆会跑不通：{untracked}"


def test_every_referenced_fixture_exists():
    """反向检查：测试代码里引用的每个夹具文件名都必须真实存在。"""
    pat = re.compile(r'["\']([A-Za-z0-9_.-]+\.(?:json|csv|parquet))["\']')
    missing = []
    for t in (ROOT / "tests").glob("test_*.py"):
        src = t.read_text()
        if "fixtures" not in src:
            continue
        for name in set(pat.findall(src)):
            # 只查看起来像夹具名的（排除 data/ 下的产物路径）
            if "/" in name:
                continue
            p = ROOT / "tests" / "fixtures" / name
            if not p.exists() and "fixtures" in src:
                cand = [c.name for c in (ROOT / "tests" / "fixtures").iterdir()]
                if name not in cand and name.startswith("window"):
                    missing.append(f"{t.name} → {name}")
    assert not missing, f"引用了不存在的夹具：{missing}"


def test_no_tracked_file_is_gigantic():
    """防止误提交大文件（OE 的年度 CSV 有 70-110 MB）。"""
    big = []
    for rel in _tracked():
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > 5_000_000:
            big.append(f"{rel} ({p.stat().st_size // 1048576} MB)")
    assert not big, f"版本库里有超过 5 MB 的文件：{big}"
