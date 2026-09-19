"""阶段 0：发现 Oracle's Elixir Google Drive 文件夹里的年度 CSV 文件 ID。

输出 data/oe_files.json，供 day0_fetch.sh 使用。
不依赖 lolwp 包，只需 requests。
"""
import json
import re
import sys

import requests

FOLDER = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
UA = "lolwp/1.0 (qxing@uchicago.edu)"
NAME_RE = re.compile(r"(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv")


def main() -> int:
    html = requests.get(
        f"https://drive.google.com/embeddedfolderview?id={FOLDER}#list",
        headers={"User-Agent": UA},
        timeout=30,
    ).text

    # 分条目切分后再匹配，避免一条大正则被 HTML 结构变动整体打掉
    entries = re.findall(r'/file/d/([\w-]+)/view.*?flip-entry-title">([^<]+)<', html, re.DOTALL)
    out = []
    for fid, title in entries:
        m = NAME_RE.match(title.strip())
        if m:
            out.append({"year": int(m.group(1)), "file_id": fid, "title": title.strip()})

    out.sort(key=lambda r: r["year"])
    seen = {r["year"] for r in out}
    if len(out) != len(seen):
        print(f"FAIL: 年份重复 {sorted(out, key=lambda r: r['year'])}", file=sys.stderr)
        return 1
    # 不硬编码 13：要求年份连续且覆盖到当前年
    if not out:
        print("FAIL: 一个年度文件都没匹配到，Drive 目录页结构可能已变", file=sys.stderr)
        return 1
    years = [r["year"] for r in out]
    if years != list(range(years[0], years[-1] + 1)):
        print(f"FAIL: 年份不连续 {years}", file=sys.stderr)
        return 1

    with open("data/oe_files.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    with open("data/oe_files.tsv", "w") as fh:
        fh.writelines(f"{r['year']}\t{r['file_id']}\n" for r in out)

    for r in out:
        print(r["year"], r["file_id"])
    print(f"OK: {len(out)} 个年度文件，{years[0]}-{years[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
