#!/usr/bin/env bash
# 阶段 0：下载 OE 年度 CSV。
# 关键修正 vs 手册 v1.1：
#   1. curl 从 stdin 继承会吞掉 while-read 的输入 → 显式 </dev/null
#   2. curl -f 只拦 4xx/5xx；配额耗尽是 HTTP 200 + HTML，必须落地后自己查
set -uo pipefail
UA="lolwp/1.0 (qxing@uchicago.edu)"
RAW=data/raw
mkdir -p "$RAW"
fail=0
while IFS=$'\t' read -r YEAR FID; do
  [ -z "${YEAR:-}" ] && continue
  OUT="$RAW/${YEAR}_LoL_esports_match_data_from_OraclesElixir.csv"
  if [ -s "$OUT" ]; then echo "skip  $YEAR (已存在 $(wc -c < "$OUT") bytes)"; continue; fi
  echo "fetch $YEAR ..."
  curl -sSL --max-time 600 -A "$UA" \
    "https://drive.usercontent.google.com/download?id=${FID}&export=download&confirm=t" \
    -o "${OUT}.tmp" </dev/null
  sz=$(wc -c < "${OUT}.tmp" 2>/dev/null || echo 0)
  if head -c 512 "${OUT}.tmp" | grep -qi "Quota exceeded\|<!DOCTYPE html"; then
    echo "  FAIL $YEAR: 拿到配额错误页 (${sz} bytes)，保留为 .quota 供排查"
    mv "${OUT}.tmp" "${OUT}.quota"; fail=$((fail+1)); sleep 5; continue
  fi
  if [ "$sz" -lt 1000000 ]; then
    echo "  FAIL $YEAR: 文件过小 ${sz} bytes"; mv "${OUT}.tmp" "${OUT}.bad"; fail=$((fail+1)); continue
  fi
  mv "${OUT}.tmp" "$OUT"
  echo "  ok   $YEAR ${sz} bytes"
  sleep 2
done < data/oe_files.tsv
echo "=== 完成，失败 $fail 个 ==="
ls -la "$RAW" | tail -20
exit $fail
