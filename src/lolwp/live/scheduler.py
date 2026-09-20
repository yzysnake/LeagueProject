"""后台调度：每日增量拉数 + 实时轮询（docker compose 的 online profile）。

**它不是主链路的一部分。** `docker compose up` 默认不起它，
纯离线的手动摆阵容完全可用——这是设计，不是妥协（§7.8）。

两个循环：
  · 每日一次：拉 OE 当年文件 → 五重校验 → 若有更新则重建 bronze/silver
  · 持续：轮询 lolesports，把预测结果写进共享状态供 /live 读取
任何一个循环挂掉都不能带倒另一个。
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

LOG = logging.getLogger("scheduler")
DAILY_INTERVAL_S = float(os.environ.get("LOLWP_DAILY_INTERVAL_S", "86400"))
LIVE_TICK_S = float(os.environ.get("LOLWP_LIVE_TICK_S", "30"))
_STOP = False


def _handle(signum, frame) -> None:
    global _STOP
    _STOP = True
    LOG.info("收到信号 %s，准备退出", signum)


def daily_ingest() -> bool:
    """拉数 + 校验。**失败不抛**，返回 False 让调度继续跑。"""
    try:
        r = subprocess.run([sys.executable, "scripts/day0_validate.py"],
                           capture_output=True, text=True, timeout=1800, check=False)
        LOG.info("每日校验退出码 %s\n%s", r.returncode, r.stdout[-800:])
        return r.returncode == 0
    except Exception:
        LOG.exception("每日校验异常")
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    poller = None
    try:
        from lolwp.live.poller import build_default_poller
        poller = build_default_poller()
        LOG.info("轮询器就绪")
    except Exception:
        LOG.exception("轮询器构建失败，只跑每日拉数")

    last_daily = 0.0
    while not _STOP:
        now = time.time()
        if now - last_daily >= DAILY_INTERVAL_S:
            LOG.info("开始每日拉数")
            daily_ingest()
            last_daily = now
        if poller is not None:
            try:
                poller.tick()
            except Exception:
                LOG.exception("轮询 tick 失败（继续）")
        time.sleep(LIVE_TICK_S)
    LOG.info("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
