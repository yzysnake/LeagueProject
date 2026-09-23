PY := PYTHONPATH=src .venv/bin/python
SNAPSHOT ?= latest

.PHONY: help setup data bronze silver icons artifact app test lint gate monitor reproduce all

help:           ## 列出所有命令
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

setup:          ## 建 Python 3.11 环境并装依赖（首次）
	uv sync --frozen

data:           ## 下载 OE 年度 CSV 并做五重校验
	bash scripts/day0_fetch.sh && $(PY) scripts/day0_validate.py

bronze:         ## CSV → Parquet，物理丢弃 135 个赛后统计列
	$(PY) -m lolwp.store.bronze

silver:         ## 折叠为一场一行 + 赛区归一 + series_id + patch + crosswalk
	$(PY) -m lolwp.store.silver

icons:          ## 下载 173 张英雄图标供离线使用
	$(PY) -c "import json,os,requests; \
	cm=json.load(open('data/ddragon/champion_map.json')); v=cm['version']; \
	os.makedirs(f'frontend/public/dd/{v}',exist_ok=True); \
	[open(f'frontend/public/dd/{v}/'+c['dd_id']+'.png','wb').write( \
	  requests.get(f'https://ddragon.leagueoflegends.com/cdn/{v}/img/champion/'+c['dd_id']+'.png',timeout=20).content) \
	 for c in cm['champions']]"

artifact:       ## 训练生产模型 BT4_LR 并打包（约 40 分钟，BT 需全量拟合）
	$(PY) -m lolwp.models.artifact

app:            ## 启动本地站点（离线可用，进程内推理）
	PYTHONPATH=src .venv/bin/streamlit run src/lolwp/app/main.py

api:            ## 启动 FastAPI（单 worker：artifact 每 worker 一份）
	PYTHONPATH=src .venv/bin/uvicorn lolwp.api.main:app --host 127.0.0.1 --port 8000 --workers 1

up:             ## docker compose：api + web，纯离线可用
	docker compose up

up-online:      ## docker compose + 每日拉数与实时轮询
	docker compose --profile online up

test:           ## 跑全部测试
	$(PY) -m pytest tests/ -q

lint:           ## ruff 全仓
	.venv/bin/ruff check src/ tests/ scripts/

gate:           ## 上线门禁自检：注入退化模型，必须被拒
	$(PY) scripts/gate_check.py --selftest

watch:          ## 实时盯盘：抓 BP 后阵容出正式预测，落盘供赛后对账
	$(PY) scripts/live_watch.py --hours 16

reconcile:      ## 赛后对账：预测 vs 实际，输出命中率 / log loss / Brier
	$(PY) scripts/live_watch.py --reconcile

install-watch:  ## 把盯盘挂成 launchd（崩溃自动重启，关终端不影响）
	@mkdir -p logs ~/Library/LaunchAgents
	@sed 's|__REPO__|$(PWD)|g' deploy/com.lolwp.livewatch.plist \
	  > ~/Library/LaunchAgents/com.lolwp.livewatch.plist
	@launchctl unload ~/Library/LaunchAgents/com.lolwp.livewatch.plist 2>/dev/null || true
	@launchctl load ~/Library/LaunchAgents/com.lolwp.livewatch.plist
	@echo "盯盘已挂："; launchctl list | grep livewatch

uninstall-watch: ## 停掉盯盘
	@launchctl unload ~/Library/LaunchAgents/com.lolwp.livewatch.plist 2>/dev/null || true
	@rm -f ~/Library/LaunchAgents/com.lolwp.livewatch.plist
	@echo "已停"

retest:         ## 重测被搁置的假设（协同/克制/比分差/熟练度），追加进台账
	$(PY) scripts/retest.py

retest-history: ## 只看台账的历史趋势，不重跑
	$(PY) scripts/retest.py --history

install-launchd: ## 【推荐】装 launchd 年度重测：睡眠错过会在唤醒后补跑
	@mkdir -p logs ~/Library/LaunchAgents
	@sed 's|__REPO__|$(PWD)|g' deploy/com.lolwp.retest.plist \
	  > ~/Library/LaunchAgents/com.lolwp.retest.plist
	@launchctl unload ~/Library/LaunchAgents/com.lolwp.retest.plist 2>/dev/null || true
	@launchctl load ~/Library/LaunchAgents/com.lolwp.retest.plist
	@echo "已装 launchd（每年 1 月 5 日 09:00，睡眠时唤醒后补跑）："
	@launchctl list | grep lolwp || echo "  （未列出，检查 plist）"

uninstall-launchd: ## 卸掉 launchd 任务
	@launchctl unload ~/Library/LaunchAgents/com.lolwp.retest.plist 2>/dev/null || true
	@rm -f ~/Library/LaunchAgents/com.lolwp.retest.plist
	@echo "已卸载"

install-cron:   ## 装 cron 年度重测（注意：睡眠时不补跑，年度任务建议用 launchd）
	@line="0 9 5 1 * cd $(PWD) && PYTHONPATH=src .venv/bin/python scripts/retest.py >> logs/retest.log 2>&1"; \
	( crontab -l 2>/dev/null | grep -v "scripts/retest.py"; echo "$$line" ) | crontab -; \
	echo "已装 cron："; crontab -l | grep retest

monitor:        ## 滚动漂移监控（阈值由抽样分布推出，不是拍脑袋）
	$(PY) -c "import warnings; warnings.filterwarnings('ignore'); \
	import pandas as pd; from lolwp.ops.monitor import rolling_monitor; \
	p=pd.read_parquet('data/cache/stage8_preds.parquet'); m=p[p.model=='BT4_LR'].copy(); \
	m['kickoff_utc']=pd.to_datetime(m.kickoff_utc,utc=True); \
	r=rolling_monitor(m.y.to_numpy(), m.p.to_numpy(), m.kickoff_utc, window=1000, step=1000); \
	print(r.round(5).to_string(index=False)); \
	print('\n阈值:', r.attrs['thresholds']); \
	print('告警', (r.alert!='-').sum(), '/', len(r))"

reproduce:      ## 从 bronze 一路重建并核对关键数字（SNAPSHOT=<manifest 里的 sha>）
	$(PY) scripts/reproduce.py --snapshot $(SNAPSHOT)

all: bronze silver icons artifact test  ## 从 bronze 重建到可用站点
