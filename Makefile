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
