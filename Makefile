PY := PYTHONPATH=src .venv/bin/python

.PHONY: help data bronze silver artifact icons app test lint all

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

data:          ## 下载 OE 年度 CSV 并做五重校验
	bash scripts/day0_fetch.sh && .venv/bin/python scripts/day0_validate.py

bronze:        ## CSV → Parquet，物理丢弃赛后统计列
	$(PY) -m lolwp.store.bronze

silver:        ## 折叠为一场一行 + 赛区归一 + series_id + patch + crosswalk
	$(PY) -m lolwp.store.silver

icons:         ## 下载 173 张英雄图标供离线使用
	$(PY) -c "import json,os,requests; \
	cm=json.load(open('data/ddragon/champion_map.json')); v=cm['version']; \
	os.makedirs(f'frontend/public/dd/{v}',exist_ok=True); \
	[open(f'frontend/public/dd/{v}/'+c['dd_id']+'.png','wb').write( \
	  requests.get(f'https://ddragon.leagueoflegends.com/cdn/{v}/img/champion/'+c['dd_id']+'.png',timeout=20).content) \
	 for c in cm['champions']]"

artifact:      ## 训练生产模型并打包
	$(PY) -m lolwp.models.artifact

app:           ## 启动本地站点（离线可用）
	PYTHONPATH=src .venv/bin/streamlit run src/lolwp/app/main.py

test:          ## 跑全部测试
	$(PY) -m pytest tests/ -q

lint:          ## ruff
	.venv/bin/ruff check src/ tests/ scripts/

all: bronze silver icons artifact test  ## 从 bronze 一路重建到可用站点
