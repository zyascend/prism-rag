.PHONY: help install ingest-vidore eval-vidore eval-ragas fetch-indexes clean

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	uv venv .venv --python 3.11
	uv pip install -e ".[dev]"

.env: ## 从模板创建 .env
	cp -n .env.example .env || true

ingest-vidore: .env ## 导入 ViDoRe Industrial 子集（构建索引）
	python scripts/ingest_vidore.py --dataset vidore/vidore_v3_industrial

eval-vidore: .env ## 运行 ViDoRe 消融评测（全量）
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial

eval-vidore-quick: .env ## 运行 ViDoRe 快速消融（10 条查询）
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --max-queries 10

eval-vidore-skip-index: .env ## 跳过索引构建，直接评测
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --skip-index

eval-ragas: .env ## 运行 RAGAS 拒答 sanity
	python scripts/run_ragas_sanity.py

fetch-indexes: ## 从 GitHub Release 拉取预编码索引
	python scripts/fetch_indexes.py

clean: ## 清理索引和评测结果
	rm -rf indexes/ results/