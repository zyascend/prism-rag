.PHONY: help install ingest-vidore eval-vidore eval-full eval-ragas demo fetch-indexes clean lint test

help: ## 显示帮助
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	uv venv .venv --python 3.11
	uv pip install -e ".[dev]"

.env: ## 从模板创建 .env
	cp -n .env.example .env || true

ingest-vidore: .env ## 导入 ViDoRe Industrial 子集（构建索引）
	python scripts/ingest_vidore.py --dataset vidore/vidore_v3_industrial

eval-vidore: .env ## 运行 ViDoRe 消融评测（全量）
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial

eval-vidore-quick: .env ## 运行 ViDoRe 快速消融（50 条查询，跳过索引构建）
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --max-queries 50 --skip-index

eval-vidore-skip-index: .env ## 跳过索引构建，直接评测
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --skip-index

eval-full: .env ## 全 8 子集 ViDoRe 评测（长跑，按需租云）
	python scripts/run_eval.py --dataset vidore/vidore_v3_synthetic
	python scripts/run_eval.py --dataset vidore/vidore_v3_industrial
	python scripts/run_eval.py --dataset vidore/vidore_v3_health
	python scripts/run_eval.py --dataset vidore/vidore_v3_energy
	@echo "全量子集评测完成，结果在 results/ 目录"

eval-ragas: .env ## 运行 RAGAS 拒答 sanity
	python scripts/run_ragas_sanity.py

eval-ragas-metrics: .env ## 运行 RAGAS 生成层评测（Faithfulness + Answer Relevancy）
	python scripts/run_ragas_metrics.py

eval-ragas-metrics-quick: .env ## 快速 RAGAS 生成层评测（10 条查询）
	python scripts/run_ragas_metrics.py --max-queries 10

demo: .env ## 启动 Docker Compose 在线 Demo
	docker compose up -d
	@echo "Demo 已启动: http://localhost:8000"
	@echo "Health check: http://localhost:8000/health"

fetch-indexes: ## 从 GitHub Release 拉取预编码索引
	python scripts/fetch_indexes.py

lint: ## 代码检查
	python -m ruff check src/ tests/

test: ## 运行测试
	python -m pytest tests/ -v --tb=short

clean: ## 清理索引和评测结果
	rm -rf indexes/ results/