# PrismRAG Eval Protocol v1

> 冻结评测口径，保证不同 run 的数字可辩护、可复现。  
> 关联计划：`docs/superpowers/plans/2026-07-20-bullet-strengthening-roadmap.md`（Cloud Boot Packing）。  
> 云上默认走 **Boot-A**：黄金消融（无 HyDE）+ 增量验收，同一次 GPU 开机。

---

## 1. 数据集

| 项 | 值 |
|----|-----|
| name | `vidore/vidore_v3_industrial` |
| language | **english only**（`--language en`） |
| query_count | **283**（`--expected-query-count 283`；全量评测且未设 `--max-queries` 时校验） |
| 相关判定 | qrels → `page_id` 集合；指标按 **page 级** 计算 |

冒烟（本地，≤10q，不计正式数字）：

```bash
python scripts/run_eval.py --max-queries 10 --skip-index --language en \
  --config-filter Full_zerank --visual-model colqwen2
```

---

## 2. 指标

| 指标 | 定义 |
|------|------|
| **NDCG@10** | 折扣 `1/log2(i+1)`（实现为 `math.log2(pos+2)`，与 pytrec_eval 一致）；**同一 page 仅首次出现计分** |
| NDCG@5 / Recall@5 / Recall@10 / MRR | 同 page 去重规则 |
| avg_latency_ms | 该配置下每 query 检索墙钟平均（含各路+融合+重排，视配置而定） |

**禁止**与 2026-07-02 及更早使用 `1/(i+1)` 的 run **直接对比绝对 NDCG**。相对结论（如「精排是瓶颈」）可在新协议下重验后沿用。

实现：`src/evaluation/ablation.py` → `compute_ndcg` / `compute_recall` / `compute_mrr`。

---

## 3. 索引与模型冻结（每个正式 run README 必填）

| 项 | Boot-A 默认 |
|----|-------------|
| visual_model | `colqwen2`（`vidore/colqwen2-v1.0`） |
| FAISS | `storage.faiss.colqwen2_index_path` |
| Dense | BGE-large-en-v1.5 + pgvector |
| BM25 | 自维护统计 / 与 pg 文本一致 |
| Reranker | 消融含 BGE 与 zerank-2 |
| table_summary_enabled | **以当次 ingest 配置为准，写入 run README**（选定后本 Boot 内不改） |
| git SHA | `git rev-parse HEAD` |
| 是否重编 Visual | **默认否**（`--skip-index`） |

---

## 4. 黄金消融配置

### 4.1 默认：`GOLDEN_NO_HYDE`（Boot-A Job1）

| name | bm25 | dense | visual | rerank | reranker | hyde |
|------|:----:|:-----:|:------:|:------:|----------|:----:|
| BM25_only | ✓ | | | | | |
| Dense_only | | ✓ | | | | |
| Visual_only | | | ✓ | | | |
| BM25_Dense | ✓ | ✓ | | | | |
| BM25_Dense_Visual | ✓ | ✓ | ✓ | | | |
| Full_no_rerank | ✓ | ✓ | ✓ | | | |
| Full_with_rerank | ✓ | ✓ | ✓ | ✓ | bge | |
| Full_zerank2 | ✓ | ✓ | ✓ | ✓ | zerank | |

```bash
python scripts/run_eval.py --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 --no-hyde \
  --output-dir runs/YYYYMMDD-bootA/golden-ablation
```

### 4.2 可选（默认不跑）

- `Full_BGE_HyDE` / `Full_zerank2_HyDE`：历史结论本场景 HyDE Δ≈0；仅 Full 档余量时补。
- 全量 `ABLATION_CONFIGS`（含 HyDE）：去掉 `--no-hyde`。

---

## 5. Boot-A 产物布局

```text
runs/YYYYMMDD-bootA/
  README.md                 # 总览：SHA、环境、结论三句话
  env.txt                   # git SHA、models 摘要
  golden-ablation/          # Job1 消融 raw + ndcg_table.md
  incremental/              # Job2 幽灵召回 / page-diff / 漂移
  summary.md                # 一页数字（给简历用）
```

### 5.1 黄金表必须回答的问题

1. `Full_no_rerank` vs `Full_zerank2`（或 `Full_with_rerank`）ΔNDCG@10 是否仍支持「瓶颈在精排」？
2. `BM25_only` vs 融合无 rerank 是否接近？
3. 本 run 的 table_summary / visual / git SHA？

### 5.2 增量验收（同机 Job2，详见 `docs/incremental-verification-runbook.md`）

| 检查 | 标准 |
|------|------|
| 幽灵召回 | 删 doc 后相关 query 不再命中该 doc 页 |
| NDCG 不漂移 | 增量后 Full_zerank2 相对 Job1 同配置 \|Δ\| &lt; 0.005（或 README 报告实测 Δ） |
| page-diff | 重编码页数 ≈ 变更页；记录墙钟 |

**省跑：** Job1 的 `Full_zerank2` 作 baseline；增量后 **只再跑 1 次** Full_zerank2（`--config-filter Full_zerank`）。

---

## 6. 一键脚本

```bash
# 仅在云上 GPU 机执行（本地勿跑全量）
bash scripts/cloud_boot_a.sh
# 可选环境变量：
#   BOOT_DATE=20260720
#   MAX_QUERIES=   # 空=全量 283；调试可设 10
#   SKIP_INCREMENTAL=1
#   VISUAL_MODEL=colqwen2
```

---

## 7. 版本

| 版本 | 日期 | 说明 |
|------|------|------|
| v1 | 2026-07-20 | 首版；Boot-A 默认 GOLDEN_NO_HYDE |
