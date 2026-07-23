# Text re-ingest 操作手册（Phase A1/A3）

> 目标：在 **不重编 ColQwen2/FAISS** 的前提下，重建 pgvector 文本索引：  
> 新分块元数据（`section_path` / prev-next）+ 可选 **上下文表摘要**。  
> 日期：2026-07-23 · 分支 `feat/content-pipeline-phase-ab`

---

## 1. 为什么必须 TRUNCATE

`insert_chunks` 使用 **`ON CONFLICT (chunk_id) DO NOTHING`**。  
同一 `chunk_id` 再次写入不会更新 `table_summary` / `section_path`。  
因此 Text re-ingest 必须：

```text
TRUNCATE chunks → 全量文本路重写
FAISS 文件不动（--skip-faiss）
```

评测入口每次会 `BM25.fit_from_pgvector`，无需单独持久化 BM25。

---

## 2. 成本粗算（ViDoRe industrial · ~5244 页）

| 步骤 | 资源 | 粗估 |
|------|------|------|
| 分块 + section/neighbors | CPU | 数分钟～十几分钟 |
| **表摘要 LLM**（~2300 table） | Ollama qwen2:7b | **1～4 h**（视缓存重复表） |
| BGE encode ~8k–12k chunk | GPU 或 CPU | 十几～四十分钟 |
| ColQwen2 / FAISS | — | **跳过** |

**建议顺序：** `smoke`（无 LLM）→ `smoke-llm`（20 页）→ `full`。

---

## 3. 云上前置检查

```bash
source /root/prism-rag/scripts/cloud_env.sh   # HF OFFLINE
export PATH=/root/miniconda3/bin:$PATH
pg_isready
# 表摘要需要：
pgrep -x ollama || nohup ollama serve &
ollama list   # 需 qwen2:7b
ls /root/autodl-tmp/huggingface/hub/datasets--vidore--vidore_v3_industrial
ls /root/autodl-tmp/indexes/colqwen2-v1.0-vidore-industrial.faiss
```

代码需含：

- `scripts/ingest_vidore.py --replace-text --table-context --skip-faiss`
- `scripts/cloud_text_reingest.sh`
- `PgVectorStore.truncate_chunks`

同步方式：本地 `pack_for_cloud.sh --upload` + 云上 `cloud_apply_upload.sh`。

---

## 4. 命令

### 4.1 冒烟（快，无 LLM）

```bash
cd /root/prism-rag
source scripts/cloud_env.sh
export PATH=/root/miniconda3/bin:$PATH
MODE=smoke MAX_PAGES=20 bash scripts/cloud_text_reingest.sh
```

验收：`section_path_filled` / `prev_filled` > 0（有标题页时）；`chunks_after` 明显小于全库。

### 4.2 冒烟 + 上下文表摘要

```bash
MODE=smoke-llm MAX_PAGES=20 bash scripts/cloud_text_reingest.sh
```

验收：`table_summary_filled` > 0；sample 摘要非空。

### 4.3 全量 Text re-ingest（贵）

```bash
# 建议 nohup；表摘要阶段日志很长
nohup env MODE=full bash scripts/cloud_text_reingest.sh \
  > /root/autodl-tmp/runs/text-reingest-full.log 2>&1 &
tail -f /root/autodl-tmp/runs/text-reingest-full.log
```

---

## 5. 重灌后评测（对照 Boot-CP Arm-A）

```bash
# 默认关 expand/boost，与 Boot-CP Arm-A 同协议
python scripts/run_eval.py \
  --skip-index --language en --visual-model colqwen2 \
  --config-filter Full_zerank2 --no-hyde \
  --max-queries 100 \
  --output-dir runs/$(date +%Y%m%d)-post-text-reingest/arm-A
```

对比：`runs/20260723-content-pipeline/arm-A` NDCG@10 = **0.3575**（同 100q 切片）。

可选再开 B1/B2（page expand 对 page NDCG 仍可能无感，见 Boot-CP README）。

---

## 6. 风险与回滚

| 风险 | 缓解 |
|------|------|
| TRUNCATE 后中途失败 → 空库 | 先 smoke；full 前确认磁盘/ollama；失败需重跑 full |
| 表摘要拖死 GPU 排队 | 摘要走 Ollama CPU/GPU 共享；可先 `--no-table-summary` 只灌结构 |
| chunk_id 变化 | 正常；BM25 随评测重建 |
| FAISS 与 pg page_id 不一致 | corpus_id 仍作 page_id，与旧 Visual 对齐 |

**没有自动从 TRUNCATE 快照回滚。** 旧文本只存在于此前 dump（若有）。云上当前无 SQL dump 时，失败即重跑 re-ingest。

---

## 7. 本地不跑全量

本机 macOS **禁止** 5244 页 + 大模型（Agents.md）。本地只改脚本/单测；全量只在云上。
