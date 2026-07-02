# Visual 路 CUDA OOM 修复复盘

> 日期: 2026-07-02 | 关联分支: feat/visual-path-fix-and-eval-alignment
> 运行记录: runs/20260702-visual-fix/

---

## 1. 问题背景

RTX 4090 24GB 上运行 ViDoRe V3 Industrial 全量评测（5244 页），Visual 路（ColPali + FAISS MaxSim）持续 CUDA OOM，导致 Visual_only 指标全为 0，含 Visual 的融合配置退化等同于 BM25+Dense。

---

## 2. 根因分析

经过逐步排查，发现**两个独立的显存问题**叠加：

### 2.1 ColPali 模型与 FAISS 向量显存叠加（设计文档已覆盖）

| 显存占用 | 大小 |
|----------|------|
| ColPali v1.3 模型 (bfloat16) | ~5.6 GB |
| BGE-large + BGE-Reranker | ~3.3 GB |
| FAISS 向量 (5.38M patches × 128d × float32) | ~2.6 GB |
| **合计（若不分离）** | **~11.5 GB** |

`--skip-index` 路径中，旧代码先 `faiss_store.load()` 将向量搬上 GPU，再在 query loop 中逐条 `colpali.encode_query()`，模型常驻与 MaxSim 向量同时存在。虽然 11.5 GB < 24 GB，但 PyTorch CUDA 分配器碎片化导致分配失败。

**修复**: `run_eval.py` 改为 Phase A (ColPali 预编码 283 条 query → unload) → Phase B (FAISS load)，两者不同时占用 GPU。

### 2.2 MaxSim 全量 matmul 中间矩阵 OOM（**新发现，设计文档未覆盖**）

这是真正的 OOM 根因。ColPali query 编码输出约 **1032 个 patches**（非预期的 ~10 个）。

`_maxsim_torch()` 中全量 matmul：

```python
scores = q_flat @ self._vectors_torch.T
# q_flat:      [1032, 128]
# vectors.T:   [128, 5_380_344]
# scores:      [1032, 5_380_344] → float32 → 1032 × 5.38M × 4B ≈ 21 GB
```

**21 GB 的中间矩阵** 即使 ColPali 完全卸载也无法在 24 GB 卡上分配（还需 ~5.9 GB 给 BGE + Reranker + FAISS 向量，总计 ~27 GB > 24 GB）。

手动测试用 `torch.randn(1, 10, 128)` 只有 10 个 patch，中间矩阵仅 ~200 MB，因此未触发 OOM，导致排查方向一度偏离。

**修复**: `_maxsim_torch()` 改为按页 batch 计算。每 200 页一组，每组中间矩阵 `[1032, ~200k]` ≈ 800 MB，安全可控。

```python
# 分页批处理伪代码
PAGE_BATCH = 200
for batch_start in range(0, num_pages, PAGE_BATCH):
    batch_vectors = vectors_torch[batch_start_patch:batch_end_patch]
    batch_scores = q_flat @ batch_vectors.T  # [1032, ~200k] ≈ 800 MB
    for page in batch:
        max_per_query = batch_scores[:, page_start:page_end].max(dim=1)
        page_scores[page_id] = max_per_query.mean()
```

---

## 3. 显存占用演进

| 阶段 | ColPali | BGE+Reranker | FAISS向量 | 合计 | 备注 |
|------|---------|-------------|-----------|------|------|
| 启动 | - | 3.3 GB | - | 3.3 GB | |
| Phase A (预编码) | 5.6 GB | 3.3 GB | - | 8.9 GB | |
| Phase A 完成 (unload后) | - | 3.3 GB | - | 3.3 GB | ✅ |
| Phase B (FAISS load) | - | 3.3 GB | 2.6 GB | 5.9 GB | |
| 逐query MaxSim | - | 3.3 GB | 2.6 GB | 5.9 GB + ~0.8 GB | ✅ |

**峰值**: ~9.7 GB，远低于 24 GB 上限。

---

## 4. 修复文件清单

| 文件 | 改动 | 说明 |
|------|------|------|
| `src/store/faiss_store.py` | `_maxsim_torch()` 重写 | 全量 matmul → 分页批处理 (200页/批) |
| `src/ingestion/encoders.py` | 新增 `unload()`, `encode_queries_batch()`, `_require_loaded()` | ColPali 生命周期管理 |
| `src/retrieval/visual_retriever.py` | 新增 `search_with_embedding()` | 接受预编码 query embedding，跳过现场编码 |
| `src/evaluation/vidore_adapter.py` | `search()` / `search_with_trace()` 扩展 `visual_query_embedding` 参数 | 透传预编码向量 |
| `src/evaluation/ablation.py` | `load_eval_data()` 语言过滤, `run_ablation()` 接收 `pre_encoded_visual` | 评测口径对齐 |
| `scripts/run_eval.py` | Phase A/B 分离, `--language` / `--expected-query-count` | 编排修正 |

---

## 5. 结果

### 5.1 消融指标 (English 283 queries, RTX 4090)

| Config | NDCG@10 | 备注 |
|--------|---------|------|
| BM25_only | 0.4432 | |
| Dense_only | 0.3938 | |
| **Visual_only** | **0.0988** | ✅ 不再全 0 |
| BM25_Dense | 0.4528 | |
| BM25_Dense_Visual | 0.4388 | Visual 参与融合 |
| **Full_with_rerank** | **0.5362** | 🏆 最佳 |

### 5.2 稳定性

- CUDA OOM: **0 次**（上次运行: 每条 query 均触发）
- Visual_only 延迟: **267 ms/query**（分块 MaxSim，27 批 × ~10ms）

---

## 6. 教训

1. **不要假设 tensor shape**。ColPali query 实际产出 1000+ patches（与页面分辨率相关），在小数据量测试时用 `torch.randn(1, 10, 128)` 会掩盖问题。
2. **分块计算是 GPU 大矩阵乘法的标准范式**。5244 页 × 1032 patches 的 matmul 即使在 A100 80GB 上也不应全量计算。
3. **评测口径对齐需要验证数据 schema**。`query_lang` 实际字段是 `language`，值是 `"english"` 而非 `"en"`。
4. **显存排查要逐阶段测量**。手动逐步测试 (load BGE → load ColPali → encode → unload → load FAISS → search) 是定位 OOM 的最有效方法。
