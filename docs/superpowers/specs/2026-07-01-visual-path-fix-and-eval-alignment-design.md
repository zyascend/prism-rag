# Visual 路修复 & 评测口径对齐 — 修正版设计文档

> 日期: 2026-07-01 | 状态: 待审批
> 关联: handoff.md §6 下一步 #2/#3, longterm-roadmap §3.1

---

## 1. 问题

### 1.1 Visual 路 CUDA OOM 不是单点问题

RTX 4090 24GB 上，Visual 路当前会在两个位置触发显存冲突：

1. **`--skip-index` 评测启动阶段**
   - `scripts/run_eval.py` 先实例化 `ColPaliEmbedder()`。
   - 随后 `faiss_store.load()` 会把全部 ColPali page vectors 搬到 GPU。
   - 模型常驻显存和 MaxSim 向量显存同时存在，进 query loop 之前就可能 OOM。

2. **逐 query 检索阶段**
   - `PrismRAGRetriever.search()` 内部调用 `VisualRetriever.search(query)`。
   - `VisualRetriever.search()` 先 `colpali.encode_query()`，再 `faiss.maxsim_search()`。
   - query 编码和 GPU MaxSim 共用同一张卡，显存峰值再次叠加。

当前量级估算：

| 占显存的 | 大小 |
|----------|------|
| ColPali v1.3 模型 | ~11.4 GB |
| FAISS 向量 (5244 页 × ~1600 patches × 128 dim × 4B) | ~21 GB |
| **合计** | **~32.4 GB > 24 GB** |

结论：问题不只是 `ablation.py` 里“每条 query 先 encode 再 MaxSim”，还包括 `run_eval.py --skip-index` 的初始化顺序。

### 1.2 评测口径当前不对齐论文基线

当前跑的是全部 1698 条 query（283 英文 + 5 种翻译）。ViDoRe V3 论文 Table 1 的 SOTA 分数是**英文 query 子集**。若不显式过滤语言并校验子集规模，结果不能直接对标论文。

---

## 2. 设计原则

1. **不在 `ablation.py` 里复制融合逻辑**
   - BM25 / Dense / Visual 的 route 组合、RRF 融合、rerank、trace 仍由 `PrismRAGRetriever` 统一负责。
   - 预编码 query 只是给 Visual route 增加一个可选输入，不拆散现有检索主链。

2. **把显存生命周期控制放到评测编排层**
   - `run_eval.py` 负责决定：何时创建 ColPali、何时预编码 query、何时卸载模型、何时开始使用 GPU MaxSim。
   - `ablation.py` 只消费已经准备好的 query / qrels / 可选预编码结果。

3. **显式校验英文子集，而不是只“假设字段存在”**
   - 不仅要按 `query_lang` 过滤，还要校验英文子集数量与预期一致。
   - 结果文件中记录 `language` 和 `num_queries`，避免后续误读。

---

## 3. 方案

### 3.1 Visual 路改为“预编码 + 统一检索入口透传”

核心思路：

```text
Phase A: 仅占用 ColPali 模型显存
  run_eval.py 先加载 queries
  -> ColPali 批量预编码 query
  -> 得到 {query_id / q_idx -> tensor[1, n_q, 128]}
  -> unload ColPali

Phase B: 仅占用 GPU MaxSim 向量显存
  -> 加载/复用 FAISS GPU 向量
  -> run_ablation() 调 PrismRAGRetriever.search(..., visual_query_embedding=q_emb)
  -> PrismRAGRetriever 内部仍负责 visual + bm25 + dense + fusion + rerank
```

这样避免了两类峰值叠加：

- `--skip-index` 场景：先预编码、再 `faiss.load()`
- query loop 场景：Visual route 不再现场调用 `encode_query()`

### 3.2 语言过滤改为显式参数 + 子集校验

评测入口新增 `--language`：

- `--language en`：默认，论文对齐模式
- `--language all`：保留当前 1698 query 全量模式

过滤后增加校验：

- 若 `language == "en"`，默认要求 query 数为 **283**。
- 若数量不符，直接 fail fast，而不是静默继续。
- 输出结果中写入 `language` 和 `num_queries`。

---

## 4. 改动清单

### 4.1 `src/ingestion/encoders.py` — `ColPaliEmbedder`

| 新增/改动 | 签名 | 说明 |
|-----------|------|------|
| `unload()` | `() -> None` | 删除 `self.model`，执行 `torch.cuda.empty_cache()`，并把对象状态标记为 unloaded |
| `encode_queries_batch()` | `(texts: List[str], batch_size: int) -> Dict[int, torch.Tensor]` | 批量编码 query，返回 `{idx: tensor[1, n_q, 128]}` |
| `_require_loaded()` | `() -> None` | 在 `encode_query()` / `encode_pages()` / `encode_queries_batch()` 前统一检查模型是否仍可用 |

约束：

- `unload()` 后再次调用 `encode_query()` / `encode_pages()` / `encode_queries_batch()`，必须抛出清晰的 `RuntimeError`，不能让对象处于“半可用”状态。
- 本次**不设计自动 reload**，避免在评测路径里偷偷重新拉起 11GB 模型。

### 4.2 `src/retrieval/visual_retriever.py` — `VisualRetriever`

| 新增 | 签名 | 说明 |
|------|------|------|
| `search_with_embedding()` | `(q_emb: torch.Tensor, k: int = 20) -> List[dict]` | 跳过 `encode_query()`，直接执行 MaxSim + grounding |

要求：

- `search()` 保留现有 API，供在线/API 场景继续使用。
- `search_with_embedding()` 与 `search()` 在相同 query embedding 下，返回的 page grounding / chunk 拼装规则必须一致。

### 4.3 `src/evaluation/vidore_adapter.py` — `PrismRAGRetriever`

这是本次设计的关键接口改动。

| 改动 | 签名 | 说明 |
|------|------|------|
| 扩展 `search()` | `search(..., visual_query_embedding: Optional[torch.Tensor] = None)` | 对外仍是统一检索入口 |
| 扩展 `search_with_trace()` | 同上 | 当 `use_visual=True` 且传入 `visual_query_embedding` 时，Visual route 走 `search_with_embedding()` |

行为约束：

- `use_visual=False` 时，忽略 `visual_query_embedding`。
- `use_visual=True` 且 `visual_query_embedding is not None` 时，**不得**再调用 `self.visual.search(query)`。
- `fusion` / `rerank` / `trace` 逻辑仍全部保留在 `PrismRAGRetriever` 内，不在 `ablation.py` 里复制一份。

### 4.4 `src/evaluation/ablation.py` — 评测执行器

职责收缩：从“自己加载一切并直接驱动 query 编码”改为“消费准备好的评测输入”。

建议改法：

| 改动 | 说明 |
|------|------|
| 新增 `load_eval_data()` | 负责加载 `queries` / `qrels`，应用 `max_queries` 和 `language` 过滤，返回 `queries_ds` 与 `qrel_map` |
| `run_ablation()` 接收 `queries_ds` / `qrel_map` | 避免它自己在不受控时机重新读数据/触发 visual 编码 |
| 新增 `pre_encoded_visual` 参数 | 类型 `Optional[Dict[int, torch.Tensor]]`，供 visual 配置使用 |

伪代码：

```python
def load_eval_data(dataset_path: str, max_queries: int | None, language: str):
    queries_ds = load_dataset(dataset_path, "queries", split="test")
    qrels_ds = load_dataset(dataset_path, "qrels", split="test")

    if language != "all":
        queries_ds = queries_ds.filter(lambda x: x["query_lang"] == language)

    if max_queries:
        queries_ds = queries_ds.select(range(min(max_queries, len(queries_ds))))

    qrel_map = build_qrel_map(qrels_ds)
    return queries_ds, qrel_map


def run_ablation(retriever, queries_ds, qrel_map, output_dir="results", pre_encoded_visual=None, language="en"):
    for config in ABLATION_CONFIGS:
        for q_idx in range(len(queries_ds)):
            q = queries_ds[q_idx]
            query_text = str(q["query"])
            visual_q_emb = None

            if config.use_visual and pre_encoded_visual is not None:
                visual_q_emb = pre_encoded_visual[q_idx]

            retrieved = retriever.search(
                query=query_text,
                k=10,
                use_bm25=config.use_bm25,
                use_dense=config.use_dense,
                use_visual=config.use_visual,
                use_rerank=config.use_rerank,
                visual_query_embedding=visual_q_emb,
            )
```

说明：

- `run_ablation()` 不再直接调用 `retriever.colpali.*`。
- 这样 `ablation.py` 不需要知道 ColPali 生命周期，只负责遍历配置和算指标。

### 4.5 `scripts/run_eval.py` — 编排修正

这里负责真正消除 `--skip-index` 的启动 OOM。

#### `--skip-index` 路径

顺序必须改为：

```text
1. load_eval_data(language=...)
2. 如果有 visual 配置:
     2.1 创建 ColPaliEmbedder
     2.2 encode_queries_batch()
     2.3 unload()
3. faiss_store.load()
4. 构造 VisualRetriever / PrismRAGRetriever
5. run_ablation(..., pre_encoded_visual=...)
```

重点：**`faiss_store.load()` 必须在 query 预编码和 `colpali.unload()` 之后。**

#### 非 `--skip-index` 路径

顺序建议：

```text
1. 用 ColPali 完成 ingest（现有逻辑）
2. load_eval_data(language=...)
3. 若有 visual 配置，复用同一个 ColPali 对 queries 做预编码
4. colpali.unload()
5. run_ablation(..., pre_encoded_visual=...)
```

说明：

- `run_full_cloud.sh` 仍可不改，因为它本来就是单独调用 `scripts/run_eval.py --skip-index`。
- 但 `run_eval.py` 内部初始化顺序必须修正，否则 shell 脚本不改也会继续 OOM。

### 4.6 `scripts/run_eval.py` — CLI

| 新增参数 | 说明 |
|---------|------|
| `--language` | 默认 `en`，支持 `all` |
| `--expected-query-count` | 默认对 `en` 使用 `283`；`all` 不校验 |

---

## 5. 不改的文件

- `src/store/faiss_store.py` — MaxSim 算法本身不改，GPU/CPU 路径不改
- `src/store/pgvector_store.py` — 不变
- `src/retrieval/fusion.py`, `src/retrieval/reranker.py` — 不变
- `scripts/run_full_cloud.sh` — 不改脚本调用方式，但依赖 `run_eval.py` 内部顺序修正生效

---

## 6. 验收标准

### 6.1 功能验收

1. `python scripts/run_eval.py --skip-index --language en --max-queries 10`
   - 能启动并完成，不出现 CUDA OOM。

2. `python scripts/run_eval.py --skip-index --language en`
   - 输出结果元数据里 `language == "en"`
   - `num_queries == 283`

3. `Visual_only` 配置
   - 指标不再是全 0。
   - `visual_top5` trace 非空（至少在命中样本上）。

4. 含 Visual 的融合配置（`BM25_Dense_Visual` / `Full_no_rerank` / `Full_with_rerank`）
   - 能正常执行 fusion / rerank，不需要在 `ablation.py` 手写第二套融合逻辑。

### 6.2 测试验收

至少补以下测试：

1. `tests/test_encoders.py`
   - `encode_queries_batch()` 输出 shape 合法。
   - `unload()` 后调用 `encode_query()` 抛 `RuntimeError`。

2. `tests/test_visual_retriever.py`
   - `search_with_embedding()` 会调用 `faiss.maxsim_search()`，且返回结果结构与 `search()` 一致。

3. 新增 `tests/test_vidore_adapter.py`
   - 当传入 `visual_query_embedding` 时，`PrismRAGRetriever.search_with_trace()` 走 `search_with_embedding()` 而不是 `search()`。
   - `use_visual=False` 时不会误用预编码向量。

4. 新增 `tests/test_ablation.py`
   - `load_eval_data(language="en")` 会过滤语言并保留正确的 `query_id -> qrels` 对应关系。
   - `run_ablation()` 在 visual 配置下会把 `visual_query_embedding` 透传给 retriever。

---

## 7. 风险 & 降级

| 风险 | 概率 | 处理 |
|------|------|------|
| 英文子集数量与预期不符 | 中 | 默认 fail fast，打印实际 `query_lang` 分布和 query 数 |
| `visual_query_embedding` 接口引入分支回归 | 中 | 用单测锁定 `PrismRAGRetriever` 的 route 选择逻辑 |
| `unload()` 后仍有代码误调 `encode_query()` | 中 | 统一抛出明确 `RuntimeError`，不要静默 fallback |
| Visual query 预编码耗时增加 | 低 | 283 条 query 量级可接受，且换来显存稳定性 |

---

## 8. 自检

- [x] 设计已覆盖 `--skip-index` 启动 OOM，而不只是 query loop OOM
- [x] 保持 `PrismRAGRetriever` 为统一融合入口，不在 `ablation.py` 复制 fusion/rerank
- [x] 语言过滤不再只停留在字段假设，增加英文子集数量校验
- [x] 明确了新增测试点和可执行验收标准
- [x] `run_full_cloud.sh` 可保持调用方式不变，但已说明其依赖前提
