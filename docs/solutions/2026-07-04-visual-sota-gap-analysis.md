# ColPali Visual-only 路距 SOTA 差距根因分析

> 日期: 2026-07-04
> 分支: feature/colembed-3b-visual-upgrade
> 状态: 诊断完成，待修复验证

---

## 1. 问题陈述

在 ViDoRe V3 Industrial 数据集（English 283 queries）上，Visual_only 配置的 NDCG@10 只有 **~0.13**，而预期 ColPali-v1.3 自身在该子集上至少应达到 **~0.47**（3.6× 差距）。

所有消融运行的一致表现：

| 运行 | 日期 | Visual_only NDCG@10 | 完整管道 (zerank2) |
|------|------|---------------------|-------------------|
| runs/20260702-page-query-fix | 7/2 | 0.1302 | - |
| runs/20260702-query-fix | 7/2 | 0.1313 | 0.5507 |
| runs/20260702-visual-fix | 7/2 | 0.0988 | 0.5362 |
| runs/20260702_1902 | 7/2 | 0.1302* | 0.5715 |

> *注：20260702_1902 云端日志确认加载了完整索引:
> `FAISS 索引已加载: (5,406,564 patches, 5,244 pages, type=flat)`

更关键的是：**添加 Visual 路会拉低 BM25+Dense 融合结果**:
- BM25_Dense = 0.4528
- BM25_Dense_Visual = 0.4402（**下降** 0.0126）

这意味着 ColPali 视觉嵌入本质上是**噪声**——它把相关页面排在后面，把不相关页面排在前面。

---

## 2. 排查路径

### 2.1 排除项（已验证正确的环节）

| 环节 | 结论 | 证据 |
|------|------|------|
| FAISS 索引大小 | ❌ 非本地索引太小问题 | 云端有完整 5244 页索引但分数一致 |
| MaxSim 评分公式 | ✅ 与官方一致 | `_maxsim_torch` 计算 `mean(max_j dot(q_i, p_j))`，官方 `score_multi_vector` 用 `sum(max_j dot(q_i, p_j))`，查询内排名等价 |
| 编码器 API 调用 | ✅ 与 vidore_benchmark 参考一致 | [参考代码](../.venv/lib/python3.14/site-packages/vidore_benchmark/retrievers/colpali_retriever.py) 同样调用 `processor.process_images()` / `processor.process_queries()` |
| 模型前向输出 | ✅ L2 归一化 + attention_mask 正确 | `proj = proj / proj.norm(dim=-1, keepdim=True)` + `proj *= attention_mask` |
| NDCG 计算公式 | ✅ 自洽（二分相关性） | `compute_ndcg()` 用 `1/(i+1)` 折扣，非标准 `log2(i+1)` 但查询内排名等价 |

### 2.2 已修复的历史问题

| 提交 | 问题 | 修复 |
|------|------|------|
| `72d8822` | Query 编码用 `process_images(白图)` → 1024 个无效 image patch 淹没 ~20 个文本 token | 改用 `process_queries()` |
| `b787c8a` | Page 编码传 `text=[""]` → 训练用的 "Describe the image." prompt 缺失 | 改用 `process_images()` |
| `be2d581` | MaxSim 全量 matmul 中间矩阵 21GB → 4090 24GB OOM | 分页 batch 计算（200 页/batch） |

---

## 3. 根因分析

### 3.1 Root Cause A（最可能）：索引构建与查询编码的模型/环境版本不一致

**核心证据：**

FAISS 索引在**一次运行中构建**，查询编码在**另一次运行中完成**。两次运行之间的 `colpali_engine`、`transformers` 或 `PIL` 版本可能不同，导致：

- 页面编码时 `processor.process_images()` 产生一组潜空间嵌入
- 查询编码时 `processor.process_queries()` 产生另一组潜空间嵌入
- 两组嵌入不在同一个空间中 → MaxSim 结果是噪声

**云端环境快照：**
- `colpali_engine==0.3.13`
- `torch==2.8.0+cu128`
- `transformers` 版本未知（运行期间可能已更新）

**本地环境快照：**
- `colpali_engine==0.3.8`
- `torch==2.11.0 (MPS)`

两套环境间 `colpali_engine` 差距 0.3.8 → 0.3.13，涉及多处 processor 和 model 改动。

### 3.2 Root Cause B（很可能）：`use_fast=False` 慢速图像处理器

运行日志报出：
```
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model.
```

`PaliGemmaImageProcessor` 使用慢速代码路径。`use_fast=True` 在 `transformers v4.52` 将成为默认行为。当前版本（v4.51.x？）的慢速和快速处理器可能对图像 resize/padding 有不同的处理方式。

如果索引编码使用了一个版本的图像处理，而查询编码使用了另一个（或在不同的 transformer 版本中运行），图像到 patch 的映射可能不一致。

### 3.3 Root Cause C（可能）：`max_length=50` 截断查询

`process_queries()` 默认 `max_length=50`。工业级查询如：
```
"What is the minimum and maximum RPM for the 1/2HP motor when operating at 115V/60Hz?"
```
编码后 token 数约 35-45 tokens（含 `<bos>Query: ` 前缀和后缀填充 token）。长查询超 50 会被截断，导致：
- 10 个 `pad_token` 缓冲区被切掉（"reasoning buffers" 失效）
- 查询文本后部截断（语义丢失）

截断后的查询质量下降，MaxSim 无法找到相关页面。

### 3.4 Root Cause D（次要/被排除）：无法独立复现

已排除的原因：
- 评分/归一化错误：模型已输出 L2 归一化向量
- chunk grounding 去重：Visual_only 不需要融合，page_id 去重正确
- pgvector/BGE 干扰：Visual_only 只用 FAISS，不走文本路
- GPU 精度：bfloat16 是 ColPali 官方推荐

---

## 4. 修复措施

### 4.1 已实施的修复（commit `16fcd7f`）

| 修复 | 文件 | 改动 |
|------|------|------|
| `use_fast=True` | `src/ingestion/encoders.py:56` | `ColPaliProcessor.from_pretrained(..., use_fast=True)` |
| `max_length=128` | `src/ingestion/encoders.py:106` | `process_queries([text], max_length=128)` |
| `max_length=128` | `src/ingestion/encoders.py:119` | `process_queries(batch, max_length=max_length)` |

### 4.2 验证方法

在云端清除旧索引后从头重建，**同一次运行内**完成索引 + 评测：

```bash
cd /root/prism-rag
python scripts/ingest_vidore.py          # 构建 FAISS 索引
python scripts/run_eval.py --skip-index  # 同次运行编码 query 并评测
```

对比 Visual_only 分数是否从 ~0.13 提升到 ~0.30-0.47 范围。

---

## 5. 关键文件清单

| 文件 | 作用 |
|------|------|
| `src/ingestion/encoders.py` | ColPaliEmbedder（query/page 编码） |
| `src/store/faiss_store.py` | FAISS MaxSim（GPU/CPU 双路径） |
| `src/ingestion/vidore_ingestor.py` | 索引构建管道 |
| `src/evaluation/ablation.py` | 消融运行器 + NDCG/Recall/MRR 计算 |
| `scripts/run_eval.py` | 评测入口（预编码 query → 加载 FAISS → 消融） |

---

## 附录：与官方基准的对比

| 对比项 | 官方 vidore_benchmark | PrismRAG |
|--------|----------------------|----------|
| 模型 | ColPali-v1.3 | 相同 |
| 处理器 | `ColPaliProcessor` | 相同 |
| 图像处理 | `process_images()` | 相同 |
| 查询处理 | `process_queries(max_length=50)` | 相同（已改为 128） |
| 评分 | `score_multi_vector()` → sum | `_maxsim_torch()` → mean |
| 评估公式 | `pytrec_eval` | 自实现 `1/(i+1)` |

两者差距仅在于评分集聚合方式（sum vs mean）和 NDCG 折扣公式（log vs linear），**这些不影响查询内排名**，因此不解释 3.6× 差距。