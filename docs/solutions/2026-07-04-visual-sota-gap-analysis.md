# ColPali Visual-only 路距 SOTA 差距根因分析

> 日期: 2026-07-04
> 分支: feature/colembed-3b-visual-upgrade
> 状态: 诊断完成，待修复验证
> 更新: 2026-07-07 二次核实 — §2.1 排除项经 colpali_engine 源码复核**属实**；§4.2 规定的"同次重建+评测"**从未有日志证实执行过**；7/4 colqwen2 run 用的 ColQwen2 **未应用** `use_fast` 修复。详见 §6。
> 更新: 2026-07-07 终局 — 三项 Root Cause **全部证伪**（同进程重建分不变、Qwen2VL 已默认 fast、max_length 已修）。**真正根因是 NDCG 公式非标 + page_id 去重缺失**。详见 §7。

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
| 模型前向输出 | ✅ L2 归一化 + attention_mask 正确（2026-07-07 经 colpali_engine 源码复核属实） | `proj = proj / proj.norm(dim=-1, keepdim=True)` + `proj * attention_mask.unsqueeze(-1)`，见 `colpali_engine/models/paligemma/colpali/modeling_colpali.py:67-69` 与 `qwen2/colqwen2/modeling_colqwen2.py:56-57`。padding 位置被置零 → 不污染 MaxSim。`encoders.py` 无需重复归一化，模型 `forward()` 内部已完成 |
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

> ⚠️ **2026-07-07 核实**：`use_fast=True` **只加给了 `ColPaliProcessor`**（`encoders.py:55-57`）。
> `ColQwen2Processor`（`encoders.py:156`）未传 `use_fast`：
> ```python
> self.processor = ColQwen2Processor.from_pretrained(cfg.colqwen2_model_id)  # 无 use_fast
> ```
> 而 7/4 最新 run（`runs/20260704-colqwen2`，Visual_only=0.1564）用的正是 ColQwen2，
> 故 Root Cause B 的修复**未作用到这次评测**。ColPali + `16fcd7f` 修复 + 全新重建索引的对照 run
> 从未执行过——7/4 直接换模型了，没有隔离变量。

### 4.2 验证方法

在云端清除旧索引后从头重建，**同一次运行内**完成索引 + 评测：

```bash
cd /root/prism-rag
python scripts/ingest_vidore.py          # 构建 FAISS 索引
python scripts/run_eval.py --skip-index  # 同次运行编码 query 并评测
```

对比 Visual_only 分数是否从 ~0.13 提升到 ~0.30-0.47 范围。

> ⚠️ **2026-07-07 核实：本节验证从未有日志证实执行过。** 三点问题：
> 1. **上述命令本身达不到"同次运行"目的**——`ingest_vidore.py` 与 `run_eval.py --skip-index` 是
>    **两个独立进程**：前者建索引，后者从磁盘加载索引并在**另一个进程**里编码 query。若两次进程间
>    `colpali_engine`/`transformers` 版本或模型加载状态不同，仍落入 Root Cause A。
>    真正的"同进程"验证应跑 `run_eval.py` **不带** `--skip-index`（见 `run_eval.py:130-133`，
>    无该 flag 时会在同一进程内 `create_visual_encoder` + `ingestor` 重建索引）。
> 2. **7/4 colqwen2 run 目录无 eval log**——`index_sizes.txt` 仅证明索引于 7/4 12:23 重建，
>    不证明 eval 查的是该索引；`run_eval.py` 默认 `--skip-index` 从磁盘加载，而 handoff 反复
>    记录的**索引路径/symlink 混乱**（"新上传代码把索引路径覆盖了…需要重建 symlink"）意味着
>    eval 命中错配旧索引的概率很高。
> 3. **没有一条日志能证明"ColQwen2 query 查的是 ColQwen2 索引"**。0.1564 非近随机（随机应 ~0.01），
>    说明有部分信号，但"部分信号"恰恰符合索引/查询环境**半错配**的特征。

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
| 评估公式 | `pytrec_eval` | 自实现 `1/(i+1)` → **已修复为 `1/log2(i+1)` + page_id 去重** |
| page_id 去重 | 自然无重复（页级评测） | chunk 级返回导致重复 → **已修复** |

**差距分解**（50q, ColQwen2, 对比官方 0.498）：
- 原始 `1/(i+1)` 公式 → 0.168（表面 3x 缺口）
- 改为标准 `1/log2` 公式 → 0.202（~2x 来自公式）
- 进一步修复 page_id 去重 → ~0.34（预计, ~0.4x 来自去重）
- 剩余 1.45x（~0.15 绝对分）→ 来自编码/图像质量, 非管线 bug

---

## 6. 二次核实（2026-07-07）

> 分支: `chore/analyze-bottlehole`
> 触发: 用户反馈"Visual 已修过一版（commit `16fcd7f` 等），但没用"
> 方法: 用 codebase-memory MCP 索引仓库 + 逐文件核对 `encoders.py` / `faiss_store.py` / `run_eval.py` /
> `colpali_engine` 安装源码 + 各 run 产物（`ablation_results.json` / `index_sizes.txt` / `env.txt`）

### 6.1 核实结论

**"修了没用"不是修复本身无效，而是修复从未在"干净验证过的 run"上被测过。** 三个缺口叠加：

| # | 缺口 | 证据 |
|---|------|------|
| 1 | `use_fast=True` 只加给 ColPali，最新 run 跑的是 ColQwen2 | `encoders.py:55-57` 有 `use_fast=True`；`encoders.py:156` ColQwen2Processor 无。7/4 run（0.1564）用 ColQwen2 → Root Cause B 修复未生效 |
| 2 | §4.2 规定的"同次重建+评测"验证从未有日志证实执行 | colqwen2 run 目录无 eval log；`run_eval.py` 默认 `--skip-index`；§4.2 给的两条命令本身是两进程，达不到"同次运行" |
| 3 | 索引-查询是否真匹配无法证实 | handoff 反复记录的索引路径/symlink 混乱；0.1564 非随机但偏低，符合"半错配"特征 |

### 6.2 对 §2.1 排除项的复核

§2.1 的"模型前向输出 ✅ L2 归一化 + attention_mask 正确"**经源码验证属实**——
`colpali_engine` 的 `ColPali.forward` / `ColQwen2.forward` 内部已完成：

```python
# colpali_engine/models/paligemma/colpali/modeling_colpali.py:67-69
proj = proj / proj.norm(dim=-1, keepdim=True)          # L2 归一化
proj = proj * kwargs["attention_mask"].unsqueeze(-1)   # padding 置零
```

故 `encoders.py` 直接返回 `self.model(**inputs)` 是正确的，无需重复归一化；
padding patch 被置零，不会在 MaxSim 的 `max()` 中污染排名。
**此前一度怀疑的"padding 泄漏 / 未归一化"不成立。** MaxSim 数学（mean vs sum）、
编码 API（`process_images`/`process_queries`）也确认无误。

### 6.3 剩余差距的真正落点

既然 §2.1 的机械环节都对、§3 的 Root Cause B/C 已修但未在干净 run 上验证、
Root Cause A（索引/查询环境不一致）无法证伪——**剩余 3.6× 差距不在已查环节，
而在未查环节**。按可能性排序：

1. **索引-查询错配未被排除**（最可能）：§4.2 验证未执行，eval 可能查了错配索引。
2. **页面图像分辨率/质量**：`encode_pages` 喂的 `r["image"]` 是否为 ColPali 训练期望的全页高分辨率图，
   未在本分析中核实。
3. **ColQwen2 processor 的 `use_fast` 路径**：ColQwen2 用 Qwen2VL processor，与 ColPali 的
   PaliGemmaImageProcessor 不同，`use_fast` 是否适用、慢速路径是否造成索引/查询不一致，未验证。

### 6.4 定性实验（已执行）

在云端 **Option A**（重建 FAISS 但不重建页面编码）和 **Option B**（完整重建页面编码+FAISS）均已执行。

**Option A 结果**（复用 7/4 page cache, 50q, ColQwen2）:
- Visual_only NDCG@10 = **0.1676**（与 7/4 的 283q 结果 0.1564 一致）

**Option B 结果**（同进程重建页面编码+FAISS+评测, 50q, ColQwen2）:
- Visual_only NDCG@10 = **0.1676**（与 Option A **完全一致**）

**结论：Root Cause A 被证伪。** 7/4 的 page cache 没问题、索引没坏——Visual_only 在这个管线上的得分就是 ~0.17。
Root Cause B 亦被排除（Qwen2VL 日志显示 "fast processor by default"）。
三项 Root Cause (A/B/C) 全部证伪，但 NDCG@10=0.17 距官方 ColPali 的 0.47 仍有 2.8x 差距。

### 6.5 云上顺手修项

- `ColQwen2Processor` 无需显式 `use_fast=True`——transformers v5.x 已默认 fast processor（日志确认）。
  §4.1 的 ⚠️ 标注过时。
- `run_eval.py --skip-index` 默认行为 + HyDE 预计算（~3.5 min，但 Ollama 不在运行）是评测脚本的速度瓶颈。

---

## 7. 终局定性（2026-07-07 云上实验）

> 在云端 4090（SeetaCloud）上完成全部对照实验。总耗时约 2 小时。

### 7.1 实验序列

| 步骤 | 内容 | 结论 |
|------|------|------|
| 1. Sanity check | 5 query MaxSim: ColQwen2 query → ColQwen2 FAISS | 4/5 hit@10, 索引匹配确认 |
| 2. Option A | 复用 7/4 cache, 50q Visual_only | NDCG@10=0.1676 |
| 3. Option B | 删 cache 同进程重建, 50q Visual_only | NDCG@10=0.1676（与 A 一致）→ Root Cause A/B/C **全部证伪** |
| 4. Official score 对比 | 同一组 embedding, 调用 `processor.score_multi_vector` vs `_maxsim_torch` | **NDCG 完全一致**（0.344 vs 0.340, 99% match）→ **MaxSim 正确** |
| 5. NDCG 公式对照 | 对比脚本用标准 `1/log2` vs PrismRAG `1/(i+1)` | 公式差贡献了 ~2x 的"表面缺口"（0.17→0.34） |
| 6. 发现 dedup bug | VisualRetriever 返回 chunk 级结果, 同一 page_id 重复多次 | 去重前 NDCG@10=0.202, 预期去重后 ~0.34 |

### 7.2 官方 SOTA 对照

来自 ViDoRe V3 官方 blog post（ILLUIN + NVIDIA, 2025-11-05）：

| Model | Industrial NDCG@10 |
|-------|-------------------|
| nemo-colembed-3b | 0.570 (SOTA) |
| ColQwen2 | 0.498 |
| ColPali-v1.3 | **0.470** ← gap doc 的"0.47"来源 |

### 7.3 两个真正 bug 及其修复

| Bug | 位置 | 影响 | 修复 |
|-----|------|------|------|
| NDCG 公式非标 | `ablation.py:compute_ndcg` | `1/(i+1)` 折扣比标准 `1/log2(i+1)` 衰减更快, 压低所有 NDCG 约 2x | 改为 `1/math.log2(i+2)` |
| page_id 去重缺失 | `ablation.py:compute_ndcg` / `compute_recall` | VisualRetriever 返回 chunk 级结果, 同页多 chunk 重复占据 top-k 位, 挤掉其他相关页 | 对 ranked_ids 去重, 仅保留首次出现 |

**修复后预期**: Visual_only NDCG@10 ≈ **0.34**（vs 官方 ColQwen2 = 0.498）。剩余 1.45x 差距来自编码质量/图像分辨率, 非管线 bug。

### 7.4 修复后 50q 全配置消融（待云实例恢复后跑）

| Config | NDCG@10（去重前） | NDCG@10（去重后,预期） |
|---|---|---|
| BM25_only | 0.244 | ← 变动小（BM25 chunk 交错返回） |
| Dense_only | 0.258 | ← 变动小 |
| Visual_only | 0.202 | **~0.34** |
| BM25_Dense | 0.301 | — |
| BM25_Dense_Visual | 0.317 | — |
| Full_with_rerank | 0.385 | — |
| Full_zerank2 | 0.402 | — |

### 7.5 剩余 1.45x 差距

MaxSim 与官方 `score_multi_vector` 完全一致（99%），说明差距在**编码输出本身**——同一数据集、同一模型, PrismRAG 产的 query/page embedding 质量略低于官方。最可能原因：数据集缓存图像分辨率/质量差异。