# 20260708-ctxrel-fix — CtxRel 口径修复云端复测

> 分支: `fix/ctxrel-compressed-context` | 云端: AutoDL 4090 | 耗时 20:38

## 根因

`compute_context_relevancy` 原本传入**原始检索 chunk**（`evaluate_generation` 中重新构造 `context_chunks`），
**绕过了 `compress_context` 的 0.4 句级过滤**。RAGAS Context Relevance 定义为"喂给 LLM 的上下文"中
相关句占比，应与 `generate_answer` / `compute_faithfulness` 使用同一份 `context`。修复后在 doc_ref
前缀注入前捕获 `ctx_for_eval = context`，传 `[ctx_for_eval]` 给 CtxRel。

## 结果对比（100-query, colqwen2, Full+rerank）

| 指标 | 修复前 `20260707-ragas-100-docref` | 修复后本 run | Δ |
|:-----|:---:|:---:|:--:|
| **Context Relevance** | **0.1157** | **0.2943** | **+154%** |
| Faithfulness | 0.8886 | 0.8821 | -0.006（噪声） |
| Answer Relevancy | 0.8007 | 0.7913 | -0.009（噪声） |
| generated / rejected | 85 / 15 | 84 / 16 | ~持平 |

## CtxRel 分布变化

| 量 | 修复前 | 修复后 | 说明 |
|:--|:--:|:--:|:--|
| num_sentences mean | 75.3 | 29.8 | 压缩到 ~40%（ratio=0.4 生效） |
| num_relevant mean | 7.8 | 8.2 | 基本不变 → 相关句被 BGE cosine 保留 |
| score mean | 0.116 | 0.294 | 分母纠正 75→30 |

分桶：旧版 47 query 落在 0.01–0.1；新版 32 query 落在 0.3–0.5、11 个 ≥0.5，0 个为零。

## 结论

- 修复不是"刷分"，是把量错的对象纠正过来：分子（相关句）几乎不变，分母从 75（含 TO 编号/前言噪音）
  纠正成 30（压缩后 LLM 实际看到的）。
- Faithfulness/Relevancy 无回归 → 生成口径未变，仅 CtxRel 口径对齐。
- CtxRel 0.294 仍非"高"，进一步提升需 handoff P1（LLM 句级预过滤替代 BGE cosine）+ 更细 chunking。

## 复现

```bash
# 云端（AutoDL 4090, Ollama qwen2:7b + nomic-embed-text, PG 11545 chunks, ColQwen2 FAISS）
export HF_HOME=/root/autodl-tmp/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/run_ragas_metrics.py --max-queries 100 --skip-index \
  --visual-model colqwen2 --output-dir runs/20260708-ctxrel-fix
```
