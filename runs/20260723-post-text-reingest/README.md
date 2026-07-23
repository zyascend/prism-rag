# Post Text re-ingest — Full_zerank2 @100q

| 项 | 值 |
|----|-----|
| 日期 | 2026-07-23 |
| 机器 | SeetaCloud 4090D |
| 索引 | Text re-ingest full（`table_context=on`，FAISS 未动） |
| 协议 | `--skip-index` · Full_zerank2 · ColQwen2 · **100q** en · expand/boost **off** |

## 对照

| 臂 | NDCG@5 | **NDCG@10** | R@10 | MRR | latency |
|----|-------:|------------:|-----:|----:|--------:|
| Boot-CP Arm-A（re-ingest **前**） | 0.3437 | **0.3575** | 0.3867 | 0.4821 | 1094 ms |
| **Post re-ingest Arm-A** | 0.3452 | **0.3589** | 0.3872 | 0.4835 | 1049 ms |
| **Δ** | +0.0015 | **+0.0014** | +0.0005 | +0.0014 | −45 ms |

## 解读

1. **同一切片协议**下，上下文表摘要 + 邻居元数据重灌后 **NDCG@10 微升 +0.14pt**，未回退。  
2. 幅度在 100q 噪声带内，**不足以单独宣称大胜**；也说明重灌 **没有伤检索主表**。  
3. `section_path` 仍几乎全空（语料缺 `#` 标题）；增益更可能来自 **表摘要质量/向量** 而非章节路径。  
4. 默认 expand/boost **继续关**。

## 产物

- `arm-A/ablation_results.json` · `run.log`  
- 重灌侧见 `runs/20260723-text-reingest-full/`（云）/ 本地同步目录  

## 决策

| 项 | 内容 |
|----|------|
| 表摘要 + context | 可保留为 **ingest 默认能力**（配置仍可关）；生产是否默认开 context 需 283q/E2E 再定 |
| 下一步（可选） | 283q 定稿 / 表子集 / E2E；关机省钱 |
