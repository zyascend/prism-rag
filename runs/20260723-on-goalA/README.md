# Goal-A — context-ON 索引正式数字（仅 Full_zerank2 + E2E）

| 项 | 值 |
|----|-----|
| 日期 | 2026-07-23 |
| 机器 | SeetaCloud 4090D |
| 索引 | Text re-ingest full · **table_summary_context ON** · FAISS ColQwen2 未重编 |
| 检索 | **仅** Full_zerank2 · expand/boost **关** · `--skip-index` |
| 生成 | E2E 走现网 generator（无 RAGAS） |
| 退出码 | NDCG_EXIT=0 · E2E_EXIT=0 · GOALA_DONE |

---

## 1. L1 检索 — Full_zerank2 · 283q en

| 指标 | 本 run | Boot-A 黄金（283q，旧文本） | Δ |
|------|-------:|----------------------------:|--:|
| **NDCG@10** | **0.5337** | **0.5318** | **+0.19pt** |
| NDCG@5 | 0.5320 | （见 bootA 表） | — |
| Recall@10 | 0.5432 | — | — |
| MRR | 0.6618 | — | — |
| latency | 1099 ms | — | — |
| n | 283 | 283 | |

机器可读：`ndcg283/ablation_results.json`

**解读：** 同协议 v1、同 283 英文集上，**上下文表摘要重灌后检索主表不降、略升**，与 100q 切片 +0.14pt 方向一致，且已是可对外引用的 **283 主表**。

对照 100q 切片（不可与 283 绝对值混比）：

| 切片 | NDCG@10 |
|------|--------:|
| Boot-CP 重灌前 @100 | 0.3575 |
| 重灌后 @100 | 0.3589 |
| **本 run 正式 283** | **0.5337** |

---

## 2. L3 E2E — 50 可答 + 20 应拒

| 指标 | 本 run | Gate2 OFF post-P0（handoff） | 历史 E2E（2026-07-05） |
|------|-------:|----------------------------:|------------------------:|
| **Correct** | **0.66** | 0.60 | 0.64 |
| **Reject accuracy** | **0.95** | 0.90 | 0.95 |
| combined | 0.747 | — | — |
| latency | 2.49 s | 2.24 s | ~2.2 s |
| 可答题误拒 | **9** | 6（post-P0） | — |

机器可读：`e2e/e2e_qa_results.json` → `summary`

**解读：**

- Correct **0.66** 高于 post-P0 0.60 与历史 0.64，方向好。  
- Reject **0.95** 健康。  
- **可答题误拒 9** 偏高（系统说不够答、金标可答）——值得 Failure Clinic 打标，可能与摘要变严/证据变「干净」有关，**不等于** NDCG 差。

---

## 3. 结论（目标 A）

| 问题 | 答案 |
|------|------|
| 当前 ON 索引检索主表如何？ | **NDCG@10 = 0.5337 @283**，≥ Boot-A 黄金 0.5318 |
| 端到端如何？ | **Correct 0.66 / Reject 0.95** |
| 是否改默认 `table_summary_context=true`？ | **仍建议另做 OFF 双臂**（本 run 只有 ON）；仅凭 ON 可写「生产可开、主表不降」 |
| RAGAS / 其它消融 | **未跑**（按计划） |

---

## 4. 产物

```
runs/20260723-on-goalA/
  env.txt
  README.md
  ndcg283/{ablation_results.json,run.log}
  e2e/{e2e_qa_results.json,run.log,...}
```
