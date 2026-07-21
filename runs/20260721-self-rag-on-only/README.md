# Self-RAG Gate2 — ON-only re-run + 干净对照（post-P0）

| 项 | 值 |
|----|----|
| 日期 | 2026-07-21 |
| 主机 | SeetaCloud 4090 · `connect.cqa1.seetacloud.com:44683` |
| 代码 | `feat/self-rag-gate2`（含 `src/rejection.py`、Faith/Rel 排除拒答、`trigger`） |
| 本 run | **仅 ON**：`SKIP_OFF=1` · `SELF_RAG_TRIGGER=always` · RAGAS 100q + E2E 50/20 |
| 产物目录 | `runs/20260721-self-rag-on-only/on/` |

## 1. 本 run 在线结果（ON-new）

### RAGAS 100q（在线 summary，口径已是 post-P0）

| 指标 | 值 |
|------|---:|
| Faithfulness | **0.9277**（n=83 放行） |
| Answer Relevancy | **0.8144** |
| CtxRel | **0.2606** |
| 拒答 / 生成 | **17 / 83** |

### E2E 50+20

| 指标 | 值 |
|------|---:|
| Correctness | **0.62** |
| Rejection accuracy | **0.95** |
| 可答被拒 | 6 |
| avg latency | 3.81 s |
| combined | 0.719 |

---

## 2. 干净对照（同一 post-P0 口径）

> **OFF 未重跑 LLM**：对 `20260721-self-rag-gate2/off` 的 JSON 用 `src.rejection.is_rejection` 重算。  
> **ON-old**：同日首次 always 跑，同样重算。  
> 机器可读：`comparison_post_p0.json`。

### 2.1 RAGAS

| 臂 | 来源 | Faith | Rel | CtxRel | 拒答数 | 放行 n |
|----|------|------:|----:|-------:|-------:|-------:|
| **OFF**（重算） | gate2/off 旧生成 | **0.9188** | 0.8157 | 0.2549 | 17 | 83 |
| **ON-old**（重算） | gate2/on 旧生成 | 0.9122 | 0.8130 | 0.2551 | 16 | 84 |
| **ON-new**（在线） | **本 run** | **0.9277** | 0.8144 | 0.2606 | 17 | 83 |
| Δ (ON-new − OFF重算) | | **+0.009** | −0.001 | +0.006 | 0 | 0 |

**原始存储 summary（污染口径，仅归档，勿写简历）：**

| 臂 | Faith 原始 | 拒答原始 |
|----|-----------:|---------:|
| OFF | 0.8295 | 2（漏检） |
| ON-old | 0.7857 | 0（Gate2 拒答未识别） |

### 2.2 E2E

| 臂 | Correctness | Reject Acc | 可答被拒 | latency |
|----|------------:|-----------:|---------:|--------:|
| OFF 重算 | 0.60 | **0.90**（原始 0.25） | 6 | 2.24 s |
| ON-old 重算 | 0.60 | 0.95 | 4 | 3.96 s |
| **ON-new** | **0.62** | **0.95** | 6 | 3.81 s |

---

## 3. 结论（定稿口径）

1. **P0 修复有效**  
   - 拒答不再记 Faith=0 拉垮均值；OFF/ON 拒答规模都在 **~16–17/100**。  
   - OFF 原始 Faith 0.83 / ON 0.79 的「Gate2 伤 Faith」叙事 **作废**。

2. **Gate2 always 相对 OFF（干净对照）**  
   - Faith **+0.9pt**（0.919 → 0.928），Rel 持平，CtxRel 微升。  
   - **不是**质变；放行子集都更「干净」，两臂拒答率接近（OFF 靠软拒答句，ON 靠硬 Gate2 句）。

3. **E2E**  
   - 拒答集：新检测后 OFF 也有 **0.90**，ON **0.95**（硬拒更稳一点）。  
   - Correctness：ON-new **0.62** vs OFF 0.60，仍在噪声级；**检索 badcase 仍是主因**。  
   - 延迟：OFF ~2.2s → ON ~3.8s（**×1.7**）。

4. **产品默认**  
   - 维持 `enabled: false` 或生产用 `trigger: low_rerank`。  
   - 简历：**不要**写旧 0.83→0.79；若写 Gate2，写「口径修复后 Faith 微升 + 拒答硬拦截，延迟 ×1.7」，或只口述阴性/边际。

---

## 4. 文件清单

```
runs/20260721-self-rag-on-only/
├── README.md                 # 本文件
├── comparison_post_p0.json   # OFF/ON-old 重算 + ON-new
├── pipeline.log
├── env.txt / launch.txt
└── on/
    ├── models.boot.yaml
    ├── ragas/ragas_metrics_default.json
    └── e2e/e2e_qa_results.json
```

旧完整双臂目录仍保留：`runs/20260721-self-rag-gate2/`（含污染口径原始 summary，分析见 `badcase_analysis.md`）。

---

## 5. 运维

结果已在本地；云上可 **关机**。
