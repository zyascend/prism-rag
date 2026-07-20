# Boot-A — 黄金消融 + 漂移复跑（2026-07-20）

> 协议：`docs/eval-protocol.md` v1  
> 主机：SeetaCloud RTX 4090（跑数时有卡；结果在无卡模式拉回）  
> 代码：`feat/bullet-strengthening`（含 `--no-hyde` / HyDE 跳过修复）

## 配置冻结

| 项 | 值 |
|----|-----|
| dataset | vidore_v3_industrial，english，**283** queries |
| NDCG | `1/log2(i+1)` + page 去重 |
| visual | colqwen2，`--skip-index`（复用 `/root/autodl-tmp/indexes`） |
| 消融 | `GOLDEN_NO_HYDE`（8 路，无 HyDE） |
| PG chunks | 8835（text 6530 + table 2305） |
| HF | `HF_HUB_OFFLINE=1`，缓存数据盘 |

## Job1 — 黄金消融 NDCG@10

| config | NDCG@10 | Recall@5 | MRR | avg_latency_ms |
|--------|---------|----------|-----|----------------|
| BM25_only | 0.4063 | 0.3884 | 0.4993 | 16.5 |
| Dense_only | 0.3638 | 0.3231 | 0.4718 | 112.3 |
| Visual_only | 0.1590 | 0.1020 | 0.1727 | 218.4 |
| BM25_Dense | 0.4208 | 0.3837 | 0.5245 | 127.2 |
| BM25_Dense_Visual | 0.4201 | 0.4029 | 0.4973 | 370.8 |
| Full_no_rerank | 0.4201 | 0.4029 | 0.4973 | 0.2* |
| Full_with_rerank (BGE) | 0.5161 | 0.4631 | 0.6307 | 717.7 |
| **Full_zerank2** | **0.5318** | 0.4730 | 0.6601 | 1190.0 |

\* 与 `BM25_Dense_Visual` 配置等价，L3 缓存命中导致延迟失真，分数可信。

### 结论（简历）

- **Full_no_rerank → Full_zerank2：Δ NDCG@10 = +0.1117**（瓶颈在精排）
- **BM25_only → Full_zerank2：Δ = +0.1255**
- 加 Visual 路对 NDCG 几乎无增益（0.4208 → 0.4201）

## Job2 — 同索引漂移

| | Full_zerank2 NDCG@10 |
|--|---------------------|
| Job1 | 0.5318 |
| Job2 | 0.5318 |
| **Δ** | **0.000000**（&lt; 0.005 验收通过） |

## 文件

```text
golden-ablation/ablation_results.json
golden-ablation/ndcg_table.md
golden-ablation/run.log
incremental/drift_eval/ablation_results.json
bootA_20260720.log / bootA_20260720_job2c.log
env.txt
```

## 未做（本 Boot 范围外）

- 幽灵删除 / page-diff 计时（runbook 手工项，可选补）
- Boot-B 路由对照
- RAGAS 生成层
