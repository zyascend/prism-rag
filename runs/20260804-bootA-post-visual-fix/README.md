# Boot-A 重跑 · Visual 页序修复后（283q · en）

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-04 |
| 机器 | SeetaCloud 4090D |
| 代码 | main（含 #40 Visual 页序修复）· skip-index · colqwen2 · `--no-hyde` |
| 协议 | v1（`1/log2` + page 去重） |
| 对照 | 旧 Boot-A [`runs/20260720-bootA/`](../20260720-bootA/) |

## 黄金消融表（NDCG@10）

| config | 旧 Boot-A | **本次** | Δ |
|--------|----------:|---------:|--:|
| BM25_only | 0.4063 | **0.4063** | 0 |
| Dense_only | 0.3638 | **0.3724** | +0.009 |
| Visual_only | 0.1590 | **0.4776** | **+0.319** |
| Visual_only_pages | — | **0.5076** | 新臂 |
| BM25_Dense | 0.4208 | **0.4249** | +0.004 |
| BM25_Dense_Visual | 0.4201 | **0.4827** | **+0.063** |
| Full_no_rerank | 0.4201 | **0.4827** | **+0.063** |
| Full_with_rerank (BGE) | 0.5161 | **0.5227** | +0.007 |
| **Full_zerank2** | **0.5318** | **0.5454** | **+0.014** |

## 主结论

1. **Full_zerank2 = 0.5454**（新主表），相对旧 Boot-A **+1.4pt**。
2. **Visual 从负贡献变正贡献**：BM25_Dense 0.425 → +Visual **0.483**（+6pt）；修前曾略掉点。
3. **精排仍是主增益**：Full_no_rerank 0.483 → Full_zerank2 **0.545**（**+6.3pt**）。
4. Visual pure 与专项 run 一致：0.478 / pages 0.508。

## 命令

```bash
python scripts/run_eval.py \
  --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 --no-hyde \
  --output-dir runs/20260804-bootA-post-visual-fix/
```
