# Visual 页序修复验证（283q · en）

| 项 | 值 |
|----|-----|
| 日期 | 2026-08-04 |
| 机器 | SeetaCloud 4090D |
| 分支 | `fix/visual-page-rank-order` |
| 协议 | v1（`1/log2` + page 去重） |
| 模型 | ColQwen2 · `--skip-index` · 现有 industrial FAISS |
| 配置 | `--config-filter Visual_only` → `Visual_only` + `Visual_only_pages` |

## 结果

| config | NDCG@5 | **NDCG@10** | Recall@5 | Recall@10 | MRR | lat (ms) |
|--------|-------:|------------:|---------:|----------:|----:|---------:|
| **Visual_only**（修序后生产路径） | 0.4939 | **0.4776** | 0.4585 | 0.4808 | 0.5983 | 141 |
| **Visual_only_pages**（页级 MaxSim） | 0.4999 | **0.5076** | 0.4676 | 0.5283 | 0.6284 | 139 |

## 对照

| 对照 | NDCG@10 |
|------|--------:|
| Boot-A Visual_only（修前） | **0.1590** |
| 本 run Visual_only（修序后） | **0.4776**（**+0.32**） |
| 本 run Visual_only_pages | **0.5076** |
| 官方 ColQwen2 Industrial EN（论文） | **~0.498** |

## 结论

1. **页序 bug 实锤**：修前 0.16 → 修后生产路径 **0.48**，主因是 grounding 乱序 + RRF 位次，不是 ColQwen2 能力只有 0.16。
2. **页级 MaxSim ≈ 官方 pure visual**：`Visual_only_pages` **0.5076** ≈ 官方 ColQwen2 **~0.50**，编码/索引与官方对齐。
3. 生产路径略低于页级（0.48 vs 0.51）：chunk expand + 单路 RRF 仍有小损失，可接受。

## 命令复现

```bash
python scripts/run_eval.py \
  --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 --no-hyde \
  --config-filter Visual_only \
  --output-dir runs/20260804-visual-page-order/
```
