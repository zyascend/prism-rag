# 增量索引云验收 Runbook

> 归属 **Boot-A Job2**（与黄金消融同一次 GPU 开机）。  
> 协议：`docs/eval-protocol.md`。脚本：`scripts/cloud_boot_a.sh`。

---

## 前置

- 索引已在数据盘（ColQwen2 FAISS + pgvector），**不重编**全库
- 代码含 `delete_document` 三路编排与 page_hash 增量（PR #26）
- 本地已：`pytest tests/test_p2_incremental.py tests/test_lifecycle.py -q` 全绿

---

## 三幕验收

### A — 删除一致性（幽灵召回）

1. 选一个已知 `doc_id`（云上 PG：`SELECT doc_id, count(*) FROM chunks GROUP BY 1 LIMIT 5;`）
2. 记录删除前：用含该文档内容的 query 检索，确认 top-k 中有其 `page_id`
3. 调用 `PrismRAGRetriever.delete_document(doc_id)`（或 API）
4. 同一 query 再检索：top-k **不得**再出现该 doc 的 page

通过标准：BM25 / Dense / Visual 融合结果均无该 doc 页。

### B — NDCG 不漂移

1. **Baseline**：直接使用 Boot-A Job1 黄金消融中的 `Full_zerank2` NDCG@10（**不要为 baseline 再跑一遍**）
2. 做一次「无语义变更」操作（可选：touch 无关元数据 / 或仅 invalidate 后再搜）**或** 完成 A 的删除后对**剩余语料**评测时需知会「语料已变、NDCG 不可比」
3. **推荐可比路径**：在 **未删业务 doc** 的前提下，对同一索引连续：
   - 仅 `run_eval --config-filter Full_zerank` 一次作为 post-check，与 Job1 Full_zerank2 比
4. 通过标准：\|ΔNDCG@10\| **&lt; 0.005**（283q）；若只跑 100q，在 README 标明并放宽解读

> 注意：若 Job2 先执行了大规模 delete，则 Full_zerank 与 Job1 **不可比**。顺序建议：  
> **先 Job1 消融 → 再 2c 同索引 Full_zerank 复跑（漂移）→ 最后 A 删除抽查（可破坏语料）→ page-diff 用副本或接受单独记时。**

`cloud_boot_a.sh` 默认顺序：Job1 → Job2c 漂移复跑 →（可选）删除抽查说明写入 README。

### C — page-diff 省时

1. 选 1 个 doc，修改约 10% 页文本后 re-ingest（或用测试 PDF）
2. 日志记录 `pages_reencoded` / `pages_skipped` / 墙钟
3. 对比「全页重编码」估计或历史全量 encode 时间
4. 通过标准：重编码页数 ≈ 变更页；README 写清节省比例（有数即可，不要求精确到 1%）

---

## 产物

```text
runs/YYYYMMDD-bootA/incremental/
  README.md          # A/B/C 结论与数字
  drift_eval/        # 可选：第二次 Full_zerank raw
  ghost_check.json   # 可选：删前后 top-k
```

---

## 简历可用句式（有数后填空）

- 增量/同索引复跑 Full_zerank2 NDCG@10 漂移 **&lt; 0.005**（283q）
- page 级 diff 重编码约占变更页，相对全量重建节省约 **X%** 墙钟
- 删文档后三路统一编排，抽查 query 无幽灵召回
