# Bullet 强化路线图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按简历/面试 ROI 优先级，把 4 条简历 bullet 背后的系统能力做厚：可辩护评测协议与黄金消融、增量索引云验收、Visual 按需路由、LLM 上下文过滤、全量生成/E2E 评测硬化。

**Architecture:** 逻辑上分 5 波（Wave），**云上按「开机窗口 Boot」打包**，默认目标 **1～2 次 GPU 开机** 完成 S/A 级（见下节 Cloud Boot Packing）。本地 macOS：协议文档、代码、单测、≤10q 冒烟；全量 283q / RAGAS / 增量 GPU 验证一律上云，遵守 `Agents.md`。

**Tech Stack:** 现有 PrismRAG（`src/evaluation/*`, `src/retrieval/*`, `src/generation/*`, `src/store/*`）、ViDoRe Industrial、pgvector、FAISS、ColQwen2、BGE/zerank-2、RAGAS 自实现、FastAPI Trace/Cache。

**关联简历 bullet:**

| Bullet | 强化目标 |
|--------|----------|
| ① 混合检索 / 精排瓶颈 | 现行协议黄金消融表 + Visual 路由延迟–质量 |
| ② 三层评测 / 修尺子 | eval protocol + 门禁 + 全量/扩集数字 |
| ③ 上下文 / CtxRel | 干净归因 +（可选）LLM 过滤 |
| ④ 一致性 / Trace / 缓存 | 云上不漂移 + 省 GPU% |

**明确不做（本计划范围外）:** Self-RAG 全量实现、多租户 ACL、换一堆 embedding 模型名、本地全量 5244 页索引、不可复现的「超 SOTA」特调、**默认不跑 HyDE 全量**（已有阴性结论，除非 Boot 有余量）。

---

## Cloud Boot Packing（减小开机次数 · 主策略）

> **计费单位是「开机窗口」不是「Wave」。**  
> 原则：① 同一索引状态能串的 job 一次跑完；② **代码改完再开机**，避免为半成品空转；③ 生成侧（RAGAS）比检索贵，能砍则砍或缩到 100q；④ HyDE / 重复 Full_zerank 能并则并。

### 三档预算（选一执行）

| 档位 | GPU 开机 | 覆盖 | 简历够用度 |
|------|:--------:|------|------------|
| **Minimal** | **1 次** | Boot-A：协议已本地写好 → 黄金消融（无 HyDE）+ 增量验收（与消融共用索引） | ①④ 硬；②③ 沿用已有数字 |
| **Standard（推荐）** | **2 次** | Boot-A 同上 + 本地做完路由/过滤代码 → **Boot-B**：路由对照 + RAGAS 100q（BGE 基线，LLM 可选） | ①②③④ 都有新料 |
| **Full** | **2～3 次** | Standard + Boot-B/C 加 RAGAS 283 + E2E 扩集 | 最厚，钱最多 |

**默认按 Standard：2 次开机。** Minimal 若只开 1 次也算计划成功。

### Boot-A —「锁检索数字」（第 1 次开机，必做）

**本地先做完（0 云费）：** `docs/eval-protocol.md`、增量 runbook/脚本、单测绿、代码已推到云可拉的 commit。索引 **不要重编**（复用已有 ColQwen2 + pgvector）。

**上机后一条龙（单 shell 串联，中途不关机）：**

```text
Boot-A pipeline（同一进程环境 / 同一索引）
│
├─ Job1  黄金消融 283q，配置=GOLDEN_NO_HYDE（约 8 路）
│        → runs/$D-bootA/golden-ablation/
│
├─ Job2  增量验收（轻）
│        ├─ 2a 删 1 doc → 抽 5～10 条 query 确认无幽灵召回（秒～分钟级）
│        ├─ 2b page-diff：改少量页 re-ingest，记 reencode 页数与耗时（可不全量 NDCG）
│        └─ 2c 【省跑】不漂移：直接引用 Job1 的 Full_zerank2 作 baseline；
│              增量操作后只再跑 1 次 Full_zerank2 283q（或 100q 若极省）
│        → runs/$D-bootA/incremental/
│
└─ （可选余量）Job3  同机再打 50 条重复 query 看 L3 cache 命中率（CPU/检索，便宜）
```

| 项 | 约定 |
|----|------|
| 消融配置 | **去掉两个 HyDE**；保留 BM25/Dense/Visual/组合/Full_no_rerank/Full_with_rerank/Full_zerank2 |
| Full_zerank 次数 | Boot-A 内 **最多 2 次**（Job1 表内 1 次 + Job2c 增量后 1 次）；禁止「连跑两次纯为确认噪声」 |
| 预估墙钟 | 检索为主约 **1.5–2.5 h**（视 rerank 而定） |
| 关机 | 结果 scp/`pull_from_cloud` 后 **立刻关机** |

**Boot-A 交付：** 黄金表 + 漂移/幽灵/page-diff 数字 → bullet ①④。

### Boot-B —「新特性对照」（第 2 次开机，Standard）

**本地先做完（0 云费）：** Wave3 VisualRouter +（可选）Wave4 LLM filter 全部单测通过并推送；**一次开机评两个特性**，不要「改路由开一次、再改过滤又开一次」。

**上机后一条龙：**

```text
Boot-B pipeline（代码已含 router ± context_filter）
│
├─ Job1  检索：Full_zerank2 ×2
│        ├─ visual_routing=always（或 disabled）
│        └─ visual_routing=heuristic
│        → 283q 理想；预算紧改为 **100q 同一 seed/前 100 条英文** 并在 README 标明
│
├─ Job2  生成：RAGAS **100q**（默认只跑当前默认压缩=BGE）
│        └─ 若 Wave4 已合入：同 100q 再跑 mode=llm 或 bge_then_llm（+1 job）
│        → 不在此 Boot 跑满 283（留给 Full 或永久不做）
│
└─ （可选）Job3  E2E 现有 50+20 快速回归（若 Job2 已很长则跳过）
```

| 项 | 约定 |
|----|------|
| 路由与过滤 | **同 commit、同开机**；过滤若未就绪则 Boot-B 只跑路由，过滤并进下次（尽量避免第 3 次） |
| RAGAS | **默认 100q 不是 283** |
| 预估墙钟 | **2–3.5 h** |
| 关机 | 拉结果后立刻关机 |

**Boot-B 交付：** 路由延迟–NDCG；可选 Faith/CtxRel 对照 → bullet ① 补强 + ③。

### Boot-C —「评测扩容」（仅 Full 档，可永久跳过）

```text
Boot-C（可选）
├─ RAGAS 283 全量（当前默认生成配置 only，禁止再开 2 模式×283）
└─ E2E 扩集 100+40（若数据集未扩，本地先扩再上云，避免占 GPU 做数据标注）
```

预估 **2–4 h**。简历不依赖本 Boot 也可讲完整故事。

### 整合后的次数对照

| | 旧（按 Wave 拆机） | **新（Boot 打包）** |
|--|-------------------|---------------------|
| Minimal | Wave1 机 + Wave2 机 = 2 | **1 次 Boot-A** |
| Standard | 4～5 次机 | **2 次（A+B）** |
| Full | 5+ 次机 | **2～3 次（A+B+可选 C）** |
| 云上 job 数 | 9～11 | Minimal **2～3**；Standard **4～6**；Full **5～7** |

### 省钱硬规则（执行时勾选）

- [ ] 开机前：索引已在数据盘、代码已拉、依赖已装（**无卡时段或上次关机前装好**）
- [ ] 开机前：本地 `pytest` 相关单测全绿
- [ ] **不重编** ColQwen2，除非 page-diff 实验必须
- [ ] 黄金消融 **默认无 HyDE**
- [ ] 增量不漂移：**复用**消融里的 Full_zerank 分，只追加 **1** 次评测
- [ ] Boot-B 前：router+filter **合并进同一分支/同一 commit 序列** 再开第二机
- [ ] RAGAS 默认 **100q**；283 仅 Boot-C
- [ ] 每个 Boot 一个 `runs/YYYYMMDD-bootA|B|C/` 总 README，避免碎片 run 导致重跑

### 推荐脚本（减少手滑空转）

- Create: `scripts/cloud_boot_a.sh` — 串 Job1 消融 → Job2c 增量后 Full_zerank → 写 summary  
- Create: `scripts/cloud_boot_b.sh` — 串 always/heuristic 两跑 + RAGAS 100q（读 env `RUN_LLM_FILTER=0|1`）  
- 本地只维护脚本；**真正 `bash` 仅在云上 GPU 机执行**

---

## Global Constraints

- 分支：从 `main` 拉 `feat/bullet-strengthening`（或每波独立子分支再合并）。**禁止直接在 main 改代码。**
- 本地禁止：下大模型、全量 ingest、全量 283 消融、全量 RAGAS（除非用户明确要求）。
- 云上：先查缓存再下载；**有卡时段只跑评测/必要编码**；结果 `runs/` + README；**Boot 结束立即关机**。
- 数字口径：一律 **标准 NDCG `1/log2(i+1)` + page 去重**；禁止与旧公式 run 混比。
- 每 Boot 结束：更新 `handoff.md` + `local/resume-prismrag.md`（`local/` 不入库）。
- 提交粒度：本地 Task 结束 commit；云上结果拉回后单独 commit。

---

## File Structure（跨波会碰的文件）

| 文件 | 责任 |
|------|------|
| `docs/eval-protocol.md` | **新建** 评测协议（数据集、公式、配置冻结、产物路径） |
| `docs/superpowers/plans/2026-07-20-bullet-strengthening-roadmap.md` | 本计划 |
| `scripts/cloud_boot_a.sh` | **新建** Boot-A 一键串（消融+增量） |
| `scripts/cloud_boot_b.sh` | **新建** Boot-B 一键串（路由±RAGAS） |
| `src/evaluation/ablation.py` | 消融配置；`GOLDEN_NO_HYDE` 子集 |
| `scripts/run_eval.py` | Layer1；建议支持跳过 HyDE 或 `--config-filter` 批跑 |
| `scripts/run_ragas_metrics.py` | Layer2 |
| `scripts/run_e2e_qa.py` | Layer3 |
| `scripts/verify_incremental_cloud.sh` | 增量验收（可被 boot_a 调用） |
| `src/retrieval/visual_router.py` | Wave3 / Boot-B 代码 |
| `src/evaluation/vidore_adapter.py` | 接入 router |
| `src/generation/context_filter.py` 等 | Wave4 / Boot-B 可选 |
| `tests/test_visual_router.py` / `test_llm_context_filter.py` / `test_ndcg_metric.py` | 本地门禁 |
| `runs/YYYYMMDD-bootA\|B\|C/` | **按 Boot 归档**，不按碎片 job 散落 |
| `handoff.md` | 进度与数字 |

---

## Wave ↔ Boot 映射（逻辑波次仍保留，云上不跟波次开机）

| Wave | 名称 | ROI | **在哪次开机跑** | 本地先做 |
|:----:|------|:---:|------------------|----------|
| **1** | 协议 + 黄金消融 | S | **Boot-A Job1** | protocol 文档 |
| **2** | 增量验收 | S | **Boot-A Job2**（与 Job1 同机） | runbook/脚本 |
| **3** | Visual 路由 | A | **Boot-B Job1** | router 代码+单测 |
| **4** | LLM 上下文过滤 | A | **Boot-B Job2**（可选） | filter 代码+单测；与 3 同 commit 再开机 |
| **5** | RAGAS283/E2E | A | **Boot-C 或跳过** | E2E 数据扩集、NDCG 单测 |

**时间线（推荐）：**

```text
本地: protocol + boot 脚本 + 单测
   → GPU Boot-A（1.5–2.5h）→ 关机
本地: router +（可选）filter 开发合并
   → GPU Boot-B（2–3.5h）→ 关机
（可选）本地扩 E2E 数据 → GPU Boot-C → 关机
```

---

# Wave 1 — 评测协议 + 黄金消融表（S）
> 云上归属：**Boot-A Job1**（与 Wave2 同机，勿单独开机）

### Task 1.1: 冻结并文档化 eval protocol

**Files:**
- Create: `docs/eval-protocol.md`
- Modify: `handoff.md`（链到 protocol）

- [ ] **Step 1: 确认当前代码中的硬约束（只读）**

```bash
# NDCG 已是 log2 + page 去重
rg -n "log2|seen" src/evaluation/ablation.py | head -20
# 消融配置列表
rg -n "ABLATION_CONFIGS|Full_zerank2|Full_no_rerank" src/evaluation/ablation.py
```

Expected: `compute_ndcg` 使用 `math.log2(pos + 2)` 与 `seen` 去重；存在 `Full_no_rerank` / `Full_zerank2`。

- [ ] **Step 2: 创建 `docs/eval-protocol.md`**

内容必须包含（勿省略字段）：

```markdown
# PrismRAG Eval Protocol v1

## 数据集
- name: vidore/vidore_v3_industrial
- language: english only
- query_count: 283（`--language en`，`--expected-query-count 283`）

## 指标
- NDCG@10: 1/log2(i+1)，page_id 首次出现计分（见 ablation.compute_ndcg）
- 同时报 Recall@5、MRR
- 延迟: 每配置平均 query 延迟（ms）

## 索引与模型冻结（黄金 run 必须写进 run README）
- visual_model: colqwen2（默认）
- table_summary_enabled: <true|false 二选一，黄金表选定后锁定>
- reranker: bge + zerank 均需出现在消融表
- 禁止与 2026-07-02 旧公式 run 直接比绝对 NDCG

## 黄金消融默认集合（GOLDEN_NO_HYDE，Boot-A）
BM25_only, Dense_only, Visual_only,
BM25_Dense, BM25_Dense_Visual,
Full_no_rerank, Full_with_rerank, Full_zerank2
# HyDE 两组默认不跑（历史已证明本场景无效）；仅 Full 档余量时补

## 产物（按 Boot 归档）
- runs/YYYYMMDD-bootA/golden-ablation/ndcg_table.md
- runs/YYYYMMDD-bootA/README.md（含 git commit、models.yaml 摘要、增量摘要）

## 冒烟
- 本地: python scripts/run_eval.py --max-queries 10 --skip-index --config-filter Full_zerank
```

- [ ] **Step 3: 本地确认冒烟命令可解析（无需全量）**

```bash
python scripts/run_eval.py --help
```

Expected: 含 `--max-queries`, `--skip-index`, `--config-filter`, `--visual-model`。

- [ ] **Step 4: Commit**

```bash
git checkout -b feat/bullet-strengthening   # 若尚未建分支
git add docs/eval-protocol.md handoff.md
git commit -m "docs: add eval protocol v1 for golden ablation"
```

---

### Task 1.2: 云上跑黄金消融（283q）

**Files:**
- Create: `runs/YYYYMMDD-golden-ablation/README.md`（日期用实际跑数日）
- Create: `runs/YYYYMMDD-golden-ablation/ndcg_table.md`

**前置（云上）:** Phase1 环境已好；ColQwen2 索引与 pgvector 数据就绪；代码为含 protocol 的 commit。

- [ ] **Step 1: 记录环境指纹到 README 草稿**

```bash
cd /root/prism-rag   # 或实际路径
git rev-parse HEAD
python -c "from src.config import cfg; print(cfg.get('models', {})); print(cfg.get('embedding', {}))"
ls -la indexes/ 2>/dev/null; ls /root/autodl-tmp/indexes/ 2>/dev/null
```

- [ ] **Step 2: 跑全量消融（无 HyDE 可先全量再单独补 HyDE；若时间紧用 config-filter 分两批）**

```bash
# 批次 A：主表（建议）
python scripts/run_eval.py --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 \
  --output-dir runs/$(date +%Y%m%d)-golden-ablation/raw

# 若 HyDE 太慢：单独
# python scripts/run_eval.py --skip-index --language en --config-filter HyDE \
#   --visual-model colqwen2 --output-dir runs/$(date +%Y%m%d)-golden-ablation/raw-hyde
```

Expected: 产出各配置 NDCG@10 / Recall@5 / MRR / latency。

- [ ] **Step 3: 整理 `ndcg_table.md` 并写结论段**

必须回答：

1. `Full_no_rerank` vs `Full_zerank2`（或 `Full_with_rerank`）ΔNDCG@10 是否仍支持「瓶颈在精排」？
2. `BM25_only` vs 融合无 rerank 是否接近？
3. HyDE Δ 是否仍 &lt; 0.01？
4. 本 run 的 `table_summary_enabled` / visual 模型 / git SHA。

- [ ] **Step 4: 拉回本地、归档、commit**

```bash
# 本地
bash scripts/pull_from_cloud.sh <host> <port> <password>   # 若项目脚本适用
git add runs/YYYYMMDD-golden-ablation docs/eval-protocol.md
git commit -m "results: golden ablation under eval protocol v1"
```

- [ ] **Step 5: 更新 `local/resume-prismrag.md` bullet① 主数字为黄金表（不入库）**

用 **本 run** 的 no-rerank → full+zerank 绝对分替换旧 0.44/0.57（若相对结论变了，同步改正文结论，禁止硬凑）。

**Wave1 完成定义:** protocol 文档合入；`runs/*-golden-ablation` 有表+结论；简历 bullet① 数字与协议一致。

---

# Wave 2 — 增量索引云验收（S）
> 云上归属：**Boot-A Job2**（复用 Job1 的 Full_zerank 作 baseline，只追加 1 次评测）

### Task 2.1: 编写可重复的增量验收脚本

**Files:**
- Create: `scripts/verify_incremental_cloud.sh`
- Create: `docs/incremental-verification-runbook.md`
- Test (本地逻辑): 已有 `tests/test_p2_incremental.py`、`tests/test_lifecycle.py` — 先全绿

- [ ] **Step 1: 本地跑增量相关单测**

```bash
make test
# 或
pytest tests/test_p2_incremental.py tests/test_lifecycle.py tests/test_faiss_lifecycle.py -q
```

Expected: PASS。失败则先修测试/代码再上云。

- [ ] **Step 2: 新增 runbook `docs/incremental-verification-runbook.md`**

写清三幕验收：

| 幕 | 操作 | 通过标准 |
|----|------|----------|
| A 删除一致性 | 删除 1 个 doc_id 后对含其内容的 query 检索 | BM25/Dense/Visual 均不应再命中该 doc 页 |
| B NDCG 不漂移 | 对**同一索引状态** `run_eval --skip-index` 两次；或「增量更新后」vs「全量重建」同配置 Full_zerank2 | \|ΔNDCG@10\| &lt; 0.005（283q） |
| C page-diff 省时 | 改 1 份 PDF 的 ~10% 页内容后 re-ingest | 日志中重编码页数 ≈ 变更页；wall time 对比全量 re-encode 有明确比例 |

- [ ] **Step 3: 实现 `scripts/verify_incremental_cloud.sh` 骨架**

脚本职责（bash，可调 python 子命令）：

1. 打印 git SHA、索引路径。
2. 调用现有 delete/ingest API 或 python 片段完成 A（若无现成 CLI，用 `python -c` 调 `PrismRAGRetriever.delete_document`）。
3. 调用 `run_eval.py --skip-index --config-filter Full_zerank --max-queries`（云上全量去掉 max-queries）。
4. 把关键数字 append 到 `runs/$STAMP-incremental-verify/summary.md`。

最小可提交骨架示例：

```bash
#!/usr/bin/env bash
set -euo pipefail
STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-runs/${STAMP}-incremental-verify}
mkdir -p "$OUT"
echo "git=$(git rev-parse HEAD)" | tee "$OUT/env.txt"
# 全量 NDCG 基线
python scripts/run_eval.py --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 --config-filter Full_zerank \
  --output-dir "$OUT/ndcg-baseline" 2>&1 | tee "$OUT/ndcg-baseline.log"
echo "Fill A/B/C manually if interactive steps remain — see docs/incremental-verification-runbook.md" \
  | tee "$OUT/README.md"
```

- [ ] **Step 4: Commit 脚本与 runbook**

```bash
git add scripts/verify_incremental_cloud.sh docs/incremental-verification-runbook.md
git commit -m "chore: incremental verification runbook and cloud script"
```

---

### Task 2.2: 云上执行 A/B/C 并归档数字

- [ ] **Step 1: 云上跑 B（不漂移）** — 同一索引连续两次 Full_zerank2 283q，或增量后 vs 重建后。
- [ ] **Step 2: 云上跑 A（删文档）** — 记录删除前后 top-k 是否含 tombstone 页。
- [ ] **Step 3: 云上跑 C（page-diff）** — 记录 `pages_reencoded` / `pages_skipped` / 分钟数。
- [ ] **Step 4: 写 `runs/YYYYMMDD-incremental-verify/README.md`**，三条全部有数字。
- [ ] **Step 5: 更新 bullet④ 可用句式：**  
  `增量更新后 Full_zerank2 NDCG 漂移 < 0.005；变更 10% 页时重编码页占比 ≈ 10%，相对全量重建节省约 X% 墙钟时间。`

**Wave2 完成定义:** runbook + 至少一次云验收 README；bullet④ 从功能描述升级为带数字。

---

# Wave 3 — Visual 按需路由（A）
> 云上归属：**Boot-B Job1**；代码必须与 Wave4 尽量同分支合并后再开第二机

### Task 3.1: VisualRouter 单元测试与实现

**Files:**
- Create: `src/retrieval/visual_router.py`
- Create: `tests/test_visual_router.py`
- Modify: `src/evaluation/vidore_adapter.py`（`search_with_trace` 在 `use_visual=True` 时先问 router）
- Modify: `src/config.py` / `config/models.yaml`（`retrieval.visual_routing.enabled`, `mode: heuristic|always|never`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_visual_router.py
from src.retrieval.visual_router import VisualRouter

def test_heuristic_skips_plain_definitional_query():
    r = VisualRouter(mode="heuristic")
    assert r.should_use_visual("What is the definition of hydraulic pressure?") is False

def test_heuristic_enables_table_or_figure_query():
    r = VisualRouter(mode="heuristic")
    assert r.should_use_visual("According to the table, what is the max torque?") is True
    assert r.should_use_visual("In the diagram, which port is the inlet?") is True

def test_always_and_never_modes():
    assert VisualRouter(mode="always").should_use_visual("hello") is True
    assert VisualRouter(mode="never").should_use_visual("see the figure") is False
```

- [ ] **Step 2: 跑测试确认失败**

```bash
pytest tests/test_visual_router.py -v
```

Expected: FAIL import or missing class.

- [ ] **Step 3: 最小实现**

```python
# src/retrieval/visual_router.py
from __future__ import annotations
import re

_VISUAL_CUES = re.compile(
    r"\b(table|figure|fig\.|diagram|chart|graph|schematic|drawing|image|plot|"
    r"illustration|page\s+\d+|see\s+(the\s+)?(figure|table))\b",
    re.I,
)

class VisualRouter:
    def __init__(self, mode: str = "heuristic") -> None:
        if mode not in ("heuristic", "always", "never"):
            raise ValueError(mode)
        self.mode = mode

    def should_use_visual(self, query: str) -> bool:
        if self.mode == "always":
            return True
        if self.mode == "never":
            return False
        return bool(_VISUAL_CUES.search(query or ""))
```

- [ ] **Step 4: 接入 `PrismRAGRetriever.search_with_trace`**

逻辑：配置 `visual_routing.enabled` 且调用方 `use_visual=True` 时：

```python
effective_visual = use_visual
if self.visual_router and use_visual:
    effective_visual = self.visual_router.should_use_visual(query)
# 后续 visual 分支用 effective_visual
# trace 增加 "visual_routed": effective_visual
```

- [ ] **Step 5: 单测通过 + commit**

```bash
pytest tests/test_visual_router.py tests/test_retrieval_cache.py -q
git add src/retrieval/visual_router.py tests/test_visual_router.py \
  src/evaluation/vidore_adapter.py config/models.yaml src/config.py
git commit -m "feat(retrieval): heuristic visual route skip for non-visual queries"
```

---

### Task 3.2: 云上对比 always vs heuristic

- [ ] **Step 1:** 配置 `mode=always`（基线）与 `mode=heuristic` 各跑 283q Full_zerank2（或至少 100q 若时间紧，但 README 标明）。
- [ ] **Step 2:** 统计：NDCG@10、平均延迟、Visual 跳过率（`visual_routed=false` 占比）。
- [ ] **Step 3:** `runs/YYYYMMDD-visual-routing/README.md` 结论模板：

```text
heuristic vs always:
- ΔNDCG@10: ...
- latency_ms: ... → ... ( -X% )
- visual_skip_rate: Y%
Resume line: 对非图表 query 跳过 Visual，延迟 -X% 且 NDCG 变化 < Z
```

**Wave3 完成定义:** 单测绿；云上有 Pareto 数字；bullet① 可追加路由句。

---

# Wave 4 — LLM 句级上下文过滤（A）
> 云上归属：**Boot-B Job2（可选）**；禁止为 Wave4 单独加第 3 次开机——未就绪则 Boot-B 只跑路由，过滤并入下次或放弃

### Task 4.1: 过滤接口与单测（mock LLM）

**Files:**
- Create: `src/prompts/prompts/context_sentence_filter.yaml`
- Modify: `src/evaluation/ragas_metrics.py`（或新建 `src/generation/context_filter.py` 更清晰）
- Modify: `src/generation/generator.py`（在 `compress_context` 后或替代路径调用）
- Create: `tests/test_llm_context_filter.py`

**设计约束:**

- 过滤 LLM 与 CtxRel **评分** prompt/模型解耦（不同 yaml；配置项 `context_filter.model`）。
- 表格 chunk（`chunk_type=table` 或现有大表保护路径）**跳过**句级 LLM 过滤，与现网大表保护一致。
- 失败降级：LLM 超时/解析失败 → 回退 BGE `compress_context`。

- [ ] **Step 1: 写 mock 单测**

```python
# tests/test_llm_context_filter.py
from src.generation.context_filter import filter_sentences_llm

def test_filter_keeps_marked_sentences(monkeypatch):
    def fake_complete(prompt: str) -> str:
        return '{"keep": [0, 2]}'

    text = "Relevant torque is 50 Nm.\nNoise line about company history.\nSee table note on page 3."
    out = filter_sentences_llm(text, query="max torque?", complete_fn=fake_complete)
    assert "50 Nm" in out
    assert "company history" not in out

def test_filter_fallback_on_bad_json(monkeypatch):
    def fake_complete(prompt: str) -> str:
        return "not-json"

    text = "A\nB\nC"
    out = filter_sentences_llm(text, query="q", complete_fn=fake_complete, fallback=lambda t, q: t)
    assert out == text
```

- [ ] **Step 2: 实现 `src/generation/context_filter.py` + prompt yaml**
- [ ] **Step 3: Generator 配置开关 `context_filter.mode: off|bge|llm|bge_then_llm`**
- [ ] **Step 4: pytest 通过并 commit**

```bash
pytest tests/test_llm_context_filter.py -q
git commit -am "feat(generation): LLM sentence context filter with BGE fallback"
```

---

### Task 4.2: 云上对照实验（BGE vs LLM）

- [ ] **Step 1:** 固定检索配置 Full_zerank2；生成侧 `bge` vs `llm`（或 `bge_then_llm`）。
- [ ] **Step 2:** RAGAS 至少 100q（与 `20260708-ctxrel-fix` 同量级）；记录 Faith / Rel / CtxRel / 拒答数 / 平均 latency。
- [ ] **Step 3:** 若 Faith 掉 &gt; 0.02 且 E2E 不升，**默认保持 bge**，LLM 作可选；阴性结果写进 README（同样有面试价值）。
- [ ] **Step 4:** 更新 bullet③：只有主指标提升才写进简历；否则写「对比实验结论保留 BGE」。

**Wave4 完成定义:** 代码可开关；对照 README；简历只写正向或诚实阴性。

---

# Wave 5 — 评测扩容与硬化（A）
> 云上归属：**Boot-C（可永久跳过）**；NDCG 单测 / Makefile 仅本地

### Task 5.1: 本地 eval smoke 门禁

**Files:**
- Modify: `Makefile`
- Create: `tests/test_ndcg_metric.py`（纯函数，防回归）

- [ ] **Step 1: 固化 NDCG 单测**

```python
# tests/test_ndcg_metric.py
from src.evaluation.ablation import compute_ndcg

def test_ndcg_log2_first_rank_perfect():
    # 唯一相关在 rank0 → DCG=1, IDCG=1
    assert abs(compute_ndcg({"p1"}, ["p1", "p2"], k=10) - 1.0) < 1e-9

def test_ndcg_dedupes_repeated_pages():
    # 重复 page 不应增加位置
    s1 = compute_ndcg({"p1"}, ["p1", "p1", "p2"], k=10)
    s2 = compute_ndcg({"p1"}, ["p1", "p2"], k=10)
    assert abs(s1 - s2) < 1e-9
```

- [ ] **Step 2: Makefile 增加**

```makefile
eval-smoke:
        python scripts/run_eval.py --max-queries 10 --skip-index --config-filter Full_zerank --language en
```

（需索引时文档注明 skip；无索引则 CI 只跑 `test_ndcg_metric`。）

- [ ] **Step 3: Commit**

```bash
git add tests/test_ndcg_metric.py Makefile
git commit -m "test: lock NDCG log2+dedupe; add eval-smoke target"
```

---

### Task 5.2: 云上 RAGAS 283 + E2E 扩集

- [ ] **Step 1:** `python scripts/run_ragas_metrics.py --skip-index`（或项目等价参数）全量 283，输出 `runs/YYYYMMDD-ragas-full-283/`。
- [ ] **Step 2:** 扩 E2E：用 `scripts/generate_e2e_qa.py` 或手工扩到 **≥100 可答 + ≥40 拒答**，跑 `scripts/run_e2e_qa.py`。
- [ ] **Step 3:** Bad case 分布表（检索缺失 / 数值错 / judge 误杀）写入 run README。
- [ ] **Step 4:** 更新 bullet② 数字；handoff 三层评测表。

**Wave5 完成定义:** 全量 Faith/Rel/CtxRel；扩集拒答/正确率；metric 单测防再踩坑。

---

# 收尾 Task — 简历与 handoff 同步

### Task F: 汇总 4 条 bullet 终态

**Files:**
- Modify: `handoff.md`
- Modify: `local/resume-prismrag.md`（不入库）

- [ ] **Step 1:** 按 Wave1–5 实际数字重写 4 条 bullet（只写已验证结论）。
- [ ] **Step 2:** handoff 增加「Bullet 强化进度」表：各 Wave 状态 / run 路径 / 主数字。
- [ ] **Step 3:** 确认无「旧公式 0.57」与「黄金表」混用。

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 黄金表不再支持「0.44→0.57」 | 改写结论，不伪造；相对 Δ 仍可能成立 |
| 云上 GPU 贵 | Wave1/2 合并同一开机窗口；HyDE 可裁 |
| LLM 过滤伤 Faith | 默认回退 BGE；阴性结果可写实验 |
| 路由误杀 Visual 有用 query | 先启发式 + 跳过率监控；可加 allowlist 词 |
| 主分支误改 | 全程 feat 分支；PR 合并 |

---

## Self-Review（对照原优先级）

| 原建议 | 计划位置 |
|--------|----------|
| 干净对照 / 黄金消融 / eval protocol | Wave1 Task1.1–1.2 |
| 增量 NDCG 不漂移 + 省 GPU | Wave2 |
| Visual 按需路由 | Wave3 |
| LLM 句级预过滤 | Wave4 |
| RAGAS 283 + E2E 扩集 | Wave5 |
| 2×2 表格摘要归因 | Wave1 锁定 `table_summary`；若需显式 2×2，在 Wave1 后附录跑二次 ingest（可选，不阻塞） |
| Self-RAG / 多租户 | 明确不做 |
| 缓存命中率数字 | Wave2 summary 可选附加：对固定 query 集打 L3/L4 命中（若时间够，挂在 2.2 Step4） |

**Placeholder 扫描:** 无 TBD；云上路径以 `runs/YYYYMMDD-*` 为准由执行日填充。

---

## 执行方式建议

1. **默认 Standard：2 次 GPU 开机（Boot-A → 本地开发 → Boot-B）**，见文首 Cloud Boot Packing。  
2. 预算紧只开 **Boot-A（Minimal）** 即可交 ①④。  
3. Boot-C / RAGAS283 / HyDE **默认不做**。  
4. 实现顺序：本地 protocol + `cloud_boot_*.sh` → Boot-A → router(+filter) 代码 → Boot-B → 更新 `local/resume-prismrag.md`。
