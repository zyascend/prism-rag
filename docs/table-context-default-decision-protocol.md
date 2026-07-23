# 决策协议：是否默认打开 `table_summary_context`

> **问题：** `ingestion.table_summary_context_enabled` 默认该 `true` 还是 `false`？  
> **已有信号：** Text re-ingest + context ON 后，Full_zerank2 **100q** NDCG@10 **0.3575 → 0.3589（+0.14pt）**，噪声带内弱阳性。  
> **本协议：** 用 **更大检索样本 + E2E** 做可辩护 Go/No-Go。  
> 日期：2026-07-23 · 分支 `feat/content-pipeline-phase-ab`

---

## 1. 决策对象（只定这一件事）

| 配置键 | 含义 |
|--------|------|
| `ingestion.table_summary_context_enabled` | 表摘要 LLM 是否注入同页邻段文本 |

**不在本协议内：** expand / modality_boost / CRAG / Gate2 / 是否关表摘要本身（`table_summary_enabled` 保持 true，与历史一致）。

两臂定义：

| 臂 | 摘要 | 上下文 | 备注 |
|----|------|--------|------|
| **OFF** | on | **off** | 旧默认；Boot-CP 前索引形态 |
| **ON** | on | **on** | 当前云上 full re-ingest 形态 |

两臂均：`--skip-faiss` 文本重灌；FAISS ColQwen2 **不动**；检索 `Full_zerank2` · expand/boost **关**。

---

## 2. 成功标准（先写死，再跑数）

### 2.1 主门禁（必须）

| # | 指标 | Go（可默认 true） | No-Go（保持 false） |
|---|------|-------------------|---------------------|
| M1 | Full_zerank2 **283q** en NDCG@10 | ON − OFF ≥ **0**（不降）且 |Δ| 写入 README | ON − OFF &lt; **−0.005**（掉超 0.5pt） |
| M2 | E2E **Correct**（50 可答） | ON − OFF ≥ **0** | ON − OFF ≤ **−0.04**（约 −2 题） |
| M3 | E2E **Reject accuracy**（20 应拒） | ON ≥ OFF − 0.05 | ON 明显更爱胡答（Reject 掉 &gt;0.05） |

### 2.2 加分项（非必须，但影响「默认开」叙事）

| # | 指标 | 说明 |
|---|------|------|
| S1 | 表子集（`data/table_subset_queries.json` 30 条可答）E2E 子集 Correct | ON 升更有说服力 |
| S2 | 100q 切片与已有 0.3589 一致（漂移验收） | 同 ON 索引复跑 |ΔNDCG@10| &lt; 0.005 |
| S3 | 延迟 | 检索侧应 ≈；生成侧与摘要无关。ingest 成本已付，**不进在线默认否决** |

### 2.3 一票决策规则

```text
IF M1 不降 AND M2 不降 AND M3 不降:
  IF (M1 升 ≥ 0.005) OR (M2 升 ≥ 0.04) OR (S1 升 ≥ 0.05):
    → 默认 true（可写「有可辩护增益」）
  ELSE:
    → 默认仍 false，但文档写「可生产开；默认关因增益未过阈值」
ELIF M1 或 M2 硬掉:
  → 默认 false，context 仅作实验开关
ELSE:
  → 默认 false，记入 handoff 待更大 E2E/badcase
```

**禁止：** 只用 100q +0.14pt 改默认；禁止 CtxRel 单独定案。

---

## 3. 样本与协议冻结

| 层 | 样本 | 命令骨架 | 主指标 |
|----|------|----------|--------|
| L1 检索 | **283q** en | `run_eval.py --skip-index --language en --expected-query-count 283 --visual-model colqwen2 --config-filter Full_zerank2 --no-hyde` | NDCG@10 |
| L3 E2E | **50+20** 全量 | `run_e2e_qa.py --skip-index --visual-model colqwen2` | Correct / Reject |
| 切片 | 100q | 同上 + `--max-queries 100` | 与历史对照 |
| 表敏感 | 30 条 | 从 `table_subset_queries.json` 滤 e2e 可答（脚本或手工子集） | Correct@subset |

口径：`docs/eval-protocol.md` v1；拒答不进 Faith 均值（E2E 本身有 Reject 指标）。

索引盐：每个臂的 run README 写明：

- `table_summary_enabled` / `table_summary_context_enabled`
- `chunks` 行数、`table_summary` 非空数、git/tarball 标识  
- FAISS 路径未改

---

## 4. 云上执行顺序（**1 次 GPU 开机**）

当前云库若已是 **ON（context 重灌后）**，顺序如下，避免无谓来回灌：

```text
Boot-Decide-TableContext（单开机）
│
├─ 确认：pg chunks=8835 · FAISS 在 · ollama/qwen2（E2E 生成若走 ollama）
│
├─ Job1  【当前索引 = ON】
│     ├─ Full_zerank2 283q → runs/$D-table-ctx-decide/on/ragas-or-ndcg/
│     ├─ Full_zerank2 100q → on/ndcg100/   （漂移 vs 0.3589）
│     └─ E2E 全量         → on/e2e/
│
├─ Job2  Text re-ingest OFF
│     MODE 等价：
│       python scripts/ingest_vidore.py --skip-faiss --replace-text
│       # 不要 --table-context；保持 table_summary 开（默认）
│     → runs/$D-table-ctx-decide/off/ingest/
│
├─ Job3  【索引 = OFF】
│     ├─ Full_zerank2 283q → off/ndcg283/
│     ├─ Full_zerank2 100q → off/ndcg100/
│     └─ E2E 全量         → off/e2e/
│
└─ Job4  写 comparison.json + README 决策段 → 关机
```

**估时（4090 级）：**

| Job | 粗估 |
|-----|------|
| 283q ×2 | ~2×（加载+~5–8 min 检索）≈ 0.5–1 h |
| E2E ×2 | 视生成；约 2×10–25 min（若 ollama 7B） |
| OFF re-ingest | 表摘要无 context 仍要 LLM；可能 **1–3 h**（可复用摘要缓存思路有限，因 prompt 不同） |
| **合计** | **约 3–6 h 墙钟**（表摘要仍是大头） |

省钱变体（不推荐作最终默认依据）：

- 只跑 **283 ON vs 已有 100q ON** + **E2E ON only**，OFF 用历史 Boot-CP 前索引数字（0.3575@100）——**不严谨**（100≠283，且 E2E 无 OFF）。

---

## 5. 命令清单（复制用）

### 5.1 环境

```bash
cd /root/prism-rag
source scripts/cloud_env.sh
export PATH=/root/miniconda3/bin:$PATH
pg_isready
pgrep -x ollama || nohup ollama serve &
D=$(date +%Y%m%d)
ROOT_OUT=runs/${D}-table-ctx-decide
mkdir -p "$ROOT_OUT"/{on,off}
```

### 5.2 ON 臂评测（当前库已是 context ON 时）

```bash
# L1 283
python scripts/run_eval.py --skip-index --language en --expected-query-count 283 \
  --visual-model colqwen2 --config-filter Full_zerank2 --no-hyde \
  --output-dir "$ROOT_OUT/on/ndcg283"

# L1 100 漂移
python scripts/run_eval.py --skip-index --language en --max-queries 100 \
  --visual-model colqwen2 --config-filter Full_zerank2 --no-hyde \
  --output-dir "$ROOT_OUT/on/ndcg100"

# E2E 全量
python scripts/run_e2e_qa.py --skip-index --visual-model colqwen2 \
  --output-dir "$ROOT_OUT/on/e2e"
```

### 5.3 切到 OFF 索引

```bash
# 不要 --table-context；会 TRUNCATE 后重灌
python scripts/ingest_vidore.py --skip-faiss --replace-text \
  2>&1 | tee "$ROOT_OUT/off/ingest.log"
```

### 5.4 OFF 臂评测（同 5.2，目录换 off/）

### 5.5 汇总

```bash
python - <<'PY'
# 手工或 scripts/compare_table_ctx_decide.py（见下）读 json 打表
print("fill comparison in README")
PY
```

自动化入口（推荐）：

```bash
bash scripts/cloud_decide_table_context.sh
# Env: SKIP_OFF_INGEST=1 若已在 OFF；ONLY=on|off|all
```

---

## 6. 表子集 E2E（S1）

`data/table_subset_queries.json` 的 id 来自 `e2e_qa.json`。

做法二选一：

1. **滤文件：** 用 id 列表生成 `data/e2e_qa_table_subset.json`，`--qa-file` 指向它。  
2. **全量 E2E 后切片：** 在 `e2e_qa_results.json` 里按 id 聚合 Correct。

优先 2（少一次生成），实现成本低。

---

## 7. 结果落档模板

`runs/YYYYMMDD-table-ctx-decide/README.md` 必含：

1. 两臂 ingest 配置与 chunk 统计  
2. NDCG@10 283 表 + Δ  
3. E2E Correct / Reject 表 + Δ  
4. 100q 与 0.3589 漂移  
5. **Decision: default true | false** + 套用的规则编号（M1/M2/…）  
6. 是否改 `config/models.yaml` 的 commit 链接  

---

## 8. 改默认时的代码动作（仅 Go 且「可写增益」时）

```yaml
# config/models.yaml
ingestion:
  table_summary_context_enabled: true   # 原 false
```

- 更新 `docs/architecture/content-pipeline.md` 默认表  
- handoff 写「默认开 + 283/E2E 数字指针」  
- **不**默认开 expand/boost  

若仅「不降但未过增益阈值」：yaml **保持 false**，README 写「生产可开」。

---

## 9. 风险

| 风险 | 缓解 |
|------|------|
| OFF re-ingest 覆盖 ON 库 | 先跑完 ON 全部评测再 TRUNCATE |
| E2E 与检索不同步（eval_via_generator） | 两臂同一 `models.yaml` 生成配置；只改索引 |
| 283 与 100 结论冲突 | **以 283 + E2E 为准**；100 仅漂移 |
| 表摘要 OFF 也要 LLM 时间 | 可接受；或接受与「摘要关」不是同一臂（本协议 **不比** 无摘要） |

---

## 10. 与已有数字的关系

| Run | 用途 |
|-----|------|
| `20260723-content-pipeline` Arm-A 0.3575@100 | OFF 近似（旧摘要无 context）**仅 100q** |
| `20260723-post-text-reingest` 0.3589@100 | ON @100 |
| **本协议 283+E2E** | **唯一改默认的依据** |
