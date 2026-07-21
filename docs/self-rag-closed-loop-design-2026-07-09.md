# Self-RAG 闭环设计文档（v2 · MVP）

> **日期**：2026-07-09 初稿 · **2026-07-20 修订为 v2**  
> **状态**：Draft v2 — **MVP 仅 Gate2（答案忠实性门）**；Gate1 为 Phase 2  
> **实现状态**：**MVP 代码已实现**（`feat/self-rag-gate2`）：`src/generation/self_rag.py` + prompts + `/ask` 接入；默认 `enabled: false`  
> **分支**：`feat/self-rag-gate2`

### 关联文档

| 文档 | 关系 |
|------|------|
| `docs/incremental-update-optimization-spec-2026-07-16.md` | 检索侧一致性已落地（幽灵召回 / 墓碑 / page-diff）；**不再是 Self-RAG 前置阻塞** |
| `docs/cache-retrieval-spec-2026-07-18.md` | L3/L4 缓存 + `index_version` 失效；闭环答案缓存边界见 §6 |
| `docs/eval-protocol.md` | 评测口径；Self-RAG 对照实验须遵守 |
| `docs/superpowers/plans/2026-07-20-bullet-strengthening-roadmap.md` | Bullet 强化主路径（Boot-A/B）；Self-RAG 为 Boot 之后的 **可选 A 级** 能力 |
| `local/resume-prismrag.md`（不入库） | 简历叙事：补「生成侧闭环」维度 |

### v1 → v2 变更摘要

| 项 | v1 | v2 |
|----|----|----|
| 默认实现范围 | 两道门骨架并排 | **仅 Gate2**；Gate1 明确 Phase 2 |
| 门1 语义 | 直接复用 `compute_context_relevancy` | **CtxRel ≠ sufficiency**；Phase 2 另设充分性判据 |
| 门1 反馈 | 含默认 HyDE | **去掉默认 HyDE**（本场景已阴性） |
| 与现网模块 | 未写 | 补 `context_filter` / L4 cache / Trace / PromptRegistry |
| 成本 | 「+2~3 次 LLM」偏乐观 | claim 级 vs 整答级、judge 独立模型、超时降级 |
| API / 行号 | `generator.generate`、过时行号 | 对齐 `Generator.answer`、`POST /ask` |
| 前置依赖 | 等增量 P0/P1 | 增量已合 main；优先级改为 Boot-A/B 之后可选 |

---

## 1. 背景与目标

### 1.1 现状（开环）

线上生成路径（`src/generation/generator.py` · `Generator.answer`）：

```text
retrieved top-k
  → 表格 chunk 跳过句级压缩（结构保护）
  → 非表格：context_filter（默认 bge 句级 cosine）
  → PromptRegistry「answer_generation」→ LLM（temperature=0）
  → 返回 answer + citations（引用由检索 chunk 构造，非模型自报）
```

`POST /ask`（`src/api/routes.py`）再包一层 L4 Answer 缓存与 `retrieval_trace`。

**没有**：生成后对「答案是否被 context 支撑」的独立实测，也没有失败时的重试 / 强制拒答。

### 1.2 问题

| 问题 | 表现 |
|------|------|
| Context 不充分仍生成 | 模型硬答或严重遗漏 |
| 幻觉 | 仅靠 prompt「仅依据 context」无法杜绝 |
| 开环 | 质量完全取决于单次生成；无法自纠 |

### 1.3 目标

在「检索 → 生成」外叠加 **可开关的自检闭环**：

- **MVP（本版）**：生成后 **Gate2 — 答案忠实性**；不通过则重生 1 次或 abstain  
- **Phase 2**：生成前 **Gate1 — 证据充分性**（非 CtxRel 噪声比）；不通过则扩 k / 轻量改写后再检索  

把「靠 prompt 祈祷」升级为「生成后实测，不过就重做或拒答」。

**非目标（本设计不做）：**

- 论文原版 Self-RAG 微调与反思 token  
- 多租户 / ACL  
- 默认打开的高延迟全链路 claim 级评测上线  
- 用 Self-RAG 替代 Boot-A/B 检索数字工作  

---

## 2. 两种 Self-RAG 形态（命名澄清）

### 2.1 论文原版（Asai et al., 2023）

微调模型边生成边吐 `<RET>` / `IsREL` / `IsSUP` / `IsUSE`。检索与生成交错，由模型驱动。

→ **本项目不做。**

### 2.2 工程落地版（本项目）

不微调；把「反思」拆成 **独立 LLM-as-Judge**，挂在主流程外当门卫。

| 门 | 职责 | 状态 |
|----|------|------|
| **Gate2** | 答案是否被 **喂给生成器的 context** 支撑 | **MVP 必做** |
| **Gate1** | 检索 context 是否 **足以回答问题** | Phase 2 |

**结论**：Self-RAG@PrismRAG = 主生成器 + 门卫判据 + 有界反馈弧 + Trace，**无需改模型权重**。

---

## 3. MVP 闭环结构（仅 Gate2）

```text
User query
   │
   ▼
Retrieve + RRF + Rerank（现有 PrismRAGRetriever）
   │
   ▼
Generator.answer（压缩 + 生成，现有）
   │
   ▼
[Gate 2: Grounded in ctx?]
   │ yes ──► 返回 answer（可写 L4 cache）
   │ no
   ▼
  action = regenerate | abstain
   │ regenerate（最多 1 次，合计 max 2 次生成）
   │   ──► 更强约束 prompt 重生 ──► 再过 Gate2
   │       仍失败 ──► abstain
   └── abstain ──► 固定拒答文案 + 空/保留 citations 策略见 §5
```

护栏：

- `generation.self_rag.max_generate_attempts = 2`（含首次）  
- Gate2 超时 / 解析失败 → **降级策略**见 §5.4（默认：放行并打 `gate_degraded=true`，或按配置强制 abstain）  
- **不在 MVP 做重检索**（避免与 Visual 路由、L3 缓存、延迟预算纠缠）

### 3.1 Phase 2 目标形态（仅文档预留，不实现）

```text
Retrieve
   ▼
[Gate 1: Evidence sufficient?]── no ──► k+=Δ / 轻量 reformulate ──► 再 Retrieve
   │ yes                                 （禁止默认 HyDE）
   ▼
Generate → Gate2 → ...
```

---

## 4. Gate2 — 答案忠实性（MVP 详细设计）

### 4.1 输入 / 输出

| | 定义 |
|--|------|
| **输入** | `query`, `answer`, `context`（必须与生成时 **同一份** 最终 prompt context，含压缩后文本） |
| **输出** | `passed: bool`, `score: float`, `detail: dict`, `latency_ms` |

`context` 口径与 CtxRel 修复一致：**评入模内容**，不是原始未压缩 chunk。

### 4.2 判据策略（两档）

#### 档 A — 整答一次判定（**MVP 默认**）

单次 judge 调用：

- 问题：答案中的事实主张是否均被 context 支持？  
- 输出：JSON，例如 `{"grounded": true|false, "score": 0.0-1.0, "unsupported": ["..."]}`  
- `passed = score >= faith_threshold`（默认 **0.8**）

**优点**：延迟可控（+1 次 LLM）。  
**缺点**：粒度粗于 claim 级。

#### 档 B — Claim 级（可选升级，对齐 RAGAS）

复用评测思路（`src/evaluation/ragas_metrics.py`）：

1. `decompose_claims(answer)`  
2. 每 claim `verify_claim(claim, context)`，`≥0.5` 算支撑  
3. `score = supported / total`  

**硬约束（上线必带）：**

- `max_claims`（默认 8）：超出则合并或只验前 N  
- 并行上限 / 总超时  
- **禁止**在热路径无 cap 地直接调用完整 `compute_faithfulness`  

实现时：抽 **纯函数判据层**（可被 RAGAS 与 Self-RAG 共用），而不是 online 直接 import 整份评测脚本副作用。

### 4.3 反馈弧

| 条件 | 动作 |
|------|------|
| `passed` | 返回原答案 |
| 未通过且 `attempt < max` | **regenerate**：同一 context，换更强 system/user 约束（独立 prompt id，见 §8） |
| 仍未通过或配置 `on_fail=abstain` | **abstain**：固定文案（与现网拒答风格一致，英文默认） |

注意：Gate2 **不回检索**——假设病在生成，不在证据集合（证据问题留给 Phase 2 Gate1）。

### 4.4 与 RAGAS Faithfulness 的关系

| | RAGAS 评测 | Online Gate2 |
|--|------------|--------------|
| 目的 | 离线指标 | 在线拦截 |
| 默认算法 | claim 级 | **整答级**（可升级 claim） |
| 模型 | 评测 LLM | **`self_rag.judge_model` 可独立** |
| 失败 | 记分 | regenerate / abstain |

对照实验应报告：**开/关 Self-RAG 的 Faith / Correctness / 拒答率 / 延迟**，而不是假设 online score ≡ RAGAS score。

---

## 5. 配置、API、降级

### 5.1 建议配置（`config/models.yaml`）

```yaml
generation:
  self_rag:
    enabled: false                 # 默认关；消融与线上显式打开
    mode: gate2_only               # MVP 固定；预留 gate1_and_gate2
    judge_model: null              # null = 与主生成 model 相同；生产建议更小/更快
    judge_base_url: null           # 可选独立 endpoint
    faith_threshold: 0.8
    max_generate_attempts: 2       # 含首次生成
    on_fail: abstain               # abstain | regenerate_then_abstain
    gate_timeout_ms: 8000
    on_judge_error: degrade_pass   # degrade_pass | abstain
    verdict_mode: whole_answer     # whole_answer | claim_level
    max_claims: 8                  # 仅 claim_level
```

### 5.2 API 集成

- 入口：`POST /ask` 在 `gen.answer(...)` 外包一层 `SelfRAGOrchestrator.answer(...)`（或 `Generator` 内可选闭环，**推荐独立 orchestrator 保持 Generator 单测简单**）。  
- 开关：`generation.self_rag.enabled`；请求级覆盖可选（调试用，非必须）。  
- 响应：可在 `AskResponse` 增加可选字段（向后兼容）：

```json
{
  "self_rag": {
    "enabled": true,
    "passed": false,
    "score": 0.5,
    "attempts": 2,
    "final_action": "abstain",
    "gate_degraded": false
  }
}
```

### 5.3 拒答文案

与现网一致，例如：

`I don't have enough information to answer that question.`

abstain 时：

- `citations`：默认 **清空**（避免「拒答却带出处」误导）；Trace 里仍保留检索与 context 便于排障。

### 5.4 Judge 失败 / 超时

| `on_judge_error` | 行为 |
|------------------|------|
| `degrade_pass`（默认） | 返回最近一次生成答案，`gate_degraded=true`，打告警 |
| `abstain` | 拒答，偏保守 |

---

## 6. 与现有模块的边界（必读）

| 模块 | 职责 | 与 Self-RAG 边界 |
|------|------|------------------|
| **`context_filter`**（bge / llm / bge_then_llm） | **生成前**去噪，减少入模无关句 | **不做**充分性判定；Gate2 的 context 必须是 filter **之后**的文本 |
| **拒答 prompt** | 模型自觉说「无法回答」 | Gate2 是 **强制** abstain；两者可叠加 |
| **L3 检索缓存** | 缓存融合/精排结果 | Gate2 不改变检索 key；重试生成仍可用同批 `results` |
| **L4 Answer 缓存** | `temperature==0` 时可缓存整答 | key 必须含 **`self_rag.enabled` + mode + threshold + verdict_mode`**（或 `self_rag` 开时 **禁用 L4**，二选一；**推荐 key 加盐**，避免开/关串答案） |
| **Trace / `GET /trace/{id}`** | 二分检索 vs 生成 | 必填 span：`self_rag.gate2`（score、passed、attempts、final_action、**attempts_detail**、degraded）；子 span `self_rag.gate2.attempt.{n}` 每轮一条（answer 截断 500 字、score、unsupported、action） |
| **PromptRegistry** | YAML 版本化 prompt | Gate2 judge + regenerate 约束 **新建** prompt id，禁止硬编码散落 |
| **Visual 路由** | 是否跑 Visual 路 | MVP 不重检索，无交互；Phase 2 重检索须 **复用同一路由决策** 或显式记录 |
| **HyDE** | 查询改写实验 | **MVP/Phase2 默认不用**（消融阴性） |

---

## 7. Gate1 — 证据充分性（Phase 2，仅设计）

### 7.1 为什么不直接用 CtxRel

| | Context Relevance（现网） | 充分性（Gate1 需要） |
|--|--------------------------|----------------------|
| 定义 | 相关句 / 总句（噪声比） | **证据是否覆盖问题** |
| 压缩后 | 常系统性升高（CtxRel+154% 故事） | 压缩可能 **删掉** 关键句 → 充分性下降 |
| 用作重检索门 | **易误判** | 应使用 sufficiency 问法 |

v1 文档「`relevance_score < 0.3~0.5` 则改写/HyDE/+k」**废止**作为默认设计。

### 7.2 Phase 2 建议判据

单次 judge：

> 仅根据 context，是否有足够信息完整回答该问题？  
> `{"sufficient": bool, "score": 0-1, "missing": "..."}`

### 7.3 反馈弧（Phase 2）

按优先级：

1. `k_cur += Δ`（最便宜）  
2. 可选：轻量 `reformulate(query)`（**独立小 prompt**，非 HyDE）  
3. 仍不足 → 进入生成前直接 abstain（可配置）  

**禁止默认开启 HyDE**；若实验，须单独开关且写入 run README。

### 7.4 触发时机

在 **context_filter 之后、首次生成之前**，对 **即将入模** 的 context 判定。

---

## 8. 落地骨架（实现参考）

### 8.1 文件

| 路径 | 职责 |
|------|------|
| `src/generation/self_rag.py` | `SelfRAGOrchestrator` + Gate2 判据 |
| `src/prompts/prompts/self_rag_gate2_verdict.yaml` | 整答 grounded 判定 |
| `src/prompts/prompts/self_rag_regenerate.yaml` | 失败后重生约束（可选与 answer_generation 分版本） |
| `tests/test_self_rag_gate2.py` | mock judge：pass / fail→abstain / fail→regen→pass / timeout 降级 |
| `config/models.yaml` + `src/config.py` | `generation.self_rag.*` |
| `src/api/routes.py` | `/ask` 接入 orchestrator |

### 8.2 伪代码（MVP）

```python
class SelfRAGOrchestrator:
    def __init__(self, generator: Generator, cfg_self_rag: dict):
        self.generator = generator
        self.cfg = cfg_self_rag

    def answer(self, query: str, retrieved: list, k_context: int = 5) -> dict:
        if not self.cfg.get("enabled"):
            return self.generator.answer(query, retrieved, k_context=k_context)

        max_attempts = int(self.cfg.get("max_generate_attempts", 2))
        last = None
        for attempt in range(1, max_attempts + 1):
            # attempt==1: 默认 prompt；attempt>1: regenerate 强约束
            last = self.generator.answer(
                query, retrieved, k_context=k_context,
                # 实现时可用 prompt_id 或 extra_system 注入
            )
            verdict = self._gate2(query, last["answer"], last["context"])
            if verdict.get("degraded"):
                last["self_rag"] = {**verdict, "attempts": attempt, "final_action": "degrade_pass"}
                return last
            if verdict["passed"]:
                last["self_rag"] = {**verdict, "attempts": attempt, "final_action": "return"}
                return last
        # abstain
        return {
            "answer": "I don't have enough information to answer that question.",
            "citations": [],
            "context": last["context"] if last else "",
            "self_rag": {
                "passed": False,
                "score": verdict.get("score"),
                "attempts": max_attempts,
                "final_action": "abstain",
            },
        }

    def _gate2(self, query: str, answer: str, context: str) -> dict:
        # whole_answer: 1× judge JSON；claim_level: cap max_claims
        # 超时 → on_judge_error
        ...
```

依赖：

- `Generator.answer` — 已有  
- PromptRegistry — 已有  
- 可选复用 `decompose_claims` / `verify_claim` 逻辑 — 仅 `verdict_mode=claim_level`  
- **不需要** HyDE、不需要改 FAISS/BM25  

---

## 9. 与「好 Prompt」的本质区别

| | 好 Prompt | Self-RAG Gate2 |
|--|-----------|----------------|
| 时机 | 生成前静态约束 | 生成后独立审计 |
| 失败 | 无人检查 | 可重生 / 强制拒答 |
| 可观测 | 难 | score / attempts / action 进 Trace |

二者叠加，不互斥。

---

## 10. 代价与风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| 延迟 / 费用 | 整答 +1 judge；claim 级 ×N | 默认 whole_answer；threshold + max_attempts；默认 `enabled: false` |
| Judge 误杀 | 正确回答被 abstain | 阈值可调；E2E 看 Correctness/拒答；degrade 策略 |
| Judge 漏放 | 幻觉仍放出 | 阈值与 prompt 迭代；claim 级升级路径 |
| 与 L4 串缓存 | 开/关 Self-RAG 返回不同答案 | key 加盐或关 L4 |
| 与 RAGAS 指标纠缠 | online 分数当 Faith | 报告分离；对照实验用同一 RAGAS 管道 |
| 只补生成期 | 不修检索漏检 | Boot-A/B 仍是主路径；Gate1 后续 |

---

## 11. 验证计划

### 11.1 本地（必做，0 GPU 大模型下载可 mock）

```bash
pytest tests/test_self_rag_gate2.py -q
```

覆盖：pass、fail→abstain、fail→regen→pass、judge 超时 degrade、disabled 透传、L4 key 含 self_rag 盐（若实现缓存）。

### 11.2 云上 / 有 LLM 环境（小样本即可）

| 实验 | 配置 | 指标 |
|------|------|------|
| 基线 | `self_rag.enabled=false` | Faith / Rel / CtxRel（RAGAS **100q**） |
| 处理 | `enabled=true`, whole_answer, threshold=0.8 | 同上 + 平均 latency + abstain 率 |
| E2E | 现有 50 可答 + 20 拒答 | Correctness / Rejection Accuracy |

**通过标准（建议，可按结果调整）：**

- Faithfulness **不降** 或 ↑ ≥ 0.02；或  
- 拒答准确率不降且 可答 Correctness 不降 ≥ 0.03；  
- p95 延迟增幅在预算内（文档化实际数字）  

**阴性同样有价值**：Faith↑ 但延迟翻倍、Correctness 掉 → 默认保持关闭，高风险场景开关打开；写 run README，简历只写已验证结论。

产物：`runs/YYYYMMDD-self-rag-gate2/README.md`。

---

## 12. 优先级与排期建议

| 阶段 | 内容 | 何时做 |
|------|------|--------|
| **现在** | 本设计 v2 定稿 | ✅ |
| **Boot-A / Boot-B** | 检索黄金表 + 路由/过滤数字 | **优先于** Self-RAG 实现（简历 ①④③ 主路径） |
| **MVP 实现** | Gate2 only + Trace + 配置 + 单测 | Boot 之后 1～2 天级本地实现 + 小样本评测 |
| **Phase 2** | Gate1 sufficiency + 有界重检索 | 有明确检索漏检 badcase 且预算允许时 |

**相对 bullet：**

- 强化 **③ / 生成质量** 或口述「开环→闭环」  
- **不替代** ① 精排瓶颈、② 修尺子、④ 一致性数字  

---

## 13. 待确认项（实现前勾选）

- [ ] 延迟预算：整答 Gate2 +1 次 judge 是否可接受；p95 上限多少  
- [ ] `on_fail` 默认 `abstain` 还是 `regenerate_then_abstain`  
- [ ] `on_judge_error`：`degrade_pass` vs `abstain`  
- [ ] L4：key 加盐 vs Self-RAG 开启时禁用  
- [ ] Judge 是否独立小模型（推荐）  
- [ ] 阈值是否先在 20～50 条样本上扫分布再定 0.8  
- [ ] 响应体是否对前端暴露 `self_rag` 字段  

---

## 14. 明确不做（复述）

- 论文微调 Self-RAG  
- MVP 内 Gate1 + HyDE 重检索  
- 热路径无 cap 的全量 claim 级 `compute_faithfulness`  
- 用 Self-RAG 数字冒充实测检索 NDCG  
- 为 Self-RAG 单独再开一轮与 Boot-A 无关的全量 283 检索消融  

---

## 附录 A — v1 废止条款

以下 v1 表述在 v2 **不再作为默认设计**：

1. 用 `compute_context_relevancy` 的 score 直接充当「Context enough」门限  
2. Gate1 失败默认开启 HyDE  
3. 优先级「必须等增量删除做完」——增量已完成  
4. 骨架默认两门循环、API 名 `generator.generate`  
5. 成本估算「每轮仅 +2~3 次调用」在 claim 级无 cap 时不成立  

---

## 附录 B — 面试一句话

> 我们没做论文里的微调 Self-RAG，而是工程化闭环：检索生成后用独立 LLM 做忠实性门，不过就重生或强制拒答；和「只靠 system prompt」比，这是可测、可 Trace 的动态校验。默认整答判定控制延迟，和 RAGAS 声明级评测解耦。
