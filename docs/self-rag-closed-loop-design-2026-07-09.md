# Self-RAG 闭环设计文档

> 日期：2026-07-09
> 状态：设计阶段（暂未实现）
> 关联文档：`docs/incremental-update-delete-design-2026-07-09.md`（更新/删除设计，第 3 层的前置基础）

## 1. 背景与目标

当前运行时生成是**单趟直出**（开环）：`src/generation/generator.py` 取 top-k context → 压缩 → 一句"仅依据 context 作答"的 prompt → `temperature=0` 出答案。没有任何自检或重试。

问题：
- 检索捞到的 context 可能**不充分**，但生成器照答，结果答非所问或严重遗漏；
- 生成器可能**编造**（hallucination）——尤其当它"觉得"答案合理时，单靠 prompt 约束无法杜绝；
- 没有反馈机制，生成质量完全依赖一次生成的好坏。

**目标**：在"检索 → 生成"的开环上叠加两道**独立的 LLM 事实核查门**，构成可自我纠错的**闭环（closed loop）**，把"靠 prompt 祈祷模型别胡说"升级为"生成后实测、不过就重做"。

## 2. 两种 Self-RAG 形态

### 2.1 论文原版（Asai et al., 2023）
微调一个模型，使其边生成边吐"反思 token"：
- `<RET>`：是否检索（按段落动态决定，而非一上来就检索）
- `IsREL`：检索段落是否相关
- `IsSUP`：某声明是否被检索内容支撑
- `IsUSE`：最终回答是否有用

检索与生成交错，由模型自身驱动。这是"训练出来的自我反思"。

### 2.2 工程落地版（本项目的选择）
不微调，将"反思"拆为**独立的 LLM-as-Judge 判据**，挂在主流程外当门卫。本项目 `src/evaluation/ragas_metrics.py` 已含两个现成判据，可直接复用：
- `compute_context_relevancy`（575 行）→ 充当 Context 充分性门
- `compute_faithfulness`（266 行）→ 充当答案忠实性门

**结论**：Self-RAG 在本项目 = 主生成器 + 两个现成 LLM 判据 + 反馈弧，**无需改动模型**。

## 3. 闭环结构（两道门）

```
User query
   │
   ▼
Retrieve + RRF + Rerank ──(BM25 + Dense + Visual, 已有)──┐
   │                                                      │
   ▼                                                      │
[Gate 1: Context enough?]── no ──► Reformulate / HyDE / +k ┘
   │ yes
   ▼
Generate answer ──(LLM draft, 已有)──┐
   │                                  │
   ▼                                  │
[Gate 2: Grounded in ctx?]── no ──► Regenerate / Abstain ┘
   │ yes
   ▼
Return answer
```
护栏：每道门 `max 2 iterations`，避免无限重检索/重写。

## 4. 门 1 — Context 充分性（复用 `compute_context_relevancy`）

### 算法（代码 575–620+）
1. `split_context_to_sentences(context_chunks)` 把召回 context 按句切；
2. 每批 ≤20 句（`_CONTEXT_RELEVANCE_BATCH_SIZE=20`），逐批问 LLM"这句跟问题相关吗"；
3. `relevance_score = 相关句数 / 总句数`。

### 闭环节点
若 `relevance_score < 阈值`（建议 0.3~0.5），**不进入生成**，触发反馈弧：
- 改写 query（更具体）；
- 调大 `k`（多捞候选）；
- 开启 HyDE（用 LLM 编假答案当额外 query，见 `src/retrieval/hyde.py`）。

即"系统根据对自身输出的不满，自适应调整检索行为"。

## 5. 门 2 — 答案忠实性（复用 `compute_faithfulness`）

### 算法（代码 266–301）
1. `decompose_claims(answer)` 把答案拆为**原子声明**；
2. 每声明 `verify_claim(claim, context)` 问 LLM"是否被 context 支持"，返回 0~1，`≥0.5` 算支撑；
3. `faithfulness_score = 被支撑声明数 / 总声明数`。

### 为什么先拆声明再逐条验证
整体问"答案对不对"太粗糙——9 句对、1 句编，整体判断易被带过。拆原子声明后，编造句被单独抓出。

### 闭环节点
若 `faithfulness_score < 阈值`（建议 0.7~0.8），触发反馈弧：
- 带更强"仅依据 context，禁止外推"约束**重写答案**；
- 或**降级为"信息不足，无法回答"**（abstain）。

注意：门 2 不回检索，回"生成"——因为 context 本身够，病在生成飘了。

## 6. 与"好 Prompt"的本质区别

- **好 Prompt = 静态约束**：生成前给指令，模型遵不遵守无人检查。
- **Self-RAG = 动态校验**：生成后独立拆答案、逐条拿 context 对质（事后审计），不依赖生成模型的自觉性。

## 7. 代价与风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| 成本线性翻倍 | 每多一轮循环 +1~3 次 LLM 调用 | `max 2 iterations` 护栏 |
| 判据本身会犯错 | `verify_claim` 是 LLM 评委，可能误判 | 拆声明 + 高阈值 |
| 阈值需调参 | 太高→延迟爆炸，太低→形同虚设 | 门1≈0.3~0.5，门2≈0.7~0.8，按集上调 |
| 只补生成期 | 不解决检索层短板（如字符级分块丢上下文） | 配合 `incremental-update-delete-design` 第 1 层改动 |

## 8. 落地骨架（实现时参考）

`src/generation/self_rag_orchestrator.py`：

```python
class SelfRAGOrchestrator:
    def __init__(self, retriever, generator, max_retries: int = 2):
        self.retriever = retriever
        self.generator = generator
        self.max_retries = max_retries
        self.ctx_threshold = 0.4
        self.faith_threshold = 0.8

    def answer(self, query: str, k: int = 8):
        k_cur, q = k, query
        for _ in range(self.max_retries):
            fused = self.retriever.search(q, k=k_cur)
            ctx_texts = [c["text"] for c in fused]
            rel = compute_context_relevancy(q, ctx_texts).relevance_score
            if rel < self.ctx_threshold:          # 门 1 不达标
                k_cur += 4
                q = reformulate(q)                 # + 可选 HyDE
                continue
            answer = self.generator.generate(q, fused)
            faith = compute_faithfulness(answer, "\n".join(ctx_texts)).faithfulness_score
            if faith >= self.faith_threshold:      # 门 2 达标
                return answer
            # 否则循环：带更强约束重写（generator 内调整 prompt）
        return "信息不足，无法基于现有文档可靠作答"
```

依赖（均已存在，无需新写算法）：
- `retriever.search` / `generator.generate` — 已有
- `compute_context_relevancy` / `compute_faithfulness` — `src/evaluation/ragas_metrics.py`
- `HyDEGenerator` — `src/retrieval/hyde.py`（门 1 可选增强）
- `reformulate` — 需补充一个轻量 query 改写函数（或用 HyDE 答案替代）

## 9. 集成方式

- 在 `src/api/routes.py` 的 `search/generate` 端点，将直接调用 `generator.generate` 改为调用 `SelfRAGOrchestrator.answer`；
- 判据调用走 LLM（与评测同套 `call_llm`），注意线上延迟预算；
- 建议以开关 `use_self_rag` 控制，先与 `use_rerank` / `use_hyde` 同列，便于消融对比。

## 10. 落地优先级与建议

- **优先级：P2**（晚于"增量 BM25 + FAISS 逻辑删除"P0/P1 与"标题层级分块"）。理由：生成校验层零件已现成，收益高但非阻塞，且依赖检索层先稳定。
- **先做**：门 2（faithfulness）单独上线——它直接止血"幻觉进答案"，且不需要改检索；
- **后做**：门 1（context relevancy）触发重检索，需先确认检索层有自适应改写/HyDE 能力。
- **验证**：用 `runs/` 下 RAGAS 报告对比 `with_self_rag` vs `no_self_rag` 的 faithfulness / answer relevancy 提升。

## 11. 待确认项

- [ ] 线上 LLM 调用延迟预算（每轮 +2~3 次调用是否可接受）
- [ ] `reformulate` 函数实现方式（轻量 LLM 改写 vs 复用 HyDE 答案）
- [ ] 阈值初值是否在样本集上先跑分布再定
- [ ] 是否需要把"门结果"回填进 observability（已存在 `src/observability/`），便于线上监控幻觉率
