# RAGAS 评测 Bad Case 分析

> 日期: 2026-07-05
> 环境: AutoDL RTX 4090 | 全量检索 (BM25+Dense+Visual+Rerank) | 50 queries
> 评测指标: Faithfulness=0.8867, Relevancy=0.8147

---

## 🟡 类型A — 合理拒答（4 条）

模型正确拒绝回答 Faithfulness=0 是误伤 —— 拒答句子被声明分解后，
LLM judge 判为不被上下文支持。

| # | Query | 拒答原因 |
|:-:|-------|---------|
| 1 | How do visual indicators in technical diagrams communicate revision updates | 文档无此内容 |
| 2 | 15 minutes fire department notification fueling operations passenger count... | 特定规程未提及 |
| 3 | What two standards regulate insulating matting requirements for high-voltage... | 标准编号未出现在文档 |
| 4 | aircraft fire training facility emergency shutoff vs fuel servicing grounding... | 对比性需求无法满足 |

**改进方向:** `evaluate_generation()` 应对拒答直接跳过 `compute_faithfulness()`，不计入平均分。

---

## 🟠 类型B — 真正 Hallucination（4 条）

### B1. 氮气罐颜色代码编造 ★★★★★

```
Query:  identify the color code used for nitrogen tank identifiers...
Answer: "SAE-AMS-STD-595, gray color number 16187"

  ✅ Nitrogen tank identifiers use a specific color code.     ← 空泛成立
  ❌ The standardization follows SAE-AMS-STD-595.             ← 编造
  ❌ The color assigned is gray.                              ← 编造
  ❌ The exact shade is number 16187.                         ← 编造
```

- **根因:** 检索未召回含氮气罐颜色标识的文档
- **解读:** LLM 用训练知识填补空白，生成看似专业的假内容
- **修复:** 改善检索召回（BM25/Dense/Visual 权重或 rerank 策略）

### B2. Marine Corps 安全协议

```
Query:  What specific safety protocols are outlined for Marine Corps personnel
Answer: "context does not contain enough information..." (部分拒答)
  ❌ 声明1-2: context 缺乏信息
  ❌ 声明3: 一般性指引存在（模糊正确）
  ❌ 声明4: 无详细 Marine Corps 安全协议
```

- **根因:** 文档不含 Marine Corps 专有内容
- **解读:** LLM 在拒答和生成之间摇摆，产生矛盾声明
- **修复:** `evaluate_generation()` 拒答检测阈值过松，部分生成漏网

### B3. AFTF 紧急关机响应时间

```
Query:  AFTF emergency shutdown response time requirement
Answer: "no specific time requirement"
  ❌ 声明1: 无时间要求（LLM judge 判为不支持）
  ✅ 声明2: 关机流程被提及
  ❌ 声明3: 时间范围缺失
```

- **根因:** 同类型A，要求精确数值的 query 文档未覆盖
- **修复:** 在 Faithfulness 计算中区分"信息缺失"和"信息错误"

### B4. 蓄水池容量计算

```
Query:  Calculate the total water capacity of the water conservation pond
Answer: 拼凑了 propane tank 容量 (10,000 gal) + 燃烧区水深 (1 inch)
  - 模型从无关数据推断水池容量
```

- **根因:** 聚合计算类 query，文档不会直接提供
- **解读:** 当前 pipeline 不适合需要多步推理或计算的 query
- **修复:** 可增加拒答规则覆盖"计算/估算"类请求

---

## 🔵 类型C — Relevancy 标尺偏差（5 条）

Relevancy 偏低但 Faithfulness 良好，说明 cosine 相似度标尺不准。

| Query | R | F | 问题 |
|-------|:-:|:-:|------|
| What are the specific applications of each plating method... | 0.66 | **1.00** | 生成问题偏"什么用途"，原 query 问"每种方法的具体应用"——语义等价，词面不同 |
| Which precipitation-hardening stainless steel alloy utilizes molybdenum... | 0.67 | **1.00** | 生成问题聚焦该钢材特性 → 与原 query 词覆盖少 |
| Explain how the concentric ring design in AFTF burn areas contributes... | 0.69 | 0.86 | 生成问题更泛化，偏离"concentric ring"焦点 |

**根本问题:** `compute_answer_relevancy()` 使用 BGE/nomic-embed-text 的
cosine 相似度，对词面不同但语义等价的 pair 区分度不足。

**修复方向:** 用 LLM 直接评分替代 cosine similarity，或使用更强的 embedding
模型（如 `gte-Qwen2` 等）。

---

## 📊 总结

| 类别 | 数量 | 对总分影响 | 优先级 |
|:----|:----:|:----------:|:------:|
| 合理拒答/误伤 | 4 | -0.02 Faithfulness | 低 |
| 真正 Hallucination | 4 | -0.03 Faithfulness | **高** |
| Relevancy 标尺偏差 | 5 | -0.02 Relevancy | 中 |

- 50 条中仅 **1 条严重幻觉**（氮气罐标准号编造），幻觉率 2%
- 拒答 5 条全部合理，但被误计入 Faithfulness 压低平均分
- 下阶段建议：修拒答计数 + 查氮气罐的检索路径为何召回失败