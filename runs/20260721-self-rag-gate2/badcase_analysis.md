# Self-RAG Gate2 A/B Bad Case 分析

> 数据：`runs/20260721-self-rag-gate2/{off,on}/`  
> 日期：2026-07-21

---

## 0. 总览结论（先看这个）

| 现象 | 根因（本分析） | 是否该怪 Gate2 算法本身 |
|------|----------------|-------------------------|
| RAGAS Faith **0.830→0.786** | **13 条 Gate2 拒答被记 Faith=0**，且 **未进 rejected_count**（文案与 RAGAS 拒答短语不对齐） | **主要是评测口径 bug** |
| 排除拒答后 Faith | off **0.827** → on **0.903**（**+7.6pt**） | Gate2 对「仍放行答案」有正向过滤 |
| E2E Correctness 同为 0.60 | **同一 20 题双错、同一 30 题双对**，**0 题被 Gate2 翻盘** | Gate2 几乎没改「对/错」集合 |
| E2E Reject 0.25→0.95 | OFF 软拒答（I don't know）**匹配不到** is_rejected；ON 硬拒答命中 | 拒答检测 + Gate2 强制文案 |
| 延迟 ×1.8 | +judge（及少量 regen） | 预期代价 |

**一句话：**  
Gate2 更像「**多拒答 + 放行子集更干净**」，不是「同一答案变更忠实」。汇总 Faith 被拒答句 **口径污染**；E2E 主错在 **检索/错 chunk/答不全**，Gate2 治不好。

---

## 1. E2E 可答题（50）— 几乎零翻盘

| 集合 | 题数 | 含义 |
|------|-----:|------|
| 两臂都对 | 30 | 检索+生成足够，Gate2 不动 |
| 两臂都错 | 20 | **共享 hard cases**，Gate2 未修好 |
| 仅 off 错 / 仅 on 错 | **0 / 0** | 无 correctness 级翻盘 |

### 1.1 共享 20 错 — 类型分布（按 judge + 答案形态）

| 类型 | 约 | 典型题 | 与 Gate2 关系 |
|------|---:|--------|----------------|
| **错 chunk / 答偏主题** | ~5 | Fuel Servicing「禁物」→ 答成 ignition sources；TC 被听成 TCTO/torque | 证据集合错，**Gate2 无法补检索** |
| **答不全 / 漏关键点** | ~5 | oxygen cart 只答接地；revision 指示只答 pointing hands | 部分 grounded 也会过门 |
| **事实冲突 / 实体错** | ~4–6 | Type D vs Type K TC；Type I vs Type II 55 PSI；合金成分 Ni vs Mo | 上下文里可能有相近干扰句，**整答 judge 难拦** |
| **Gate2 硬拒答（on）** | 3 | TUS instrumentation；VCI 6 inches；DAFMAN 91-223 标题 | off 也是错/软拒，correctness 仍错 |

### 1.2 On 侧 3 条可答硬拒（误拒 / 证据不足拒）

| 问题摘要 | 预期 | On 行为 | 解读 |
|----------|------|---------|------|
| TUS interval 仪器类型 + 管压 | Type B/C/D + 管压计算 | 硬拒答 | 复合问；context 可能只盖一半 → Gate2 偏保守 **合理倾向**，但 E2E 记错 |
| VCI 是否 6 inches 无效 | 实际规范是 **12 inches** | 硬拒答 | 问的是「6 inches」真假；证据是 12" 规则 → 模型/门卫不敢下结论 |
| DAFMAN 91-223 标题 | Water and Fuel Systems | 硬拒答 | 标题类短事实，**召回失败** 时拒答；off 也是「context 没有」 |

→ 这 3 条 **不是「本来对、Gate2 弄错」**，而是「本来也对不上 expected」；Gate2 只是改成标准拒答句。

### 1.3 Off 软拒 vs On 硬拒（可答集）

- Off：约 2 条「context 没有 / I don't know」类  
- On：3 条标准 `I don't have enough information...`  
- 对 **correctness** 无帮助（expected 仍是实体答案）

---

## 2. E2E 拒答集（20）— Gate2 主收益区

| arm | rejection_accuracy | 失败形态 |
|-----|-------------------:|----------|
| off | **0.25 (5/20)** | 大量 **软拒**（"I don't know" / "context does not contain"）**未命中** `is_answer_rejected`；还有写诗等 **硬答** |
| on | **0.95 (19/20)** | 几乎全部硬拒；**唯一失败**：仍写「conveyor belts 诗」 |

### 2.1 Off 假阴性（该拒未识别）样例

| 问题 | 实际输出 | 检测 |
|------|----------|------|
| capital of France | "I don't know..." | is_rejected=**False** |
| current US president | "I don't know." | False |
| 2022 World Cup | context 无关 + don't know | False |
| poem about conveyor belts | **真写诗** | False（真失败） |

→ **0.25 严重低估**「系统是否拒绝瞎答」；一半是 **评测短语表过窄**。

### 2.2 On 仍漏的 1 条

- 「写传送带诗」→ 仍创作 → Gate2 **整答 grounded 判定对「创意写作」无效**（无事实声明可验，或 judge 放行）。

---

## 3. RAGAS Faith — 口径污染 vs 真实效果

### 3.1 原始均值（被污染）

| arm | Faith 原始 | Faith=0 桶 | RAGAS rejected_count |
|-----|----------:|-----------:|---------------------:|
| off | 0.830 | 9 | 2 |
| on | 0.786 | **15** | **0** ← 异常 |

### 3.2 关键 Bug：Gate2 拒答文案 ∉ RAGAS 拒答短语

| 系统 | 拒答句 |
|------|--------|
| Gate2 `ABSTAIN_ANSWER` | `I don't have enough information to answer that question.` |
| RAGAS 检测短语 | `cannot answer` / `not enough information` / `based on the available` / … |

`"don't have enough information"` **不含** `"not enough information"` 子串 →  
**13 条 on 拒答**：`rejected_count` 不计，却进 Faith，且 **score=0.0**。

粗算：13 × 0 拖均值 ≈ 拉低约 0.11 量级，足以解释 0.830→0.786。

### 3.3 排除「enough information / don't know」类后的 Faith

| arm | 排除条数 | 剩余 n | **调整后 mean Faith** |
|-----|--------:|-------:|----------------------:|
| off | 2 | 98 | **0.827** |
| on | 13 | 87 | **0.903** |

→ **放行答案上，Gate2 与更严生成约束使 Faith ↑**；汇总表显示 ↓ 是 **指标定义问题**。

### 3.4 同 query Faith Δ（on−off）

| | 题数 |
|--|-----:|
| 大跌 >0.2 | 9（多变为拒答 → 0） |
| 大涨 >0.2 | 4（如 Type I 喷嘴压力 off 含糊 → on 答对且 1.0） |
| 均值 Δ | −0.027（含拒答 0 分） |

---

## 4. 根因分层（指导改什么）

```text
用户可见错误
├─ L1 检索/错页/错 chunk     ← E2E 共享 20 错主力（TC/TCTO、禁物、印章标题…）
├─ L2 生成偏题/不全/实体混淆 ← 同 chunk 下仍错；整答 Gate2 不敏感
├─ L3 Gate2 硬拒答           ← 覆盖不足或证据模糊；E2E 记错，产品或可接受
└─ L4 评测口径               ← 拒答短语不一致；Faith 含拒答 0 分
```

| 优先级 | 动作 | 状态 |
|--------|------|------|
| **P0** | 统一拒答文案 + 短语表 + **拒答不进 Faith/Rel 均值** | ✅ `src/rejection.py`；RAGAS/E2E/Gate2 共用 |
| **P1** | Gate2 `trigger=low_rerank`（max score &lt; 阈值才过门） | ✅ 默认 `low_rerank` / `0.35`；A/B 脚本可用 `always` |
| **P2** | 检索侧：错实体/错章节（TC vs torque、Type I/II） | ⏳ 未做（下一迭代） |
| **P2** | claim 级 Gate2 或 citation 硬约束 | ⏳ 未做 |

---

## 5. 对简历 / 产品的建议

| 场景 | 建议 |
|------|------|
| 简历数字 | **不要**写 Faith 0.83→0.79；若写 Gate2，写「拒答准确率 0.25→0.95」需注明 off 含检测假阴性，或写「强制 grounded 拒答后拒答检测 0.95」 |
| 口述 | 「对照发现汇总 Faith 被拒答句拖累；排除拒答后放行答案 Faith 约 0.90；E2E 主错在检索，Gate2 翻不了 correctness」 |
| 默认配置 | 维持 `enabled=false` 或 **仅 rejection/低置信路径 enable** |

---

## 6. 附录：共享 E2E 错题指纹（便于回归）

两臂都错、且形态稳定的代表：

1. aircraft fuel servicing safety components — 错章节清单  
2. hot brakes 手势对、下一步错（关右发 vs procedure c）  
3. gaseous oxygen cart — 只答接地  
4. hydrant specialized defueling — 答成车型/飞机而非 Type II/III hydrant  
5. FSSZ prohibited — ignition vs explosives/cargo  
6. TC type D vs type K  
7. TC purpose — 听成 TCTO/torque  
8. Briner 2905 干湿膜比  
9. live-fire physical strain — 环境因素 vs 运动损伤  
10. aircraft tire markings — 答成 caution 标签  
11. LOX/LIN PPE — 氮侧过度具体  
12. Type I nozzle 55 PSI — off 含糊 / on 曾 Type II 混淆（RAGAS 上 on 有改善）  
13. torque wrench Step A — 调最低读数 vs 选套筒  
14. 8630 合金最高成分 — Mo vs Ni  
15. sealant gun kit — MIL 化合物 vs PN 尺寸  
16. ambulance vs recruiting 涂装/标志  

（完整文本见 `off|on/e2e/badcase_e2e_qa_analysis.md`）
