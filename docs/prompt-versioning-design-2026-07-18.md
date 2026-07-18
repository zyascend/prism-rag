# Prompt 版本管理设计方案（方案 A：集中管理 + 版本记录）

> 状态：设计稿（待评审，未实现）
> 范围：仅"集中管理 + 版本记录"，不做 A/B、不做在线热切换/回退
> 日期：2026-07-18

## 1. 背景与现状

经源码核查，当前项目内 prompt 以**硬编码常量**方式散落于多处，无任何版本概念：

| 位置 | prompt 角色 | 现状 |
|---|---|---|
| `src/generation/generator.py:76-81` | 生成主 prompt（system + user） | `answer()` 内硬编码 |
| `src/evaluation/ragas_metrics.py` | 6 个评测指标 prompt（`CLAIM_DECOMPOSITION_PROMPT:157`、`CLAIM_VERIFICATION_PROMPT:166`、`REVERSE_QUESTION_PROMPT:177`、`GENERATION_PROMPT:186`、`_RELEVANCY_FALLBACK_PROMPT:387`、`CONTEXT_RELEVANCE_PROMPT:486`） | 模块级常量，`.format()` 注入变量 |
| `src/retrieval/hyde.py` | HyDE 视觉改写 prompt | 代码内嵌 |
| `src/ingestion/table_summarizer.py` | 表格摘要 prompt | 代码内嵌 |

**问题**：改 prompt = 改代码 = 重新部署；无法追溯"哪版长什么样、谁改的、为何改"；无 code review 锚点；回滚只能靠 git revert 整个文件。

## 2. 目标边界（方案 A）

**做（In scope）**
- 外置：散落 prompt 迁移到独立文件，代码从文件加载。
- 集中：统一目录、统一加载入口（`PromptRegistry`）。
- 版本记录：每条 prompt 带 `version` / `created_at` / `author` / `changelog`。
- 可追溯：版本历史随文件进 git，可 diff、可 review、可 blame。

**不做（Out of scope，明确边界）**
- 不做运行时多版本共存 / A/B 分流。
- 不做在线热切换（改生效版本仍需发版）。
- 不做回退 API（回退 = git revert + 重新部署）。
- 不接外部 Prompt 平台（LangSmith / PromptLayer）。

> 设计意图：先把"版本可追溯"这块最刚需、成本最低的能力做扎实，为将来升级到方案 B（运行时多版本）保留平滑演进接口，但不提前承担其复杂度。

## 3. 总体设计

```
src/prompts/
├── __init__.py          # 导出 get_active / list_prompts
├── models.py            # PromptVersion / Prompt 数据类
├── loader.py            # 从 YAML 加载、校验、解析
├── registry.py          # 全局注册表，get_active(id)
└── prompts/             # 版本化 prompt 资源文件（*.yaml）
    ├── answer_generation.yaml
    ├── claim_decomposition.yaml
    ├── claim_verification.yaml
    ├── reverse_question.yaml
    ├── ragas_generation.yaml
    ├── relevancy_fallback.yaml
    ├── context_relevance.yaml
    ├── hyde.yaml
    └── table_summary.yaml
```

加载时序（进程启动时一次）：
1. `registry.init()` 扫描 `src/prompts/prompts/*.yaml`。
2. 每个文件解析为一个 `Prompt`（含 `id` + 多版本列表）。
3. 校验：每个 prompt 必须有且仅有一个 `active: true` 版本；`version` 在文件内唯一。
4. 建立 `dict[prompt_id] -> PromptVersion(active)` 索引，供调用点 `get_active(id)` 取用。

## 4. Prompt YAML Schema

```yaml
id: answer_generation
description: 生成最终答案的 system + user prompt
versions:
  - version: 1
    created_at: "2026-07-18"
    author: yang
    changelog: "初始版本，从 generator.py:76-81 迁移，文本零改动"
    active: true
    system: |
      You are a precise assistant for industrial PDF question answering.
      Only answer using the provided context. If the context is insufficient,
      say you don't have enough information.
    user: |
      Context:
      {context}

      Question: {question}

      Answer concisely and cite the page where each fact comes from.
```

字段说明：

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 全局唯一，调用点据此取用 |
| `description` | 否 | 人类可读说明 |
| `versions[].version` | 是 | 文件内正整数，唯一 |
| `versions[].created_at` | 是 | ISO 日期，便于追溯 |
| `versions[].author` | 否 | 变更人（git blame 补充） |
| `versions[].changelog` | 是 | 本次变更摘要 |
| `versions[].active` | 是 | 布尔，仅一个为 true |
| `versions[].system` / `user` / `template` | 视 prompt 而定 | 保留原有 `{var}` 占位符，`.format()` 照常可用 |

## 5. 核心组件

**`models.py`**（约 40 LOC）
- `PromptVersion`：dataclass，`version/created_at/author/changelog/active` + 模板字段。
- `Prompt`：dataclass，`id/description/versions: list[PromptVersion]`，提供 `active_version` 属性。

**`loader.py`**（约 60 LOC）
- `load_prompt_file(path) -> Prompt`：用标准库 `yaml.safe_load` 解析。
- 校验：缺 `id`、无 `active`、多个 `active`、`version` 重复 → 抛 `PromptConfigError`。
- 零外部依赖（项目已依赖 PyYAML）。

**`registry.py`**（约 80 LOC）
- `init(prompts_dir: str | None = None)`：扫描目录，构建索引。
- `get_active(prompt_id: str) -> PromptVersion`：取生效版本；未找到抛 `PromptNotFound`。
- `list_prompts() -> dict`：返回所有 prompt 当前 active 摘要（供可选只读端点）。
- 模块级单例，首次调用惰性初始化。

**调用点改造**（约 8 文件，每处 5–10 行）
- `generator.py:76-81`：`system, user = get_active("answer_generation").system, .user`
- `ragas_metrics.py`：6 个常量改为 `get_active("<id>").template`（保留 `.format()` 注入）
- `hyde.py` / `table_summarizer.py`：同上
- 删除原硬编码常量，import 改为从 `src.prompts` 引入。

## 6. 与现有体系融合

- **配置风格**：沿用项目既有 YAML + 精确钉死（类比 `models.yaml`）的工程习惯，不引入新范式。
- **可寻址目录**：`prompts_dir` 默认 `src/prompts/prompts/`，可通过 `cfg`（`config.py`）新增 `prompts.dir` 覆盖，支持多环境（如 `local-dev` 覆盖评测 prompt）。
- **可观测性（可选复用）**：若将来要在 trace 里标注 `prompt_version`，`registry` 已天然提供该字段，接入 `tracer` span 成本为 0（本方案不强制，列入演进）。

## 7. 迁移策略（零行为变更）

1. 提取每个硬编码 prompt 的**原文**，作为 `version: 1` / `active: true` 写入对应 YAML，**不修改任何 prompt 文本**。
2. 调用点改为从 registry 读取，逻辑等价替换。
3. 用既有 `make test` + 评测（`--max-queries 10` 轻量验证）确认答案/指标与原硬编码完全一致——保证本次重构无语义回归。

## 8. 测试策略

- `tests/test_prompt_registry.py`：
  - 加载正常文件 → `get_active` 返回正确版本。
  - 无 `active` / 多 `active` / `version` 重复 → 抛 `PromptConfigError`。
  - `get_active("unknown")` → 抛 `PromptNotFound`。
- 迁移回归：断言各调用点取到的模板字符串与重构前硬编码值一致（快照比对）。
- 端到端：跑一次 `--max-queries 10` 评测，分数与基线无差异。

## 9. 明确不做的边界（防止范围蔓延）

- 无运行时版本路由、无实验切片、无热更新 watch。
- 无管理 API（写入/激活端点不做；仅可选只读 `GET /prompts` 便于排查，见 §10）。
- 无外部平台依赖、无额外服务进程。

## 10. 演进路径（A → B 平滑升级）

方案 A 的数据模型（`id` + 多 `version` + `active` 标记）已天然具备升级到方案 B 的底座：

- **在线切换**：将 `active` 标记从"文件内静态"升级为"可持久化的活跃版本状态"（小表 / JSON / config），`activate(id, version)` 改之即生效。
- **A/B 分流**：在 `get_active` 之上加一层路由（实验 key / 流量比例），返回非活跃版本。
- 上述升级**无需重构 YAML schema 与 registry 加载逻辑**，仅扩展"选择"层。

## 11. 工作量估算

| 项 | 量级 |
|---|---|
| `prompts/` 包（models/loader/registry） | ~180 LOC 新增 |
| 9 个版本化 YAML 文件 | ~300 行（含 v1 原文） |
| 调用点改造（8 文件） | ~60 LOC 改动 |
| 测试 | ~150 LOC 新增 |
| **合计** | **~350–500 LOC（含测试），零新依赖** |

折算：agent 实现约 1 个中等 PR；人力约 0.5–1 个工作日。显著低于方案 B（600–900 LOC + 路由/API/归因）。

## 12. 待确认的设计决策（已按推荐默认设定，可调整）

| 决策点 | 推荐默认 | 备选 |
|---|---|---|
| 生效版本判定（原 Q2） | **文件内 `active: true` 标记**（单文件自包含，diff 一目了然） | 全局 config 指向；最高版本号即生效 |
| 存储格式 | **YAML**（多行 prompt 可读、可注释） | JSON |
| 是否加只读 `GET /prompts` 端点 | 推荐加（轻量，便于线上排查当前生效版本） | 不加 |

---

## 附：面试话术（版本管理短板 → 加分）

> "prompt 版本管理是我们之前明确的缺口：生成和评测的 prompt 都是代码内嵌常量。我已经设计了一套轻量方案落地——把散落各处的 prompt 外置成带 `version`/`created_at`/`changelog` 的 YAML，用 `PromptRegistry` 统一加载，`get_active(id)` 取生效版，历史版本随文件进 git 可追溯、可 code review。
>
> 我特意把范围收敛在'集中管理+版本记录'，先不做 A/B 和在线回退——因为那块收益要靠实验体系支撑，当前最痛的是'改 prompt 必发版且不可追溯'。数据模型我留了升级口：将来要做运行时多版本，registry 的 `id+多version+active` 结构不用重构，只加一层选择路由。配置和 trace 体系也都能直接复用。"
