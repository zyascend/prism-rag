# 端到端 QA 首次上云跑 — 教训记录

> 日期: 2026-07-05 | 场景: 端到端 QA 第三层（Layer 3）首次上云评测
> 参与: 50 条可回答 + 20 条拒答，Full 检索（BM25+Dense+ColQwen2 Visual+RRF+BGE Reranker）

---

## 问题清单

### 1. 代码未上传（低级）

| 环节 | 现象 |
|------|------|
| SSH 后 `python scripts/run_e2e_qa.py` | `No such file or directory` |
| **根因** | 云上部署的是旧代码，`run_e2e_qa.py`, `e2e_qa.py`, `e2e_qa.json` 都不存在 |

**教训：** 不要默认云端和本地同步。每次上云前显式 tar + scp，并在 SSH 后 `ls scripts/` 验证新文件。

---

### 2. structlog 依赖缺失

| 环节 | 现象 |
|------|------|
| `from src.observability import get_tracer` | `ModuleNotFoundError: No module named 'structlog'` |
| **根因** | 可观测性模块新增了 `structlog` / `rich` 依赖，云端 conda 环境没有 |

**教训：** 依赖变更应同步更新 `requirements-cloud.txt` 或走 `pip install`。`cloud_setup.sh` 的 conda 复用逻辑只补缺失包，不会重装已有包——补装本身没问题，但没人记得手动跑。

---

### 3. ColPali 基座模型缓存不完整（核心障碍）

| 环节 | 现象 |
|------|------|
| `ColPali.from_pretrained("vidore/colpali-v1.3")` | XetHub 401 (Unauthorized) |
| **根因** | `colpaligemma-3b-pt-448-base` 的权重 shard 文件（`model-00001-of-00002.safetensors` 等）从未完整下载——缓存里只有索引文件 + 两个 0 字节 `.incomplete` |

**教训：**
- `cloud_setup.sh` 的模型预下载（Phase 1）之前通过代理下载基座模型时就已经被 XetHub 401 拦了，但 `snapshot_download` 没有报错退出（`.incomplete` 静默创建）
- **Phase 1 结束时应该验证模型文件完整性**：检查 `.incomplete` 文件数量，或者 `from_pretrained(local_files_only=True)` 试加载
- 后续改用 `HF_ENDPOINT=https://hf-mirror.com` 下载，绕过了 XetHub 问题

---

### 4. 网络代理的双面性

| 场景 | 现象 |
|------|------|
| 走代理 (`source /etc/network_turbo`) | 访问 huggingface.co 正常，但 XetHub 文件下载返回 401 |
| 不走代理 | huggingface.co 直连 `Network is unreachable` |
| `HF_HUB_OFFLINE=1` + 不走代理 | 本地缓存模型可正常加载 |

**教训：**
- 对于已缓存的模型，设 `HF_HUB_OFFLINE=1` + `unset http_proxy` 是最稳的
- 对于需要下载的模型（特别是 XetHub 存储的大模型），用 `HF_ENDPOINT=https://hf-mirror.com`
- `source /etc/network_turbo` 只在数据集下载（走 `datasets` 库）时需要
- 模型下载和模型加载是两套策略，不要混用

---

### 5. macOS tar 覆盖云端 symlink（经典复现）

| 环节 | 现象 |
|------|------|
| 第二次跑时 FAISS 索引找不到 | `WARNING: FAISS 索引不存在，跳过 visual 检索` |
| **根因** | 云端 `indexes/` `results/` `logs/` 是指向 `/root/autodl-tmp/` 的 symlink。macOS tar 包里的同名目录（普通目录）覆盖了 symlink，导致 2GB ColQwen2 FAISS 索引不可见 |

**教训：**
- handoff.md 已经记录过这个坑（"macOS tar 的 HF symlink 到 Linux 断链"），但针对的是 HF cache，没覆盖自定义 symlink
- **修复方法**：`rm -rf indexes results logs && ln -sf /root/autodl-tmp/indexes indexes && ...`
- 预防：tar 时 `--exclude=indexes --exclude=results --exclude=logs`，或改用单文件 scp 而非全量 tar

---

### 6. Judge 解析 bug（代码缺陷）

| 环节 | 现象 |
|------|------|
| `compute_answer_correctness()` 返回 | 50 条全 `is_correct=False`，正确率 0% |
| **根因** | Judge prompt 末尾以 `JUDGMENT:` 结尾，Ollama 正常填写 `YES` 但**不重复前缀**。代码只解析以 `JUDGMENT:` 开头的行，导致 `judgment_text` 为空→判错 |

```python
# 修复前
if line_stripped.upper().startswith("JUDGMENT:"):
    judgment_line = line_stripped           # ← 找不到任何行

# 修复后
if not judgment_line:
    for line in lines:
        ls = line.strip().upper()
        if ls in ("YES", "NO") or ls.startswith("YES") or ls.startswith("NO"):
            judgment_line = f"JUDGMENT: {ls}"
            break
```

**教训：** LLM 输出格式解析不能假设模型会严格重复模板。回退逻辑必须兜底。

---

### 7. 可观测性数据纯内存

| 环节 | 现象 |
|------|------|
| 评测跑完想查延迟 | 只有 `avg_latency_seconds: 2.19`，没有拆分逐 span 的延迟/命中数据 |
| **根因** | `get_collector()` 是内存单例，`ingest_trace()` 数据从未落盘。`reporter.generate_report()` 已实现但无人调用 |

**教训：** 设计文档写了 tracer → collector → reporter 的闭环，但实现只到 collector。实测时才发现 reporter 那端是断的。已在 `dump_collector()` 公共函数补上。

---

## 改进项汇总

| 优先级 | 项目 | 方案 |
|--------|------|------|
| P0 | Cloud 模型验证 | Phase 1 结束后检查 `.incomplete` 文件 + `local_files_only=True` 试加载 |
| P0 | symlink 保护 | tar 打包时排除 `indexes/` `results/` `logs/` 或改用 scp 单文件 |
| P1 | 网络策略文档化 | 明确三类场景（直连/hf-mirror/代理）的适用范围和 env 变量设置 |
| P1 | Judge 解析健壮性 | 已修复：回退到无前缀的 YES/NO 识别 |
| P2 | 全链路可观测性落盘 | 已修复：`dump_collector()` 统一入口，三个 CLI 脚本已集成 |