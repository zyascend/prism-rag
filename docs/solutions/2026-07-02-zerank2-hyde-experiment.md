# zerank-2 Reranker + HyDE 查询改写实验复盘

> 日期: 2026-07-02 | 关联: main 分支
> 运行记录: runs/20260702_1902/

---

## 1. 实验背景

Full+Rerank (BGE) 已达 NDCG@10 = 0.5506，超过 pipeline SOTA (0.532)。继续抠 Visual 路 ROI 越来越低，更高的杠杆在换 Reranker 和查询改写。

| 措施 | 编码成本 | 预期 ΔNDCG@10 |
|---|---|---|
| BGE → zerank-2 | 零 | +0.03~0.08 |
| 查询改写 HyDE | 零 | +0.02~0.05 |
| ColPali → ColEmbed-3B | 需重编码 | +0.10~0.16 |

---

## 2. 实验结果

### 完整消融

| Config | NDCG@10 | Δ vs 基线 |
|---|---|---|
| Full_with_rerank (BGE) | 0.5506 | 基线 |
| **Full_zerank2** | **0.5715** | **+0.0209** |
| Full_BGE_HyDE | 0.5458 | -0.0048 |
| Full_zerank2_HyDE | 0.5733 | +0.0018 (≈0) |

### 结论

- **zerank-2: 有效**。+0.0209 在预期下沿（+0.03~0.08），代价是推理延迟 ~2x（逐条预测，详见 §4）
- **HyDE: 无效**。ViDoRe Industrial 短技术查询不适合 HyDE — 生成的假设文档偏离原文风格，引入噪声而非信号

---

## 3. 踩坑记录

### 3.1 zerank-2 分数随机（NDCG@10 = 0.17）

**现象**: 首次运行 Full_zerank2 得 0.1737，接近随机。

**根因**: `sentence-transformers` 版本 3.4.1 不支持 zerank-2 的自定义 `LogitScore` 模块（模型用 ST 5.4.0 保存）。旧版 CrossEncoder 回退到 `AutoModelForSequenceClassification` 默认头，`score.weight` 随机初始化。

**修复**: 升级 `sentence-transformers >= 5.4`（实际装 5.6.0）。注意 `vidore-benchmark` 要求 `<4.0.0`，但我们自有 eval 适配器不受影响。

```bash
pip install 'sentence-transformers>=5.4'
```

**教训**: 加载社区模型前先检查 `config_sentence_transformers.json` 中的 `__version__`，确保 ST 版本兼容。

### 3.2 zerank-2 不支持 batch > 1

**现象**: `ValueError: Cannot handle batch sizes > 1 if no padding token is defined.`

**根因**: zerank-2 底层 Qwen3-4B 无 padding token，CrossEncoder 默认批量推理触发 transformers 校验。

**修复**: `Reranker.rerank()` 改为逐条 `model.predict([pair])`。缺陷：40 个候选从 1 次 batch 变 40 次推理，延迟 ~2x（1192ms vs 544ms）。

**教训**: Qwen 系列模型无 padding token 是已知限制，未来换模型时需提前验证。

### 3.3 Ollama 500 错误

**现象**: HyDE 预计算阶段所有请求返回 500。

**根因**: Ollama 安装时 `llama-server` 二进制下载失败（网络问题），推理引擎缺失。后续重装解决。

**教训**: `curl -fsSL https://ollama.com/install.sh | sh` 无报错不代表安装完整。部署后必须用 `ollama run <model> --verbose` 验证。

### 3.4 CPU 模式 cgroup 内存限制

**现象**: GPU 模式下 Ollama 正常，CPU 模式下 llama-server 被 kill。

**根因**: AutoDL CPU 模式实例有 cgroup 内存限制（检测到 2GB），qwen2:7b 需 4.2GB 无法加载。GPU 模式下模型加载到显存绕过限制。

**教训**: HyDE 必须在 GPU 模式下完成预计算，Ollama 需 GPU 加速。

### 3.5 三模型同时加载 OOM

**现象**: BGE embedder + BGE reranker + zerank-2 + ColPali 同时加载 → 24GB OOM。

**修复**:
1. zerank-2 延迟到 ColPali 卸载后加载
2. BGE embedder 保持 GPU（检索需要速度）
3. zerank-2 强制 bf16（`model_kwargs={"torch_dtype": torch.bfloat16}`）

**显存分配**: BGE(~1.3G) + BGE-reranker(~1.2G) → ColPali(~7G) → unload → zerank-2-bf16(~8G) + FAISS(~2.6G) = ~13G

---

## 4. 架构变更

### 新增文件
- `src/retrieval/hyde.py` — HyDEGenerator，Ollama 生成假设文档，支持预计算缓存

### 修改文件
- `config/models.yaml` — 新增 `zerank_reranker`, `llm`
- `src/config.py` — 新增 `zerank_reranker_model_id`, `llm_model_id` 属性
- `src/retrieval/reranker.py` — 支持可选 `model_id` + `model_kwargs`；逐条预测兼容 zerank-2
- `src/evaluation/vidore_adapter.py` — 支持 `use_hyde` + `reranker_type` 双维度
- `src/evaluation/ablation.py` — `AblationConfig` 扩展 `reranker_type` + `use_hyde`；新增 `--quick` 模式
- `scripts/run_eval.py` — HyDE 预计算流程 + 双 Reranker 延迟加载 + `--quick` flag

### HyDE 预计算流程

```
ColPali unload → Ollama GPU 推理 (283 条缓存) → kill Ollama 释放显存
→ 加载 zerank-2 → 加载 FAISS → eval（HyDE 从缓存读取，无 Ollama 依赖）
```

---

## 5. 下一步

1. **换 ColEmbed-3B**: 预期 +0.10~0.16，是当前最高杠杆项
2. **zerank-2 加速**: 研究是否可加 padding token 恢复批量推理
3. **HyDE 改进**: 尝试更适合工业文档的 prompt，或用 ColPali 的多向量直接匹配 HyDE 文本
