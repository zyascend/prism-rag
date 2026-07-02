# Design: ColPali v1.3 → Colembed-3B Visual 路升级

> 日期: 2026-07-02 | 状态: Draft | 分支: 待创建

---

## 1. 动机

当前 Visual 路使用 `vidore/colpali-v1.3`（Paligemma-3B 骨干），ViDoRe V1 NDCG@5 = 84.8，ViDoRe V2 = ~54.6。在本项目的 ViDoRe V3 Industrial 数据集上，Visual_only NDCG@10 仅 0.1302（远低于官方 ~0.47，存在 3.6x 编码/grounding 差距），导致三路融合中 Visual 路成为负向贡献（BM25+Dense 0.4528 → 加 Visual 降为 0.4402）。

`nvidia/llama-nemoretriever-colembed-3b-v1` 是基于 Llama 3.2-3B + SigLIP2 的 ColBERT-style late-interaction 模型，ViDoRe V1 NDCG@5 = 91.0（#1），ViDoRe V2 = 63.5（#1）。输出 128-dim 多向量，与 ColPali v1.3 格式完全一致。

**目标**：以最小侵入方式替换 Visual 路 embedding 模型，预期全链路 NDCG@10 从 0.5715 提升至 0.62~0.68。

---

## 2. 方案：方案 A — 最小侵入 Encoder 替换

只替换 `src/ingestion/encoders.py` 中的视觉编码器，保持 FAISS 存储 / MaxSim 搜索 / pgvector grounding 完全不变。

### 与 ColPali 的差异对照

| 维度 | ColPali v1.3 | Colembed-3B |
|------|-------------|-------------|
| 加载方式 | `colpali-engine` 包 | `AutoModel.from_pretrained(trust_remote_code=True)` |
| 骨干 | Paligemma-3B | Llama 3.2-3B + SigLIP2 |
| 注意力 | 因果 (causal) | 双向 (bidirectional) |
| 输出维度 | 128 | 128 |
| 编码 API | `processor.process_images()` + `model.forward()` | `model.forward_passages(images)` + `model.forward_queries(texts)` |
| 原生评分 | 无（需自行实现 MaxSim） | `model.get_scores(q, d)` |
| 依赖 | `colpali-engine` | `transformers>=4.49.0` + `flash-attn==2.6.3` |

---

## 3. 详细设计

### 3.1 ColembedEncoder 类

位置：`src/ingestion/encoders.py`，与 `ColPaliEmbedder` 并列。

```python
class ColembedEncoder:
    """NVIDIA llama-nemoretriever-colembed-3b-v1 multi-vector encoder.

    接口对标 ColPaliEmbedder，下游零改动：
      encode_pages(images) → List[np.ndarray] shape [n_tokens, 128]
      encode_query(text)   → np.ndarray       shape [n_tokens, 128]
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        max_input_tiles: int = 2,
        use_fp16: bool = True,
    ):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()
        self.device = device
        self.max_input_tiles = max_input_tiles

    def encode_pages(
        self, images: List[Image.Image], batch_size: int = 4
    ) -> List[np.ndarray]:
        embeddings = self.model.forward_passages(
            images,
            batch_size=batch_size,
            max_input_tiles=self.max_input_tiles,
        )
        # embeddings: torch.Tensor [batch, n_tokens, 128]
        return [
            emb.float().cpu().numpy() for emb in embeddings
        ]

    def encode_query(self, text: str) -> np.ndarray:
        embeddings = self.model.forward_queries(
            [text], batch_size=1
        )
        return embeddings[0].float().cpu().numpy()

    @staticmethod
    def verify_scoring_equivalence(
        encoder: "ColembedEncoder",
        test_pages: List[Image.Image],
        test_queries: List[str],
    ) -> dict:
        """验证 FAISS MaxSim 与模型原生 get_scores() 排名一致性。

        编码 N 页 + M 条 query → 计算 N×M 得分矩阵 → 比较每行 top-5。
        返回 {"passed": bool, "match_rate": float, "details": str}
        """
```

**关键参数**：
- `max_input_tiles=2`：显存安全默认值，与论文训练设置一致。云端可在命令行覆盖为 4。
- `torch_dtype=bfloat16`：减少显存，float32 输出给 FAISS。

### 3.2 工厂函数

```python
def create_visual_encoder(model_name: str, device: str) -> "VisualEncoderProtocol":
    """按 model_name 前缀分发。"""
    if model_name.startswith("colembed"):
        return ColembedEncoder(
            model_id=cfg.colembed_model_id,
            device=device,
            max_input_tiles=cfg.get("embedding.colembed_max_input_tiles", 2),
        )
    else:
        return ColPaliEmbedder(
            model_id=cfg.colpali_model_id,
            device=device,
        )
```

### 3.3 配置变更

**`config/models.yaml`** 新增：

```yaml
models:
  colembed: "nvidia/llama-nemoretriever-colembed-3b-v1"

embedding:
  colembed_max_input_tiles: 2
  colembed_batch_size: 4
```

**`src/config.py`** 新增：

```python
@property
def colembed_model_id(self) -> str:
    return self.get("models.colembed")
```

### 3.4 评分一致性验证

在 `ColembedEncoder` 中内建 `verify_scoring_equivalence()` 方法：

1. 编码 10 页 test 图像 + 10 条 test query
2. **方法 A**：encode → FAISS IndexFlatIP → MaxSim → 得分矩阵
3. **方法 B**：encode → `model.get_scores(q_embs, p_embs)` → 得分矩阵
4. 比较每行 top-5 排名是否一致
5. 通过 → 后续全走 FAISS；失败 → 报告 mismatch rate，考虑切换到原生 `get_scores()`

此验证在 S2（本地）执行，云端无效执行。

### 3.5 下游适配

| 文件 | 改动 |
|------|------|
| `src/ingestion/vidore_ingestor.py` | `--visual-model` 参数（默认 `colpali`），编码器创建改用工厂函数 |
| `src/retrieval/visual_retriever.py` | Query 编码阶段同理，工厂函数加载 |
| `src/store/faiss_store.py` | 无改动。128-dim 完全兼容。index 文件名加 `colembed` 后缀避免覆盖现有 ColPali 索引 |
| `src/ingestion/progress.py` | 无改动。编码进度 pickle 自动适配 |
| `src/retrieval/dense_retriever.py` | 无改动 |
| `src/retrieval/bm25_retriever.py` | 无改动 |
| `src/retrieval/fusion.py` | 无改动 |
| `src/retrieval/reranker.py` | 无改动 |
| `src/retrieval/hyde.py` | 无改动 |

### 3.6 评测适配

```python
# 新增消融配置（仅跑相关项）
COLEMBED_ABLATIONS = {
    "Visual_only_baseline": {  # ColPali 基线，对照用
        "visual_model": "colpali",
        "bm25": False, "dense": False, "visual": True,
        "reranker": "none", "use_hyde": False,
    },
    "Visual_only_colembed": {
        "visual_model": "colembed-3b",
        "bm25": False, "dense": False, "visual": True,
        "reranker": "none", "use_hyde": False,
    },
    "Full_zerank2_colembed": {
        "visual_model": "colembed-3b",
        "bm25": True, "dense": True, "visual": True,
        "reranker": "zerank-2", "use_hyde": False,
    },
}
```

`scripts/run_eval.py` 新增 `--visual-model` / `--preset colembed` 参数。

### 3.7 云端部署

- `requirements-cloud.txt`：确认 `flash-attn==2.6.3` + `transformers>=4.49.0`
- `scripts/cloud_setup.sh` Phase 1：检测 flash-attn 是否已安装（云端缓存），未安装则 `pip install flash-attn==2.6.3 --no-build-isolation`
- 模型已缓存于 `/root/autodl-tmp/huggingface`，跳过下载
- 显存管理：Colembed 编码 → 卸载 → FAISS 加载 → BGE reranker / zerank-2 加载。编码阶段 batch_size=4（保守，24GB 4090 可调至 8）

---

## 4. 实施步骤

| 步骤 | 内容 | 产出 | 预计 |
|------|------|------|------|
| **S1** | `ColembedEncoder` 类 + 工厂函数 + models.yaml/config.py | 本地代码 | 30min |
| **S2** | 本地验证：10 页编码 + `verify_scoring_equivalence()` | 评分一致性报告 | 20min |
| **S3** | Cloud setup 更新：依赖检查 + flash-attn | 更新的 cloud_setup.sh | 10min |
| **S4** | 云端全量编码（5244 页，直接加载已缓存模型） | FAISS 索引文件 | 40min |
| **S5** | 聚焦评测：3 组消融（ColPali 基线 + Colembed_Visual + Colembed_Full） | NDCG@10/Recall/MRR | 10min |
| **S6** | 结果对比 & 决策：是否正式替换 ColPali v1.3 | 决策记录 | 10min |

> 总计约 2 小时（不含本地模型下载）

---

## 5. 风险 & 缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| flash-attn 编译失败 | 中 | 云端预装；手动 `--no-build-isolation` |
| 4090 24GB 显存不足 | 中 | batch_size=4 + max_input_tiles=2；OOM 则降 batch_size=1 |
| 模型输出与 FAISS MaxSim 排名不一致 | 低 | S2 验证环节；不一致则切换到 `model.get_scores()` |
| `trust_remote_code=True` 安全风险 | 低 | 固定 revision hash 锁定版本 |
| 本地 macOS 无法运行 | 确定 | 只做代码级验证（S1-S2），实际运行全在云端 |

---

## 6. 成功标准

1. `verify_scoring_equivalence()` 通过（match_rate ≥ 0.95）
2. `Visual_only_colembed` NDCG@10 显著高于 `Visual_only_baseline`（目标：0.35+，当前基线 0.1302）
3. `Full_zerank2_colembed` NDCG@10 高于当前最优 0.5715（目标：0.60+）
