# Handoff — 本地环境与数据状态

> 当前分支: main
> 最后 commit: `9467938 docs: 第一版设计`

---

## 1. Python 环境

| 项目 | 值 |
|------|-----|
| 管理器 | `uv`（Python 3.11，venv 在 `poc/.venv/`） |
| 系统 Python | 3.14.6（未使用） |
| 硬件 | MacBook M 系列, 32GB, GPU MPS |
| 磁盘剩余 | ~798 GB |

### 已安装的依赖（poc/.venv）

- torch 2.11.0（MPS ✅）
- colpali-engine 0.3.17
- faiss-cpu 1.14.3
- datasets 5.0.0
- transformers 5.12.1
- Pillow, numpy, psutil, tqdm

---

## 2. 已下载数据

### 2.1 ViDoRe v3 Industrial 子集

| 维度 | 值 |
|------|-----|
| HF 路径 | `vidore/vidore_v3_industrial` |
| 本地缓存 | `~/.cache/huggingface/datasets/vidore___vidore_v3_industrial/` |
| 磁盘占用 | **~1.9 GB** |
| 页面数 | **5,244** 页 |
| 文档数 | **27** 份 |
| 查询数 | **1,698** 条 |
| qrels | **9,684** 条 |
| 图片尺寸 | 1000×1600（竖版 A4 比例, PNG） |

**每页数据结构**（corpus split='test'）:
```python
{
    "corpus_id": int,           # 页级唯一 ID
    "doc_id": str,              # 文档标识（如 "AFD-091005-056"）
    "page_number_in_doc": int,  # 文档内的页码
    "image": PIL.PngImageFile,  # 页面截图（ColPali 直接输入）
    "markdown": str | None      # OCR 文本（BGE/BM25 用）
}
```

**查询数据结构**（queries split='test'）:
```python
{
    "query_id": int,
    "query": str,               # 查询原文
    "answer": str | None,       # 参考答案（部分有）
    "language": str,            # 如 "english"
    "query_types": [...],
    "content_type": [...],
    ...
}
```

**qrels 结构**（qrels split='test'）:
```python
{
    "query_id": int,
    "corpus_id": int,
    "score": int,               # 1 = 相关
    "content_type": [...],
    "bounding_boxes": [...],     # 页面内 bounding box
}
```

**开发注意**: 竖版图片（1000×1600）的 ColPali patch 数（~1600 patches/page）多于 POC 测试的横版图片（1024×768, ~1031 patches），编码时间和索引体积会大~50%。

### 2.2 POC 验证产物

| 文件 | 内容 |
|------|------|
| `poc/colpali_spike.py` | ColPali 性能验证脚本（编码吞吐、FAISS 索引、MaxSim 延迟） |
| `poc/download_dev_data.py` | 数据下载辅助脚本 |
| `poc/.venv/` | Python 3.11 虚拟环境，所有依赖已装好 |

**POC 实测关键数据**（vidore/colpali-v1.3, MPS bfloat16, 合成横版页面）:

| 指标 | 值 |
|------|-----|
| 编码吞吐 | ~1.5 pg/s（横版 1024×768） |
| 每页索引 | ~0.5 MB（IndexFlatIP, 128d） |
| MPS 显存（模型） | ~5.6 GB |
| MaxSim naïve | ~86ms/10pg |
| 首 query 冷启动 | ~1s（torch.mps 编译开销） |
| 模型参数量 | 3,491M（3.5B） |

详见设计文档 §3.4（已更新 POC 实测数据）。

---

## 3. 尚未下载的数据

这些是开发完成、全量评测时才需要：

| 数据 | 来源 | 量 | 触发条件 |
|------|------|-----|---------|
| ViDoRe 全 8 子集 | HF `vidore/vidore_v3_*` | ~24,000 页 | 跑 ViDoRe Leaderboard 评测前 |
| RealKIE（合同/发票） | GitHub | ~200 份 | Ingestion pipeline 跑通后搭 Demo |
| CHIC（发票/采购单） | GitHub | ~100 份 | 同上 |
| Siemens 设备手册 | 公开采集 | ~50 份 | Demo 丰富时 |
| 自建 QA（50 条） | 人工 + LLM | 50 条 | 端到端评测前 |
| 拒答集（20 条） | 自己写 | 20 条 | 第一阶段评测前 |

---

## 4. 项目仓库结构（当前）

```
pdf-rag/
├── docs/
│   ├── industrial-pdf-rag-design.md       # 设计文档（已含 POC 数据更新）
│   └── industrial-pdf-rag-architecture.md # 架构图（已同步 POC 数据）
├── poc/
│   ├── .venv/                              # Python 3.11 虚拟环境
│   ├── colpali_spike.py                    # POC 验证脚本
│   └── download_dev_data.py                # 数据下载脚本
├── data/                                   # 空, 数据在 HF cache 下
└── README.md (待创建)
```

---

## 5. 下一步开发建议

1. **Git 分支**: 切 `feat/ingestion` 开始开发
2. **模块目录**: 按设计文档建 `ingestion/` `retrieval/` `evaluation/`
3. **数据读取**: 用 `datasets.load_dataset()` 直接读 HF cache，不用复制数据
4. **首先开发的模块**: Ingestion pipeline（MinerU 解析 → BGE encode → ColPali encode → 写 pgvector + FAISS）
