"""BGE + ColPali 编码器封装"""

from __future__ import annotations

from typing import Dict, List

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import trange

from src.config import cfg


class BGEEmbedder:
    """BGE-large-en-v1.5 文本编码器"""

    def __init__(self, device: str | None = None):
        self.device = device if device is not None else cfg.get("embedding.bge_device", "cpu")
        # 使用 sentence-transformers 加载 BGE
        self.model = SentenceTransformer(
            cfg.bge_model_id,
            device=self.device,
        )
        self.dim = cfg.bge_dim

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> torch.Tensor:
        """编码文本列表为向量矩阵 [N, dim]"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # BGE 惯例：归一化后用内积等价余弦
        )

    def __call__(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


class ColPaliEmbedder:
    """ColPali 整页多向量编码器"""

    def __init__(self, device: str | None = None):
        self.device = device if device is not None else cfg.get("embedding.colpali_device", "cpu")
        self.model = ColPali.from_pretrained(
            cfg.colpali_model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        # 强制 use_fast=True：慢速图像处理器可能在 resize/padding 上有细微差异
        # 导致页面编码与查询编码进入不同潜空间（见 docs/solutions/.../visual-sota-gap-analysis.md）
        self.processor = ColPaliProcessor.from_pretrained(
            cfg.colpali_model_id,
            use_fast=True,
        )
        self._warmed_up = False
        self._loaded = True

    @torch.no_grad()
    def encode_pages(
        self, images: List[Image.Image], batch_size: int | None = None, show_progress: bool = False
    ) -> List[torch.Tensor]:
        """编码页面列表，每页返回 [n_patches, 128] 多向量"""
        self._require_loaded()
        if batch_size is None:
            batch_size = cfg.get("embedding.colpali_batch_size", 4)
        # 预热：首次 query 有 torch.compile 开销
        if not self._warmed_up:
            dummy = Image.new("RGB", (1000, 1600), color=255)
            self._warmup(dummy)
            self._warmed_up = True

        batches = []
        for i in trange(0, len(images), batch_size, disable=not show_progress, desc="ColPali encode"):
            batch_imgs = images[i : i + batch_size]
            # 使用 process_images 确保页面编码含 "Describe the image." prompt，
            # 与 ColPali 训练时的输入格式对齐。之前传 text=[""] 缺失了
            # 5 个关键 token (<bos>Describe the image.)，导致 page embedding
            # 语义质量下降，Visual 路 NDCG@10 损失约 30-50%。
            batch_inputs = self.processor.process_images(images=batch_imgs).to(self.device)
            batch_outputs = self.model(**batch_inputs)
            # batch_outputs: [batch, n_patches, 128]
            batches.extend(list(batch_outputs.cpu()))

        return batches

    def _warmup(self, dummy_image: Image.Image):
        """MPS 首次查询预热"""
        inputs = self.processor.process_images(images=[dummy_image]).to(self.device)
        _ = self.model(**inputs)

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """编码单条文本查询为 [1, n_patches, 128]（ColPali 查询编码）

        使用 processor.process_queries() 做纯文本编码。
        之前传 dummy white image 导致 1024 个白图 patch 淹没 ~10 个文本 token，
        MaxSim 退化到 NDCG@10 ≈ 0.1（应为 ~0.35）。
        """
        self._require_loaded()
        # max_length=128：工业级查询长达 40+ token，默认 50 会截断 pad buffer 和查询尾部
        # 导致 query embedding 语义残缺，MaxSim 找不到相关页
        inputs = self.processor.process_queries([text], max_length=128).to(self.device)
        return self.model(**inputs).cpu()

    @torch.no_grad()
    def encode_queries_batch(self, texts: List[str], batch_size: int = 4, max_length: int = 128) -> Dict[int, torch.Tensor]:
        """批量编码多个 query，返回 {idx: tensor[1, n_patches, 128]}

        纯文本编码，理由同 encode_query。
        """
        self._require_loaded()
        results: Dict[int, torch.Tensor] = {}
        for i in trange(0, len(texts), batch_size, desc="ColPali encode queries"):
            batch_texts = texts[i : i + batch_size]
            # max_length=128：默认 50 可能截断长查询（工业集含 40+ token 查询）
            inputs = self.processor.process_queries(batch_texts, max_length=max_length).to(self.device)
            batch_outputs = self.model(**inputs)  # [batch, n_q, 128]
            for j, emb in enumerate(batch_outputs):
                results[i + j] = emb.unsqueeze(0).cpu()  # -> [1, n_q, 128]
        return results

    def unload(self):
        """卸载模型并释放显存。之后调用任何 encode_* 方法都会抛 RuntimeError。"""
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        self._loaded = False

    def _require_loaded(self):
        """检查模型是否可用，不可用时抛出 RuntimeError"""
        if not self._loaded:
            raise RuntimeError(
                "ColPaliEmbedder 已通过 unload() 卸载，无法执行编码操作。"
                "请重新创建 ColPaliEmbedder 实例后再调用。"
            )


def create_visual_encoder(
    model_name: str = "colpali",
    device: str | None = None,
) -> ColPaliEmbedder:
    """工厂函数：按模型名前缀创建视觉编码器。

    model_name 前缀匹配规则:
      - "colpali" → ColPaliEmbedder (vidore/colpali-v1.3)
      - "colembed" → ColembedEncoder (需切换 feature 分支)
    """
    if not model_name.startswith("colpali"):
        raise ValueError(f"不支持的 visual model: {model_name}（主分支仅支持 colpali，colembed 在 feature 分支）")
    return ColPaliEmbedder(device=device)
