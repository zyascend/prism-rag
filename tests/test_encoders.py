"""编码器单元测试"""

import torch
from src.ingestion.encoders import BGEEmbedder, ColPaliEmbedder


def test_bge_encoder_output_shape():
    embedder = BGEEmbedder(device="cpu")
    texts = ["What is the load capacity?", "Conveyor belt specifications"]
    embs = embedder.encode(texts)
    assert isinstance(embs, torch.Tensor)
    assert embs.shape == (2, 1024)
    # 验证归一化
    norms = torch.norm(embs, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_colpali_encoder_page_output():
    from PIL import Image
    import numpy as np

    embedder = ColPaliEmbedder(device="cpu")
    # 创建模拟页面
    imgs = [Image.fromarray(np.random.randint(0, 255, (1600, 1000, 3), dtype=np.uint8)) for _ in range(2)]
    embs = embedder.encode_pages(imgs, batch_size=2)
    assert len(embs) == 2
    for emb in embs:
        assert emb.ndim == 2  # [n_patches, 128]
        assert emb.shape[-1] == 128


def test_colpali_query_output():
    embedder = ColPaliEmbedder(device="cpu")
    q_emb = embedder.encode_query("load capacity")
    assert q_emb.ndim == 3  # [1, n_patches, 128]
    assert q_emb.shape[-1] == 128


def test_colpali_encode_queries_batch():
    """encode_queries_batch() 输出格式与单条 encode_query 一致"""
    embedder = ColPaliEmbedder(device="cpu")
    texts = ["load capacity", "conveyor belt specs"]
    result = embedder.encode_queries_batch(texts, batch_size=2)
    assert isinstance(result, dict)
    assert len(result) == 2
    for idx, emb in result.items():
        assert emb.ndim == 3  # [1, n_patches, 128]
        assert emb.shape[-1] == 128


def test_colpali_unload_raises_on_encode_query():
    """unload() 后调用 encode_query() 必须抛 RuntimeError"""
    embedder = ColPaliEmbedder(device="cpu")
    embedder.encode_query("load capacity")  # 确保 loaded
    embedder.unload()
    try:
        embedder.encode_query("load capacity")
        assert False, "应该已抛出 RuntimeError"
    except RuntimeError:
        pass


def test_colpali_unload_raises_on_encode_pages():
    """unload() 后调用 encode_pages() 必须抛 RuntimeError"""
    from PIL import Image
    import numpy as np

    embedder = ColPaliEmbedder(device="cpu")
    embedder.unload()
    imgs = [Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))]
    try:
        embedder.encode_pages(imgs)
        assert False, "应该已抛出 RuntimeError"
    except RuntimeError:
        pass


def test_colpali_unload_raises_on_encode_queries_batch():
    """unload() 后调用 encode_queries_batch() 必须抛 RuntimeError"""
    embedder = ColPaliEmbedder(device="cpu")
    embedder.unload()
    try:
        embedder.encode_queries_batch(["test"])
        assert False, "应该已抛出 RuntimeError"
    except RuntimeError:
        pass