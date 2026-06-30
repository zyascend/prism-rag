#!/usr/bin/env python
"""
下载开发需要的最小数据：
1. ViDoRe Industrial 单子集 - 评测闭环必选（~3,000 页）
2. 1 份真实 PDF 用于解析管道验证（可选）
"""

import os, sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_vidore_industrial():
    """下载 ViDoRe Industrial 单子集（~3,000 页, ~500MB）"""
    from datasets import load_dataset
    print("📦 下载 ViDoRe Industrial 子集…")
    ds = load_dataset(
        "vidore/vidore-benchmark-v1",
        "industrial",
        split="corpus",
        cache_dir=str(DATA_DIR / "vidore"),
    )
    print(f"  ✅ 完成: {len(ds)} 页, 保存在 {DATA_DIR / 'vidore'}")
    print(f"  📝 结构: {ds.features}")
    print()
    # 同时下载查询和 qrels
    queries = load_dataset("vidore/vidore-benchmark-v1", "industrial", split="queries", cache_dir=str(DATA_DIR / "vidore"))
    qrels = load_dataset("vidore/vidore-benchmark-v1", "industrial", split="qrels", cache_dir=str(DATA_DIR / "vidore"))
    print(f"  ✅ 查询: {len(queries)} 条, qrels: {len(qrels)} 条")


def main():
    print("=" * 50)
    print("开发数据下载")
    print("=" * 50)

    # 1. ViDoRe Industrial 子集（核心）
    download_vidore_industrial()

    # 2. 真实 PDF（找 1 份公开的工业文档）
    print("💡 可选: 找 1 份真实 PDF 放在 data/ 下做解析管道验证")
    print("   例如从 Siemens 官网或 GitHub RealKIE 找 1 份合同/手册")
    print("   或直接从公开仓库下载:")
    print()
    print("   git clone --depth=1 https://github.com/huggingface/industreal-kie")
    d = DATA_DIR / "real_pdf"
    os.makedirs(d, exist_ok=True)
    print(f"   或手动放 1 份 PDF 到 {d}/ 即可")

    print()
    print("✅ 下载完成。开发期间的数据策略:")
    print("  - 开发调试: 只用 Industrial 子集（~3,000 页）")
    print("  - 全量评测: 开发完成后才下全部 8 个子集（~24,000 页）")
    print("  - Demo 知识库: 检索闭环跑通后再下 RealKIE/CHIC/Siemens")
    print("  - 编码时间: ~35 分钟（Industrial 子集 3,000 页 @ 1.5 pg/s）")


if __name__ == "__main__":
    main()