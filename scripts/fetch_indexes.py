#!/usr/bin/env python
"""从 GitHub Release 拉取预编码索引（首次使用）"""

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RELEASE_URL = "https://github.com/zyascend/prism-rag/releases/download/v0.1.0/indexes.zip"
INDEXES_DIR = Path("indexes")


def main():
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(INDEXES_DIR.glob("*.faiss"))
    if existing:
        logger.info(f"索引已存在: {existing}")
        logger.info("如要重新下载，请先运行 `make clean`")
        return

    zip_path = INDEXES_DIR / "indexes.zip"

    logger.info(f"下载预编码索引: {RELEASE_URL}")
    try:
        urlretrieve(RELEASE_URL, zip_path)
    except Exception as e:
        logger.warning(f"下载失败（首次发布前可跳过）: {e}")
        logger.info("请先运行 `make ingest-vidore` 自行编码")
        return

    logger.info("解压中...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(INDEXES_DIR)

    zip_path.unlink()
    logger.info(f"索引已下载到 {INDEXES_DIR.resolve()}")


if __name__ == "__main__":
    main()