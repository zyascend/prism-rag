# PrismRAG API 服务镜像
# 用法: docker build -t prismrag-api . && docker compose up

FROM python:3.11-slim

WORKDIR /app

# 系统依赖（psycopg2 + FAISS）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 源码
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY scripts/ scripts/

# 索引目录（运行时挂载或 Volume）
RUN mkdir -p /app/indexes /app/results

EXPOSE 8000
CMD ["python", "scripts/run_api.py"]
