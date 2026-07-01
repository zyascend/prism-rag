#!/usr/bin/env bash
# AutoDL 云端一键部署脚本
# 用法: SSH 到云实例后执行:
#   git clone https://github.com/zyascend/prism-rag.git && cd prism-rag && bash scripts/cloud_setup.sh
set -euo pipefail

echo "========================================"
echo " PrismRAG AutoDL 云端部署"
echo "========================================"

# ── 0. 环境确认 ──
echo ""
echo "[0/5] 环境确认..."
python3 --version
nvidia-smi | head -5 || echo "⚠️ nvidia-smi 不可用"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

# ── 1. 虚拟环境 ──
echo ""
echo "[1/5] 创建虚拟环境..."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -q

# ── 2. 安装依赖 (faiss-gpu 替换 faiss-cpu) ──
echo ""
echo "[2/5] 安装依赖..."
pip install -r requirements-cloud.txt -q
echo "  依赖安装完成 ✅"

# ── 3. 启动 PostgreSQL + pgvector ──
echo ""
echo "[3/5] 配置 PostgreSQL..."

# AutoDL 上通常已有 PostgreSQL，检查并创建数据库
if command -v psql &>/dev/null; then
    # 尝试创建用户和数据库（忽略已存在错误）
    sudo -u postgres psql -c "CREATE USER prismrag WITH PASSWORD 'prismrag'" 2>/dev/null || true
    sudo -u postgres psql -c "CREATE DATABASE prismrag OWNER prismrag" 2>/dev/null || true
    sudo -u postgres psql -d prismrag -c "CREATE EXTENSION IF NOT EXISTS vector" 2>/dev/null || true
    echo "  PostgreSQL + pgvector 就绪 ✅"
else
    echo "  ⚠️ PostgreSQL 未安装，尝试 Docker..."
    docker run -d --name prismrag-db \
        -e POSTGRES_DB=prismrag -e POSTGRES_USER=prismrag -e POSTGRES_PASSWORD=prismrag \
        -p 5432:5432 pgvector/pgvector:pg16 2>/dev/null && \
        echo "  PostgreSQL Docker 启动 ✅" || \
        echo "  ❌ 请手动安装 PostgreSQL + pgvector"
fi

# ── 4. 创建 .env ──
echo ""
echo "[4/5] 创建配置文件..."
cp -n .env.example .env 2>/dev/null || true
mkdir -p indexes results logs

# GPU 适配: 24GB 显存可提高 batch size
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    VRAM=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_mem // 1024**3)" 2>/dev/null || echo 0)
    if [ "$VRAM" -ge 20 ]; then
        echo "  检测到 ${VRAM}GB VRAM → ColPali batch_size=8"
        sed -i 's/colpali_batch_size: 4/colpali_batch_size: 8/' config/models.yaml
    fi
fi

# ── 5. 运行全量流程 ──
echo ""
echo "[5/5] 运行全量流程..."
echo "  ⏳ 下载 ViDoRe 数据集 + ColPali 模型（首次约 15-30 分钟）..."
echo "  ⏳ ColPali 编码 5244 页（约 1-1.5 小时，GPU 加速）..."
echo "  ⏳ 全量消融评测..."

python scripts/ingest_vidore.py --dataset vidore/vidore_v3_industrial 2>&1 | tee logs/ingest.log
python scripts/run_eval.py --dataset vidore/vidore_v3_industrial --skip-index 2>&1 | tee logs/eval.log

echo ""
echo "========================================"
echo " ✅ 全量流程完成！"
echo " 结果: results/ablation_results.json"
echo ""
echo " 📦 拉回本地（在 Mac 终端执行）:"
echo "    bash scripts/pull_from_cloud.sh <host> <port>"
echo ""
echo " 或手动 rsync:"
echo "    rsync -avz -e 'ssh -p <port>' root@<host>:/root/prism-rag/results/ ./results/"
echo "    rsync -avz -e 'ssh -p <port>' root@<host>:/root/prism-rag/indexes/ ./indexes/"
echo ""
echo " ⚠️  拉完记得关机，按时计费！"
echo "========================================"
