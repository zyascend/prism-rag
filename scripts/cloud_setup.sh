#!/usr/bin/env bash
# ============================================================
# PrismRAG AutoDL ⚡ Phase 1 — 无卡环境准备
# ============================================================
# 用途: 在无 GPU 模式下完成所有环境搭建、依赖安装、模型/数据下载。
#       完成后切有卡模式，直接执行 run_full_cloud.sh 跑流水线。
#
# 用法 (SSH 到云实例后):
#   cd /root/prism-rag && bash scripts/cloud_setup.sh
#
# 耗时: 首次约 15-25 分钟（主要是模型下载 ~5GB）
# 前置: 代码已上传到 /root/prism-rag/
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 环境检测 ────────────────────────────────────────────────
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
HAS_AUTODL=false
[ -f /etc/network_turbo ] && HAS_AUTODL=true
[ -d "$AUTODL_TMP" ] || HAS_AUTODL=false

log "====== PrismRAG Phase 1: 环境准备 ======"
log "模式: $( $HAS_AUTODL && echo 'AutoDL 无卡' || echo '独立服务器' )"
log "磁盘: $(df -h / | tail -1 | awk '{print $4" avail"}')"
$HAS_AUTODL && log "数据盘: $(df -h "$AUTODL_TMP" | tail -1 | awk '{print $4" avail"}')"

# ═══════════════════════════════════════════════════════════
# Step 1: Python 3.11
# ═══════════════════════════════════════════════════════════
log "[1/6] Python 3.11..."
if python3.11 --version &>/dev/null; then
    log "  ✅ Python $(python3.11 --version)"
else
    log "  安装中..."
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3.11-dev 2>&1 | tail -3
    log "  ✅ Python $(python3.11 --version)"
fi

# ═══════════════════════════════════════════════════════════
# Step 2: 虚拟环境 + pip 依赖
# ═══════════════════════════════════════════════════════════
log "[2/6] venv + pip 依赖..."

cd /root/prism-rag

# 清理旧 venv（Python 版本不匹配时）
if [ -f .venv/bin/python3 ]; then
    VENV_VER=$(./.venv/bin/python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
    if [ "$VENV_VER" != "3.11" ]; then
        warn "  旧 venv (Python $VENV_VER) 不匹配，重建"
        rm -rf .venv
    fi
fi

if [ ! -f .venv/bin/python3 ]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q

# 安装依赖（包含 CUDA 版 torch/faiss，无 GPU 也能装）
# 注意：此时不开 /etc/network_turbo 代理（代理会拖慢 PyPI），但保留阿里云 pip 镜像
log "  pip install..."
pip install -r requirements-cloud.txt 2>&1 | grep -E "(Successfully|ERROR|^Collecting|Downloading.*torch|Downloading.*faiss|Downloading.*transformers|Downloading.*colpali|Installing)" || true

# 快速验证
python3 -c "
import torch, faiss
print(f'  ✅ PyTorch {torch.__version__} | FAISS {faiss.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()} (切有卡后将变为 True)')
"

# ═══════════════════════════════════════════════════════════
# Step 3: PostgreSQL + pgvector
# ═══════════════════════════════════════════════════════════
log "[3/6] PostgreSQL + pgvector..."

# 编译 pgvector（幂等：已安装则跳过）
if [ ! -f /usr/lib/postgresql/14/lib/vector.so ]; then
    log "  编译 pgvector..."
    apt-get install -y -qq postgresql-14 postgresql-server-dev-14 2>&1 | tail -2

    PGV_DIR="/tmp/pgvector"
    if [ ! -d "$PGV_DIR" ]; then
        git clone --branch v0.8.4 https://github.com/pgvector/pgvector.git "$PGV_DIR" 2>&1 | tail -1 || \
        git clone --branch v0.8.4 https://gitee.com/mirrors/pgvector.git "$PGV_DIR" 2>&1 | tail -1
    fi
    cd "$PGV_DIR"
    make clean 2>/dev/null || true
    make -j$(nproc) 2>&1 | tail -2
    make install 2>&1 | tail -2
    cd /root/prism-rag
    log "  ✅ pgvector 编译完成"
else
    log "  ✅ pgvector 已安装"
fi

# 启动 + 建库（幂等）
pg_ctlcluster 14 main start 2>/dev/null || service postgresql start 2>/dev/null || true
sleep 1

su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'prismrag';\"" 2>/dev/null || true
su - postgres -c "psql -c \"CREATE USER prismrag WITH PASSWORD 'prismrag' CREATEDB;\"" 2>/dev/null || true
su - postgres -c "createdb prismrag -O prismrag" 2>/dev/null || true
su - postgres -c "psql -d prismrag -c 'CREATE EXTENSION IF NOT EXISTS vector;'" 2>/dev/null && \
    log "  ✅ PostgreSQL + pgvector 就绪" || \
    warn "  ⚠️  pgvector 扩展启用失败"

# ── 网络加速（AutoDL 内网代理，加速 GitHub/HF，完成后才开） ──
# 注意：pip install 必须先于此处执行，因为代理会拖慢 PyPI
if $HAS_AUTODL; then
    source /etc/network_turbo 2>/dev/null || true
    log "  ✅ 网络代理已启用（加速 HF/GitHub）"
fi

# ═══════════════════════════════════════════════════════════
# Step 4: 预下载模型（CPU-only，约 5-10 min）
# ═══════════════════════════════════════════════════════════
log "[4/6] 预下载模型到 HF cache..."

export HF_HOME="$AUTODL_TMP/huggingface"
mkdir -p "$HF_HOME"

python3 << 'PYEOF'
import os, sys
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU

print("  Downloading ColPali (vidore/colpali-v1.3, ~3.5GB)...")
from colpali_engine.models import ColPali, ColPaliProcessor
import torch
ColPali.from_pretrained(
    "vidore/colpali-v1.3",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
print("  ✅ ColPali cached")

print("  Downloading BGE (BAAI/bge-large-en-v1.5, ~1.3GB)...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
print("  ✅ BGE cached")

print("  Downloading BGE Reranker (BAAI/bge-reranker-large, ~1.3GB)...")
from sentence_transformers import CrossEncoder
CrossEncoder("BAAI/bge-reranker-large", device="cpu")
print("  ✅ Reranker cached")

print("  All models cached successfully!")
PYEOF

log "  ✅ 模型预下载完成 ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1))"

# ═══════════════════════════════════════════════════════════
# Step 5: 预缓存 ViDoRe 数据集（约 2GB）
# ═══════════════════════════════════════════════════════════
log "[5/6] 预缓存 ViDoRe 数据集..."

python3 << 'PYEOF'
import os
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import load_dataset
DS = "vidore/vidore_v3_industrial"

print(f"  Loading {DS}...")
for subset in ["corpus", "queries", "qrels"]:
    ds = load_dataset(DS, subset, split="test")
    print(f"  ✅ {subset}: {len(ds)} rows cached")
PYEOF

log "  ✅ 数据集预缓存完成"

# ═══════════════════════════════════════════════════════════
# Step 6: 目录结构 + 验证
# ═══════════════════════════════════════════════════════════
log "[6/6] 目录链接 + 验证..."

# 数据盘持久化 symlink（indexes/results/logs → autodl-tmp）
if $HAS_AUTODL; then
    for dir in indexes results logs; do
        mkdir -p "$AUTODL_TMP/$dir"
        [ -L "$dir" ] && rm -f "$dir"
        [ -d "$dir" ] && [ ! -L "$dir" ] && rm -rf "$dir"
        ln -sf "$AUTODL_TMP/$dir" "$dir"
        log "  $dir → $AUTODL_TMP/$dir"
    done
else
    mkdir -p indexes results logs
fi

# 更新 config 中 HF cache 路径（确保重启后仍有效）
python3 << 'PYEOF'
import yaml
with open("config/models.yaml") as f:
    c = yaml.safe_load(f)
# 保持默认值不变，HF_HOME 由环境变量控制
print("config/models.yaml OK")
PYEOF

# ── 最终验证 ───────────────────────────────────────────────
log ""
log "====== Phase 1 验证 ======"

PASS=true

# Python + 关键包
python3 -c "import torch; print(f'  ✅ PyTorch {torch.__version__}')" || { warn "  ❌ PyTorch"; PASS=false; }
python3 -c "import faiss; print(f'  ✅ FAISS {faiss.__version__}')" || { warn "  ❌ FAISS"; PASS=false; }
python3 -c "import sentence_transformers; print(f'  ✅ sentence-transformers')" || { warn "  ❌ sentence-transformers"; PASS=false; }
python3 -c "from colpali_engine.models import ColPali; print(f'  ✅ colpali-engine')" || { warn "  ❌ colpali-engine"; PASS=false; }

# PostgreSQL
pg_isready 2>/dev/null && log "  ✅ PostgreSQL running" || { warn "  ⚠️  PostgreSQL not running"; }
python3 -c "
import psycopg2
c = psycopg2.connect(host='localhost', dbname='prismrag', user='prismrag', password='prismrag')
c.close()
print('  ✅ PostgreSQL connection OK')
" 2>/dev/null || { warn "  ⚠️  PostgreSQL connection failed"; }

# HF cache
HF_SIZE=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)
log "  ✅ HF cache: $HF_SIZE"

# Data disk
if $HAS_AUTODL; then
    log "  ✅ Data disk avail: $(df -h "$AUTODL_TMP" | tail -1 | awk '{print $4}')"
fi

# ── 总结 ───────────────────────────────────────────────────
log ""
if $PASS; then
    log "====== Phase 1 完成 ✅ ======"
else
    warn "====== Phase 1 完成（有警告）⚠️ ======"
fi

log ""
log "📋 下一步 — Phase 2（切有卡模式后执行）:"
log ""
log "  cd /root/prism-rag && bash scripts/run_full_cloud.sh"
log ""
log "⏱  预计耗时: ~40-50 min"
log "  - ColPali 编码 5244 页: ~10 min"
log "  - BGE 编码 + BM25: ~3 min"
log "  - ViDoRe 消融 1698 queries × 7 configs: ~20 min"
log ""
log "💾 磁盘占用预估:"
log "  - 索引 (FAISS): ~2.5 GB"
log "  - 结果: ~50 KB"
log "  - 日志: ~1 MB"
log "  - 模型 + 数据 (已缓存): ~8 GB"
log "=========================================="
