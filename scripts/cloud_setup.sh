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
# 耗时: 首次约 15-20 分钟
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
# Step 1: Python 环境（智能检测 conda → 复用 或 apt 安装）
# ═══════════════════════════════════════════════════════════
log "[1/6] Python 环境..."

PYTHON_BIN=""
CONDA_BASE="/root/miniconda3"
CONDA_PYTHON="$CONDA_BASE/bin/python3"

# 优先检测 conda（AutoDL 自带 torch + CUDA，省 2GB 下载）
if [ -f "$CONDA_PYTHON" ]; then
    CONDA_TORCH=$("$CONDA_PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if [ -n "$CONDA_TORCH" ]; then
        PYTHON_BIN="$CONDA_PYTHON"
        log "  ✅ 复用 conda (Python $($PYTHON_BIN --version 2>&1), torch $CONDA_TORCH)"
        log "  ⚡ 跳过 PyTorch/CUDA 下载 (~2GB)"
    else
        log "  conda 存在但无 torch，将完整安装"
    fi
fi

# 没有可复用的 → 安装 Python 3.11
if [ -z "$PYTHON_BIN" ]; then
    if python3.11 --version &>/dev/null; then
        PYTHON_BIN="python3.11"
    else
        log "  安装 Python 3.11..."
        apt-get update -qq
        apt-get install -y -qq python3.11 python3.11-venv python3.11-dev 2>&1 | tail -3
        PYTHON_BIN="python3.11"
    fi
    log "  ✅ $($PYTHON_BIN --version 2>&1)"
fi

# ═══════════════════════════════════════════════════════════
# Step 2: 项目依赖（只装缺失的部分）
# ═══════════════════════════════════════════════════════════
log "[2/6] 项目依赖..."

cd /root/prism-rag

if [ "$PYTHON_BIN" = "$CONDA_PYTHON" ]; then
    # === conda 路径：只装缺失包 ===
    log "  检测 conda 已有包 vs requirements-cloud.txt..."

    # 列出 requirements 中需要但 conda 缺失的包
	    MISSING=$($PYTHON_BIN << 'PYEOF'
import subprocess, sys

# 获取已安装包
installed = subprocess.check_output(
    ["/root/miniconda3/bin/pip", "list", "--format=columns"],
    text=True
).lower()

# 检查 requirements-cloud.txt 每一行（只看包名，忽略版本约束）
needed = []
with open("requirements-cloud.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 提取包名（去掉版本约束、extras）
        pkg = line.split(">=")[0].split("==")[0].split("[")[0].split(";")[0].strip().lower()
        pkg = pkg.replace("_", "-")
        if pkg not in installed:
            needed.append(line)

print(" ".join(repr(l) for l in needed))
PYEOF
)

    if [ -z "$MISSING" ] || [ "$MISSING" = " " ]; then
        log "  ✅ 所有依赖已满足"
    else
        log "  需安装 $(echo "$MISSING" | wc -w) 个缺失包..."
        eval "$CONDA_PYTHON -m pip install $MISSING 2>&1 | grep -v '^Requirement already satisfied' || true"
        log "  ✅ 依赖安装完成"
    fi

else
    # === 非 conda 路径：完整 pip install ===
    if [ ! -f .venv/bin/python3 ]; then
        $PYTHON_BIN -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip -q
    log "  pip install（完整安装，约 10-15 min）..."
    pip install -r requirements-cloud.txt 2>&1
    log "  ✅ pip install 完成"
fi

# 快速验证
if [ "$PYTHON_BIN" = "$CONDA_PYTHON" ]; then
    VERIFY_PY="$CONDA_PYTHON"
else
    VERIFY_PY=".venv/bin/python3"
fi

$VERIFY_PY -c "
import torch, faiss
print(f'  ✅ PyTorch {torch.__version__} | FAISS {faiss.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()} (切有卡后将变为 True)')
" 2>&1 || warn "  ⚠️  基础包验证失败，但不阻塞流程"

# ═══════════════════════════════════════════════════════════
# Step 3: PostgreSQL + pgvector
# ═══════════════════════════════════════════════════════════
log "[3/6] PostgreSQL + pgvector..."

# 编译 pgvector（幂等：已安装则跳过）
if [ ! -f /usr/lib/postgresql/14/lib/vector.so ]; then
	    log "  编译 pgvector..."
	    apt-get update -qq 2>/dev/null
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

# ═══════════════════════════════════════════════════════════
# Step 4: 预下载模型（HF，约 5-10 min）
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# Step 4: 预下载模型（HF 镜像，约 5-10 min）
# ═══════════════════════════════════════════════════════════
log "[4/6] 预下载模型到 HF cache..."

export HF_HOME="$AUTODL_TMP/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"  # 国内镜像更快更稳
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"

$VERIFY_PY << 'PYEOF'
import os
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 强制 CPU

from huggingface_hub import snapshot_download

print("  Downloading ColPali (vidore/colpali-v1.3, ~3.5GB)...")
snapshot_download("vidore/colpali-v1.3")
print("  ✅ ColPali cached")

print("  Downloading BGE (BAAI/bge-large-en-v1.5, ~1.3GB)...")
snapshot_download("BAAI/bge-large-en-v1.5")
print("  ✅ BGE cached")

print("  Downloading BGE Reranker (BAAI/bge-reranker-large, ~1.3GB)...")
snapshot_download("BAAI/bge-reranker-large")
print("  ✅ Reranker cached")

print("  All models cached successfully!")
PYEOF

log "  ✅ 模型预下载完成 ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1))"

# ═══════════════════════════════════════════════════════════
# Step 5: 预缓存 ViDoRe 数据集（约 2GB）
# ═══════════════════════════════════════════════════════════
log "[5/6] 预缓存 ViDoRe 数据集..."

$VERIFY_PY << 'PYEOF'
import os
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# 数据集下载走代理（hf-mirror.com 对 datasets 库支持不稳定）
os.environ["http_proxy"] = "http://172.26.1.26:12798"
os.environ["https_proxy"] = "http://172.26.1.26:12798"

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

# ── 保存 Python 路径到标记文件，供 Phase 2 使用 ──
echo "$VERIFY_PY" > .python_bin
log "  Python 路径已保存: $VERIFY_PY → .python_bin"

# ── 最终验证 ───────────────────────────────────────────────
log ""
log "====== Phase 1 验证 ======"

PASS=true

$VERIFY_PY -c "import torch; print(f'  ✅ PyTorch {torch.__version__}')" 2>/dev/null || { warn "  ❌ PyTorch"; PASS=false; }
$VERIFY_PY -c "import faiss; print(f'  ✅ FAISS {faiss.__version__}')" 2>/dev/null || { warn "  ❌ FAISS"; PASS=false; }
$VERIFY_PY -c "import sentence_transformers; print(f'  ✅ sentence-transformers')" 2>/dev/null || { warn "  ❌ sentence-transformers"; PASS=false; }
$VERIFY_PY -c "from colpali_engine.models import ColPali; print(f'  ✅ colpali-engine')" 2>/dev/null || { warn "  ❌ colpali-engine"; PASS=false; }

pg_isready 2>/dev/null && log "  ✅ PostgreSQL running" || warn "  ⚠️  PostgreSQL not running"

$VERIFY_PY -c "
import psycopg2
c = psycopg2.connect(host='localhost', dbname='prismrag', user='prismrag', password='prismrag')
c.close()
print('  ✅ PostgreSQL connection OK')
" 2>/dev/null || warn "  ⚠️  PostgreSQL connection failed"

HF_SIZE=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)
log "  ✅ HF cache: $HF_SIZE"

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
log "💾 磁盘占用预估:"
log "  - 索引 (FAISS): ~2.5 GB"
log "  - 结果: ~50 KB"
log "  - 模型 + 数据 (已缓存): ~8 GB"
log "=========================================="
