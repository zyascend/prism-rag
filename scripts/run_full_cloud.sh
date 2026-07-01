#!/usr/bin/env bash
# ============================================================
# PrismRAG AutoDL ⚡ Phase 2 — 全量流水线（需要 GPU）
# ============================================================
# 前置: cloud_setup.sh (Phase 1) 已完成
# 用法:
#   1. 在 AutoDL 控制台切「有卡模式」开机
#   2. SSH 登录后执行:
#      cd /root/prism-rag && bash scripts/run_full_cloud.sh
#
# 耗时: ~40-50 min (4090 24GB)
#   - ColPali 编码 5244 页: ~10 min
#   - BGE 编码 + BM25: ~3 min
#   - ViDoRe 消融 1698 queries × 7 configs: ~20 min
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

cd /root/prism-rag

# ═══════════════════════════════════════════════════════════
# 0. 预检
# ═══════════════════════════════════════════════════════════
log "====== PrismRAG Phase 2: 全量流水线 $(date) ======"

# ── GPU 检测 ──
if ! command -v nvidia-smi &>/dev/null; then
    err "未检测到 nvidia-smi，请确认 GPU 已激活"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
log "GPU: $GPU_NAME"

# ── 网络加速 ──
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
    log "网络代理已启用"
fi

# ── venv ──
if [ ! -f .venv/bin/python3 ]; then
    err "未找到 .venv，请先运行 bash scripts/cloud_setup.sh"
fi
source .venv/bin/activate

# ── CUDA 验证 ──
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || \
    err "CUDA 不可用，检查 GPU 驱动"

VRAM=$(python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_mem // 1024**3)")
log "VRAM: ${VRAM}GB"

# ── HF cache ──
export HF_HOME="/root/autodl-tmp/huggingface"
if [ ! -d "$HF_HOME" ]; then
    warn "HF cache 未找到，模型将重新下载（可能 10+ 分钟）"
    mkdir -p "$HF_HOME"
else
    log "HF cache: $(du -sh "$HF_HOME" | cut -f1)"
fi

# ── PostgreSQL ──
pg_ctlcluster 14 main start 2>/dev/null || service postgresql start 2>/dev/null || true
sleep 1
pg_isready 2>/dev/null || err "PostgreSQL 未运行"
log "PostgreSQL 就绪"

# ═══════════════════════════════════════════════════════════
# 1. 确保数据目录链接
# ═══════════════════════════════════════════════════════════
AUTODL_TMP="/root/autodl-tmp"
for dir in indexes results logs; do
    mkdir -p "$AUTODL_TMP/$dir"
    [ -L "$dir" ] && rm -f "$dir"
    [ -d "$dir" ] && [ ! -L "$dir" ] && rm -rf "$dir"
    ln -sf "$AUTODL_TMP/$dir" "$dir"
done
log "数据目录 → $AUTODL_TMP"

# ═══════════════════════════════════════════════════════════
# 2. Ingest: 文本+视觉编码 → pgvector + FAISS
# ═══════════════════════════════════════════════════════════
log ""
log "=========================================="
log " 阶段 1/2: 数据导入 + 索引构建"
log "=========================================="
log "进度: logs/ingest.log"

> logs/ingest.log

python3 scripts/ingest_vidore.py \
    --dataset vidore/vidore_v3_industrial \
    2>&1 | tee -a logs/ingest.log
INGEST_RC=${PIPESTATUS[0]}

if [ "$INGEST_RC" -ne 0 ]; then
    err "Ingest 失败 (exit=$INGEST_RC)，查看 logs/ingest.log"
fi

log ""
log "✅ Ingest 完成"
log "  pgvector chunks: $(python3 -c "from src.store.pgvector_store import PgVectorStore; print(PgVectorStore().count())" 2>/dev/null || echo '?')"
log "  FAISS index: $(ls -lh indexes/*.faiss 2>/dev/null | awk '{print $5}')"

# ═══════════════════════════════════════════════════════════
# 3. Eval: ViDoRe 消融评测
# ═══════════════════════════════════════════════════════════
log ""
log "=========================================="
log " 阶段 2/2: ViDoRe 消融评测"
log "=========================================="
log "进度: logs/eval.log"

> logs/eval.log

python3 scripts/run_eval.py \
    --dataset vidore/vidore_v3_industrial \
    --skip-index \
    2>&1 | tee -a logs/eval.log
EVAL_RC=${PIPESTATUS[0]}

if [ "$EVAL_RC" -ne 0 ]; then
    err "Eval 失败 (exit=$EVAL_RC)，查看 logs/eval.log"
fi

# ═══════════════════════════════════════════════════════════
# 4. 产出汇总
# ═══════════════════════════════════════════════════════════
log ""
log "=========================================="
log " ✅ Phase 2 完成 $(date)"
log "=========================================="

log ""
log "📊 产出:"
log "  索引: $(du -sh indexes/ 2>/dev/null | cut -f1)"
ls -lhS indexes/ 2>/dev/null | grep -v "^total" | while read -r line; do log "    $line"; done

log ""
log "  结果: $(du -sh results/ 2>/dev/null | cut -f1)"
ls -lhS results/ 2>/dev/null | grep -v "^total" | while read -r line; do log "    $line"; done

log ""
log "  日志: $(du -sh logs/ 2>/dev/null | cut -f1)"

# 打印结果摘要
log ""
if [ -f results/ablation_results.json ]; then
    log "📈 消融结果摘要:"
    python3 -c "
import json
with open('results/ablation_results.json') as f:
    results = json.load(f)
print(f\"  {'Config':<25} {'NDCG@10':<10} {'Recall@5':<10} {'MRR':<10}\")
print(f\"  {'-'*55}\")
for r in results:
    print(f\"  {r['config']:<25} {r['ndcg@10']:<10.4f} {r['recall@5']:<10.4f} {r['mrr']:<10.4f}\")
" 2>/dev/null || true
fi

log ""
log "📦 拉回本地:"
log "  cd /path/to/pdf-rag && bash scripts/pull_from_cloud.sh"
log ""
log "⚠️  拉完记得去控制台关机！按时计费！"
log "=========================================="
