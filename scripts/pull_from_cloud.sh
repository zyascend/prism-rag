#!/usr/bin/env bash
# 从 AutoDL 云端拉取全部产出到本地
# 用法（在本地 Mac 执行）:
#   bash scripts/pull_from_cloud.sh <host> <port>
# 示例:
#   bash scripts/pull_from_cloud.sh region-1.autodl.com 12345

set -euo pipefail

HOST="${1:-}"
PORT="${2:-}"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "用法: bash scripts/pull_from_cloud.sh <host> <port>"
    echo "示例: bash scripts/pull_from_cloud.sh region-1.autodl.com 12345"
    exit 1
fi

SSH_OPTS="-p $PORT -o StrictHostKeyChecking=no"
REMOTE="root@$HOST"
REMOTE_DIR="/root/prism-rag"   # AutoDL 默认 /root

echo "========================================"
echo " 从 AutoDL 拉取产出"
echo " $REMOTE:$PORT → 本地"
echo "========================================"

# ── 1. 拉取评测结果（小文件，秒级） ──
echo ""
echo "[1/4] 拉取评测结果..."
rsync -avz --progress -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/results/" ./results/
echo "  ✅ results/"

# ── 2. 拉取索引（大文件，可能需要几分钟） ──
echo ""
echo "[2/4] 拉取 FAISS 索引..."
rsync -avz --progress -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/indexes/" ./indexes/
echo "  ✅ indexes/"

# ── 3. 拉取日志 ──
echo ""
echo "[3/4] 拉取日志..."
rsync -avz --progress -e "ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/logs/" ./logs/ 2>/dev/null || echo "  (无日志目录)"
echo "  ✅ logs/"

# ── 4. 大小统计 ──
echo ""
echo "[4/4] 产出统计:"
echo "  results: $(du -sh results/ 2>/dev/null | cut -f1)"
echo "  indexes: $(du -sh indexes/ 2>/dev/null | cut -f1)"

echo ""
echo "========================================"
echo " ✅ 全部拉取完成"
echo " 本地可直接运行: make eval-vidore-skip-index"
echo "========================================"
