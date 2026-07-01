#!/usr/bin/env bash
# ============================================================
# 从 AutoDL 云端拉取全部产出到本地
# ============================================================
# 用法:
#   方式 1（自动读取连接信息）:
#     bash scripts/pull_from_cloud.sh
#     首次运行会提示输入 SSH 连接信息并保存到 .cloud-connection
#
#   方式 2（命令行指定）:
#     bash scripts/pull_from_cloud.sh <host> <port> <password>
#     示例: bash scripts/pull_from_cloud.sh connect.cqa1.seetacloud.com 14215 insIDvdAmzip
#
# 产出:
#   - results/    ← 消融评测结果
#   - indexes/    ← FAISS 索引 + id_map
#   - logs/       ← 运行日志
# ============================================================
set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[pull]${NC} $*"; }

# ── 读取连接信息 ───────────────────────────────────────────
CONFIG_FILE="$(dirname "$0")/../.cloud-connection"

if [ $# -ge 3 ]; then
    HOST="$1"
    PORT="$2"
    PASSWORD="$3"
else
    # 尝试从配置文件读取
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
        log "读取连接信息: $CLOUD_HOST:$CLOUD_PORT"
        HOST="${CLOUD_HOST:-}"
        PORT="${CLOUD_PORT:-}"
        PASSWORD="${CLOUD_PASSWORD:-}"
    fi

    # 如果还是空，交互式输入
    if [ -z "${HOST:-}" ] || [ -z "${PORT:-}" ]; then
        echo "首次使用，请输入 AutoDL 实例连接信息:"
        read -r -p "  SSH Host (如 connect.cqa1.seetacloud.com): " HOST
        read -r -p "  SSH Port (如 14215): " PORT
        read -r -s -p "  SSH Password: " PASSWORD
        echo ""
    fi
fi

# 验证
if [ -z "${HOST:-}" ] || [ -z "${PORT:-}" ] || [ -z "${PASSWORD:-}" ]; then
    echo "用法: bash scripts/pull_from_cloud.sh <host> <port> <password>"
    echo "示例: bash scripts/pull_from_cloud.sh connect.cqa1.seetacloud.com 14215 insIDvdAmzip"
    exit 1
fi

# ── 保存连接信息（方便下次使用） ──────────────────────────
if [ ! -f "$CONFIG_FILE" ] || [ "${SAVE_CONNECTION:-1}" = "1" ]; then
    cat > "$CONFIG_FILE" << EOF
# AutoDL 连接信息（自动生成，请勿提交到 Git）
CLOUD_HOST="$HOST"
CLOUD_PORT="$PORT"
CLOUD_PASSWORD="$PASSWORD"
EOF
    chmod 600 "$CONFIG_FILE"
    log "连接信息已保存到 $CONFIG_FILE（.gitignore 已排除）"
fi

# ── 检查 sshpass ───────────────────────────────────────────
if ! command -v sshpass &>/dev/null; then
    echo "需要 sshpass，正在安装..."
    if command -v brew &>/dev/null; then
        brew install hudochenkov/sshpass/sshpass
    else
        echo "请手动安装 sshpass:"
        echo "  macOS: brew install hudochenkov/sshpass/sshpass"
        echo "  Linux: sudo apt-get install sshpass"
        exit 1
    fi
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT"
REMOTE="root@$HOST"
REMOTE_DIR="/root/prism-rag"

# ── 连接测试 ───────────────────────────────────────────────
log "连接 $REMOTE:$PORT..."
if ! sshpass -p "$PASSWORD" ssh $SSH_OPTS "$REMOTE" "echo connected" 2>/dev/null; then
    echo "❌ SSH 连接失败，请检查 Host/Port/Password"
    exit 1
fi
log "连接成功"

# ── 拉取 ───────────────────────────────────────────────────
SSH_CMD="sshpass -p '$PASSWORD' ssh $SSH_OPTS '$REMOTE'"

log ""
log "====== 拉取云端产出 ======"
log ""

# 1. 结果
log "[1/3] 拉取评测结果..."
mkdir -p results
rsync -avz --progress -e "sshpass -p '$PASSWORD' ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/results/" ./results/ 2>&1 || log "  (无 results/ 目录，跳过)"
log "  ✅ results/"

# 2. 索引
log "[2/3] 拉取 FAISS 索引..."
mkdir -p indexes
rsync -avz --progress -e "sshpass -p '$PASSWORD' ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/indexes/" ./indexes/ 2>&1 || log "  (无 indexes/ 目录，跳过)"
log "  ✅ indexes/"

# 3. 日志
log "[3/3] 拉取日志..."
mkdir -p logs
rsync -avz --progress -e "sshpass -p '$PASSWORD' ssh $SSH_OPTS" \
    "$REMOTE:$REMOTE_DIR/logs/" ./logs/ 2>&1 || log "  (无 logs/ 目录，跳过)"
log "  ✅ logs/"

# ── 统计 ───────────────────────────────────────────────────
log ""
log "====== 产出统计 ======"
log "  results: $(du -sh results/ 2>/dev/null | cut -f1)"
log "  indexes: $(du -sh indexes/ 2>/dev/null | cut -f1)"
log "  logs:    $(du -sh logs/ 2>/dev/null | cut -f1)"

log ""
log "====== 拉取完成 ✅ ======"
log ""
log "本地可直接运行:"
log "  make eval-vidore-skip-index    # 用拉回的索引做评测"
log ""
log "⚠️  数据已拉回本地，记得去 AutoDL 控制台关机！"
