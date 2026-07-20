#!/usr/bin/env bash
# 本地打包代码 → 云上 scp。故意不带 indexes / HF / 大 run，避免覆盖云端 symlink。
#
# 用法:
#   bash scripts/pack_for_cloud.sh
#   SSHPASS='...' bash scripts/pack_for_cloud.sh --upload -p 44683 root@connect.cqa1.seetacloud.com
#
# 上传后在云上执行:
#   bash scripts/cloud_apply_upload.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-/tmp/prism-rag-code.tar.gz}"
UPLOAD=0
SCP_PORT=""
SCP_TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload) UPLOAD=1; shift ;;
    -p) SCP_PORT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -*)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
    *)
      SCP_TARGET="$1"
      shift
      ;;
  esac
done

# macOS: 禁止打包 ._ 资源叉 / xattr（Linux 解压会刷屏 LIBARCHIVE 警告）
export COPYFILE_DISABLE=1
TAR_EXTRA=()
if tar --help 2>&1 | grep -q -- '--no-xattrs'; then
  TAR_EXTRA+=(--no-xattrs)
fi
if tar --help 2>&1 | grep -q -- '--no-mac-metadata'; then
  TAR_EXTRA+=(--no-mac-metadata)
fi

echo "==> Packing $ROOT → $OUT"
# 明确排除：索引、缓存、虚拟环境、巨型产物（云端已有或在数据盘）
tar czf "$OUT" "${TAR_EXTRA[@]}" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='.codebase-memory' \
  --exclude='.firecrawl' \
  --exclude='.superpowers' \
  --exclude='.workbuddy' \
  --exclude='indexes' \
  --exclude='indexes/*' \
  --exclude='results' \
  --exclude='results/*' \
  --exclude='logs' \
  --exclude='logs/*' \
  --exclude='runs' \
  --exclude='runs/*' \
  --exclude='data/raw' \
  --exclude='data/processed' \
  --exclude='data/interim' \
  --exclude='models' \
  --exclude='.huggingface' \
  --exclude='hf_cache' \
  --exclude='storage' \
  --exclude='minio_data' \
  --exclude='*.faiss' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='page_embeddings_cache.pkl' \
  --exclude='local' \
  --exclude='*.egg-info' \
  --exclude='.env' \
  --exclude='.cloud-connection' \
  -C "$ROOT" \
  .

ls -lh "$OUT"
echo "==> Archive listing (top-level):"
tar tzf "$OUT" | head -40
echo "..."
# 安全检查：包内不得出现真实 indexes 内容
if tar tzf "$OUT" | grep -E '^(\./)?indexes/[^/]+' | grep -v '/$' | head -5 | grep -q .; then
  echo "ERROR: tarball contains files under indexes/ — abort" >&2
  exit 1
fi
echo "==> OK: no indexes/* payload in tarball"

if [[ "$UPLOAD" -eq 1 ]]; then
  if [[ -z "$SCP_TARGET" || -z "$SCP_PORT" ]]; then
    echo "Usage: SSHPASS=... $0 --upload -p PORT user@host" >&2
    exit 1
  fi
  if [[ -z "${SSHPASS:-}" ]]; then
    echo "Set SSHPASS for non-interactive scp" >&2
    exit 1
  fi
  echo "==> scp → ${SCP_TARGET}:/root/prism-rag-code.tar.gz (port $SCP_PORT)"
  sshpass -e scp -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password \
    -o PubkeyAuthentication=no -P "$SCP_PORT" "$OUT" "${SCP_TARGET}:/root/prism-rag-code.tar.gz"
  echo "==> Upload done. On cloud run:"
  echo "    cd /root/prism-rag && bash scripts/cloud_apply_upload.sh"
  echo "    # 若 scripts 尚未更新，先:"
  echo "    tar xzf /root/prism-rag-code.tar.gz -C /root/prism-rag scripts/cloud_apply_upload.sh scripts/cloud_env.sh"
  echo "    bash scripts/cloud_apply_upload.sh"
fi
