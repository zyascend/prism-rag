#!/usr/bin/env bash
# 云上环境变量：数据盘缓存 + 默认强制 HF/Transformers/Datasets 离线
# 用法: source scripts/cloud_env.sh
# 若必须临时联网下载: HF_ALLOW_ONLINE=1 source scripts/cloud_env.sh

# 可被重复 source
AUTODL_TMP="${AUTODL_TMP:-/root/autodl-tmp}"
export HF_HOME="${HF_HOME:-$AUTODL_TMP/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$AUTODL_TMP/torch}"
# 关闭 hf_transfer / Xet 怪异问题
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if [[ "${HF_ALLOW_ONLINE:-0}" == "1" ]]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE 2>/dev/null || true
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  echo "[cloud_env] ONLINE allowed: HF_ENDPOINT=$HF_ENDPOINT HF_HOME=$HF_HOME"
else
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  echo "[cloud_env] OFFLINE: HF_HOME=$HF_HOME (set HF_ALLOW_ONLINE=1 to download)"
fi

# Python 选择：优先项目 .python_bin → conda → python3
if [[ -f /root/prism-rag/.python_bin ]]; then
  # shellcheck disable=SC1091
  export PATH="$(dirname "$(cat /root/prism-rag/.python_bin)"):$PATH"
fi
if [[ -x /root/miniconda3/bin/python ]]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi

# 工作目录提示
export PRISM_ROOT="${PRISM_ROOT:-/root/prism-rag}"
