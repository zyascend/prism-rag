#!/usr/bin/env bash
# Boot-B: Visual 路由对照（always vs heuristic）+ 可选 RAGAS 100q
# 仅在云上 GPU 机执行。本地须先合入 visual_router（及可选 context_filter）。
#
# Usage:
#   bash scripts/cloud_boot_b.sh
# Env:
#   BOOT_DATE=20260720
#   MAX_QUERIES=100          # 预算紧用 100；全量 283 则留空并设 FULL=1
#   FULL=0|1                # 1 → 283q 且 expected-query-count
#   RUN_RAGAS=1|0           # 默认 1：跑 BGE 压缩 RAGAS
#   RUN_LLM_FILTER=0|1      # 1：再跑 context_filter.mode=llm|bge_then_llm（更贵）
#   VISUAL_MODEL=colqwen2
#   PYTHON=python
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-bootB}"
VISUAL_MODEL="${VISUAL_MODEL:-colqwen2}"
PYTHON="${PYTHON:-python}"
RUN_RAGAS="${RUN_RAGAS:-1}"
RUN_LLM_FILTER="${RUN_LLM_FILTER:-0}"
FULL="${FULL:-0}"
MAX_QUERIES="${MAX_QUERIES:-100}"
MODELS_BAK=""

restore_models_yaml() {
  if [[ -n "${MODELS_BAK}" && -f "${MODELS_BAK}" ]]; then
    mv -f "${MODELS_BAK}" config/models.yaml
    MODELS_BAK=""
  fi
}
trap restore_models_yaml EXIT

mkdir -p "$OUT/routing-always" "$OUT/routing-heuristic" "$OUT/ragas"

{
  echo "boot=B"
  echo "date=${BOOT_DATE}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "visual_model=${VISUAL_MODEL}"
  echo "full=${FULL}"
  echo "max_queries=${MAX_QUERIES}"
  echo "run_ragas=${RUN_RAGAS}"
  echo "run_llm_filter=${RUN_LLM_FILTER}"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT/env.txt"

run_eval_routing() {
  local mode="$1"
  local dest="$2"
  echo "==> Full_zerank with visual_routing.mode=${mode} → ${dest}"
  # 运行时覆盖：写临时 yaml snippet 不现实；用环境变量 + 小 Python 补丁配置
  VISUAL_ROUTING_MODE="$mode" "$PYTHON" - <<PY
import os, yaml
from pathlib import Path
p = Path("config/models.yaml")
data = yaml.safe_load(p.read_text())
data.setdefault("retrieval", {}).setdefault("visual_routing", {})
data["retrieval"]["visual_routing"]["enabled"] = True
data["retrieval"]["visual_routing"]["mode"] = os.environ["VISUAL_ROUTING_MODE"]
# 仅本次进程：通过 PRISM_CONFIG_JSON 不存在时，写 boot 旁路文件
Path("${dest}").mkdir(parents=True, exist_ok=True)
boot_cfg = Path("${dest}/models.boot.yaml")
boot_cfg.write_text(yaml.dump(data))
print("wrote", boot_cfg)
PY
  local args=(
    --skip-index
    --language en
    --visual-model "$VISUAL_MODEL"
    --config-filter Full_zerank
    --output-dir "$dest"
  )
  if [[ "$FULL" == "1" ]]; then
    args+=(--expected-query-count 283)
  else
    args+=(--max-queries "$MAX_QUERIES")
  fi
  # run_eval 只读默认 models.yaml：临时替换，trap 保证恢复
  MODELS_BAK="$OUT/models.yaml.bak"
  cp config/models.yaml "$MODELS_BAK"
  cp "${dest}/models.boot.yaml" config/models.yaml
  set +e
  "$PYTHON" scripts/run_eval.py "${args[@]}" 2>&1 | tee "${dest}/run.log"
  local rc=$?
  set -e
  restore_models_yaml
  if [[ $rc -ne 0 ]]; then
    echo "run_eval failed for mode=${mode}" >&2
    exit $rc
  fi
}

run_eval_routing always "$OUT/routing-always"
run_eval_routing heuristic "$OUT/routing-heuristic"

if [[ "$RUN_RAGAS" == "1" ]]; then
  echo "==> RAGAS (default context_filter.mode=bge) max_queries=${MAX_QUERIES}"
  # 确保 routing 关闭，避免与生成对照混淆（检索用默认 full visual）
  "$PYTHON" scripts/run_ragas_metrics.py --skip-index --max-queries "${MAX_QUERIES}" \
    --output-dir "$OUT/ragas/bge" 2>&1 | tee "$OUT/ragas/bge.log"
fi

if [[ "$RUN_LLM_FILTER" == "1" ]]; then
  echo "==> RAGAS with context_filter.mode=bge_then_llm (expensive)"
  MODELS_BAK="$OUT/models.yaml.bak"
  cp config/models.yaml "$MODELS_BAK"
  "$PYTHON" - <<'PY'
import yaml
from pathlib import Path
p = Path("config/models.yaml")
data = yaml.safe_load(p.read_text())
data.setdefault("context_filter", {})["mode"] = "bge_then_llm"
p.write_text(yaml.dump(data))
PY
  set +e
  "$PYTHON" scripts/run_ragas_metrics.py --skip-index --max-queries "${MAX_QUERIES}" \
    --output-dir "$OUT/ragas/bge_then_llm" \
    2>&1 | tee "$OUT/ragas/bge_then_llm.log"
  rc=$?
  set -e
  restore_models_yaml
  [[ $rc -eq 0 ]] || exit $rc
fi

{
  echo "# Boot-B summary (${BOOT_DATE})"
  echo
  echo "- git: \`$(git rev-parse HEAD 2>/dev/null || echo unknown)\`"
  echo "- routing: always vs heuristic under \`${OUT}/routing-*\`"
  echo "- ragas: \`${OUT}/ragas/\`"
  echo
  echo "## Fill in"
  echo
  echo "| mode | NDCG@10 | avg_latency_ms | visual_skip_rate |"
  echo "|------|---------|----------------|------------------|"
  echo "| always | | | 0% |"
  echo "| heuristic | | | |"
  echo
  echo "Resume: 对非图表 query 跳过 Visual，延迟 -X% 且 NDCG 变化 < Z"
} | tee "$OUT/summary.md"

echo "==> Boot-B finished: $OUT — pull results then shutdown GPU."
