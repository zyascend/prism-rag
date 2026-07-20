#!/usr/bin/env bash
# Boot-A: 黄金消融（GOLDEN_NO_HYDE）+ 可选 Full_zerank 漂移复跑
# 仅在云上 GPU 机执行。本地全量禁止（见 Agents.md / docs/eval-protocol.md）。
#
# Usage:
#   bash scripts/cloud_boot_a.sh
# Env:
#   BOOT_DATE=20260720          # 默认当天 YYYYMMDD
#   MAX_QUERIES=                # 空=全量 283；调试可 10
#   SKIP_DRIFT=0|1              # 1=跳过第二次 Full_zerank
#   VISUAL_MODEL=colqwen2
#   PYTHON=python               # 或 .venv/bin/python
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-bootA}"
VISUAL_MODEL="${VISUAL_MODEL:-colqwen2}"
PYTHON="${PYTHON:-python}"
SKIP_DRIFT="${SKIP_DRIFT:-0}"
MAX_QUERIES="${MAX_QUERIES:-}"

mkdir -p "$OUT/golden-ablation" "$OUT/incremental"

{
  echo "boot=A"
  echo "date=${BOOT_DATE}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "visual_model=${VISUAL_MODEL}"
  echo "max_queries=${MAX_QUERIES:-full283}"
  echo "host=$(hostname)"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT/env.txt"

echo "==> Job1: GOLDEN_NO_HYDE ablation → $OUT/golden-ablation"
JOB1_ARGS=(
  --skip-index
  --language en
  --visual-model "$VISUAL_MODEL"
  --no-hyde
  --output-dir "$OUT/golden-ablation"
)
if [[ -z "${MAX_QUERIES}" ]]; then
  JOB1_ARGS+=(--expected-query-count 283)
else
  JOB1_ARGS+=(--max-queries "$MAX_QUERIES")
fi

"$PYTHON" scripts/run_eval.py "${JOB1_ARGS[@]}" 2>&1 | tee "$OUT/golden-ablation/run.log"

if [[ "$SKIP_DRIFT" != "1" ]]; then
  echo "==> Job2c: Full_zerank drift re-run (same index) → $OUT/incremental/drift_eval"
  JOB2_ARGS=(
    --skip-index
    --language en
    --visual-model "$VISUAL_MODEL"
    --config-filter Full_zerank
    --output-dir "$OUT/incremental/drift_eval"
  )
  if [[ -z "${MAX_QUERIES}" ]]; then
    JOB2_ARGS+=(--expected-query-count 283)
  else
    JOB2_ARGS+=(--max-queries "$MAX_QUERIES")
  fi
  "$PYTHON" scripts/run_eval.py "${JOB2_ARGS[@]}" 2>&1 | tee "$OUT/incremental/drift_eval.log"
else
  echo "==> SKIP_DRIFT=1: skip second Full_zerank"
fi

# 从结果 JSON 抽一行摘要（若存在）
SUMMARY="$OUT/summary.md"
{
  echo "# Boot-A summary (${BOOT_DATE})"
  echo
  echo "- git: \`$(git rev-parse HEAD 2>/dev/null || echo unknown)\`"
  echo "- visual: \`${VISUAL_MODEL}\`"
  echo "- protocol: \`docs/eval-protocol.md\` v1"
  echo "- golden dir: \`${OUT}/golden-ablation\`"
  echo "- drift dir: \`${OUT}/incremental/drift_eval\`"
  echo
  echo "## Next (manual)"
  echo
  echo "1. Fill \`ndcg_table.md\` from golden raw results."
  echo "2. Ghost-delete check + page-diff timing → \`incremental/README.md\` (see docs/incremental-verification-runbook.md)."
  echo "3. Compare Job1 Full_zerank2 vs drift_eval Full_zerank2; expect |ΔNDCG@10| < 0.005."
  echo "4. scp results, shutdown GPU."
} | tee "$SUMMARY"

echo "==> Boot-A pipeline finished: $OUT"
echo "    Remember: pull results, then shutdown the instance."
