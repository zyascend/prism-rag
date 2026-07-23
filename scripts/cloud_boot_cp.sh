#!/usr/bin/env bash
# Boot-CP: Content Pipeline Phase A/B 云验收（默认 1 次开机）
# skip-index：验证 B1/B2 与代码加载不回归；A1/A3 需 re-ingest 后另开 Arm。
#
# Usage (cloud GPU only):
#   source scripts/cloud_env.sh
#   bash scripts/cloud_boot_cp.sh
# Env:
#   BOOT_DATE=20260723
#   MAX_QUERIES=100
#   FULL=0|1              # 1 → 283q
#   VISUAL_MODEL=colqwen2
#   PYTHON=python
#   RUN_E2E=0|1           # 默认 0；1 则在各臂后再跑 E2E（更贵）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-content-pipeline}"
VISUAL_MODEL="${VISUAL_MODEL:-colqwen2}"
PYTHON="${PYTHON:-python}"
FULL="${FULL:-0}"
MAX_QUERIES="${MAX_QUERIES:-100}"
RUN_E2E="${RUN_E2E:-0}"
MODELS_BAK=""

restore_models_yaml() {
  if [[ -n "${MODELS_BAK}" && -f "${MODELS_BAK}" ]]; then
    mv -f "${MODELS_BAK}" config/models.yaml
    MODELS_BAK=""
  fi
}
trap restore_models_yaml EXIT

mkdir -p "$OUT"/{arm-A,arm-B1,arm-B2}

{
  echo "boot=CP"
  echo "date=${BOOT_DATE}"
  echo "mode=skip-index-three-arm"
  echo "visual_model=${VISUAL_MODEL}"
  echo "full=${FULL}"
  echo "max_queries=${MAX_QUERIES}"
  echo "run_e2e=${RUN_E2E}"
  echo "host=$(hostname)"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo "note=A1/A3 metadata empty until text re-ingest; B1 page-mode + B2 use existing chunk_type"
} | tee "$OUT/env.txt"

write_boot_cfg() {
  local dest="$1"
  local expand_on="$2"   # true|false
  local boost_on="$3"    # true|false
  mkdir -p "$dest"
  EXPAND_ON="$expand_on" BOOST_ON="$boost_on" DEST="$dest" "$PYTHON" - <<'PY'
import os, yaml
from pathlib import Path
p = Path("config/models.yaml")
data = yaml.safe_load(p.read_text())
ret = data.setdefault("retrieval", {})
ne = ret.setdefault("neighbor_expand", {})
ne["enabled"] = os.environ["EXPAND_ON"] == "true"
ne.setdefault("mode", "page")
ne.setdefault("max_extra", 2)
ne.setdefault("stage", "post_rerank")
mb = ret.setdefault("modality_boost", {})
mb["enabled"] = os.environ["BOOST_ON"] == "true"
mb.setdefault("table_score_bonus", 0.02)
mb.setdefault("image_score_bonus", 0.02)
mb.setdefault("force_visual_on_visual_intent", False)
# keep ingestion defaults (context off for skip-index arm)
ing = data.setdefault("ingestion", {})
ing.setdefault("table_summary_context_enabled", False)
boot = Path(os.environ["DEST"]) / "models.boot.yaml"
boot.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False))
print("wrote", boot)
print("expand", ne["enabled"], "boost", mb["enabled"])
PY
}

run_arm() {
  local name="$1"
  local dest="$2"
  local expand_on="$3"
  local boost_on="$4"
  echo ""
  echo "========== Arm ${name}: expand=${expand_on} boost=${boost_on} =========="
  write_boot_cfg "$dest" "$expand_on" "$boost_on"
  MODELS_BAK="$OUT/models.yaml.bak"
  cp -f config/models.yaml "$MODELS_BAK"
  cp -f "${dest}/models.boot.yaml" config/models.yaml

  local args=(
    --skip-index
    --language en
    --visual-model "$VISUAL_MODEL"
    --config-filter Full_zerank2
    --no-hyde
    --output-dir "$dest"
  )
  if [[ "$FULL" == "1" ]]; then
    args+=(--expected-query-count 283)
  else
    args+=(--max-queries "$MAX_QUERIES")
  fi

  set +e
  "$PYTHON" scripts/run_eval.py "${args[@]}" 2>&1 | tee "${dest}/run.log"
  local rc=$?
  set -e
  restore_models_yaml
  if [[ $rc -ne 0 ]]; then
    echo "Arm ${name} FAILED rc=${rc}" >&2
    exit $rc
  fi

  if [[ "$RUN_E2E" == "1" ]]; then
    echo "==> E2E arm ${name}"
    MODELS_BAK="$OUT/models.yaml.bak"
    cp -f config/models.yaml "$MODELS_BAK"
    cp -f "${dest}/models.boot.yaml" config/models.yaml
    set +e
    "$PYTHON" scripts/run_e2e_qa.py --skip-index --visual-model "$VISUAL_MODEL" \
      --output-dir "${dest}/e2e" 2>&1 | tee "${dest}/e2e.log"
    set -e
    restore_models_yaml
  fi
}

# 确保 PG schema 有 A3 列
"$PYTHON" - <<'PY'
from src.config import cfg
cfg.load()
from src.store.pgvector_store import PgVectorStore
PgVectorStore().create_schema()
print("pg schema ok, chunks=", PgVectorStore().count())
PY

run_arm "A"  "$OUT/arm-A"  false false
run_arm "B1" "$OUT/arm-B1" true  false
run_arm "B2" "$OUT/arm-B2" false true

# 摘要
SUMMARY="$OUT/README.md"
{
  echo "# Boot-CP summary (${BOOT_DATE})"
  echo
  echo "- mode: skip-index · Full_zerank2 · max_queries=\`${MAX_QUERIES}\` (FULL=${FULL})"
  echo "- visual: \`${VISUAL_MODEL}\`"
  echo "- arms: A (off/off) · B1 (expand) · B2 (boost)"
  echo "- note: section_path/neighbors 在旧索引中为空；B1 用 **page** 模式仍可扩同页 chunk"
  echo
  echo "## NDCG@10 (from ablation_results.json)"
  echo
  "$PYTHON" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
for arm in ["arm-A", "arm-B1", "arm-B2"]:
    p = out / arm / "ablation_results.json"
    if not p.exists():
        # some runs nest under results
        cands = list((out / arm).rglob("ablation_results.json"))
        p = cands[0] if cands else None
    if not p or not p.exists():
        print(f"| {arm} | MISSING |")
        continue
    data = json.loads(p.read_text())
    # data may be list of configs or dict
    rows = data if isinstance(data, list) else data.get("results", data.get("configs", [data]))
    if isinstance(rows, dict):
        rows = [rows]
    ndcg = None
    for r in rows:
        name = r.get("name") or r.get("config") or ""
        if "zerank" in str(name).lower() or name == "Full_zerank2":
            ndcg = r.get("ndcg@10") or r.get("ndcg_at_10") or r.get("metrics", {}).get("ndcg@10")
            break
    if ndcg is None and rows:
        r0 = rows[0]
        ndcg = r0.get("ndcg@10") or r0.get("ndcg_at_10")
    print(f"| {arm} | {ndcg} | file={p}")
PY
  echo
  echo "## Next"
  echo
  echo "1. Compare Arm-A vs Boot-A Full_zerank2 baseline"
  echo "2. Compare B1/B2 vs Arm-A"
  echo "3. Optional: text re-ingest for A1/A3 then re-run Arm-A"
  echo "4. scp results, shutdown GPU"
} | tee "$SUMMARY"

echo "==> Boot-CP finished: $OUT"
echo "    Remember: pull results, then shutdown the instance."
