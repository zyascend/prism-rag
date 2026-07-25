#!/usr/bin/env bash
# Boot-R: P0 Refiner soft_rank + P1 Search Planning + Crossref Expand 云验收
# 方案: docs/superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md
#
# 默认 1 次开机、多臂、--skip-index（不改 embed，无需 re-index）。
# 冻结: CRAG off · Gate2 off · neighbor_expand/boost 关 · eval_via_generator=true
#
# Usage（云 GPU）:
#   source scripts/cloud_env.sh
#   service postgresql start   # 或 pg_ctlcluster …
#   pgrep -x ollama >/dev/null || nohup ollama serve &
#   bash scripts/cloud_boot_r.sh
#
# Env:
#   BOOT_DATE=20260725
#   OUT=runs/20260725-boot-r
#   MAX_QUERIES=100
#   VISUAL_MODEL=colqwen2
#   PYTHON=python
#   RUN_RAGAS=1|0          # 默认 1
#   RUN_E2E=1|0            # 默认 1
#   ARMS=base,r1,pl,cr     # 默认四臂；可加 best
#   SKIP_BASE=0|1 …
#   FAIL_FAST=1|0          # 默认 1：一臂失败即停；0=继续
#   SOFT_KEEP_RATIO=0.5
#   SOFT_MIN_SIM=0.25
#
# 本地: Agents.md 禁止全量；MAX_QUERIES=3 冒烟仅当有索引+LLM。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
[[ -f scripts/cloud_env.sh ]] && source scripts/cloud_env.sh

BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-boot-r}"
VISUAL_MODEL="${VISUAL_MODEL:-colqwen2}"
PYTHON="${PYTHON:-python}"
MAX_QUERIES="${MAX_QUERIES:-100}"
RUN_RAGAS="${RUN_RAGAS:-1}"
RUN_E2E="${RUN_E2E:-1}"
ARMS="${ARMS:-base,r1,pl,cr}"
FAIL_FAST="${FAIL_FAST:-1}"
SOFT_KEEP_RATIO="${SOFT_KEEP_RATIO:-0.5}"
SOFT_MIN_SIM="${SOFT_MIN_SIM:-0.25}"
MODELS_BAK=""

restore_models_yaml() {
  if [[ -n "${MODELS_BAK}" && -f "${MODELS_BAK}" ]]; then
    mv -f "${MODELS_BAK}" config/models.yaml
    MODELS_BAK=""
    echo "[boot-r] restored config/models.yaml"
  fi
}
trap restore_models_yaml EXIT

mkdir -p "$OUT"

{
  echo "boot=R"
  echo "job=boot_r_refiner_planning_crossref"
  echo "date=${BOOT_DATE}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "visual_model=${VISUAL_MODEL}"
  echo "max_queries=${MAX_QUERIES}"
  echo "run_ragas=${RUN_RAGAS}"
  echo "run_e2e=${RUN_E2E}"
  echo "arms=${ARMS}"
  echo "soft_keep_ratio=${SOFT_KEEP_RATIO}"
  echo "soft_min_sim=${SOFT_MIN_SIM}"
  echo "host=$(hostname 2>/dev/null || echo unknown)"
  echo "skip_index=1"
  echo "plan=docs/superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT/env.txt"

# ── 写臂配置：在仓库 models.yaml 基础上只改实验变量 ──────────
# arm: base | r1 | pl | cr | best
write_boot_cfg() {
  local arm="$1"
  local dest_yaml="$2"
  ARM="$arm" DEST_YAML="$dest_yaml" \
  SOFT_KEEP_RATIO="$SOFT_KEEP_RATIO" SOFT_MIN_SIM="$SOFT_MIN_SIM" \
  "$PYTHON" - <<'PY'
import os, yaml
from pathlib import Path

arm = os.environ["ARM"].strip().lower()
data = yaml.safe_load(Path("config/models.yaml").read_text()) or {}

# ── 冻结：评测走 Generator；CRAG / Gate2 关；expand/boost 关 ──
gen = data.setdefault("generation", {})
gen["eval_via_generator"] = True
sr = gen.setdefault("self_rag", {})
sr["enabled"] = False

ret = data.setdefault("retrieval", {})
crag = ret.setdefault("crag", {})
crag["enabled"] = False
vr = ret.setdefault("visual_routing", {})
vr["enabled"] = False
ne = ret.setdefault("neighbor_expand", {})
ne["enabled"] = False
mb = ret.setdefault("modality_boost", {})
mb["enabled"] = False

# 默认全关实验位
ref = data.setdefault("refiner", {})
ref["enabled"] = True
ref["mode"] = "bge"
ref["protect_table_chunks"] = True
ref["emit_trace"] = True
soft = ref.setdefault("soft_rank", {})
soft["min_sim"] = float(os.environ.get("SOFT_MIN_SIM", "0.25"))
soft["prune_below"] = None
soft["keep_ratio"] = float(os.environ.get("SOFT_KEEP_RATIO", "0.5"))
ref.setdefault("adaptive_ratio", {})["enabled"] = False
data.setdefault("context_filter", {})["mode"] = "bge"

plan = ret.setdefault("search_planning", {})
plan["enabled"] = False
plan["mode"] = "heuristic"
plan["allow_skip_retrieval"] = False
plan["table_prefers_text"] = True
vis = plan.setdefault("visual", {})
vis["on_cues"] = True
vis["default_visual"] = False

xref = ret.setdefault("crossref_expand", {})
xref["enabled"] = False
xref["stage"] = "post_rerank"
xref["max_extra"] = 3
xref["max_per_hit"] = 1
xref["same_doc_only"] = True

# 云上 LLM 走 Ollama（与历史 A/B 一致）
llm = data.setdefault("llm", {})
llm.setdefault("base_url", os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
llm.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "ollama"))
llm.setdefault(
    "model",
    os.environ.get("LLM_MODEL", data.get("models", {}).get("llm", "qwen2:7b")),
)

if arm == "base":
    pass  # 现状
elif arm == "r1":
    ref["mode"] = "soft_rank"
    data["context_filter"]["mode"] = "soft_rank"
elif arm == "pl":
    plan["enabled"] = True
    plan["mode"] = "heuristic"
    vis["default_visual"] = False
elif arm == "cr":
    xref["enabled"] = True
elif arm == "best":
    ref["mode"] = "soft_rank"
    data["context_filter"]["mode"] = "soft_rank"
    plan["enabled"] = True
    plan["mode"] = "heuristic"
    vis["default_visual"] = False
    xref["enabled"] = True
else:
    raise SystemExit(f"unknown arm: {arm!r} (base|r1|pl|cr|best)")

out = Path(os.environ["DEST_YAML"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
print(
    "wrote", out,
    "arm=", arm,
    "refiner.mode=", ref.get("mode"),
    "planning=", plan.get("enabled"),
    "crossref=", xref.get("enabled"),
    "llm=", llm.get("model"),
)
PY
}

should_skip_arm() {
  local arm="$1"
  local var="SKIP_${arm^^}"
  # bash 3.2 兼容：不用 ${arm^^} 在 mac；云上 bash4+ 有
  case "$arm" in
    base) var=SKIP_BASE ;;
    r1)   var=SKIP_R1 ;;
    pl)   var=SKIP_PL ;;
    cr)   var=SKIP_CR ;;
    best) var=SKIP_BEST ;;
    *)    var="SKIP_UNKNOWN" ;;
  esac
  local val="${!var:-0}"
  [[ "$val" == "1" ]]
}

run_arm() {
  local arm="$1"
  local dest="$OUT/${arm}"
  mkdir -p "$dest/ragas" "$dest/e2e"

  if should_skip_arm "$arm"; then
    echo "==> SKIP arm=${arm} (SKIP_*=1)"
    return 0
  fi

  echo ""
  echo "========== Boot-R arm=${arm} → ${dest} =========="
  write_boot_cfg "$arm" "${dest}/models.boot.yaml"

  MODELS_BAK="$OUT/models.yaml.bak"
  cp -f config/models.yaml "$MODELS_BAK"
  cp -f "${dest}/models.boot.yaml" config/models.yaml

  local rc=0
  if [[ "$RUN_RAGAS" == "1" ]]; then
    echo "  → RAGAS max_queries=${MAX_QUERIES} (eval_via_generator=true)"
    set +e
    "$PYTHON" scripts/run_ragas_metrics.py \
      --skip-index \
      --language en \
      --visual-model "$VISUAL_MODEL" \
      --max-queries "$MAX_QUERIES" \
      --output-dir "$dest/ragas" \
      2>&1 | tee "$dest/ragas/run.log"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      restore_models_yaml
      echo "RAGAS FAILED arm=${arm} rc=${rc}" | tee -a "$OUT/failures.log"
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit "$rc"
      fi
    fi
  fi

  if [[ "$RUN_E2E" == "1" ]]; then
    echo "  → E2E QA"
    set +e
    "$PYTHON" scripts/run_e2e_qa.py \
      --skip-index \
      --visual-model "$VISUAL_MODEL" \
      --output-dir "$dest/e2e" \
      2>&1 | tee "$dest/e2e/run.log"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      restore_models_yaml
      echo "E2E FAILED arm=${arm} rc=${rc}" | tee -a "$OUT/failures.log"
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit "$rc"
      fi
    fi
  fi

  restore_models_yaml
  echo "  ✓ arm ${arm} done"
}

# ── 预检 ──────────────────────────────────────────────────
preflight() {
  echo "==> preflight"
  if command -v pg_isready >/dev/null 2>&1; then
    if ! pg_isready -q 2>/dev/null; then
      echo "  starting postgresql..."
      pg_ctlcluster 14 main start 2>/dev/null \
        || service postgresql start 2>/dev/null \
        || true
    fi
  fi
  if ! pgrep -x ollama >/dev/null 2>&1; then
    echo "  WARN: ollama not running — start: nohup ollama serve &"
  fi
  "$PYTHON" - <<'PY'
from src.config import cfg
cfg.load()
from src.store.pgvector_store import PgVectorStore
PgVectorStore().create_schema()
n = PgVectorStore().count()
print(f"  pg chunks={n}")
if n < 100:
    print("  WARN: few chunks; confirm index loaded")
PY
}

preflight

# ── 跑臂 ──────────────────────────────────────────────────
IFS=',' read -ra ARM_LIST <<< "$ARMS"
for arm in "${ARM_LIST[@]}"; do
  arm="$(echo "$arm" | tr -d '[:space:]')"
  [[ -z "$arm" ]] && continue
  run_arm "$arm"
done

# ── 摘要 ──────────────────────────────────────────────────
SUMMARY="$OUT/README.md"
{
  echo "# Boot-R summary (${BOOT_DATE})"
  echo
  echo "| 项 | 值 |"
  echo "|----|----|"
  echo "| git | \`$(git rev-parse HEAD 2>/dev/null || echo unknown)\` |"
  echo "| arms | \`${ARMS}\` |"
  echo "| visual | \`${VISUAL_MODEL}\` |"
  echo "| RAGAS | max_queries=\`${MAX_QUERIES}\` · skip-index |"
  echo "| 冻结 | eval_via_generator=true · CRAG off · Gate2 off · expand/boost off |"
  echo "| 方案 | [p0-p1-refiner-planning-impl](../../docs/superpowers/plans/2026-07-25-p0-p1-refiner-planning-impl.md) |"
  echo
  echo "## 臂含义"
  echo
  echo "| arm | 变更 |"
  echo "|-----|------|"
  echo "| base | refiner=bge · planning off · crossref off |"
  echo "| r1 | refiner=**soft_rank** (prune=null) |"
  echo "| pl | search_planning heuristic · default_visual=false |"
  echo "| cr | crossref_expand on · post_rerank |"
  echo "| best | soft_rank + planning + crossref（可选） |"
  echo
  echo "## 指标（自动解析；缺文件则 —）"
  echo
  echo "| arm | Faith | Rel | CtxRel | RAGAS拒答 | E2E Correct | E2E Reject | E2E latency | 误拒(可答) |"
  echo "|-----|------:|----:|-------:|----------:|------------:|-----------:|------------:|-----------:|"
  "$PYTHON" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
arms = [a.strip() for a in "${ARMS}".split(",") if a.strip()]

def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def ragas_row(d):
    if not d:
        return "—", "—", "—", "—"
    s = d.get("summary") or d
    faith = s.get("faithfulness") or s.get("faithfulness_mean") or d.get("faithfulness")
    if isinstance(faith, dict):
        faith = faith.get("mean") or faith.get("score")
    rel = s.get("answer_relevancy") or s.get("relevancy") or d.get("relevancy")
    if isinstance(rel, dict):
        rel = rel.get("mean") or rel.get("score")
    ctx = s.get("context_relevancy") or s.get("context_relevance") or d.get("context_relevancy")
    if isinstance(ctx, dict):
        ctx = ctx.get("mean") or ctx.get("score")
    rej = s.get("rejected_count") or s.get("num_rejected") or "—"
    def fmt(x):
        if x is None or x == "—":
            return "—"
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)
    return fmt(faith), fmt(rel), fmt(ctx), str(rej)

def e2e_row(d):
    if not d:
        return "—", "—", "—", "—"
    s = d.get("summary") or d
    c = s.get("avg_correctness")
    r = s.get("rejection_accuracy")
    lat = s.get("avg_latency_seconds")
    miss = s.get("rejected_count_answerable")
    def fmt(x, nd=3):
        if x is None:
            return "—"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return str(x)
    return fmt(c), fmt(r), fmt(lat, 2), str(miss if miss is not None else "—")

for arm in arms:
    base = out / arm
    # ragas
    ragas = None
    for cand in [
        base / "ragas" / "ragas_metrics_default.json",
        base / "ragas" / "ragas_metrics_debug.json",
    ]:
        ragas = load_json(cand)
        if ragas:
            break
    if ragas is None:
        cands = list((base / "ragas").rglob("ragas_metrics*.json")) if (base / "ragas").exists() else []
        if cands:
            ragas = load_json(cands[0])
    e2e = load_json(base / "e2e" / "e2e_qa_results.json")
    if e2e is None and (base / "e2e").exists():
        cands = list((base / "e2e").rglob("e2e_qa_results.json"))
        if cands:
            e2e = load_json(cands[0])
    f, rel, ctx, rej = ragas_row(ragas)
    ec, er, elat, em = e2e_row(e2e)
    print(f"| {arm} | {f} | {rel} | {ctx} | {rej} | {ec} | {er} | {elat} | {em} |")
PY
  echo
  echo "## Go / No-Go（相对 base）"
  echo
  echo "| 臂 | Go | No-Go |"
  echo "|----|----|-------|"
  echo "| r1 soft_rank | Correct ≥ base−0.02；Faith ≥ base−0.02；latency ≤ base×1.15 | Correct 掉 >2pt → 保持 mode=bge |"
  echo "| pl planning | Correct ≥ base−0.02；latency ↓≥10% 或 Visual 调用率↓ | Correct 掉 >2pt → planning 关 |"
  echo "| cr crossref | Correct ≥ base−0.02；latency ≤ base×1.10；子集 miss 不恶化 | 全量大掉 → crossref 关 |"
  echo "| CtxRel | 可观察 | **不作上线否决** |"
  echo
  echo "## 产物"
  echo
  echo "- \`${OUT}/env.txt\`"
  echo "- 每臂: \`models.boot.yaml\` · \`ragas/\` · \`e2e/\`"
  echo "- 失败日志: \`${OUT}/failures.log\`（若有）"
  echo
  echo "## 决策模板（跑完填）"
  echo
  echo "| 开关 | 建议 | 理由 |"
  echo "|------|------|------|"
  echo "| refiner.mode=soft_rank | 开 / 关 | |"
  echo "| search_planning.enabled | 开 / 关 | |"
  echo "| crossref_expand.enabled | 开 / 关 | |"
  echo
  echo "## 下一步"
  echo
  echo "1. scp/pull \`${OUT}\` 到本地"
  echo "2. 更新 handoff 默认建议"
  echo "3. **关机省钱**"
} | tee "$SUMMARY"

echo ""
echo "==> Boot-R finished: $OUT"
echo "    Pull results, update handoff, then shutdown GPU."
