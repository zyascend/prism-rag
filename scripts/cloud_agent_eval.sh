#!/usr/bin/env bash
# Phase2 Agentic RAG 双臂云验收：pipeline vs agent
# Spec: docs/superpowers/specs/2026-08-03-agentic-rag-langgraph-design.md §7
#
# 纪律：
#   - --skip-index（与 Boot 同索引 / 同 backbone）
#   - 不改默认 agent.enabled（eval 直接调 graph）
#   - 黄金 NDCG / 283 消融不走本脚本
#   - 先查缓存再触网（AGENTS.md）
#
# Usage（云 GPU）:
#   source scripts/cloud_env.sh
#   service postgresql start   # 或 pg_ctlcluster …
#   pgrep -x ollama >/dev/null || nohup ollama serve &
#   bash scripts/cloud_agent_eval.sh
#
# Env:
#   BOOT_DATE=20260806
#   OUT=runs/20260806-agent-eval
#   MAX_QUERIES=          # 空=全量 agent_eval_qa
#   VISUAL_MODEL=colqwen2
#   JUDGE=llm|heuristic  # 默认 llm
#   PYTHON=python
#   RUN_GRADE_OFF=1|0    # 默认 1：再跑 grade off 子目录
#   SKIP_VISUAL=0|1
#   NO_RERANK=0|1
#   QA_FILE=data/agent_eval_qa.json
#   TAGS=                # 可选 multi_hop,atomic,reject
#   FORCE_LOCAL=0|1
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
[[ -f scripts/cloud_env.sh ]] && source scripts/cloud_env.sh

BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-agent-eval}"
VISUAL_MODEL="${VISUAL_MODEL:-colqwen2}"
PYTHON="${PYTHON:-python}"
JUDGE="${JUDGE:-llm}"
QA_FILE="${QA_FILE:-data/agent_eval_qa.json}"
RUN_GRADE_OFF="${RUN_GRADE_OFF:-0}"   # 默认 0：先出主双臂；需要再设 1
SKIP_VISUAL="${SKIP_VISUAL:-0}"
NO_RERANK="${NO_RERANK:-0}"
FORCE_LOCAL="${FORCE_LOCAL:-0}"
TAGS="${TAGS:-}"
MAX_QUERIES="${MAX_QUERIES:-}"

# Ollama OpenAI-compatible (Generator / judge)
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"
export LLM_MODEL="${LLM_MODEL:-qwen2:7b}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

{
  echo "job=cloud_agent_eval"
  echo "date=${BOOT_DATE}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git branch --show-current 2>/dev/null || echo unknown)"
  echo "visual_model=${VISUAL_MODEL}"
  echo "judge=${JUDGE}"
  echo "qa_file=${QA_FILE}"
  echo "max_queries=${MAX_QUERIES:-all}"
  echo "run_grade_off=${RUN_GRADE_OFF}"
  echo "skip_visual=${SKIP_VISUAL}"
  echo "no_rerank=${NO_RERANK}"
  echo "host=$(hostname 2>/dev/null || echo unknown)"
  echo "skip_index=1"
  echo "agent_enabled_default=false"
  echo "spec=docs/superpowers/specs/2026-08-03-agentic-rag-langgraph-design.md"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT/env.txt"

# ── 缓存 / 依赖自检（不自动下载）──────────────────────────────
echo "[agent-eval] cache check (no auto-download)"
if [[ -d "${HF_HOME:-/root/autodl-tmp/huggingface}" ]]; then
  echo "  HF_HOME=${HF_HOME:-/root/autodl-tmp/huggingface}"
  ls "${HF_HOME:-/root/autodl-tmp/huggingface}/models" 2>/dev/null | head -5 || true
else
  echo "  WARN: HF_HOME missing — ensure models cached before --execute"
fi
if command -v ollama >/dev/null 2>&1; then
  ollama list 2>/dev/null | head -20 || echo "  WARN: ollama list failed"
else
  echo "  WARN: ollama CLI not found"
fi

COMMON=(
  --execute
  --skip-index
  --qa-file "$QA_FILE"
  --judge "$JUDGE"
  --visual-model "$VISUAL_MODEL"
  --k 5
)
if [[ -n "$MAX_QUERIES" ]]; then
  COMMON+=(--max-queries "$MAX_QUERIES")
fi
if [[ -n "$TAGS" ]]; then
  COMMON+=(--tags "$TAGS")
fi
if [[ "$SKIP_VISUAL" == "1" ]]; then
  COMMON+=(--skip-visual)
fi
if [[ "$NO_RERANK" == "1" ]]; then
  COMMON+=(--no-rerank)
fi
if [[ "$FORCE_LOCAL" == "1" ]]; then
  COMMON+=(--force-local)
fi

echo "[agent-eval] arm=dual (pipeline + agent grade default)"
set +e
$PYTHON scripts/run_agent_eval.py \
  "${COMMON[@]}" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
RC=${PIPESTATUS[0]}
set -e

if [[ $RC -ne 0 ]]; then
  echo "[agent-eval] dual-arm FAILED rc=$RC — see $OUT/run.log"
  exit "$RC"
fi

if [[ "$RUN_GRADE_OFF" == "1" ]]; then
  GOUT="${OUT}/grade_off"
  mkdir -p "$GOUT"
  echo "[agent-eval] arm=agent grade_off isolation → $GOUT"
  set +e
  $PYTHON scripts/run_agent_eval.py \
    "${COMMON[@]}" \
    --grade-off \
    --output-dir "$GOUT" \
    2>&1 | tee "$GOUT/run.log"
  GRC=${PIPESTATUS[0]}
  set -e
  if [[ $GRC -ne 0 ]]; then
    echo "[agent-eval] grade_off arm FAILED rc=$GRC (dual-arm results still in $OUT)"
  fi
fi

# ── 决议摘录 ──────────────────────────────────────────────────
if [[ -f "$OUT/results.json" ]]; then
  $PYTHON - <<'PY' "$OUT/results.json" | tee -a "$OUT/env.txt"
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
s = data.get("summary") or {}
go = s.get("go_nogo") or {}
print("go_nogo_verdict=" + str(go.get("verdict")))
for c in go.get("checks") or []:
    print(f"check_{c.get('id')}={c.get('ok')} :: {c.get('detail')}")
arms = s.get("arms") or {}
for name, arm in arms.items():
    print(f"{name}_correct_rate={arm.get('correct_rate')}")
    print(f"{name}_reject_accuracy={arm.get('reject_accuracy')}")
    print(f"{name}_false_reject={arm.get('false_reject_count')}")
    if arm.get("avg_searches") is not None:
        print(f"{name}_avg_searches={arm.get('avg_searches')}")
print("readme=" + str(p.parent / "README.md"))
print("NOTE=do not flip agent.enabled without human Go + config PR")
PY
fi

echo "[agent-eval] done → $OUT"
echo "  results: $OUT/results.json"
echo "  readme:  $OUT/README.md"
echo "  decision: keep agent.enabled=false unless GO + separate PR"
