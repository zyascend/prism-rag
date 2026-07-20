#!/usr/bin/env bash
# Self-RAG Gate2 开/关对照（云上 GPU）
# 设计：docs/self-rag-closed-loop-design-2026-07-09.md
#
# 两臂均开 generation.eval_via_generator=true（走生产 Generator），
# 仅切换 self_rag.enabled，隔离 Gate2 效应。
# 默认关闭 visual_routing，避免与路由对照混淆。
#
# Usage（云上）:
#   source scripts/cloud_env.sh   # 可选
#   bash scripts/cloud_self_rag_ab.sh
#
# Env:
#   BOOT_DATE=20260721
#   MAX_QUERIES=100           # RAGAS 默认 100；预算紧可 50
#   RUN_E2E=1|0              # 默认 1：再跑 data/e2e_qa.json
#   RUN_RAGAS=1|0            # 默认 1
#   VISUAL_MODEL=colqwen2
#   PYTHON=python
#   SKIP_OFF=0|1             # 1=只跑 ON 臂（已有 off 结果时）
#   SKIP_ON=0|1
#
# 本地：禁止全量；可用 MAX_QUERIES=3 冒烟（需索引+LLM）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-self-rag-gate2}"
VISUAL_MODEL="${VISUAL_MODEL:-colqwen2}"
PYTHON="${PYTHON:-python}"
MAX_QUERIES="${MAX_QUERIES:-100}"
RUN_RAGAS="${RUN_RAGAS:-1}"
RUN_E2E="${RUN_E2E:-1}"
SKIP_OFF="${SKIP_OFF:-0}"
SKIP_ON="${SKIP_ON:-0}"
MODELS_BAK=""

restore_models_yaml() {
  if [[ -n "${MODELS_BAK}" && -f "${MODELS_BAK}" ]]; then
    mv -f "${MODELS_BAK}" config/models.yaml
    MODELS_BAK=""
  fi
}
trap restore_models_yaml EXIT

mkdir -p "$OUT/off" "$OUT/on"

{
  echo "job=self_rag_ab"
  echo "date=${BOOT_DATE}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "visual_model=${VISUAL_MODEL}"
  echo "max_queries=${MAX_QUERIES}"
  echo "run_ragas=${RUN_RAGAS}"
  echo "run_e2e=${RUN_E2E}"
  echo "host=$(hostname 2>/dev/null || echo unknown)"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT/env.txt"

# 写入临时 models.yaml：eval_via_generator + self_rag.enabled + 关 visual routing
write_boot_cfg() {
  local enabled="$1"   # true|false
  local dest_yaml="$2"
  ENABLED_SR="$enabled" DEST_YAML="$dest_yaml" "$PYTHON" - <<'PY'
import os, yaml
from pathlib import Path

root = Path(".")
src = root / "config" / "models.yaml"
data = yaml.safe_load(src.read_text()) or {}
gen = data.setdefault("generation", {})
gen["eval_via_generator"] = True
sr = gen.setdefault("self_rag", {})
sr["enabled"] = os.environ["ENABLED_SR"].lower() in ("1", "true", "yes")
# 隔离变量：A/B 不混 visual 路由
vr = data.setdefault("retrieval", {}).setdefault("visual_routing", {})
vr["enabled"] = False
# 生成侧保持默认 bge 压缩
data.setdefault("context_filter", {})["mode"] = data.get("context_filter", {}).get("mode", "bge")

out = Path(os.environ["DEST_YAML"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))
print("wrote", out, "self_rag.enabled=", sr["enabled"])
PY
}

run_arm() {
  local arm="$1"          # off | on
  local enabled="$2"      # false | true
  local dest="$OUT/${arm}"
  mkdir -p "$dest/ragas" "$dest/e2e"

  echo "==> Arm ${arm}: self_rag.enabled=${enabled} → ${dest}"
  write_boot_cfg "$enabled" "${dest}/models.boot.yaml"

  MODELS_BAK="$OUT/models.yaml.bak"
  cp config/models.yaml "$MODELS_BAK"
  cp "${dest}/models.boot.yaml" config/models.yaml

  # 失效 Config 单例缓存（若进程内曾 load）
  # 每个 python 子进程重新 load，无需处理单例

  local rc=0
  if [[ "$RUN_RAGAS" == "1" ]]; then
    echo "  → RAGAS max_queries=${MAX_QUERIES}"
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
      echo "RAGAS failed arm=${arm}" >&2
      exit $rc
    fi
  fi

  if [[ "$RUN_E2E" == "1" ]]; then
    echo "  → E2E QA (data/e2e_qa.json)"
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
      echo "E2E failed arm=${arm}" >&2
      exit $rc
    fi
  fi

  restore_models_yaml
  echo "  ✓ arm ${arm} done"
}

if [[ "$SKIP_OFF" != "1" ]]; then
  run_arm off false
else
  echo "==> SKIP_OFF=1"
fi

if [[ "$SKIP_ON" != "1" ]]; then
  run_arm on true
else
  echo "==> SKIP_ON=1"
fi

# 摘要骨架（人工填表 / 后续可脚本化解析 JSON）
SUMMARY="$OUT/README.md"
{
  echo "# Self-RAG Gate2 A/B (${BOOT_DATE})"
  echo
  echo "| 项 | 值 |"
  echo "|----|----|"
  echo "| git | \`$(git rev-parse HEAD 2>/dev/null || echo unknown)\` |"
  echo "| visual | \`${VISUAL_MODEL}\` |"
  echo "| RAGAS max_queries | \`${MAX_QUERIES}\` |"
  echo "| eval_via_generator | **true**（两臂） |"
  echo "| 隔离 | visual_routing.enabled=false；仅 self_rag.enabled 不同 |"
  echo
  echo "## 指标表（填）"
  echo
  echo "| arm | Faith | Rel | CtxRel | 拒答数 | E2E Correctness | E2E Rejection | avg latency |"
  echo "|-----|-------|-----|--------|--------|-----------------|---------------|-------------|"
  echo "| off (Gate2 关) | | | | | | | |"
  echo "| on  (Gate2 开) | | | | | | | |"
  echo "| Δ | | | | | | | |"
  echo
  echo "## 产物"
  echo
  echo "- \`${OUT}/off/ragas/\` / \`${OUT}/on/ragas/\`"
  echo "- \`${OUT}/off/e2e/\` / \`${OUT}/on/e2e/\`"
  echo "- 各 arm 的 \`models.boot.yaml\` 冻结配置"
  echo
  echo "## 分析提示"
  echo
  echo "1. Trace：开臂看 \`self_rag.gate2.attempts_detail\`（fail→regen 回放）。"
  echo "2. 简历：仅当 Faith↑ 或 瞎答↓ 且 Correctness 不掉时写入 bullet ③。"
  echo "3. 阴性：延迟×N 且指标平 → 默认保持 enabled=false，口述对照实验。"
  echo "4. **拉结果后立刻关机。**"
  echo
  echo "## 本地冒烟（可选）"
  echo
  echo "\`\`\`bash"
  echo "MAX_QUERIES=3 RUN_E2E=0 bash scripts/cloud_self_rag_ab.sh"
  echo "\`\`\`"
} | tee "$SUMMARY"

echo "==> Self-RAG A/B finished: $OUT"
echo "    Pull results, fill README metrics, shutdown GPU."
