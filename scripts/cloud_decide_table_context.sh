#!/usr/bin/env bash
# 决策：table_summary_context 是否默认打开
# 协议：docs/table-context-default-decision-protocol.md
#
# 云 GPU only。默认假设：当前 pg 已是 context ON 的 full re-ingest。
#
# Usage:
#   source scripts/cloud_env.sh
#   bash scripts/cloud_decide_table_context.sh
#
# Env:
#   ONLY=all|on|off     默认 all：先 ON 评测 → OFF re-ingest → OFF 评测
#   SKIP_OFF_INGEST=0|1 1=跳过 OFF 重灌（库已是 OFF 时）
#   RUN_283=1|0         默认 1
#   RUN_100=1|0         默认 1
#   RUN_E2E=1|0         默认 1
#   PYTHON=python
#   BOOT_DATE=YYYYMMDD
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
[[ -f scripts/cloud_env.sh ]] && source scripts/cloud_env.sh

PYTHON="${PYTHON:-python}"
ONLY="${ONLY:-all}"
SKIP_OFF_INGEST="${SKIP_OFF_INGEST:-0}"
RUN_283="${RUN_283:-1}"
RUN_100="${RUN_100:-1}"
RUN_E2E="${RUN_E2E:-1}"
BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-table-ctx-decide}"

mkdir -p "$OUT"/{on,off}

{
  echo "job=decide-table-context"
  echo "only=${ONLY}"
  echo "skip_off_ingest=${SKIP_OFF_INGEST}"
  echo "run_283=${RUN_283} run_100=${RUN_100} run_e2e=${RUN_E2E}"
  echo "host=$(hostname)"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
  echo "protocol=docs/table-context-default-decision-protocol.md"
} | tee "$OUT/env.txt"

preflight() {
  if ! pg_isready -q 2>/dev/null; then
    pg_ctlcluster 14 main start 2>/dev/null || service postgresql start 2>/dev/null || true
    sleep 2
  fi
  pg_isready
  if [[ "$RUN_E2E" == "1" ]] || [[ "$SKIP_OFF_INGEST" != "1" ]]; then
    if ! pgrep -x ollama >/dev/null; then
      nohup ollama serve > /tmp/ollama-serve.log 2>&1 &
      sleep 2
    fi
  fi
  "$PYTHON" - <<'PY' | tee "$OUT/preflight_counts.txt"
from src.config import cfg
cfg.load()
from src.store.pgvector_store import PgVectorStore
pg = PgVectorStore()
print("chunks", pg.count())
import psycopg2
c = psycopg2.connect(pg.conn_string)
cur = c.cursor()
cur.execute("SELECT count(*) FROM chunks WHERE table_summary <> ''")
print("table_summary_filled", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM chunks WHERE section_path <> ''")
print("section_path_filled", cur.fetchone()[0])
print("context_cfg", cfg.get("ingestion.table_summary_context_enabled", False))
PY
}

run_ndcg() {
  local dest="$1"
  local nq="$2"  # empty = 283 expected
  mkdir -p "$dest"
  local args=(
    --skip-index --language en --visual-model colqwen2
    --config-filter Full_zerank2 --no-hyde
    --output-dir "$dest"
  )
  if [[ -z "$nq" ]]; then
    args+=(--expected-query-count 283)
  else
    args+=(--max-queries "$nq")
  fi
  echo "==> NDCG → $dest (max_queries=${nq:-283})"
  "$PYTHON" scripts/run_eval.py "${args[@]}" 2>&1 | tee "$dest/run.log"
}

run_e2e() {
  local dest="$1"
  mkdir -p "$dest"
  echo "==> E2E → $dest"
  "$PYTHON" scripts/run_e2e_qa.py --skip-index --visual-model colqwen2 \
    --output-dir "$dest" 2>&1 | tee "$dest/run.log"
}

run_arm_eval() {
  local arm="$1"  # on|off
  local base="$OUT/$arm"
  mkdir -p "$base"
  if [[ "$RUN_283" == "1" ]]; then
    run_ndcg "$base/ndcg283" ""
  fi
  if [[ "$RUN_100" == "1" ]]; then
    run_ndcg "$base/ndcg100" 100
  fi
  if [[ "$RUN_E2E" == "1" ]]; then
    run_e2e "$base/e2e"
  fi
  # 标签
  echo "arm=${arm}" > "$base/arm_label.txt"
}

ingest_off() {
  echo "==> Text re-ingest OFF (summary on, context off)"
  mkdir -p "$OUT/off"
  "$PYTHON" scripts/ingest_vidore.py --skip-faiss --replace-text \
    2>&1 | tee "$OUT/off/ingest.log"
}

write_comparison() {
  "$PYTHON" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")

def load_ndcg(path: Path):
    p = path / "ablation_results.json"
    if not p.exists():
        cands = list(path.rglob("ablation_results.json"))
        p = cands[0] if cands else None
    if not p or not p.exists():
        return None
    d = json.loads(p.read_text())
    r = d[0] if isinstance(d, list) else d
    return {
        "ndcg@10": r.get("ndcg@10"),
        "ndcg@5": r.get("ndcg@5"),
        "recall@10": r.get("recall@10"),
        "mrr": r.get("mrr"),
        "avg_latency_ms": r.get("avg_latency_ms"),
        "num_queries": r.get("num_queries"),
    }

def load_e2e(path: Path):
    # 现网: e2e_qa_results.json → summary.avg_correctness / rejection_accuracy
    for name in ("e2e_qa_results.json", "results.json"):
        cands = list(path.rglob(name))
        if not cands:
            continue
        d = json.loads(cands[0].read_text())
        s = d.get("summary") if isinstance(d, dict) else None
        if isinstance(s, dict):
            return {
                "correct": s.get("avg_correctness", s.get("answer_correctness")),
                "reject_accuracy": s.get("rejection_accuracy", s.get("reject_accuracy")),
                "num_answerable": s.get("num_answerable"),
                "num_rejection": s.get("num_rejection"),
                "path": str(cands[0]),
            }
        if isinstance(d, dict) and "metrics" in d:
            m = d["metrics"]
            return {
                "correct": m.get("avg_correctness", m.get("answer_correctness")),
                "reject_accuracy": m.get("rejection_accuracy"),
                "path": str(cands[0]),
            }
        if isinstance(d, dict):
            return {"raw_keys": list(d.keys())[:30], "path": str(cands[0])}
    return None

comp = {
    "on": {
        "ndcg283": load_ndcg(out / "on" / "ndcg283"),
        "ndcg100": load_ndcg(out / "on" / "ndcg100"),
        "e2e": load_e2e(out / "on" / "e2e"),
    },
    "off": {
        "ndcg283": load_ndcg(out / "off" / "ndcg283"),
        "ndcg100": load_ndcg(out / "off" / "ndcg100"),
        "e2e": load_e2e(out / "off" / "e2e"),
    },
    "historical_100q": {
        "off_approx_bootcp": 0.3575,
        "on_post_reingest": 0.3589,
    },
}

def delta(a, b, key):
    if not a or not b or a.get(key) is None or b.get(key) is None:
        return None
    return round(float(a[key]) - float(b[key]), 4)

# 决策（仅当两边 283+e2e 齐全）
decision = "incomplete"
reason = []
on283, off283 = comp["on"]["ndcg283"], comp["off"]["ndcg283"]
one2e, offe2e = comp["on"]["e2e"], comp["off"]["e2e"]
if on283 and off283 and on283.get("ndcg@10") is not None and off283.get("ndcg@10") is not None:
    d_ndcg = float(on283["ndcg@10"]) - float(off283["ndcg@10"])
    reason.append(f"M1 NDCG@10 ON-OFF={d_ndcg:+.4f}")
    m1_ok = d_ndcg >= -0.005
    m1_gain = d_ndcg >= 0.005
else:
    m1_ok = m1_gain = False
    reason.append("M1 missing")

m2_ok = m2_gain = False
m3_ok = True
if isinstance(one2e, dict) and isinstance(offe2e, dict):
    c_on = one2e.get("correct")
    c_off = offe2e.get("correct")
    r_on = one2e.get("reject_accuracy")
    r_off = offe2e.get("reject_accuracy")
    if c_on is not None and c_off is not None:
        d_c = float(c_on) - float(c_off)
        reason.append(f"M2 Correct ON-OFF={d_c:+.4f}")
        m2_ok = d_c >= -0.04
        m2_gain = d_c >= 0.04
    if r_on is not None and r_off is not None:
        d_r = float(r_on) - float(r_off)
        reason.append(f"M3 Reject ON-OFF={d_r:+.4f}")
        m3_ok = d_r >= -0.05
else:
    reason.append("M2/M3 e2e incomplete or parse failed — check raw json")

if "missing" not in " ".join(reason) and on283 and off283:
    if m1_ok and m2_ok and m3_ok:
        if m1_gain or m2_gain:
            decision = "default_true"
        else:
            decision = "default_false_but_safe_to_enable"
    else:
        decision = "default_false"

comp["decision"] = decision
comp["reason"] = reason
(out / "comparison.json").write_text(json.dumps(comp, indent=2, ensure_ascii=False) + "\n")

lines = [
    f"# Table-context default decision ({out.name})",
    "",
    f"- decision: **`{decision}`**",
    f"- reason: {'; '.join(reason)}",
    "",
    "## NDCG@10",
    "",
    "| arm | nq | NDCG@10 | R@10 | latency |",
    "|-----|---:|--------:|-----:|--------:|",
]
for arm in ("on", "off"):
    for key, nq in (("ndcg283", 283), ("ndcg100", 100)):
        m = comp[arm][key]
        if m:
            lines.append(
                f"| {arm} | {nq} | {m.get('ndcg@10')} | {m.get('recall@10')} | {m.get('avg_latency_ms')} |"
            )
lines += [
    "",
    "## E2E",
    "",
    "```json",
    json.dumps({"on": one2e, "off": offe2e}, indent=2, ensure_ascii=False),
    "```",
    "",
    "## Rules",
    "",
    "See docs/table-context-default-decision-protocol.md §2.",
    "",
    f"- default_true: 不降且 (NDCG+≥0.5pt 或 Correct+≥0.04)",
    f"- default_false_but_safe_to_enable: 不降但增益未过阈值 → yaml 仍 false",
    f"- default_false: 硬掉",
]
(out / "README.md").write_text("\n".join(lines) + "\n")
print("wrote", out / "comparison.json")
print("decision", decision)
for r in reason:
    print(" ", r)
PY
}

# ── main ──────────────────────────────────────────────────
preflight

case "$ONLY" in
  on)
    run_arm_eval on
    ;;
  off)
    if [[ "$SKIP_OFF_INGEST" != "1" ]]; then
      ingest_off
    fi
    run_arm_eval off
    ;;
  all)
    run_arm_eval on
    if [[ "$SKIP_OFF_INGEST" != "1" ]]; then
      ingest_off
    fi
    run_arm_eval off
    ;;
  *)
    echo "ONLY must be all|on|off" >&2
    exit 1
    ;;
esac

write_comparison
echo "==> Done: $OUT"
echo "    Read $OUT/README.md and comparison.json"
echo "    Then shutdown GPU if idle."
