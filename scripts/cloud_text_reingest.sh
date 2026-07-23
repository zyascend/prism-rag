#!/usr/bin/env bash
# 云上 Text re-ingest：TRUNCATE chunks → 分块 +（可选）上下文表摘要 + BGE → pgvector
# FAISS / ColQwen2 索引不动。供 Phase A1/A3 验收。
#
# 仅在云 GPU 机执行。见 Agents.md：先查 HF/ollama 缓存。
#
# Usage:
#   source scripts/cloud_env.sh
#   # 冒烟 20 页（关表摘要 LLM，快）
#   MODE=smoke bash scripts/cloud_text_reingest.sh
#   # 冒烟 20 页 + 表摘要（测 ollama 链路）
#   MODE=smoke-llm bash scripts/cloud_text_reingest.sh
#   # 全量 5244 页 + 上下文表摘要（贵：表 LLM 可能数小时）
#   MODE=full bash scripts/cloud_text_reingest.sh
#
# Env:
#   MODE=smoke|smoke-llm|full
#   MAX_PAGES=20          # smoke 默认 20；full 忽略
#   TABLE_CONTEXT=1       # full/smoke-llm 默认 1；smoke 默认 0
#   PYTHON=python
#   OUT=runs/YYYYMMDD-text-reingest
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
[[ -f scripts/cloud_env.sh ]] && source scripts/cloud_env.sh

MODE="${MODE:-smoke}"
PYTHON="${PYTHON:-python}"
BOOT_DATE="${BOOT_DATE:-$(date +%Y%m%d)}"
OUT="${OUT:-runs/${BOOT_DATE}-text-reingest}"
mkdir -p "$OUT"

case "$MODE" in
  smoke)
    MAX_PAGES="${MAX_PAGES:-20}"
    TABLE_CONTEXT="${TABLE_CONTEXT:-0}"
    NO_SUMMARY=1
    ;;
  smoke-llm)
    MAX_PAGES="${MAX_PAGES:-20}"
    TABLE_CONTEXT="${TABLE_CONTEXT:-1}"
    NO_SUMMARY=0
    ;;
  full)
    MAX_PAGES=""
    TABLE_CONTEXT="${TABLE_CONTEXT:-1}"
    NO_SUMMARY=0
    ;;
  *)
    echo "Unknown MODE=$MODE (smoke|smoke-llm|full)" >&2
    exit 1
    ;;
esac

{
  echo "job=text-reingest"
  echo "mode=${MODE}"
  echo "max_pages=${MAX_PAGES:-all}"
  echo "table_context=${TABLE_CONTEXT}"
  echo "no_summary=${NO_SUMMARY}"
  echo "host=$(hostname)"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT/env.txt"

echo "==> Preflight"
# PG
if ! pg_isready -q 2>/dev/null; then
  pg_ctlcluster 14 main start 2>/dev/null || service postgresql start 2>/dev/null || true
  sleep 2
fi
pg_isready

# Ollama（表摘要需要）
if [[ "$NO_SUMMARY" != "1" ]]; then
  if ! pgrep -x ollama >/dev/null; then
    nohup ollama serve > /tmp/ollama-serve.log 2>&1 &
    sleep 2
  fi
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "ERROR: ollama not reachable; table summary needs qwen2:7b" >&2
    exit 1
  fi
  ollama list | tee "$OUT/ollama_list.txt" || true
fi

# HF offline 提示
if [[ "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  echo "HF offline OK (cache required for dataset)"
else
  echo "WARN: HF not offline — prefer source cloud_env.sh"
fi

# 备份当前 chunk 计数
"$PYTHON" - <<PY | tee "$OUT/pre_counts.txt"
from src.config import cfg
cfg.load()
from src.store.pgvector_store import PgVectorStore
pg = PgVectorStore()
pg.create_schema()
print("chunks_before", pg.count())
import psycopg2
c = psycopg2.connect(pg.conn_string)
cur = c.cursor()
cur.execute("SELECT chunk_type, count(*) FROM chunks GROUP BY 1 ORDER BY 1")
for row in cur.fetchall():
    print("type", row[0], row[1])
cur.execute("SELECT count(*) FROM chunks WHERE section_path <> ''")
print("section_path_filled", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM chunks WHERE table_summary <> ''")
print("table_summary_filled", cur.fetchone()[0])
PY

ARGS=(
  --skip-faiss
  --replace-text
)
if [[ -n "${MAX_PAGES}" ]]; then
  ARGS+=(--max-pages "$MAX_PAGES")
fi
if [[ "$TABLE_CONTEXT" == "1" ]]; then
  ARGS+=(--table-context)
fi
if [[ "$NO_SUMMARY" == "1" ]]; then
  ARGS+=(--no-table-summary)
fi

echo "==> Running: $PYTHON scripts/ingest_vidore.py ${ARGS[*]}"
set +e
"$PYTHON" scripts/ingest_vidore.py "${ARGS[@]}" 2>&1 | tee "$OUT/ingest.log"
RC=${PIPESTATUS[0]}
set -e
if [[ $RC -ne 0 ]]; then
  echo "ingest FAILED rc=$RC" >&2
  exit $RC
fi

"$PYTHON" - <<PY | tee "$OUT/post_counts.txt"
from src.config import cfg
cfg.load()
from src.store.pgvector_store import PgVectorStore
pg = PgVectorStore()
print("chunks_after", pg.count())
import psycopg2
c = psycopg2.connect(pg.conn_string)
cur = c.cursor()
cur.execute("SELECT chunk_type, count(*) FROM chunks GROUP BY 1 ORDER BY 1")
for row in cur.fetchall():
    print("type", row[0], row[1])
cur.execute("SELECT count(*) FROM chunks WHERE section_path <> ''")
print("section_path_filled", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM chunks WHERE prev_chunk_id <> ''")
print("prev_filled", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM chunks WHERE table_summary <> ''")
print("table_summary_filled", cur.fetchone()[0])
# sample
cur.execute("""
  SELECT chunk_type, left(section_path,60), left(table_summary,80)
  FROM chunks WHERE chunk_type='table' AND table_summary <> '' LIMIT 3
""")
print("samples:")
for row in cur.fetchall():
    print(" ", row)
PY

{
  echo "# Text re-ingest (${BOOT_DATE} · ${MODE})"
  echo
  echo "- skip_faiss: true (ColQwen2 FAISS untouched)"
  echo "- replace_text: true (TRUNCATE chunks first)"
  echo "- table_context: ${TABLE_CONTEXT}"
  echo "- table_summary: $([ "$NO_SUMMARY" = 1 ] && echo off || echo on)"
  echo "- max_pages: ${MAX_PAGES:-all}"
  echo
  echo "## Counts"
  echo '```'
  cat "$OUT/pre_counts.txt"
  echo "---"
  cat "$OUT/post_counts.txt"
  echo '```'
  echo
  echo "## Next"
  echo
  echo "1. Full_zerank2 100q vs Boot-CP Arm-A (same 100q protocol)"
  echo "2. Optional: enable expand/boost arms"
  echo "3. E2E / table subset"
} | tee "$OUT/README.md"

echo "==> Done: $OUT"
echo "    FAISS unchanged under indexes/ → /root/autodl-tmp/indexes"
