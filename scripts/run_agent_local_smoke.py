#!/usr/bin/env python
"""本机轻量真链路：pipeline vs agent，默认 5 条、上限 10。

纪律（AGENTS.md）：
  - 禁止自动 ollama pull / HF 下载
  - 缺索引或缺模型 → 清晰 exit，不硬跑
  - 单条异常记 error 继续；summary 写失败率

用法：
  export CONFIG_PROFILE=local-dev
  # ollama serve 且 ollama list 已有目标模型（不 pull）
  PYTHONPATH=. .venv/bin/python scripts/run_agent_local_smoke.py
  PYTHONPATH=. .venv/bin/python scripts/run_agent_local_smoke.py --max-queries 3 --dry-run
  PYTHONPATH=. .venv/bin/python scripts/run_agent_local_smoke.py --tags multi_hop --max-queries 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

MAX_QUERIES_HARD_CAP = 10
DEFAULT_QA = ROOT / "data" / "agent_eval_qa.json"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _check_ollama_model(model: str) -> Tuple[bool, str]:
    """Return (ok, message). Never pulls models."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except FileNotFoundError:
        return False, "ollama CLI not found; install ollama or set a non-Ollama llm.base_url"
    except subprocess.TimeoutExpired:
        return False, "ollama list timed out — is `ollama serve` running?"
    except OSError as e:
        return False, f"ollama list failed: {e}"

    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[:300]
        return False, f"ollama list exit {r.returncode}: {err or 'is ollama serve running?'}"

    # model names in list are like "qwen2:7b" — match prefix before size column
    lines = (r.stdout or "").splitlines()
    names = []
    for line in lines[1:]:  # skip header
        parts = line.split()
        if parts:
            names.append(parts[0])
    if not names:
        return False, "ollama list empty — no local models (do NOT auto-pull; pull manually if needed)"

    # accept exact or base match (qwen2:7b vs qwen2)
    target = (model or "").strip()
    if any(n == target or n.startswith(target + ":") or target.startswith(n.split(":")[0]) for n in names):
        # tighter: target must appear as a listed name or prefix
        if any(n == target or n.startswith(target) for n in names):
            return True, f"ollama model present: {target}"
        # if only base matched loosely, still require exact name
        if target in names:
            return True, f"ollama model present: {target}"

    if target in names or any(n == target for n in names):
        return True, f"ollama model present: {target}"

    # final exact substring on first column
    for n in names:
        if n == target:
            return True, f"ollama model present: {target}"

    return (
        False,
        f"model {target!r} not in `ollama list` ({', '.join(names[:8])}…). "
        f"Pull manually if desired; this script will NOT auto-pull.",
    )


def _index_paths_from_cfg() -> List[Path]:
    from src.config import cfg

    paths = []
    for key in (
        "storage.faiss.index_path",
        "storage.faiss.colqwen2_index_path",
        "storage.faiss.id_map_path",
        "storage.faiss.colqwen2_id_map_path",
    ):
        p = cfg.get(key)
        if p:
            paths.append(Path(p))
    return paths


def _check_index_present() -> Tuple[bool, str]:
    paths = _index_paths_from_cfg()
    if not paths:
        return False, "no faiss paths in config"
    # need at least one .faiss file existing
    faiss = [p for p in paths if str(p).endswith(".faiss")]
    existing = [p for p in faiss if p.is_file()]
    if existing:
        return True, f"index ok: {existing[0]}"
    tried = ", ".join(str(p) for p in faiss) or "(none)"
    profile = os.environ.get("CONFIG_PROFILE", "(unset)")
    return (
        False,
        f"FAISS index missing (CONFIG_PROFILE={profile}). Tried: {tried}. "
        f"Ingest local-demo first or point storage.faiss.* at an existing index. "
        f"This script will NOT download or rebuild a full index.",
    )


def _is_ollama_base_url(url: str) -> bool:
    u = (url or "").lower()
    return "11434" in u or "ollama" in u


def preflight(*, require_llm: bool, require_index: bool) -> List[str]:
    """Return list of blocking error messages (empty if ok)."""
    from src.config import cfg

    errors: List[str] = []
    if require_index:
        ok, msg = _check_index_present()
        if not ok:
            errors.append(msg)
        else:
            logger.info(msg)

    if require_llm:
        base = str(cfg.get("llm.base_url") or "")
        model = str(cfg.get("llm.model") or "")
        if _is_ollama_base_url(base):
            ok, msg = _check_ollama_model(model)
            if not ok:
                errors.append(msg)
            else:
                logger.info(msg)
        else:
            # non-ollama: only warn if api key empty — do not probe network
            key = cfg.get("llm.api_key") or os.environ.get("OPENAI_API_KEY") or ""
            if not key and "localhost" not in base and "127.0.0.1" not in base:
                errors.append(
                    f"llm.api_key empty for base_url={base!r}; set key or use local-dev Ollama"
                )
            else:
                logger.info("llm endpoint %s model=%s (no auto-download check)", base, model)
    return errors


def build_retriever_light(*, skip_visual: bool = False):
    """Build retriever for smoke; prefer loading existing FAISS, no re-ingest."""
    from src.config import cfg
    from src.evaluation.vidore_adapter import PrismRAGRetriever
    from src.ingestion.encoders import BGEEmbedder, create_visual_encoder
    from src.ingestion.text_chunker import TextChunker
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.fusion import RRFFusion
    from src.retrieval.reranker import Reranker
    from src.retrieval.visual_retriever import VisualRetriever
    from src.store.faiss_store import FaissColPaliStore
    from src.store.pgvector_store import PgVectorStore

    pg_store = PgVectorStore()
    backend = cfg.get("embedding.visual_backend", "colpali")
    if backend == "colqwen2":
        faiss_store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colqwen2_index_path"),
            id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
        )
    else:
        faiss_store = FaissColPaliStore()

    bge = BGEEmbedder()
    visual_encoder = None
    if not skip_visual and cfg.get("retrieval.use_visual", True):
        # Only create if index exists — still may load weights from local HF cache
        visual_encoder = create_visual_encoder(model_name=backend)

    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    visual = VisualRetriever(faiss_store, pg_store, visual_encoder) if visual_encoder else None
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()

    faiss_loaded = faiss_store.load()
    if faiss_loaded:
        try:
            bm25.fit_from_pgvector(pg_store)
        except Exception as e:
            logger.warning("bm25 fit failed: %s", e)
    else:
        logger.warning("FAISS not loaded; dense-only if pg has chunks")

    return PrismRAGRetriever(
        pg_store=pg_store,
        faiss_store=faiss_store,
        bge=bge,
        colpali=visual_encoder,
        chunker=chunker,
        bm25=bm25,
        dense=dense,
        visual=visual,
        fusion=fusion,
        reranker=reranker,
    )


def _string_match_correct(answer: str, gold: Optional[str], expect_reject: bool) -> Optional[bool]:
    """Placeholder Correct: reject phrase check or loose gold token overlap. Not Phase2 metric."""
    from src.rejection import is_rejection

    if expect_reject:
        return bool(is_rejection(answer or ""))
    if not gold:
        return None
    a = (answer or "").lower()
    # take a few content words from gold
    tokens = [t for t in gold.lower().replace(",", " ").split() if len(t) > 4][:8]
    if not tokens:
        return None
    hits = sum(1 for t in tokens if t in a)
    return hits >= max(1, len(tokens) // 3)


def run_one_item(
    item: Dict[str, Any],
    *,
    retriever,
    generator,
    k: int,
    use_visual: bool,
    use_rerank: bool,
) -> Dict[str, Any]:
    from src.agent.eval import agent_answer_for_eval
    from src.generation.self_rag import answer_for_eval

    q = item.get("query") or item.get("question") or ""
    gold = item.get("gold_answer")
    expect_reject = bool(item.get("expect_reject"))
    row: Dict[str, Any] = {
        "id": item.get("id"),
        "tag": item.get("tag"),
        "query": q,
        "expect_reject": expect_reject,
    }

    # pipeline arm
    t0 = time.perf_counter()
    try:
        hits = retriever.search(q, k=k, use_visual=use_visual, use_rerank=use_rerank)
        pipe = answer_for_eval(
            q,
            hits,
            k_context=k,
            generator=generator,
            retriever=retriever,
            use_rerank=use_rerank,
            use_visual=use_visual,
        )
        row["pipeline"] = {
            "answer": pipe.get("answer") or "",
            "n_citations": len(pipe.get("citations") or []),
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "error": None,
            "correct_placeholder": _string_match_correct(
                pipe.get("answer") or "", gold, expect_reject
            ),
        }
    except Exception as e:
        logger.exception("pipeline failed id=%s", item.get("id"))
        row["pipeline"] = {
            "answer": "",
            "n_citations": 0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "error": str(e)[:400],
            "correct_placeholder": False,
        }

    # agent arm
    t1 = time.perf_counter()
    try:
        ag = agent_answer_for_eval(
            q,
            retriever=retriever,
            generator=generator,
            k_context=k,
            use_rerank=use_rerank,
            use_visual=use_visual,
            cfg={"return_trajectory": True},
        )
        agent_meta = ag.get("agent") or {}
        row["agent"] = {
            "answer": ag.get("answer") or "",
            "n_citations": len(ag.get("citations") or []),
            "latency_ms": int((time.perf_counter() - t1) * 1000),
            "status": agent_meta.get("status"),
            "subqueries": agent_meta.get("subqueries") or [],
            "counts": agent_meta.get("counts") or {},
            "trajectory_summary": [
                {
                    "step": t.get("step"),
                    "node": t.get("node"),
                    "tool": t.get("tool"),
                }
                for t in (agent_meta.get("trajectory") or [])[:12]
                if isinstance(t, dict)
            ],
            "error": agent_meta.get("error"),
            "correct_placeholder": _string_match_correct(
                ag.get("answer") or "", gold, expect_reject
            ),
        }
    except Exception as e:
        logger.exception("agent failed id=%s", item.get("id"))
        row["agent"] = {
            "answer": "",
            "n_citations": 0,
            "latency_ms": int((time.perf_counter() - t1) * 1000),
            "status": "error",
            "subqueries": [],
            "counts": {},
            "trajectory_summary": [],
            "error": str(e)[:400],
            "correct_placeholder": False,
        }
    return row


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Local agent smoke: ≤10 queries, pipeline vs agent (no auto model download)"
    )
    parser.add_argument("--qa-file", default=str(DEFAULT_QA))
    parser.add_argument(
        "--max-queries",
        type=int,
        default=5,
        help=f"default 5, hard cap {MAX_QUERIES_HARD_CAP}",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="comma-separated tags (multi_hop,atomic,reject)",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="",
        help="default runs/local-agent-smoke-<ts>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load QA + preflight only; do not call retriever/LLM",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="dense/bm25 only (avoids loading visual encoder weights)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip index/model checks (not recommended)",
    )
    args = parser.parse_args(argv)

    n = int(args.max_queries)
    if n < 1:
        logger.error("--max-queries must be >= 1")
        return 2
    if n > MAX_QUERIES_HARD_CAP:
        logger.error(
            "--max-queries=%s exceeds hard cap %s (AGENTS.md local smoke limit)",
            n,
            MAX_QUERIES_HARD_CAP,
        )
        return 2

    profile = os.environ.get("CONFIG_PROFILE", "")
    if not profile:
        logger.warning(
            "CONFIG_PROFILE unset; recommend `export CONFIG_PROFILE=local-dev` for demo index"
        )

    from src.agent.eval import load_agent_eval_qa
    from src.config import cfg

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or None
    try:
        items = load_agent_eval_qa(args.qa_file, tags=tags, max_items=n)
    except FileNotFoundError:
        logger.error("QA file not found: %s", args.qa_file)
        return 2
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("bad QA file: %s", e)
        return 2

    if not items:
        logger.error("no items after tag/max filter")
        return 2

    logger.info(
        "smoke: %d items | profile=%s | model=%s | dry_run=%s",
        len(items),
        profile or "(default)",
        cfg.get("llm.model"),
        args.dry_run,
    )

    if not args.skip_preflight:
        errs = preflight(require_llm=not args.dry_run, require_index=not args.dry_run)
        if errs:
            for e in errs:
                logger.error("preflight: %s", e)
            logger.error(
                "Abort. Fix index/models manually (no auto-download). "
                "Or use --dry-run / unit tests (tests/test_agent_eval.py)."
            )
            return 1

    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "runs" / f"local-agent-smoke-{_utc_ts()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "n_items": len(items),
            "item_ids": [it.get("id") for it in items],
            "config_profile": profile or None,
            "llm_model": cfg.get("llm.model"),
            "note": "preflight passed or skipped; no retrieval/generation",
        }
        (out_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("dry-run wrote %s", out_dir / "results.json")
        return 0

    # Real path — may still fail if HF weights missing; we do not download.
    use_visual = bool(cfg.get("retrieval.use_visual", True)) and not args.skip_visual
    use_rerank = bool(cfg.get("retrieval.use_rerank", True))
    try:
        retriever = build_retriever_light(skip_visual=not use_visual)
        from src.generation.generator import Generator

        generator = Generator(bge_embedder=getattr(retriever, "bge", None))
    except Exception as e:
        logger.error(
            "failed to build retriever/generator (no auto-download): %s", e
        )
        return 1

    rows: List[Dict[str, Any]] = []
    n_err = 0
    for it in items:
        row = run_one_item(
            it,
            retriever=retriever,
            generator=generator,
            k=args.k,
            use_visual=use_visual,
            use_rerank=use_rerank,
        )
        if (row.get("pipeline") or {}).get("error") or (row.get("agent") or {}).get(
            "error"
        ):
            n_err += 1
        rows.append(row)
        ag = row.get("agent") or {}
        logger.info(
            "id=%s tag=%s agent.status=%s counts=%s lat_p=%sms lat_a=%sms",
            row.get("id"),
            row.get("tag"),
            ag.get("status"),
            ag.get("counts"),
            (row.get("pipeline") or {}).get("latency_ms"),
            ag.get("latency_ms"),
        )

    def _arm_stats(arm: str) -> Dict[str, Any]:
        corrects = [
            r[arm].get("correct_placeholder")
            for r in rows
            if arm in r and r[arm].get("correct_placeholder") is not None
        ]
        errs = sum(1 for r in rows if (r.get(arm) or {}).get("error"))
        lats = [r[arm]["latency_ms"] for r in rows if arm in r]
        return {
            "n": len(rows),
            "errors": errs,
            "correct_placeholder_rate": (
                sum(1 for c in corrects if c) / len(corrects) if corrects else None
            ),
            "latency_ms_mean": (sum(lats) / len(lats)) if lats else None,
        }

    summary = {
        "n_items": len(rows),
        "error_items": n_err,
        "error_rate": n_err / len(rows) if rows else 0.0,
        "pipeline": _arm_stats("pipeline"),
        "agent": _arm_stats("agent"),
        "note": (
            "correct_placeholder is string-match only — not Phase2 decision metric. "
            "Cloud dual-arm: scripts/run_agent_eval.py"
        ),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_profile": profile or None,
        "llm_model": cfg.get("llm.model"),
        "max_queries": n,
        "k": args.k,
        "use_visual": use_visual,
        "qa_file": str(args.qa_file),
        "summary": summary,
        "items": rows,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = out_dir / "README.md"
    readme.write_text(
        "# local agent smoke\n\n"
        f"- items: {len(rows)}\n"
        f"- profile: `{profile or 'default'}`\n"
        f"- model: `{cfg.get('llm.model')}`\n"
        f"- error_rate: {summary['error_rate']:.2f}\n\n"
        "Not a Phase2 go/no-go. See `results.json` for trajectories.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("wrote %s", out_path)
    return 0 if n_err == 0 else 0  # soft: still success if partial; summary has rates


if __name__ == "__main__":
    raise SystemExit(main())
