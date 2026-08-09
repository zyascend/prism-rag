#!/usr/bin/env python
"""Phase2 云上双臂：pipeline vs agent on data/agent_eval_qa.json.

纪律（AGENTS.md）：
  - 本机不要跑全量 40–50（用 --delegate-local-smoke ≤10 或 --dry-run）
  - 云上 --execute --skip-index，与 Boot 同 backbone / 同生成模型
  - 黄金 NDCG / 283 消融永不走 agent

示例：

  # 云上正式双臂（~46 条）
  PYTHONPATH=. python scripts/run_agent_eval.py \\
    --execute --skip-index \\
    --qa-file data/agent_eval_qa.json \\
    --judge llm \\
    --output-dir runs/YYYYMMDD-agent-eval

  # 第三臂：grade off（写 agent_grade_off 字段）
  PYTHONPATH=. python scripts/run_agent_eval.py \\
    --execute --skip-index --grade-off \\
    --output-dir runs/YYYYMMDD-agent-eval-grade-off

  # 本机 ≤10 真链路
  PYTHONPATH=. python scripts/run_agent_eval.py --delegate-local-smoke --max-queries 5

  # 只列计划（默认）
  PYTHONPATH=. python scripts/run_agent_eval.py --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _load_smoke_module():
    path = ROOT / "scripts" / "run_agent_local_smoke.py"
    spec = importlib.util.spec_from_file_location("run_agent_local_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _git_meta() -> Dict[str, str]:
    meta = {"git_head": "unknown", "git_branch": "unknown"}
    try:
        meta["git_head"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    try:
        meta["git_branch"] = (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    return meta


def build_retriever(
    *,
    skip_index: bool = True,
    visual_model: str = "colqwen2",
    skip_visual: bool = False,
):
    """Build PrismRAGRetriever; prefer loading existing FAISS (no re-ingest)."""
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
    backend = visual_model or cfg.get("embedding.visual_backend", "colqwen2")
    if backend == "colqwen2":
        faiss_store = FaissColPaliStore(
            index_path=cfg.get("storage.faiss.colqwen2_index_path"),
            id_map_path=cfg.get("storage.faiss.colqwen2_id_map_path"),
        )
    else:
        faiss_store = FaissColPaliStore()

    bge = BGEEmbedder()
    visual_encoder = None
    use_vis = bool(cfg.get("retrieval.use_visual", True)) and not skip_visual
    if use_vis:
        visual_encoder = create_visual_encoder(model_name=backend)

    chunker = TextChunker()
    bm25 = BM25Retriever()
    dense = DenseRetriever(pg_store, bge)
    visual = (
        VisualRetriever(faiss_store, pg_store, visual_encoder)
        if visual_encoder
        else None
    )
    fusion = RRFFusion(rrf_k=60)
    reranker = Reranker()

    faiss_loaded = faiss_store.load()
    if faiss_loaded:
        try:
            bm25.fit_from_pgvector(pg_store)
            logger.info("index loaded (FAISS + BM25)")
        except Exception as e:
            logger.warning("bm25 fit failed: %s", e)
    elif skip_index:
        logger.warning(
            "FAISS missing with --skip-index; dense/bm25-only if pg has chunks"
        )
    else:
        raise SystemExit(
            "FAISS index missing; run ingest first or pass --skip-index on cloud "
            "with existing index paths"
        )

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


def _write_readme(
    out_dir: Path,
    *,
    summary: Dict[str, Any],
    env: Dict[str, Any],
    n_items: int,
) -> None:
    arms = summary.get("arms") or {}
    go = summary.get("go_nogo") or {}
    lines = [
        "# Phase2 agent dual-arm eval",
        "",
        f"- items: **{n_items}**",
        f"- created: `{env.get('created_at')}`",
        f"- git: `{env.get('git_branch')}` @ `{env.get('git_head', '')[:12]}`",
        f"- llm: `{env.get('llm_model')}`",
        f"- visual: `{env.get('visual_model')}`",
        f"- judge: `{env.get('judge')}`",
        f"- skip_index: `{env.get('skip_index')}`",
        f"- grade_off: `{env.get('grade_off')}`",
        "",
        "## Verdict (draft)",
        "",
        f"**{go.get('verdict', 'n/a')}**",
        "",
        go.get("note") or "",
        "",
    ]
    for c in go.get("checks") or []:
        flag = {True: "PASS", False: "FAIL", None: "SKIP"}.get(c.get("ok"), "?")
        lines.append(f"- [{flag}] `{c.get('id')}`: {c.get('detail')}")
    lines += ["", "## Arms", ""]
    for name, arm in arms.items():
        lat = arm.get("latency_ms") or {}
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"| metric | value |")
        lines.append(f"|--------|------:|")
        lines.append(f"| n | {arm.get('n')} |")
        cr = arm.get("correct_rate")
        lines.append(
            f"| correct_rate | {cr:.4f} |" if cr is not None else "| correct_rate | n/a |"
        )
        ra = arm.get("reject_accuracy")
        lines.append(
            f"| reject_accuracy | {ra:.4f} |"
            if ra is not None
            else "| reject_accuracy | n/a |"
        )
        lines.append(f"| false_reject_count | {arm.get('false_reject_count')} |")
        lines.append(f"| degrade_count | {arm.get('degrade_count')} |")
        lines.append(f"| errors | {arm.get('errors')} |")
        if arm.get("avg_searches") is not None:
            lines.append(f"| avg_searches | {arm.get('avg_searches'):.3f} |")
        if arm.get("avg_llm_calls") is not None:
            lines.append(f"| avg_llm_calls | {arm.get('avg_llm_calls'):.3f} |")
        if lat.get("mean") is not None:
            lines.append(
                f"| latency_ms mean/p50/p95 | "
                f"{lat['mean']:.0f} / {lat.get('p50'):.0f} / {lat.get('p95'):.0f} |"
            )
        lines.append("")
        by_tag = arm.get("by_tag") or {}
        if by_tag:
            lines.append("| tag | n | correct_rate |")
            lines.append("|-----|--:|-------------:|")
            for tag, st in sorted(by_tag.items()):
                rate = st.get("correct_rate")
                rate_s = f"{rate:.4f}" if rate is not None else "n/a"
                lines.append(f"| {tag} | {st.get('n')} | {rate_s} |")
            lines.append("")

    delta = summary.get("delta") or {}
    if delta:
        lines += ["## Delta (agent − pipeline)", ""]
        for k in ("correct_rate", "reject_accuracy", "false_reject_count", "latency_ms_mean"):
            v = delta.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                lines.append(f"- {k}: {v:+.4f}")
            else:
                lines.append(f"- {k}: {v:+d}" if isinstance(v, int) else f"- {k}: {v}")
        for tag, st in (delta.get("by_tag") or {}).items():
            d = st.get("correct_rate")
            if d is not None:
                lines.append(f"- by_tag.{tag}.correct_rate: {d:+.4f}")
        lines.append("")

    lines += [
        "## Decision discipline",
        "",
        "- Do **not** flip `agent.enabled: true` without explicit human Go + config PR.",
        "- NDCG / Boot ablation never use agent path.",
        "- See `results.json` for per-item trajectories.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_env_txt(out_dir: Path, env: Dict[str, Any]) -> None:
    lines = [f"{k}={v}" for k, v in env.items()]
    (out_dir / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_llm_defaults() -> None:
    """Cloud Boot scripts inject llm.*; bare models.yaml often only has models.llm name.

    OpenAI SDK requires a non-empty api_key even for local Ollama.
    """
    from src.config import cfg

    cfg.load()
    data = cfg._data  # noqa: SLF001 — intentional runtime inject for eval
    llm = data.setdefault("llm", {})
    if not llm.get("base_url"):
        llm["base_url"] = os.environ.get(
            "LLM_BASE_URL", "http://localhost:11434/v1"
        )
    if not llm.get("api_key"):
        llm["api_key"] = os.environ.get("OPENAI_API_KEY") or "ollama"
    if not llm.get("model"):
        llm["model"] = os.environ.get(
            "LLM_MODEL",
            (data.get("models") or {}).get("llm") or "qwen2:7b",
        )
    logger.info(
        "llm defaults: model=%s base_url=%s",
        llm.get("model"),
        llm.get("base_url"),
    )


def execute_dual_arm(args: argparse.Namespace) -> int:
    from src.agent.config import agent_config
    from src.agent.eval import (
        load_agent_eval_qa,
        run_dual_arm_item,
        summarize_dual_arm,
    )
    from src.config import cfg
    from src.generation.generator import Generator

    _ensure_llm_defaults()

    tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None
    items = load_agent_eval_qa(
        args.qa_file, tags=tags, max_items=args.max_queries
    )
    if not items:
        logger.error("no eval items")
        return 2

    # Soft local guard: full set is cloud work
    if (
        not args.force_local
        and len(items) > 10
        and os.environ.get("CONFIG_PROFILE") == "local-dev"
    ):
        logger.error(
            "refusing %d items under CONFIG_PROFILE=local-dev "
            "(use --max-queries ≤10, --delegate-local-smoke, or --force-local on cloud)",
            len(items),
        )
        return 2

    visual_model = args.visual_model or cfg.get("embedding.visual_backend", "colqwen2")
    use_visual = bool(cfg.get("retrieval.use_visual", True)) and not args.skip_visual
    use_rerank = bool(cfg.get("retrieval.use_rerank", True)) and not args.no_rerank

    logger.info(
        "Phase2 execute: n=%d judge=%s visual=%s skip_index=%s grade_off=%s",
        len(items),
        args.judge,
        visual_model,
        args.skip_index,
        args.grade_off,
    )

    try:
        retriever = build_retriever(
            skip_index=args.skip_index,
            visual_model=visual_model,
            skip_visual=not use_visual,
        )
        generator = Generator(bge_embedder=getattr(retriever, "bge", None))
    except SystemExit:
        raise
    except Exception as e:
        logger.error("failed to build retriever/generator: %s", e)
        return 1

    agent_cfg: Dict[str, Any] = {"return_trajectory": True}
    if args.grade_off:
        agent_cfg["grade"] = {"enabled": False}
        agent_cfg["max_grade_cycles"] = 0

    # agent budget snapshot for go draft
    base_agent = agent_config()
    max_searches = float(base_agent.get("max_total_searches") or 3)

    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%d')}-agent-eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()
    for i, it in enumerate(items, 1):
        logger.info("[%d/%d] id=%s tag=%s", i, len(items), it.get("id"), it.get("tag"))
        row = run_dual_arm_item(
            it,
            retriever=retriever,
            generator=generator,
            k=args.k,
            use_visual=use_visual,
            use_rerank=use_rerank,
            judge=args.judge,
            agent_cfg=agent_cfg,
            run_pipeline=not args.agent_only,
            run_agent_arm=not args.pipeline_only,
        )
        rows.append(row)
        ag = row.get("agent") or {}
        logger.info(
            "  pipe.correct=%s agent.correct=%s status=%s counts=%s lat_p=%s lat_a=%s",
            (row.get("pipeline") or {}).get("correct"),
            ag.get("correct"),
            ag.get("status"),
            ag.get("counts"),
            (row.get("pipeline") or {}).get("latency_ms"),
            ag.get("latency_ms"),
        )

    summary = summarize_dual_arm(rows)
    # recompute go with budget from config
    if "pipeline" in (summary.get("arms") or {}) and "agent" in (
        summary.get("arms") or {}
    ):
        from src.agent.eval import go_nogo_draft

        summary["go_nogo"] = go_nogo_draft(
            summary["arms"]["pipeline"],
            summary["arms"]["agent"],
            max_total_searches=max_searches,
        )

    meta = _git_meta()
    env = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "phase2_dual_arm",
        "qa_file": str(args.qa_file),
        "n_items": len(rows),
        "k": args.k,
        "judge": args.judge,
        "skip_index": bool(args.skip_index),
        "skip_visual": bool(args.skip_visual),
        "no_rerank": bool(args.no_rerank),
        "grade_off": bool(args.grade_off),
        "visual_model": visual_model,
        "llm_model": cfg.get("llm.model"),
        "llm_base_url": cfg.get("llm.base_url"),
        "config_profile": os.environ.get("CONFIG_PROFILE") or "",
        "use_visual": use_visual,
        "use_rerank": use_rerank,
        "max_total_searches": max_searches,
        "elapsed_s": round(time.perf_counter() - t_all, 1),
        **meta,
    }

    payload = {
        "version": 1,
        "env": env,
        "summary": summary,
        "items": rows,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_env_txt(out_dir, env)
    _write_readme(out_dir, summary=summary, env=env, n_items=len(rows))

    print(
        json.dumps(
            {
                "output": str(out_path),
                "n_items": len(rows),
                "go_nogo": (summary.get("go_nogo") or {}).get("verdict"),
                "summary_arms": {
                    k: {
                        "correct_rate": v.get("correct_rate"),
                        "reject_accuracy": v.get("reject_accuracy"),
                        "avg_searches": v.get("avg_searches"),
                        "latency_ms_mean": (v.get("latency_ms") or {}).get("mean"),
                    }
                    for k, v in (summary.get("arms") or {}).items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info("wrote %s", out_path)
    return 0


def write_skeleton(args: argparse.Namespace, items: List[Dict[str, Any]]) -> int:
    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT
        / "runs"
        / f"agent-eval-skeleton-{_utc_ts()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 2,
        "phase": "skeleton",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_file": str(args.qa_file),
        "skip_index": bool(args.skip_index),
        "n_items": len(items),
        "item_ids": [it.get("id") for it in items],
        "tags_filter": [
            t.strip() for t in (args.tags or "").split(",") if t.strip()
        ]
        or None,
        "protocol": {
            "arms": ["pipeline", "agent"],
            "optional_arm": "agent_grade_off (--grade-off)",
            "metrics": [
                "correct (llm|heuristic)",
                "reject_accuracy",
                "false_reject_count",
                "latency_ms p50/p95",
                "avg_searches",
                "avg_llm_calls",
                "degrade_count",
            ],
            "decision": "do not flip agent.enabled without Phase2 Go",
            "execute": (
                "python scripts/run_agent_eval.py --execute --skip-index "
                "--judge llm --output-dir runs/YYYYMMDD-agent-eval"
            ),
            "cloud": "bash scripts/cloud_agent_eval.sh",
        },
        "items_preview": [
            {
                "id": it.get("id"),
                "tag": it.get("tag"),
                "query": (it.get("query") or "")[:120],
                "expect_reject": bool(it.get("expect_reject")),
            }
            for it in items
        ],
        "summary": {
            "pipeline_correct": None,
            "agent_correct": None,
            "note": "placeholder — not executed; use --execute on cloud",
        },
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# agent eval skeleton (Phase2)\n\n"
        f"- items listed: {len(items)}\n"
        "- status: **not executed** (skeleton)\n"
        "- execute: `python scripts/run_agent_eval.py --execute --skip-index "
        "--judge llm --output-dir runs/YYYYMMDD-agent-eval`\n"
        "- cloud: `bash scripts/cloud_agent_eval.sh`\n"
        "- local ≤10: `scripts/run_agent_local_smoke.py`\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(path), "n_items": len(items)}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Phase2 agent dual-arm eval (pipeline vs agent)"
    )
    parser.add_argument(
        "--qa-file", default=str(ROOT / "data" / "agent_eval_qa.json")
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="limit items; omit for all in file (Phase2 ~40–50)",
    )
    parser.add_argument("--tags", default="", help="comma tags filter")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="do not require rebuild; load existing FAISS/pg (cloud default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write skeleton results only",
    )
    parser.add_argument(
        "--delegate-local-smoke",
        action="store_true",
        help="forward to run_agent_local_smoke (caps at 10)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="full dual-arm on cloud (or --force-local)",
    )
    parser.add_argument(
        "--judge",
        choices=["heuristic", "llm"],
        default="llm",
        help="correctness judge (default llm for Phase2)",
    )
    parser.add_argument(
        "--visual-model",
        default="",
        help="colqwen2|colpali (default from config)",
    )
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--grade-off",
        action="store_true",
        help="disable agent grade/refine cycle (isolation arm; separate out-dir)",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="skip pipeline arm",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="skip agent arm",
    )
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="allow >10 items under local-dev (not recommended)",
    )
    args = parser.parse_args(argv)

    from src.agent.eval import load_agent_eval_qa

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or None
    try:
        items = load_agent_eval_qa(
            args.qa_file, tags=tags, max_items=args.max_queries
        )
    except FileNotFoundError:
        logger.error("QA file not found: %s", args.qa_file)
        return 2

    if not items and not args.execute:
        logger.error("no eval items")
        return 2

    if args.delegate_local_smoke:
        mod = _load_smoke_module()
        n = min(len(items), int(args.max_queries or 5), 10)
        smoke_argv = [
            "--qa-file",
            args.qa_file,
            "--max-queries",
            str(n),
            "--k",
            str(args.k),
        ]
        if args.tags:
            smoke_argv.extend(["--tags", args.tags])
        if args.output_dir:
            smoke_argv.extend(["--output-dir", args.output_dir])
        if args.dry_run:
            smoke_argv.append("--dry-run")
        if args.skip_visual:
            smoke_argv.append("--skip-visual")
        return int(mod.main(smoke_argv))

    if args.execute:
        return execute_dual_arm(args)

    # default: skeleton plan
    if not args.dry_run:
        logger.info(
            "Phase2 skeleton: writing plan only (no LLM). "
            "Pass --execute for cloud dual-arm, or --delegate-local-smoke for ≤10."
        )
    return write_skeleton(args, items)


if __name__ == "__main__":
    raise SystemExit(main())
