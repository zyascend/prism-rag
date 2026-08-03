#!/usr/bin/env python
"""Phase2 云上双臂骨架：pipeline vs agent on data/agent_eval_qa.json.

本机不要跑全量（AGENTS.md）。云上示例：

  # 索引已在数据盘；与 Boot 同 backbone / 同生成模型
  PYTHONPATH=. python scripts/run_agent_eval.py \\
    --qa-file data/agent_eval_qa.json \\
    --max-queries 50 \\
    --skip-index \\
    --output-dir runs/YYYYMMDD-agent-dual-arm

本脚本默认 **只写骨架 results.json（不调 LLM）**。
本机 ≤10 真链路请用：

  PYTHONPATH=. python scripts/run_agent_local_smoke.py --max-queries 5

或：

  PYTHONPATH=. python scripts/run_agent_eval.py --delegate-local-smoke --max-queries 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Phase2 agent dual-arm eval skeleton (cloud; not local full run)"
    )
    parser.add_argument("--qa-file", default=str(ROOT / "data" / "agent_eval_qa.json"))
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
        help="parity flag with run_eval (documented for cloud); skeleton does not rebuild index",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write skeleton results only (default behavior when not --delegate-local-smoke)",
    )
    parser.add_argument(
        "--delegate-local-smoke",
        action="store_true",
        help="forward to run_agent_local_smoke (caps at 10) for small dual-arm",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="reserved: full dual-arm on cloud (not implemented in skeleton; use smoke or Phase2)",
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

    if not items:
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
        return int(mod.main(smoke_argv))

    if args.execute:
        logger.error(
            "--execute full dual-arm is Phase2 work (expand agent_eval_qa to ~40–50 + cloud). "
            "For ≤10 local: --delegate-local-smoke. Skeleton writes plan only."
        )
        return 2

    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT
        / "runs"
        / f"agent-eval-skeleton-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        logger.info(
            "Phase2 skeleton: writing plan only (no LLM). "
            "Pass --dry-run to be explicit, or --delegate-local-smoke for ≤10 local dual-arm."
        )

    payload = {
        "version": 1,
        "phase": "skeleton",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_file": str(args.qa_file),
        "skip_index": bool(args.skip_index),
        "n_items": len(items),
        "item_ids": [it.get("id") for it in items],
        "tags_filter": tags,
        "protocol": {
            "arms": ["pipeline", "agent"],
            "metrics_placeholder": [
                "correct_placeholder (string match)",
                "reject_accuracy",
                "latency_ms",
                "avg_searches",
                "avg_llm_calls",
                "degrade_count",
            ],
            "decision": "do not flip agent.enabled without Phase2 Go",
        },
        "cloud_hint": (
            "On GPU host with index+models: dual-arm loop via "
            "agent_answer_for_eval + answer_for_eval (see run_agent_local_smoke.run_one_item); "
            "archive under runs/YYYYMMDD-agent-*/"
        ),
        "items_preview": [
            {
                "id": it.get("id"),
                "tag": it.get("tag"),
                "query": (it.get("query") or "")[:120],
                "expect_reject": bool(it.get("expect_reject")),
            }
            for it in items
        ],
        "per_item_results": [],
        "summary": {
            "pipeline_correct": None,
            "agent_correct": None,
            "note": "placeholder — not executed",
        },
    }
    path = out_dir / "results.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# agent eval skeleton (Phase2)\n\n"
        f"- items listed: {len(items)}\n"
        "- status: **not executed** (skeleton)\n"
        "- local ≤10: `scripts/run_agent_local_smoke.py`\n"
        "- entry: `src/agent/eval.py::agent_answer_for_eval`\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(path), "n_items": len(items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
