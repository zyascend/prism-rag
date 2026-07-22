#!/usr/bin/env python3
"""RAG Failure Diagnostics Clinic CLI.

用法示例::

    # 启发式（无 LLM）
    python scripts/run_failure_clinic.py --text "deleted PDF still recalled by BM25"

    # 从文件
    python scripts/run_failure_clinic.py --file runs/.../badcase.md --heuristic

    # LLM triage（需 OPENAI_API_KEY 或配置 llm.*）
    python scripts/run_failure_clinic.py --text "..." --llm

    # 列出模式
    python scripts/run_failure_clinic.py --list-patterns

输出默认打印 Markdown；``--json-out path`` 写结构化结果。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 保证仓库根在 path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_llm_complete():
    from src.config import cfg

    try:
        cfg.load()
    except Exception:
        pass
    from openai import OpenAI

    base = cfg.get("llm.base_url", "https://api.openai.com/v1")
    key = cfg.get("llm.api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    model = cfg.get("llm.model", "gpt-4o-mini")
    if not key and "localhost" not in str(base) and "127.0.0.1" not in str(base):
        # Ollama 等本地 endpoint 常不需要真 key
        key = key or "ollama"
    client = OpenAI(base_url=base, api_key=key or "x")

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    return complete


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PrismRAG Failure Diagnostics Clinic")
    p.add_argument("--text", type=str, help="Bug description text")
    p.add_argument("--file", type=str, help="Read bug description from file")
    p.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM judge (default: heuristic only)",
    )
    p.add_argument(
        "--heuristic",
        action="store_true",
        help="Force heuristic mode even if --llm set",
    )
    p.add_argument("--list-patterns", action="store_true", help="Print P01–P12 catalog")
    p.add_argument("--json-out", type=str, default="", help="Write diagnosis JSON")
    p.add_argument(
        "--example",
        choices=["p01", "p04", "p09", ""],
        default="",
        help="Use a built-in example instead of --text/--file",
    )
    args = p.parse_args(argv)

    from src.diagnostics import (
        diagnose_failure,
        format_diagnosis_markdown,
        list_patterns,
    )

    if args.list_patterns:
        for pat in list_patterns():
            print(f"{pat['id']}: {pat['name']}")
            print(f"  {pat['summary']}")
            print(f"  PrismRAG: {pat['prismrag_hints']}")
            print()
        return 0

    examples = {
        "p01": (
            "User asked if Bitcoin is accepted. Retrieved FAQ says credit cards only. "
            "Model answered yes we support Bitcoin via third-party gateway. No errors in logs."
        ),
        "p04": (
            "After delete_document, BM25 and visual search still return chunks from the "
            "deleted industrial manual. index_version was not bumped."
        ),
        "p09": (
            "Faithfulness on 50 queries is 0.89 but full 283-query run drops to 0.77. "
            "Online /ask path differs from offline RAGAS because eval_via_generator is false."
        ),
    }

    if args.example:
        bug = examples[args.example]
    elif args.file:
        bug = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        bug = args.text
    else:
        p.error("Provide --text, --file, --example, or --list-patterns")

    complete_fn = None
    if args.llm and not args.heuristic:
        complete_fn = _build_llm_complete()

    diag = diagnose_failure(bug, complete_fn=complete_fn)
    print(format_diagnosis_markdown(diag))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in diag.items() if k != "raw"}
        payload["bug_description"] = bug
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
