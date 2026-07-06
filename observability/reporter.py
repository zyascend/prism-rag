"""报告生成 — Markdown + JSON 输出到 runs/<run_id>/observability/"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


def generate_report(snapshot: dict[str, Any], run_id: str) -> Path:
    """生成完整报告到 runs/<run_id>/observability/

    Args:
        snapshot: MetricsCollector.snapshot() 返回的完整快照
        run_id: 运行标识，如 "20260705-ragas-eval"

    Returns:
        Path to the report directory
    """
    out_dir = Path("runs") / run_id / "observability"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON files
    configs = snapshot.get("configs", {})
    (out_dir / "metrics.json").write_text(
        json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    traces = snapshot.get("traces", [])
    with open(out_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    alerts = snapshot.get("alerts", [])
    (out_dir / "alerts.json").write_text(
        json.dumps(alerts, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Per-query RAGAS details
    ragas_details = snapshot.get("ragas_details", {})
    if ragas_details:
        (out_dir / "ragas_details.json").write_text(
            json.dumps(ragas_details, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Markdown report
    md = _build_markdown(snapshot, run_id)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    return out_dir


def _build_markdown(snapshot: dict[str, Any], run_id: str) -> str:
    configs = snapshot.get("configs", {})
    alerts = snapshot.get("alerts", [])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Observability Report — {now}",
        "",
        f"**Run:** `{run_id}`",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    if configs:
        lines.append(
            "| Config | N | P50 | P95 | Avg Latency | B-Hits | D-Hits | V-Hits | Faith | Relev | CtxRel |"
        )
        lines.append(
            "|--------|---|---|-----|-------------|--------|--------|--------|-------|-------|--------|"
        )
        for label, m in configs.items():
            lat = m.get("latency", {})
            hits = m.get("hits", {})
            qual = m.get("quality", {})
            faith = qual.get("avg_faithfulness")
            relev = qual.get("avg_answer_relevancy")
            ctxrel = qual.get("avg_context_relevancy")
            faith_str = f"{faith:.3f}" if faith is not None else "—"
            relev_str = f"{relev:.3f}" if relev is not None else "—"
            ctxrel_str = f"{ctxrel:.3f}" if ctxrel is not None and ctxrel > 0 else "—"
            lines.append(
                f"| {label} "
                f"| {m.get('num_queries', 0)} "
                f"| {lat.get('p50_ms', 0):.0f}ms "
                f"| {lat.get('p95_ms', 0):.0f}ms "
                f"| {lat.get('avg_ms', 0):.0f}ms "
                f"| {hits.get('avg_bm25', 0):.1f} "
                f"| {hits.get('avg_dense', 0):.1f} "
                f"| {hits.get('avg_visual', 0):.1f} "
                f"| {faith_str} "
                f"| {relev_str} "
                f"| {ctxrel_str} "
                "|"
            )

    # Alerts
    lines.append("")
    lines.append(f"## Alerts ({len(alerts)})")
    lines.append("")
    if alerts:
        for a in alerts:
            icon = "⛔" if a.get("level") == "error" else "⚠"
            lines.append(
                f"- {icon} {a.get('timestamp', '')} | "
                f"`{a.get('config_label', '')}` | "
                f"{a.get('message', '')}"
            )
    else:
        lines.append("No alerts.")

    # Per-Config detail
    lines.append("")
    lines.append("## Per-Config Detail")
    for label, m in configs.items():
        lat = m.get("latency", {})
        lines.append(f"### {label}")
        lines.append(f"- **Queries:** {m.get('num_queries', 0)}")
        lines.append(f"- **Latency:** P50={lat.get('p50_ms', 0):.0f}ms, "
                      f"P95={lat.get('p95_ms', 0):.0f}ms, "
                      f"P99={lat.get('p99_ms', 0):.0f}ms, "
                      f"Avg={lat.get('avg_ms', 0):.0f}ms "
                      f"(min={lat.get('min_ms', 0):.0f}, max={lat.get('max_ms', 0):.0f})")
        hits = m.get("hits", {})
        lines.append(f"- **Hits:** BM25={hits.get('avg_bm25', 0):.1f}, "
                      f"Dense={hits.get('avg_dense', 0):.1f}, "
                      f"Visual={hits.get('avg_visual', 0):.1f}")
        cache = m.get("cache", {})
        lines.append(f"- **HyDE cache hit rate:** {cache.get('hyde_hit_rate', 0):.1%}")
        qual = m.get("quality", {})
        if qual.get("avg_faithfulness") is not None:
            lines.append(f"- **Faithfulness:** {qual.get('avg_faithfulness', 0):.3f}")
            lines.append(f"- **Answer Relevancy:** {qual.get('avg_answer_relevancy', 0):.3f}")
        ctxrel = qual.get("avg_context_relevancy", 0)
        if ctxrel > 0:
            lines.append(f"- **Context Relevance:** {ctxrel:.3f}")
        lines.append("")

    # RAGAS distribution summary
    ragas_details = snapshot.get("ragas_details", {})
    if ragas_details:
        lines.append("## RAGAS Score Distribution")
        lines.append("")
        for label, queries in ragas_details.items():
            if not queries:
                continue
            faith_scores = [q["faithfulness"] for q in queries]
            relev_scores = [q["answer_relevancy"] for q in queries]
            ctxrel_scores = [q["context_relevancy"] for q in queries if q.get("context_relevancy", 0) > 0]
            lines.append(f"### {label} ({len(queries)} queries)")
            lines.append("")
            lines.append("| Metric | Min | P25 | P50 | P75 | Max | Mean |")
            lines.append("|--------|-----|-----|-----|-----|-----|------|")
            for name, scores in [("Faithfulness", faith_scores), ("Answer Relevancy", relev_scores)]:
                if scores:
                    s = sorted(scores)
                    n = len(s)
                    lines.append(
                        f"| {name} | {s[0]:.3f} | {_pctl(s, 25):.3f} | {_pctl(s, 50):.3f} "
                        f"| {_pctl(s, 75):.3f} | {s[-1]:.3f} | {sum(s)/n:.3f} |"
                    )
            if ctxrel_scores:
                s = sorted(ctxrel_scores)
                n = len(s)
                lines.append(
                    f"| Context Relevance | {s[0]:.4f} | {_pctl(s, 25):.4f} | {_pctl(s, 50):.4f} "
                    f"| {_pctl(s, 75):.4f} | {s[-1]:.4f} | {sum(s)/n:.4f} |"
                )
            lines.append("")

    return "\n".join(lines)


def _pctl(sorted_data: list[float], p: float) -> float:
    """百分位数辅助函数"""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]