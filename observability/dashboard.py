"""Rich 终端仪表盘 — 评测运行时实时刷新"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout

if TYPE_CHECKING:
    from src.observability.collectors import MetricsCollector


class Dashboard:
    """评测运行时终端 Live 面板

    Usage:
        dashboard = Dashboard()
        dashboard.start(collector)
        # ... run eval loop, periodically call dashboard.update() ...
        dashboard.stop()
    """

    def __init__(self):
        self._live: Live | None = None
        self._collector: MetricsCollector | None = None
        self._start_time: float = 0.0

    def start(self, collector: MetricsCollector) -> None:
        self._collector = collector
        self._start_time = time.time()
        self._layout = self._build_layout()
        self._live = Live(
            self._layout, console=Console(), refresh_per_second=2, screen=True
        )
        self._live.start()

    def update(self) -> None:
        if self._live:
            self._live.update(self._build_layout())

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="alerts"),
        )
        layout["header"].update(self._build_header())
        layout["body"].update(self._build_body())
        layout["alerts"].update(self._build_alerts())
        return layout

    def _build_header(self) -> Panel:
        runtime = time.time() - self._start_time
        mins, secs = divmod(int(runtime), 60)
        snap = self._collector.snapshot() if self._collector else {}
        configs = snap.get("configs", {})
        total_queries = sum(c.get("num_queries", 0) for c in configs.values())
        active_configs = list(configs.keys())
        current = active_configs[-1] if active_configs else "—"

        return Panel(
            f"[bold cyan]PrismRAG Observability[/bold cyan]\n"
            f"Config: [yellow]{current}[/yellow]    "
            f"Queries: [green]{total_queries}[/green]    "
            f"Runtime: [blue]{mins}m {secs}s[/blue]",
            title="Status",
        )

    def _build_body(self) -> Table:
        table = Table(title="Per-Config Metrics", expand=True)
        table.add_column("Config", style="cyan", width=20)
        table.add_column("N", justify="right", width=6)
        table.add_column("P50", justify="right", width=8)
        table.add_column("P95", justify="right", width=8)
        table.add_column("Avg", justify="right", width=8)
        table.add_column("B/D/V Hits", width=16)
        table.add_column("Faith", justify="right", width=7)
        table.add_column("Relev", justify="right", width=7)
        table.add_column("CtxRel", justify="right", width=7)

        if self._collector:
            snap = self._collector.snapshot()
            for label, m in snap.get("configs", {}).items():
                lat = m.get("latency", {})
                hits = m.get("hits", {})
                qual = m.get("quality", {})
                ctxrel = qual.get("avg_context_relevancy", 0)
                table.add_row(
                    label[:20],
                    str(m.get("num_queries", 0)),
                    f"{lat.get('p50_ms', 0):.0f}",
                    f"{lat.get('p95_ms', 0):.0f}",
                    f"{lat.get('avg_ms', 0):.0f}",
                    f"{hits.get('avg_bm25', 0):.0f}/{hits.get('avg_dense', 0):.0f}/{hits.get('avg_visual', 0):.0f}",
                    f"{qual.get('avg_faithfulness', 0):.3f}" if qual.get("avg_faithfulness") else "—",
                    f"{qual.get('avg_answer_relevancy', 0):.3f}" if qual.get("avg_answer_relevancy") else "—",
                    f"{ctxrel:.3f}" if ctxrel > 0 else "—",
                )
        return table

    def _build_alerts(self) -> Panel:
        if not self._collector:
            return Panel("", title="Alerts")
        alerts = self._collector.get_alerts()
        count_warn = sum(1 for a in alerts if a.level == "warning")
        count_err = sum(1 for a in alerts if a.level == "error")
        header = f"Alerts: [yellow]⚠ {count_warn}[/yellow] [red]⛔ {count_err}[/red]"

        lines = []
        for a in alerts[-5:]:  # last 5
            icon = "[red]⛔[/red]" if a.level == "error" else "[yellow]⚠[/yellow]"
            ts = a.timestamp.strftime("%H:%M:%S") if a.timestamp else ""
            lines.append(f"  {icon} {ts} | {a.config_label} | {a.message[:80]}")
        body = "\n".join(lines) if lines else "  No alerts"

        return Panel(body, title=header)