"""
source_attribution.py
---------------------
Tracks and displays data provenance — which source provided each data point,
and how it was validated / adjusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataPoint:
    """A single audited data point with full provenance."""

    metric: str
    value: float | None
    primary_source: str
    all_sources: dict[str, float] = field(default_factory=dict)
    is_consensus: bool = False
    was_flagged: bool = False
    spread_pct: float = 0.0
    note: str = ""

    def citation(self) -> str:
        """Return a short citation string for display."""
        if self.is_consensus:
            src_names = ", ".join(self.all_sources.keys())
            return f"Consensus ({src_names})"
        return self.primary_source


class SourceAttribution:
    """
    Registry that maps each metric to its DataPoint for a given company.

    Usage
    -----
    attribution = SourceAttribution.from_audit_report(report)
    print(attribution.cite("revenue"))
    """

    def __init__(self) -> None:
        self._points: dict[str, DataPoint] = {}

    def register(self, point: DataPoint) -> None:
        self._points[point.metric] = point

    def get(self, metric: str) -> DataPoint | None:
        return self._points.get(metric)

    def cite(self, metric: str) -> str:
        """Return a citation string for a metric, or 'Unknown Source'."""
        point = self._points.get(metric)
        return point.citation() if point else "Unknown Source"

    @classmethod
    def from_audit_report(cls, report: Any) -> "SourceAttribution":
        """Build a SourceAttribution from a DataAuditor AuditReport."""
        attribution = cls()
        for audit in report.metric_audits:
            if not audit.has_data:
                continue
            primary = next(iter(audit.source_values), "Unknown")
            point = DataPoint(
                metric=audit.metric,
                value=audit.consensus_value,
                primary_source=primary,
                all_sources=dict(audit.source_values),
                is_consensus=len(audit.source_values) > 1,
                was_flagged=audit.is_flagged,
                spread_pct=audit.spread_pct,
            )
            attribution.register(point)
        return attribution

    def provenance_lines(self) -> list[str]:
        """Return formatted provenance lines for display."""
        lines = ["  DATA PROVENANCE"]
        lines.append("  " + "-" * 50)
        for metric, point in self._points.items():
            from data_auditor import _METRIC_LABELS
            label = _METRIC_LABELS.get(metric, metric)
            flag = " 🚩" if point.was_flagged else ""
            lines.append(f"  {label:<30} ← {point.citation()}{flag}")
        return lines
