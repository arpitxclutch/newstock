"""
cross_verify.py
---------------
Cross-verifies audited financial data by comparing consensus values across
multiple sources and flagging discrepancies.

This module provides a summary of source agreement for key metrics
before passing data to the valuation engine.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def cross_verify(audit_report: Any) -> dict[str, Any]:
    """
    Run cross-verification on an AuditReport and return a summary dict.

    Returns
    -------
    dict with keys:
      - 'verified_metrics': list of metric names that passed
      - 'flagged_metrics': list of metric names that failed
      - 'consensus_data': dict of {metric: consensus_value}
      - 'source_count': number of successful sources
      - 'overall_quality': 'HIGH' | 'MEDIUM' | 'LOW'
      - 'verification_lines': list of human-readable lines
    """
    verified: list[str] = []
    flagged: list[str] = []
    consensus_data: dict[str, Any] = {}

    for audit in audit_report.metric_audits:
        if not audit.has_data:
            continue
        consensus_data[audit.metric] = audit.consensus_value
        if audit.is_flagged:
            flagged.append(audit.metric)
        else:
            verified.append(audit.metric)

    source_count = len(audit_report.successful_sources)
    total_audited = len(verified) + len(flagged)

    if total_audited == 0:
        quality = "LOW"
    elif len(flagged) == 0 and source_count >= 2:
        quality = "HIGH"
    elif len(flagged) <= 2 and source_count >= 2:
        quality = "MEDIUM"
    else:
        quality = "LOW"

    lines = _build_lines(
        audit_report, verified, flagged, source_count, quality
    )

    return {
        "verified_metrics": verified,
        "flagged_metrics": flagged,
        "consensus_data": consensus_data,
        "source_count": source_count,
        "overall_quality": quality,
        "verification_lines": lines,
    }


def _build_lines(
    report: Any,
    verified: list[str],
    flagged: list[str],
    source_count: int,
    quality: str,
) -> list[str]:
    quality_icons = {"HIGH": "✅", "MEDIUM": "⚠️", "LOW": "🚩"}
    icon = quality_icons.get(quality, "⚠️")

    lines: list[str] = []
    lines.append("-" * 70)
    lines.append(
        f"  CROSS-VERIFICATION SUMMARY  {icon} Data Quality: {quality}"
    )
    lines.append(
        f"  Sources used: {source_count}  |  "
        f"Metrics verified: {len(verified)}  |  "
        f"Metrics flagged: {len(flagged)}"
    )

    if flagged:
        lines.append(
            f"\n  ⚠️  Flagged metrics (discrepancy > threshold):"
        )
        for m in flagged:
            audit = next(
                (a for a in report.metric_audits if a.metric == m), None
            )
            if audit:
                lines.append(
                    f"    • {audit.label}: {audit.spread_pct:.1f}% spread across sources"
                )
        lines.append(
            "  → Consensus (median) values used in valuation despite discrepancies."
        )
    else:
        lines.append(
            "  ✅ All available metrics are in strong agreement across sources."
        )

    lines.append("-" * 70)
    return lines
