"""
data_auditor.py
---------------
Cross-validates financial metrics from multiple sources, flags discrepancies,
and generates consensus values for use in valuation models.

Thresholds:
  - Indian companies: flag discrepancies > 15%
  - US companies: flag discrepancies > 20%
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

logger = logging.getLogger(__name__)

# Metrics to audit across sources
_AUDITED_METRICS = [
    "revenue",
    "net_income",
    "ebitda",
    "total_debt",
    "cash",
    "free_cash_flow",
    "shares_outstanding",
    "current_price",
    "book_value",
    "dividend_per_share",
    "return_on_equity",
    "beta",
    "market_cap",
]

# Human-readable labels for display
_METRIC_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "net_income": "Net Income",
    "ebitda": "EBITDA",
    "total_debt": "Total Debt",
    "cash": "Cash & Equivalents",
    "free_cash_flow": "Free Cash Flow",
    "shares_outstanding": "Shares Outstanding",
    "current_price": "Current Price",
    "book_value": "Book Value",
    "dividend_per_share": "Dividend Per Share",
    "return_on_equity": "Return on Equity",
    "beta": "Beta",
    "market_cap": "Market Cap",
}


def _pct_spread(values: list[float]) -> float:
    """Return the max deviation from the median as a percentage of the median."""
    if len(values) < 2:
        return 0.0
    med = statistics.median(values)
    if med == 0:
        return 0.0
    return max(abs(v - med) / abs(med) for v in values) * 100


def _consensus(values: list[float]) -> float:
    """Return the consensus (median) value."""
    return statistics.median(values)


class MetricAudit:
    """Holds the audit result for a single metric."""

    def __init__(
        self,
        metric: str,
        source_values: dict[str, float],
        threshold_pct: float,
    ) -> None:
        self.metric = metric
        self.label = _METRIC_LABELS.get(metric, metric)
        self.source_values = source_values  # {source_name: value}
        self.threshold_pct = threshold_pct

        valid_values = [v for v in source_values.values() if v is not None and v != 0]
        self.available_sources = len(valid_values)
        self.consensus_value: float | None = _consensus(valid_values) if valid_values else None
        self.spread_pct: float = _pct_spread(valid_values) if valid_values else 0.0
        self.is_flagged: bool = self.spread_pct > threshold_pct
        self.has_data: bool = bool(valid_values)

    @property
    def status_icon(self) -> str:
        if not self.has_data:
            return "⚠️"
        if self.is_flagged:
            return "🚩"
        return "✅"

    def __repr__(self) -> str:
        return (
            f"MetricAudit({self.metric}, consensus={self.consensus_value:.2f}, "
            f"spread={self.spread_pct:.1f}%, flagged={self.is_flagged})"
        )


class AuditReport:
    """Full audit report for a company across all metrics and sources."""

    def __init__(
        self,
        ticker: str,
        is_indian: bool,
        metric_audits: list[MetricAudit],
        sources: list[dict[str, Any]],
        currency: str,
        company_name: str,
    ) -> None:
        self.ticker = ticker
        self.is_indian = is_indian
        self.metric_audits = metric_audits
        self.sources = sources
        self.currency = currency
        self.company_name = company_name
        self.threshold_pct = 15.0 if is_indian else 20.0

        self.flagged_metrics = [m for m in metric_audits if m.is_flagged]
        self.missing_metrics = [m for m in metric_audits if not m.has_data]
        self.clean_metrics = [
            m for m in metric_audits if m.has_data and not m.is_flagged
        ]
        self.successful_sources = [s for s in sources if s.get("fetch_success")]

    def get_consensus(self, metric: str) -> float | None:
        """Return the consensus value for a given metric."""
        for audit in self.metric_audits:
            if audit.metric == metric:
                return audit.consensus_value
        return None

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary of consensus values for all audited metrics."""
        return {
            audit.metric: audit.consensus_value
            for audit in self.metric_audits
            if audit.has_data
        }

    def summary_lines(self) -> list[str]:
        """Generate a human-readable audit summary."""
        lines: list[str] = []
        currency_symbol = "₹" if self.currency == "INR" else "$"

        lines.append("=" * 70)
        lines.append(f"  DATA AUDIT REPORT — {self.company_name} ({self.ticker})")
        lines.append("=" * 70)
        lines.append(
            f"  Sources queried : {len(self.sources)} | "
            f"Successful: {len(self.successful_sources)}"
        )
        lines.append(
            f"  Discrepancy threshold: {self.threshold_pct:.0f}%  |  "
            f"Currency: {self.currency} ({currency_symbol})"
        )
        lines.append("-" * 70)

        for audit in self.metric_audits:
            if not audit.has_data:
                lines.append(f"  {audit.status_icon}  {audit.label:<30} — No data available")
                continue

            # Build per-source values string
            src_parts = []
            for src_name, val in audit.source_values.items():
                if val is not None:
                    formatted = _fmt_value(val, audit.metric, currency_symbol)
                    src_parts.append(f"{formatted} ({src_name})")

            sources_str = "  |  ".join(src_parts)
            consensus_str = _fmt_value(audit.consensus_value, audit.metric, currency_symbol)

            if audit.is_flagged:
                lines.append(
                    f"  {audit.status_icon}  {audit.label:<30} "
                    f"Consensus: {consensus_str}  [spread {audit.spread_pct:.1f}% > {audit.threshold_pct:.0f}% threshold]"
                )
                lines.append(f"       Sources: {sources_str}")
            else:
                lines.append(
                    f"  {audit.status_icon}  {audit.label:<30} "
                    f"Consensus: {consensus_str}  [spread {audit.spread_pct:.1f}%]"
                )
                lines.append(f"       Sources: {sources_str}")

        lines.append("-" * 70)
        if self.flagged_metrics:
            lines.append(
                f"  ⚠️  {len(self.flagged_metrics)} metric(s) have high source disagreement "
                f"(>{self.threshold_pct:.0f}%): "
                + ", ".join(m.label for m in self.flagged_metrics)
            )
        else:
            lines.append("  ✅  All metrics show strong cross-source agreement.")
        lines.append("=" * 70)
        return lines


def _fmt_value(value: float | None, metric: str, currency_symbol: str) -> str:
    """Format a numeric value for display based on metric type."""
    if value is None:
        return "N/A"

    # Percentage metrics
    if metric in ("return_on_equity", "revenue_growth", "earnings_growth", "profit_margin"):
        return f"{value * 100:.1f}%"

    # Ratio / small number metrics
    if metric in ("beta", "pe_ratio", "debt_to_equity"):
        return f"{value:.2f}x"

    # Per-share price metrics (small currency values)
    if metric in ("current_price", "dividend_per_share"):
        return f"{currency_symbol}{value:,.2f}"

    # Share count — no currency symbol
    if metric == "shares_outstanding":
        if abs(value) >= 1e9:
            return f"{value / 1e9:,.2f}B shares"
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.1f}M shares"
        return f"{value:,.0f} shares"

    # Large financial figures — scale to Cr (INR) or B/M (USD)
    if currency_symbol == "₹":
        if abs(value) >= 1e7:
            return f"₹{value / 1e7:,.1f} Cr"
        return f"₹{value:,.0f}"
    else:
        if abs(value) >= 1e9:
            return f"${value / 1e9:,.2f}B"
        if abs(value) >= 1e6:
            return f"${value / 1e6:,.1f}M"
        return f"${value:,.0f}"


class DataAuditor:
    """
    Cross-validates metrics from multiple data sources and produces an AuditReport.

    Usage:
        auditor = DataAuditor(ticker, is_indian=True)
        report = auditor.audit(sources)
    """

    def __init__(self, ticker: str, is_indian: bool) -> None:
        self.ticker = ticker
        self.is_indian = is_indian
        self.threshold_pct = 15.0 if is_indian else 20.0

    def audit(self, sources: list[dict[str, Any]]) -> AuditReport:
        """
        Build an AuditReport from a list of source dicts returned by MultiSourceFetcher.

        Parameters
        ----------
        sources : list of dicts
            Each dict must have a 'source' key and 'fetch_success' key,
            plus optional financial metric keys.
        """
        successful = [s for s in sources if s.get("fetch_success")]

        # Determine currency and company name from the first successful source
        currency = "INR" if self.is_indian else "USD"
        company_name = self.ticker
        for s in successful:
            if s.get("currency"):
                currency = s["currency"]
            if s.get("company_name"):
                company_name = s["company_name"]
                break

        metric_audits: list[MetricAudit] = []
        for metric in _AUDITED_METRICS:
            source_values: dict[str, float] = {}
            for s in successful:
                val = s.get(metric)
                if val is not None:
                    try:
                        source_values[s["source"]] = float(val)
                    except (TypeError, ValueError):
                        pass
            audit = MetricAudit(metric, source_values, self.threshold_pct)
            metric_audits.append(audit)

        return AuditReport(
            ticker=self.ticker,
            is_indian=self.is_indian,
            metric_audits=metric_audits,
            sources=sources,
            currency=currency,
            company_name=company_name,
        )


def compute_revenue_cagr(historical_revenues: list[float], years: int = 3) -> float | None:
    """
    Compute CAGR from a list of historical revenues (most-recent first).

    Parameters
    ----------
    historical_revenues : list of floats (most-recent first)
    years : int — number of years for the CAGR calculation

    Returns
    -------
    CAGR as a decimal (e.g., 0.125 for 12.5%) or None if insufficient data.
    """
    if not historical_revenues or len(historical_revenues) < 2:
        return None
    n = min(years, len(historical_revenues) - 1)
    latest = historical_revenues[0]
    oldest = historical_revenues[n]
    if oldest is None or oldest <= 0 or latest is None or latest <= 0:
        return None
    return (latest / oldest) ** (1 / n) - 1
