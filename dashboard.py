"""
dashboard.py
------------
Main entry point for the NewStock analysis dashboard.

For any given ticker, the dashboard:
  1. Fetches data from 3+ credible sources (Indian: Yahoo/Screener/NSE; US: Yahoo/SEC/AlphaVantage)
  2. Audits and cross-validates all financial metrics
  3. Selects the best Damodaran valuation model with detailed rationale
  4. Runs the valuation using consensus / audited data
  5. Displays full output with source attribution

Usage:
    python dashboard.py TATAMOTORS.NS
    python dashboard.py NVDA
    python dashboard.py AAPL --av-key YOUR_ALPHA_VANTAGE_KEY
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from multi_source_fetcher import MultiSourceFetcher
from data_auditor import DataAuditor
from source_attribution import SourceAttribution
from enhanced_model_selector import EnhancedModelSelector
from valuation_engine import ValuationEngine
from cross_verify import cross_verify

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")


def run_dashboard(ticker: str, av_api_key: str | None = None, verbose: bool = False) -> None:
    """
    Run the full analysis dashboard for the given ticker and print results to stdout.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    ticker = ticker.upper().strip()
    is_indian = _is_indian_ticker(ticker)
    market_label = "🇮🇳 Indian Market (NSE/BSE)" if is_indian else "🇺🇸 US Market (NYSE/NASDAQ)"
    currency_label = "INR (₹)" if is_indian else "USD ($)"

    print("\n" + "=" * 70)
    print(f"  NEWSTOCK ANALYSIS DASHBOARD")
    print(f"  Ticker  : {ticker}")
    print(f"  Market  : {market_label}")
    print(f"  Currency: {currency_label}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Multi-source data fetch
    # ------------------------------------------------------------------ #
    print("\n  ⏳ Fetching data from multiple sources...")
    start_time = time.time()

    fetcher = MultiSourceFetcher(ticker, av_api_key=av_api_key)
    sources = fetcher.fetch_all()

    elapsed = time.time() - start_time
    successful = [s for s in sources if s.get("fetch_success")]
    print(
        f"  ✓  Data fetched from {len(successful)}/{len(sources)} sources "
        f"in {elapsed:.1f}s"
    )

    if not successful:
        print("\n  ❌  ERROR: No data sources returned valid data.")
        print("       Check your internet connection and verify the ticker symbol.")
        print("       For Indian stocks, use suffix .NS (NSE) or .BO (BSE).")
        print("       Example: TATAMOTORS.NS  or  NVDA")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 2: Data audit
    # ------------------------------------------------------------------ #
    print("\n  ⏳ Auditing data across sources...")
    auditor = DataAuditor(ticker, is_indian=is_indian)
    audit_report = auditor.audit(sources)

    # Print audit report
    print()
    for line in audit_report.summary_lines():
        print(line)

    # ------------------------------------------------------------------ #
    # Step 3: Source attribution
    # ------------------------------------------------------------------ #
    attribution = SourceAttribution.from_audit_report(audit_report)

    # ------------------------------------------------------------------ #
    # Step 4: Cross-verification
    # ------------------------------------------------------------------ #
    cv_result = cross_verify(audit_report)
    print()
    for line in cv_result["verification_lines"]:
        print(line)

    if cv_result["overall_quality"] == "LOW" and len(successful) < 2:
        print(
            "\n  ⚠️  WARNING: Only one data source available. "
            "Valuation accuracy may be limited."
        )

    # ------------------------------------------------------------------ #
    # Step 5: Enhanced model selection
    # ------------------------------------------------------------------ #
    print("\n  ⏳ Selecting valuation model...")
    selector = EnhancedModelSelector(ticker, is_indian=is_indian)
    model_result = selector.select(audit_report, attribution)

    print()
    for line in model_result.rationale_lines():
        print(line)

    # ------------------------------------------------------------------ #
    # Step 6: Valuation
    # ------------------------------------------------------------------ #
    print("\n  ⏳ Running valuation...")
    engine = ValuationEngine()
    valuation = engine.value(model_result, audit_report)

    print()
    for line in valuation.summary_lines():
        print(line)

    # ------------------------------------------------------------------ #
    # Step 7: Data provenance footer
    # ------------------------------------------------------------------ #
    print()
    for line in attribution.provenance_lines():
        print(line)

    print()
    print("  ℹ️  Note: All financial data is fetched live from public APIs.")
    print("      Valuations are based on the Damodaran framework and are for")
    print("      informational purposes only. Not investment advice.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NewStock Analysis Dashboard — Multi-source, Damodaran-powered valuation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dashboard.py TATAMOTORS.NS
  python dashboard.py NVDA
  python dashboard.py RELIANCE.NS
  python dashboard.py AAPL --av-key YOUR_KEY
  python dashboard.py TCS.NS --verbose
        """,
    )
    parser.add_argument(
        "ticker",
        help="Stock ticker symbol. For Indian stocks use .NS (NSE) or .BO (BSE) suffix.",
    )
    parser.add_argument(
        "--av-key",
        dest="av_key",
        default=None,
        help="Alpha Vantage API key (optional; get free key at alphavantage.co)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    args = parser.parse_args()
    run_dashboard(args.ticker, av_api_key=args.av_key, verbose=args.verbose)


if __name__ == "__main__":
    main()
