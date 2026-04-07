"""
dashboard.py
------------
Streamlit web application for the NewStock analysis dashboard.

For any given ticker, the dashboard:
  1. Fetches data from 3+ credible sources (Indian: Yahoo/Screener/NSE; US: Yahoo/SEC/AlphaVantage)
  2. Audits and cross-validates all financial metrics
  3. Selects the best Damodaran valuation model with detailed rationale
  4. Runs the valuation using consensus / audited data
  5. Displays full output with source attribution

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import logging
import time
from typing import Any

import streamlit as st

from multi_source_fetcher import MultiSourceFetcher
from data_auditor import DataAuditor, _fmt_value
from source_attribution import SourceAttribution
from enhanced_model_selector import EnhancedModelSelector
from valuation_engine import ValuationEngine
from cross_verify import cross_verify

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INDIAN_PRESETS = [
    "TATAMOTORS.NS", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "WIPRO.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "MARUTI.NS",
]
_US_PRESETS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX",
    "JPM", "BAC",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")


def _currency_symbol(is_indian: bool) -> str:
    return "₹" if is_indian else "$"


def _fmt_large(value: float | None, is_indian: bool) -> str:
    """Format a large financial figure for display."""
    if value is None:
        return "N/A"
    sym = _currency_symbol(is_indian)
    if is_indian:
        if abs(value) >= 1e7:
            return f"₹{value / 1e7:,.1f} Cr"
        return f"₹{value:,.0f}"
    else:
        if abs(value) >= 1e12:
            return f"${value / 1e12:,.2f}T"
        if abs(value) >= 1e9:
            return f"${value / 1e9:,.2f}B"
        if abs(value) >= 1e6:
            return f"${value / 1e6:,.1f}M"
        return f"${value:,.0f}"


def _fmt_price(value: float | None, is_indian: bool) -> str:
    if value is None:
        return "N/A"
    sym = _currency_symbol(is_indian)
    return f"{sym}{value:,.2f}"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NewStock — Institutional Equity Lab",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _build_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return user inputs."""
    st.sidebar.title("🏛️ NewStock")
    st.sidebar.caption("Damodaran DCF · Multi-Source Verification")
    st.sidebar.markdown("---")

    # Ticker input
    st.sidebar.subheader("📊 Ticker")
    ticker_input = st.sidebar.text_input(
        "Enter ticker symbol",
        value="",
        placeholder="e.g. TATAMOTORS.NS or NVDA",
        help="For Indian stocks add .NS (NSE) or .BO (BSE). For US stocks just use the ticker.",
    )

    # Preset suggestions
    with st.sidebar.expander("🇮🇳 Indian stock presets"):
        cols = st.columns(2)
        chosen_indian = None
        for i, t in enumerate(_INDIAN_PRESETS):
            if cols[i % 2].button(t, key=f"ind_{t}", use_container_width=True):
                chosen_indian = t

    with st.sidebar.expander("🇺🇸 US stock presets"):
        cols = st.columns(2)
        chosen_us = None
        for i, t in enumerate(_US_PRESETS):
            if cols[i % 2].button(t, key=f"us_{t}", use_container_width=True):
                chosen_us = t

    # Resolve ticker from presets
    ticker = ticker_input.strip().upper()
    if chosen_indian:
        ticker = chosen_indian
    elif chosen_us:
        ticker = chosen_us

    st.sidebar.markdown("---")

    # API key
    st.sidebar.subheader("🔑 Alpha Vantage API Key")
    av_key = st.sidebar.text_input(
        "API key (optional)",
        value="",
        type="password",
        placeholder="Leave blank to use free tier",
        help="Get a free key at alphavantage.co — improves US stock data quality.",
    )

    st.sidebar.markdown("---")

    # Verbose toggle
    st.sidebar.subheader("⚙️ Options")
    verbose = st.sidebar.toggle("Show verbose logging", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "All data is fetched live from public APIs. "
        "Valuations are based on the Damodaran framework and are for "
        "informational purposes only. **Not investment advice.**"
    )

    run_clicked = st.sidebar.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=(not ticker),
    )

    return {
        "ticker": ticker,
        "av_key": av_key or None,
        "verbose": verbose,
        "run": run_clicked,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header() -> None:
    st.markdown(
        """
        <h1 style='text-align:center; margin-bottom:0'>🏛️ Institutional Equity Lab</h1>
        <p style='text-align:center; color:gray; margin-top:4px'>
            Damodaran DCF &nbsp;·&nbsp; Multi-Source Verification &nbsp;·&nbsp; All Listed Companies
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


# ---------------------------------------------------------------------------
# Step 1 — Data Fetching
# ---------------------------------------------------------------------------

def _render_step1(sources: list[dict], ticker: str) -> None:
    successful = [s for s in sources if s.get("fetch_success")]
    st.subheader("✅ Step 1 — Multi-Source Data Fetch")

    cols = st.columns(len(sources))
    for col, src in zip(cols, sources):
        ok = src.get("fetch_success", False)
        icon = "✅" if ok else "❌"
        status = "Success" if ok else src.get("error", "Failed")
        col.metric(
            label=f"{icon} {src['source']}",
            value="Connected" if ok else "Failed",
            delta=status if not ok else None,
            delta_color="inverse",
        )

    if not successful:
        st.error(
            f"❌ **No data sources returned valid data for `{ticker}`.**\n\n"
            "Please check:\n"
            "- The ticker symbol is correct\n"
            "- For Indian stocks, use `.NS` (NSE) or `.BO` (BSE) suffix\n"
            "- Your internet connection\n\n"
            "**Common tickers:** `TATAMOTORS.NS`, `RELIANCE.NS`, `NVDA`, `AAPL`"
        )
        st.stop()

    # Company header
    company_name = ticker
    current_price = None
    market_cap = None
    currency = "INR" if _is_indian_ticker(ticker) else "USD"
    for s in successful:
        if s.get("company_name"):
            company_name = s["company_name"]
        if s.get("current_price") and current_price is None:
            current_price = s["current_price"]
        if s.get("market_cap") and market_cap is None:
            market_cap = s["market_cap"]
        if s.get("currency"):
            currency = s["currency"]

    is_indian = _is_indian_ticker(ticker)
    st.success(
        f"**{company_name.upper()}** (`{ticker}`) — "
        f"Current Price: **{_fmt_price(current_price, is_indian)}**  |  "
        f"Market Cap: **{_fmt_large(market_cap, is_indian)}**"
    )

    with st.expander("📋 Raw data from all sources"):
        import pandas as pd
        _DISPLAY_FIELDS = [
            "source", "company_name", "currency", "current_price", "market_cap",
            "revenue", "net_income", "ebitda", "total_debt", "cash",
            "free_cash_flow", "shares_outstanding", "beta", "pe_ratio",
            "dividend_per_share", "return_on_equity", "book_value",
        ]
        rows = []
        for s in sources:
            row = {k: s.get(k) for k in _DISPLAY_FIELDS}
            row["fetch_success"] = "✅" if s.get("fetch_success") else "❌"
            rows.append(row)
        df = pd.DataFrame(rows).set_index("source")
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Step 2 — Data Audit
# ---------------------------------------------------------------------------

def _render_step2(audit_report: Any, cv_result: dict) -> None:
    quality = cv_result["overall_quality"]
    quality_map = {"HIGH": ("✅", "success"), "MEDIUM": ("⚠️", "warning"), "LOW": ("🚩", "error")}
    icon, status = quality_map.get(quality, ("⚠️", "warning"))

    st.subheader(f"{icon} Step 2 — Data Audit Report")

    # Overall quality badge
    flagged_count = len(cv_result["flagged_metrics"])
    verified_count = len(cv_result["verified_metrics"])
    cols = st.columns(3)
    cols[0].metric("Data Quality", quality, delta=None)
    cols[1].metric("Verified Metrics", verified_count)
    cols[2].metric("Flagged Metrics", flagged_count, delta=f"{flagged_count} issues" if flagged_count else None, delta_color="inverse")

    # Per-metric audit table
    import pandas as pd
    rows = []
    currency_symbol = "₹" if audit_report.currency == "INR" else "$"
    for audit in audit_report.metric_audits:
        if not audit.has_data:
            continue
        consensus_str = _fmt_value(audit.consensus_value, audit.metric, currency_symbol)
        src_count = audit.available_sources
        spread_str = f"{audit.spread_pct:.1f}%"
        status_icon = audit.status_icon
        flagged_str = "⚠️ Yes" if audit.is_flagged else "✅ No"
        rows.append({
            "Status": status_icon,
            "Metric": audit.label,
            "Consensus Value": consensus_str,
            "Sources": src_count,
            "Spread": spread_str,
            "Flagged": flagged_str,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Source-level detail
    with st.expander("🔬 Source-level breakdown"):
        for audit in audit_report.metric_audits:
            if not audit.has_data:
                continue
            if not audit.source_values:
                continue
            st.markdown(f"**{audit.status_icon} {audit.label}** — spread: `{audit.spread_pct:.1f}%`")
            src_cols = st.columns(len(audit.source_values))
            for col, (src_name, val) in zip(src_cols, audit.source_values.items()):
                formatted = _fmt_value(val, audit.metric, currency_symbol)
                col.metric(src_name, formatted)

    # Flagged metrics warning
    if audit_report.flagged_metrics:
        st.warning(
            f"⚠️ **{len(audit_report.flagged_metrics)} metric(s) show high source disagreement** "
            f"(>{audit_report.threshold_pct:.0f}% spread): "
            + ", ".join(m.label for m in audit_report.flagged_metrics)
            + "\n\nConsensus (median) values are used in the valuation despite discrepancies."
        )
    else:
        st.success("✅ All available metrics show strong cross-source agreement.")


# ---------------------------------------------------------------------------
# Step 3 — Model Selection
# ---------------------------------------------------------------------------

def _render_step3(model_result: Any) -> None:
    st.subheader("🎯 Step 3 — Enhanced Model Selection")

    currency = "₹" if model_result.is_indian else "$"

    # Decision criteria cards
    st.markdown("#### 📊 Decision Criteria")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Economy Growth",
        f"{model_result.economy_growth * 100:.1f}%",
        help=model_result.economy_growth_source,
    )
    c2.metric(
        "Firm Growth (CAGR)",
        f"{model_result.firm_growth * 100:.1f}%",
        help=model_result.firm_growth_source,
    )
    premium = model_result.growth_premium
    c3.metric(
        "Growth Premium",
        f"{premium * 100:+.1f}%",
        delta="above economy" if premium > 0 else "below economy",
        delta_color="normal" if premium > 0 else "inverse",
    )

    c4, c5, c6 = st.columns(3)
    moat_label = f"~{model_result.moat_duration_years} yrs" if model_result.has_competitive_advantage else "None"
    c4.metric(
        "Competitive Moat",
        "YES ✅" if model_result.has_competitive_advantage else "NO ❌",
        delta=moat_label if model_result.has_competitive_advantage else None,
    )
    debt_label = "CHANGING ⚠️" if model_result.debt_changing else "STABLE ✅"
    c5.metric(
        "Debt Structure",
        debt_label,
        help=model_result.debt_trajectory,
    )
    if model_result.fcfe_per_share > 0:
        ratio = model_result.dividend_per_share / model_result.fcfe_per_share * 100
        c6.metric(
            "Dividend / FCFE",
            f"{ratio:.1f}%",
            delta="Use DDM" if model_result.dividend_covers_fcfe else "Use FCFE",
            delta_color="normal",
        )
    else:
        c6.metric("Dividend / FCFE", "N/A")

    st.markdown("---")

    # Model eligibility matrix
    st.markdown("#### 📐 Model Eligibility Matrix")

    import pandas as pd
    matrix_rows = []
    for m in model_result.model_eligibility:
        selected = m.model_name == model_result.selected_model
        matrix_rows.append({
            "Model": ("🏆 " if selected else "") + m.model_name + (" ← SELECTED" if selected else ""),
            "Eligible": "✅" if m.is_eligible else "❌",
            "Rationale": m.rationale,
        })
    if matrix_rows:
        df = pd.DataFrame(matrix_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Final selection highlight
    st.markdown("---")
    st.markdown("#### 🎯 Final Selection")

    col_a, col_b = st.columns([1, 2])
    col_a.success(f"**{model_result.selected_model}**")
    with col_b:
        st.markdown(f"> {model_result.selection_rationale}")

    st.markdown(
        f"**High-growth phase:** {model_result.high_growth_years} years "
        f"@ {model_result.high_growth_rate * 100:.1f}% growth  |  "
        f"**Terminal growth:** {model_result.terminal_growth_rate * 100:.1f}%"
    )
    if model_result.transition_growth_rate is not None:
        st.markdown(
            f"**Transition phase:** {model_result.transition_years} years, "
            f"declining to {model_result.transition_growth_rate * 100:.1f}%"
        )

    with st.expander("📄 Full rationale text"):
        st.code("\n".join(model_result.rationale_lines()), language=None)


# ---------------------------------------------------------------------------
# Step 4 — Valuation
# ---------------------------------------------------------------------------

def _render_step4(valuation: Any, audit_report: Any) -> None:
    st.subheader("💎 Step 4 — Valuation Results")

    is_indian = audit_report.is_indian
    currency = "₹" if is_indian else "$"

    if valuation.error:
        st.error(
            f"⚠️ **Valuation could not be completed:** {valuation.error}\n\n"
            "This may be due to missing or negative cash flow data. "
            "Try a different ticker or check data quality in Step 2."
        )
        with st.expander("📋 Assumptions used"):
            for k, v in valuation.key_assumptions.items():
                st.text(f"  {k}: {v}")
        return

    # Key metric cards
    c1, c2, c3 = st.columns(3)
    iv = valuation.intrinsic_value_per_share
    cp = valuation.current_price
    upside = valuation.upside_pct

    c1.metric(
        "Current Price",
        _fmt_price(cp, is_indian),
    )
    c2.metric(
        "Intrinsic Value",
        _fmt_price(iv, is_indian),
        delta=f"{upside:+.1f}%" if upside is not None else None,
        delta_color="normal" if (upside or 0) >= 0 else "inverse",
    )

    if upside is not None:
        if upside > 10:
            verdict = "BUY 📈"
            verdict_color = "success"
        elif upside < -10:
            verdict = "SELL 📉"
            verdict_color = "error"
        else:
            verdict = "HOLD ⚖️"
            verdict_color = "warning"
        c3.metric("Recommendation", verdict)
    else:
        c3.metric("Recommendation", "N/A")

    st.markdown("---")

    # Verdict banner
    if upside is not None:
        if upside > 10:
            st.success(
                f"✅ **{valuation.company_name}** appears **UNDERVALUED** — "
                f"Intrinsic value {_fmt_price(iv, is_indian)} vs current {_fmt_price(cp, is_indian)} "
                f"({upside:+.1f}% upside)"
            )
        elif upside < -10:
            st.error(
                f"📉 **{valuation.company_name}** appears **OVERVALUED** — "
                f"Intrinsic value {_fmt_price(iv, is_indian)} vs current {_fmt_price(cp, is_indian)} "
                f"({upside:+.1f}% downside)"
            )
        else:
            st.warning(
                f"⚖️ **{valuation.company_name}** appears **FAIRLY VALUED** — "
                f"Intrinsic value {_fmt_price(iv, is_indian)} vs current {_fmt_price(cp, is_indian)} "
                f"({upside:+.1f}%)"
            )

    # Assumptions
    with st.expander("📋 Key Assumptions"):
        import pandas as pd
        rows = [{"Parameter": k, "Value": v} for k, v in valuation.key_assumptions.items()]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("📄 Full valuation text"):
        st.code("\n".join(valuation.summary_lines()), language=None)


# ---------------------------------------------------------------------------
# Provenance footer
# ---------------------------------------------------------------------------

def _render_provenance(attribution: Any, cv_result: dict) -> None:
    st.markdown("---")
    with st.expander("🔗 Data Provenance"):
        for line in attribution.provenance_lines():
            st.text(line)


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    _render_header()
    inputs = _build_sidebar()

    ticker = inputs["ticker"]

    if not ticker:
        # Landing page
        st.markdown(
            """
            ## Welcome to NewStock Institutional Equity Lab 👋

            Use the **sidebar** to enter a ticker symbol and run a full valuation.

            ### What you'll get:
            | Step | Description |
            |------|-------------|
            | **1. Multi-Source Fetch** | Data from 3 live APIs per company |
            | **2. Data Audit** | Cross-validation & discrepancy detection |
            | **3. Model Selection** | Detailed Damodaran model rationale |
            | **4. Valuation** | Intrinsic value with BUY/HOLD/SELL verdict |

            ### Supported Companies:
            - 🇮🇳 **Any NSE/BSE listed Indian company** — use `.NS` (NSE) or `.BO` (BSE) suffix
            - 🇺🇸 **Any US listed company** — just the ticker (e.g. `NVDA`, `AAPL`)

            ### Quick start:
            Enter `TATAMOTORS.NS` or `NVDA` in the sidebar and click **Run Analysis**.
            """,
        )
        return

    if not inputs["run"]:
        st.info(f"ℹ️ Ticker set to **`{ticker}`** — click **🚀 Run Analysis** in the sidebar to start.")
        return

    if inputs["verbose"]:
        logging.getLogger().setLevel(logging.INFO)

    is_indian = _is_indian_ticker(ticker)
    currency_label = "INR (₹)" if is_indian else "USD ($)"
    market_label = "🇮🇳 Indian Market (NSE/BSE)" if is_indian else "🇺🇸 US Market (NYSE/NASDAQ)"

    st.markdown(
        f"### Analysis for `{ticker}` &nbsp; {market_label} &nbsp; `{currency_label}`"
    )
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Step 1: Fetch
    # ------------------------------------------------------------------ #
    with st.spinner(f"⏳ Fetching data for **{ticker}** from 3 sources…"):
        fetcher = MultiSourceFetcher(ticker, av_api_key=inputs["av_key"])
        sources = fetcher.fetch_all()

    _render_step1(sources, ticker)
    st.markdown("---")

    successful = [s for s in sources if s.get("fetch_success")]

    # ------------------------------------------------------------------ #
    # Step 2: Audit
    # ------------------------------------------------------------------ #
    with st.spinner("⏳ Cross-validating data across sources…"):
        auditor = DataAuditor(ticker, is_indian=is_indian)
        audit_report = auditor.audit(sources)
        cv_result = cross_verify(audit_report)
        attribution = SourceAttribution.from_audit_report(audit_report)

    _render_step2(audit_report, cv_result)
    st.markdown("---")

    # Warn if data quality is low
    if cv_result["overall_quality"] == "LOW" and len(successful) < 2:
        st.warning(
            "⚠️ Only one data source available — valuation accuracy may be limited."
        )

    # ------------------------------------------------------------------ #
    # Step 3: Model selection
    # ------------------------------------------------------------------ #
    with st.spinner("⏳ Selecting Damodaran valuation model…"):
        selector = EnhancedModelSelector(ticker, is_indian=is_indian)
        model_result = selector.select(audit_report, attribution)

    _render_step3(model_result)
    st.markdown("---")

    # ------------------------------------------------------------------ #
    # Step 4: Valuation
    # ------------------------------------------------------------------ #
    with st.spinner("⏳ Computing intrinsic value…"):
        engine = ValuationEngine()
        valuation = engine.value(model_result, audit_report)

    _render_step4(valuation, audit_report)

    # ------------------------------------------------------------------ #
    # Provenance footer
    # ------------------------------------------------------------------ #
    _render_provenance(attribution, cv_result)

    st.markdown("---")
    st.caption(
        "ℹ️ All financial data is fetched live from public APIs. "
        "Valuations are based on the Damodaran framework and are for "
        "informational purposes only. **Not investment advice.**"
    )


if __name__ == "__main__":
    main()
