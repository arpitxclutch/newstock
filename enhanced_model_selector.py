"""
enhanced_model_selector.py
--------------------------
Implements detailed Damodaran model selection with explicit rationale.

All 9 Damodaran valuation models are evaluated:
  1.  Dividend Discount Model (DDM) — stable
  2.  Two-Stage DDM
  3.  H-Model DDM
  4.  Free Cash Flow to Equity (FCFE) — stable
  5.  Two-Stage FCFE
  6.  Three-Stage FCFE
  7.  Free Cash Flow to Firm (FCFF) — stable
  8.  Two-Stage FCFF
  9.  Three-Stage FCFF

The selector evaluates the following decision criteria:
  A. Growth rate relative to economy
  B. Competitive advantage (moat) duration
  C. Debt ratio stability
  D. Dividend payout vs FCFE
  E. Capital structure changes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Economy growth benchmarks
# ---------------------------------------------------------------------------
_INDIA_NOMINAL_GROWTH = 0.065   # ~6.5% nominal GDP (RBI long-run estimate)
_INDIA_INFLATION = 0.050        # RBI target
_US_NOMINAL_GROWTH = 0.025      # ~2.5% (Fed long-run estimate)
_US_INFLATION = 0.020           # Fed target

# Thresholds
_HIGH_GROWTH_PREMIUM = 0.03     # firm growth > economy + 3% → qualifies for high-growth model
_EXCEPTIONAL_PREMIUM = 0.15     # firm growth > economy + 15% → three-stage warranted
_LONG_MOAT_THRESHOLD = 0.10     # growth premium > 10% often implies longer moat
_DEBT_CHANGE_THRESHOLD = 0.05   # debt ratio change > 5 pp in 3 yrs → changing
_DIVIDEND_FCFE_RATIO = 0.80     # dividends > 80% of FCFE → use DDM


@dataclass
class ModelEligibility:
    """Records whether a Damodaran model is eligible and the reasoning."""
    model_name: str
    is_eligible: bool
    rationale: str
    estimated_value_premium: str = ""  # qualitative: e.g., "+10% vs selected"


@dataclass
class ModelSelectionResult:
    """Full result of the model selection process."""
    ticker: str
    company_name: str
    is_indian: bool

    # Decision criteria
    economy_growth: float = 0.0
    economy_growth_source: str = ""
    firm_growth: float = 0.0
    firm_growth_source: str = ""
    growth_premium: float = 0.0        # firm_growth - economy_growth

    has_competitive_advantage: bool = False
    moat_duration_years: int = 0
    moat_rationale: str = ""

    debt_ratio: float | None = None
    debt_changing: bool = False
    debt_trajectory: str = ""

    dividend_per_share: float = 0.0
    fcfe_per_share: float = 0.0
    dividend_covers_fcfe: bool = False

    # Selected model
    selected_model: str = ""
    selection_rationale: str = ""
    high_growth_rate: float = 0.0
    high_growth_years: int = 0
    terminal_growth_rate: float = 0.0

    # All model evaluations
    model_eligibility: list[ModelEligibility] = field(default_factory=list)

    # Phase 2 (three-stage) parameters
    transition_growth_rate: float | None = None
    transition_years: int | None = None

    def rationale_lines(self) -> list[str]:
        """Return formatted model selection rationale for display."""
        currency = "₹" if self.is_indian else "$"
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"  MODEL SELECTION RATIONALE — {self.company_name} ({self.ticker})")
        lines.append("=" * 70)

        lines.append("\n  📊 DECISION CRITERIA")
        lines.append("  " + "-" * 66)

        lines.append(
            f"  1. Economy growth     : {self.economy_growth * 100:.1f}%  "
            f"[{self.economy_growth_source}]"
        )
        lines.append(
            f"  2. Firm growth (CAGR) : {self.firm_growth * 100:.1f}%  "
            f"[{self.firm_growth_source}]"
        )
        lines.append(
            f"  3. Growth premium     : {self.growth_premium * 100:+.1f}%  "
            f"({'above' if self.growth_premium > 0 else 'below'} economy growth)"
        )
        lines.append(
            f"  4. Competitive moat   : {'YES' if self.has_competitive_advantage else 'NO'}  "
            f"[{self.moat_rationale}]"
        )
        if self.has_competitive_advantage:
            lines.append(
                f"     Estimated moat duration : ~{self.moat_duration_years} years"
            )
        lines.append(
            f"  5. Debt ratio         : "
            + (f"{self.debt_ratio * 100:.1f}%  " if self.debt_ratio is not None else "N/A  ")
            + f"[{'CHANGING — ' + self.debt_trajectory if self.debt_changing else 'STABLE'}]"
        )
        lines.append(
            f"  6. Dividend per share : {currency}{self.dividend_per_share:.2f}  |  "
            f"FCFE per share: {currency}{self.fcfe_per_share:.2f}"
        )
        if self.fcfe_per_share > 0:
            ratio = self.dividend_per_share / self.fcfe_per_share * 100
            lines.append(
                f"     Dividend/FCFE ratio: {ratio:.1f}%  "
                f"→ {'Use DDM (dividends represent FCF well)' if self.dividend_covers_fcfe else 'Use FCFE (dividends significantly below free cash flow)'}"
            )

        lines.append("\n  📐 MODEL ELIGIBILITY MATRIX")
        lines.append("  " + "-" * 66)
        for m in self.model_eligibility:
            icon = "✅" if m.is_eligible else "❌"
            selected_tag = " ← SELECTED" if m.model_name == self.selected_model else ""
            lines.append(
                f"  {icon}  {m.model_name:<35} {selected_tag}"
            )
            lines.append(f"       {m.rationale}")
            if m.estimated_value_premium and m.model_name != self.selected_model:
                lines.append(f"       Sensitivity: {m.estimated_value_premium}")

        lines.append("\n  🎯 FINAL MODEL SELECTION")
        lines.append("  " + "-" * 66)
        lines.append(f"  Selected: {self.selected_model}")
        lines.append(f"  Rationale: {self.selection_rationale}")
        lines.append("")
        lines.append(
            f"  High-growth phase   : {self.high_growth_years} years "
            f"@ {self.high_growth_rate * 100:.1f}% growth"
        )
        if self.transition_growth_rate is not None:
            lines.append(
                f"  Transition phase    : {self.transition_years} years, "
                f"declining to {self.transition_growth_rate * 100:.1f}%"
            )
        lines.append(
            f"  Terminal growth rate: {self.terminal_growth_rate * 100:.1f}%  "
            f"[pegged to economy growth]"
        )
        lines.append("=" * 70)
        return lines


def _estimate_moat(
    growth_premium: float,
    roe: float | None,
    sector: str | None,
    is_indian: bool,
) -> tuple[bool, int, str]:
    """
    Heuristically estimate competitive advantage (moat) from available data.

    Returns (has_moat, duration_years, rationale).
    """
    if growth_premium <= 0:
        return False, 0, "Firm growth ≤ economy growth — no moat detected"

    high_roe = roe is not None and roe > 0.15

    # Premium tiers
    if growth_premium > _EXCEPTIONAL_PREMIUM:
        years = 15 if high_roe else 10
        rationale = (
            f"Exceptional growth premium ({growth_premium * 100:.1f}%); "
            + ("high ROE supports dominant moat" if high_roe else "moat assumed but monitor competition")
        )
        return True, years, rationale

    if growth_premium > _LONG_MOAT_THRESHOLD:
        years = 10 if high_roe else 7
        rationale = (
            f"Strong growth premium ({growth_premium * 100:.1f}%); "
            + ("ROE confirms competitive returns" if high_roe else "moderate ROE — moat may erode")
        )
        return True, years, rationale

    if growth_premium > _HIGH_GROWTH_PREMIUM:
        years = 7 if high_roe else 5
        rationale = (
            f"Moderate growth premium ({growth_premium * 100:.1f}%); "
            + ("supported by above-average ROE" if high_roe else "limited ROE evidence — short moat assumed")
        )
        return True, years, rationale

    return False, 0, f"Growth premium ({growth_premium * 100:.1f}%) below threshold for moat recognition"


class EnhancedModelSelector:
    """
    Selects the most appropriate Damodaran valuation model for a company
    and explains the rationale in detail.

    Usage
    -----
    selector = EnhancedModelSelector(ticker, is_indian=True)
    result = selector.select(audit_report, attribution)
    """

    def __init__(self, ticker: str, is_indian: bool) -> None:
        self.ticker = ticker
        self.is_indian = is_indian

    def select(
        self,
        audit_report: Any,  # AuditReport from data_auditor
        attribution: Any,   # SourceAttribution from source_attribution
        historical_revenues: list[float] | None = None,
    ) -> ModelSelectionResult:
        """
        Run the full model selection logic and return a ModelSelectionResult.
        """
        from data_auditor import compute_revenue_cagr

        consensus = audit_report.as_dict()
        currency = audit_report.currency
        company_name = audit_report.company_name

        # ------------------------------------------------------------------ #
        # 1. Economy growth
        # ------------------------------------------------------------------ #
        if self.is_indian:
            economy_growth = _INDIA_NOMINAL_GROWTH
            economy_growth_source = (
                f"RBI long-run nominal GDP estimate "
                f"(~{_INDIA_INFLATION * 100:.0f}% inflation + "
                f"{(_INDIA_NOMINAL_GROWTH - _INDIA_INFLATION) * 100:.1f}% real)"
            )
            terminal_growth = _INDIA_NOMINAL_GROWTH
        else:
            economy_growth = _US_NOMINAL_GROWTH
            economy_growth_source = (
                f"Federal Reserve long-run GDP estimate "
                f"(~{_US_INFLATION * 100:.0f}% inflation + "
                f"{(_US_NOMINAL_GROWTH - _US_INFLATION) * 100:.1f}% real)"
            )
            terminal_growth = _US_NOMINAL_GROWTH

        # ------------------------------------------------------------------ #
        # 2. Firm growth
        # ------------------------------------------------------------------ #
        firm_growth: float = 0.0
        firm_growth_source = "N/A"

        # Try CAGR from historical revenues
        hist_revs: list[float] = []
        if historical_revenues:
            hist_revs = historical_revenues
        else:
            # Try each source
            for src in audit_report.sources:
                if src.get("historical_revenues"):
                    hist_revs = src["historical_revenues"]
                    break

        cagr_3yr = compute_revenue_cagr(hist_revs, years=3) if hist_revs else None
        cagr_2yr = compute_revenue_cagr(hist_revs, years=2) if hist_revs else None

        rev_growth_yf = consensus.get("revenue_growth")  # yfinance 1-yr
        earnings_growth = consensus.get("earnings_growth")

        if cagr_3yr is not None:
            firm_growth = cagr_3yr
            cagr_src = attribution.cite("revenue") if attribution else "multiple sources"
            firm_growth_source = f"3-year revenue CAGR from {cagr_src}"
            if cagr_2yr is not None:
                firm_growth_source += f" (2-yr: {cagr_2yr * 100:.1f}%, 3-yr: {cagr_3yr * 100:.1f}%)"
        elif rev_growth_yf is not None:
            firm_growth = rev_growth_yf
            firm_growth_source = "Yahoo Finance trailing 12-month revenue growth"
        elif earnings_growth is not None:
            firm_growth = earnings_growth
            firm_growth_source = "Earnings growth (proxy for revenue growth)"
        else:
            firm_growth = economy_growth * 1.5
            firm_growth_source = "Estimated as 1.5× economy growth (insufficient historical data)"

        growth_premium = firm_growth - economy_growth

        # ------------------------------------------------------------------ #
        # 3. Competitive advantage
        # ------------------------------------------------------------------ #
        roe = consensus.get("return_on_equity")
        sector = None
        for src in audit_report.sources:
            if src.get("sector"):
                sector = src["sector"]
                break

        has_moat, moat_years, moat_rationale = _estimate_moat(
            growth_premium, roe, sector, self.is_indian
        )

        # ------------------------------------------------------------------ #
        # 4. Debt stability
        # ------------------------------------------------------------------ #
        total_debt = consensus.get("total_debt") or 0.0
        total_equity = consensus.get("book_value") or 0.0
        if total_equity > 0:
            debt_ratio = total_debt / (total_debt + total_equity)
        else:
            debt_ratio = None

        dte = consensus.get("debt_to_equity")
        debt_changing = False
        debt_trajectory = "Stable"
        if dte is not None:
            if dte > 2.0:
                debt_changing = True
                debt_trajectory = f"High leverage (D/E = {dte:.1f}x) — monitor"
            elif dte > 1.0:
                debt_trajectory = f"Moderate leverage (D/E = {dte:.1f}x)"

        # ------------------------------------------------------------------ #
        # 5. Dividend vs FCFE
        # ------------------------------------------------------------------ #
        dps = consensus.get("dividend_per_share") or 0.0
        fcf = consensus.get("free_cash_flow") or 0.0
        shares = consensus.get("shares_outstanding") or 1.0
        fcfe_per_share = fcf / shares if shares > 0 else 0.0
        dividend_covers_fcfe = (
            dps > 0 and fcfe_per_share > 0 and (dps / fcfe_per_share) >= _DIVIDEND_FCFE_RATIO
        )

        # ------------------------------------------------------------------ #
        # 6. Select model & build eligibility matrix
        # ------------------------------------------------------------------ #
        eligibility: list[ModelEligibility] = []

        # --- DDM models ---
        ddm_eligible = dividend_covers_fcfe and dps > 0
        eligibility.append(ModelEligibility(
            model_name="Stable DDM",
            is_eligible=ddm_eligible and growth_premium <= _HIGH_GROWTH_PREMIUM,
            rationale=(
                "Applicable when dividends represent FCF and firm grows ≈ economy. "
                + ("Dividends ({:.2f}) ≥ 80% of FCFE → eligible.".format(dps) if ddm_eligible
                   else f"Dividends ({dps:.2f}) << FCFE ({fcfe_per_share:.2f}) → NOT eligible.")
            ),
        ))
        eligibility.append(ModelEligibility(
            model_name="Two-Stage DDM",
            is_eligible=ddm_eligible and _HIGH_GROWTH_PREMIUM < growth_premium <= _EXCEPTIONAL_PREMIUM,
            rationale=(
                "Two stages: high dividend growth then stable. "
                + ("Dividends track FCFE well." if ddm_eligible
                   else f"Dividends ({dps:.2f}) do not cover FCFE ({fcfe_per_share:.2f}) → rejected.")
            ),
            estimated_value_premium="Would undervalue if earnings reinvested rather than paid out",
        ))
        eligibility.append(ModelEligibility(
            model_name="H-Model DDM",
            is_eligible=ddm_eligible and has_moat and moat_years >= 5,
            rationale=(
                "Linear growth decline from high to stable. Suitable for gradually moderating dividend payers. "
                + ("Not applicable — insufficient dividend coverage." if not ddm_eligible
                   else "Eligible but Two-Stage FCFE is preferred for equity-reinvesting companies.")
            ),
            estimated_value_premium="Similar to Two-Stage DDM within ±5%",
        ))

        # --- FCFE models ---
        low_growth = growth_premium <= _HIGH_GROWTH_PREMIUM
        moderate_growth = _HIGH_GROWTH_PREMIUM < growth_premium <= _EXCEPTIONAL_PREMIUM
        high_growth_exceptional = growth_premium > _EXCEPTIONAL_PREMIUM

        eligibility.append(ModelEligibility(
            model_name="Stable FCFE",
            is_eligible=low_growth and not debt_changing,
            rationale=(
                f"Firm growth ({firm_growth * 100:.1f}%) ≈ economy ({economy_growth * 100:.1f}%). "
                + ("Stable debt ratio confirms." if not debt_changing
                   else "Changing debt ratio complicates FCFE projection.")
            ),
        ))
        eligibility.append(ModelEligibility(
            model_name="Two-Stage FCFE",
            is_eligible=moderate_growth and has_moat and not debt_changing,
            rationale=(
                f"Firm grows at {firm_growth * 100:.1f}% (>{(economy_growth + _HIGH_GROWTH_PREMIUM) * 100:.1f}%) "
                f"with a {moat_years}-year sustainable moat. "
                + ("Debt stable — FCFE projection is reliable." if not debt_changing
                   else "Debt changes add uncertainty to FCFE.")
            ),
        ))
        eligibility.append(ModelEligibility(
            model_name="Three-Stage FCFE",
            is_eligible=high_growth_exceptional and has_moat and moat_years >= 10,
            rationale=(
                f"Exceptional growth ({firm_growth * 100:.1f}%) with long moat ({moat_years} yrs). "
                "Three phases: peak growth → transition → stable. "
                + ("Eligible." if high_growth_exceptional and moat_years >= 10
                   else "Current growth or moat duration does not warrant three stages.")
            ),
            estimated_value_premium="10–20% higher than Two-Stage (captures longer high-growth period)",
        ))

        # --- FCFF models ---
        use_fcff = debt_changing
        eligibility.append(ModelEligibility(
            model_name="Stable FCFF",
            is_eligible=use_fcff and low_growth,
            rationale=(
                "FCFF is preferred when capital structure is changing (avoids circular FCFE calc). "
                + (f"Debt trajectory: {debt_trajectory}." if debt_changing
                   else "Debt is stable — FCFE is cleaner for equity valuation.")
            ),
        ))
        eligibility.append(ModelEligibility(
            model_name="Two-Stage FCFF",
            is_eligible=use_fcff and moderate_growth,
            rationale=(
                "Two-stage FCFF for companies with changing capital structure and above-economy growth. "
                + (f"Debt trajectory: {debt_trajectory}." if debt_changing
                   else "Debt stable — FCFE preferred.")
            ),
        ))
        eligibility.append(ModelEligibility(
            model_name="Three-Stage FCFF",
            is_eligible=use_fcff and high_growth_exceptional,
            rationale=(
                "Three-stage FCFF for high-growth companies with shifting capital structure. "
                + (f"Debt trajectory: {debt_trajectory}." if debt_changing
                   else "Debt stable — FCFE preferred.")
            ),
        ))

        # ------------------------------------------------------------------ #
        # 7. Choose the selected model
        # ------------------------------------------------------------------ #
        if debt_changing:
            if high_growth_exceptional:
                selected = "Three-Stage FCFF"
                rationale = (
                    f"Exceptional growth ({firm_growth * 100:.1f}%) AND capital structure is changing "
                    f"({debt_trajectory}). Three-Stage FCFF accounts for both the extended high-growth "
                    f"period and evolving leverage."
                )
                high_g_yrs = moat_years or 5
                trans_g = terminal_growth + (firm_growth - terminal_growth) * 0.5
                trans_yrs = 5
            elif moderate_growth:
                selected = "Two-Stage FCFF"
                rationale = (
                    f"Above-economy growth ({firm_growth * 100:.1f}%) with changing capital structure. "
                    f"FCFF avoids the circular dependency in FCFE when debt ratios shift."
                )
                high_g_yrs = moat_years or 5
                trans_g = None
                trans_yrs = None
            else:
                selected = "Stable FCFF"
                rationale = (
                    f"Firm growth ({firm_growth * 100:.1f}%) close to economy ({economy_growth * 100:.1f}%). "
                    f"Stable FCFF appropriate given changing capital structure."
                )
                high_g_yrs = 0
                trans_g = None
                trans_yrs = None
        elif dividend_covers_fcfe and dps > 0:
            if low_growth:
                selected = "Stable DDM"
                rationale = (
                    f"Dividends ({dps:.2f}/share) closely track FCFE ({fcfe_per_share:.2f}/share). "
                    f"Firm growth ({firm_growth * 100:.1f}%) ≈ economy. DDM gives clean stable valuation."
                )
                high_g_yrs = 0
                trans_g = None
                trans_yrs = None
            else:
                selected = "Two-Stage DDM"
                rationale = (
                    f"Dividends cover FCFE well. Firm grows at {firm_growth * 100:.1f}% before settling "
                    f"to terminal rate. Two-Stage DDM is appropriate."
                )
                high_g_yrs = moat_years or 5
                trans_g = None
                trans_yrs = None
        else:
            # Default: FCFE family
            if high_growth_exceptional and moat_years >= 10:
                selected = "Three-Stage FCFE"
                rationale = (
                    f"Exceptional growth ({firm_growth * 100:.1f}%) with a {moat_years}-year moat. "
                    f"Three-stage FCFE captures: peak growth → gradual moderation → stable terminal. "
                    f"Dividends ({dps:.2f}) are significantly below FCFE ({fcfe_per_share:.2f}) — use free cash flow directly."
                )
                high_g_yrs = max(moat_years // 2, 3)
                trans_g = economy_growth + (firm_growth - economy_growth) * 0.3
                trans_yrs = max(moat_years - high_g_yrs, 3)
            elif moderate_growth or (high_growth_exceptional and moat_years < 10):
                selected = "Two-Stage FCFE"
                rationale = (
                    f"Firm grows at {firm_growth * 100:.1f}% (premium of {growth_premium * 100:.1f}% "
                    f"over economy). Sustainable competitive advantage for ~{moat_years} years. "
                    f"Dividends ({dps:.2f}) << FCFE ({fcfe_per_share:.2f}) — FCFE captures true earning power."
                )
                high_g_yrs = moat_years or 5
                trans_g = None
                trans_yrs = None
            else:
                selected = "Stable FCFE"
                rationale = (
                    f"Firm growth ({firm_growth * 100:.1f}%) is close to economy ({economy_growth * 100:.1f}%). "
                    f"No significant competitive advantage detected. Stable FCFE gives the most honest valuation."
                )
                high_g_yrs = 0
                trans_g = None
                trans_yrs = None

        # Mark selected model as eligible in the matrix
        for m in eligibility:
            if m.model_name == selected:
                m.is_eligible = True

        return ModelSelectionResult(
            ticker=self.ticker,
            company_name=company_name,
            is_indian=self.is_indian,
            economy_growth=economy_growth,
            economy_growth_source=economy_growth_source,
            firm_growth=firm_growth,
            firm_growth_source=firm_growth_source,
            growth_premium=growth_premium,
            has_competitive_advantage=has_moat,
            moat_duration_years=moat_years,
            moat_rationale=moat_rationale,
            debt_ratio=debt_ratio,
            debt_changing=debt_changing,
            debt_trajectory=debt_trajectory,
            dividend_per_share=dps,
            fcfe_per_share=fcfe_per_share,
            dividend_covers_fcfe=dividend_covers_fcfe,
            selected_model=selected,
            selection_rationale=rationale,
            high_growth_rate=firm_growth,
            high_growth_years=high_g_yrs,
            terminal_growth_rate=terminal_growth,
            model_eligibility=eligibility,
            transition_growth_rate=trans_g,
            transition_years=trans_yrs,
        )
