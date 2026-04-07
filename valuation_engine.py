"""
valuation_engine.py
-------------------
Core valuation calculations implementing Damodaran models.

Supported models:
  - Stable FCFE / Two-Stage FCFE / Three-Stage FCFE
  - Stable FCFF / Two-Stage FCFF / Three-Stage FCFF
  - Stable DDM / Two-Stage DDM / H-Model DDM

All inputs come from the AuditReport consensus values and ModelSelectionResult.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    """Holds the result of a Damodaran valuation."""
    ticker: str
    company_name: str
    model_used: str
    intrinsic_value_per_share: float | None
    current_price: float | None
    upside_pct: float | None
    currency: str
    key_assumptions: dict[str, Any]
    error: str | None = None

    def summary_lines(self) -> list[str]:
        currency_symbol = "₹" if self.currency == "INR" else "$"
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"  VALUATION — {self.company_name} ({self.ticker})")
        lines.append(f"  Model: {self.model_used}")
        lines.append("=" * 70)

        if self.error:
            lines.append(f"  ⚠️  Valuation could not be completed: {self.error}")
            lines.append("=" * 70)
            return lines

        if self.intrinsic_value_per_share is not None:
            lines.append(
                f"  Intrinsic Value  : {currency_symbol}{self.intrinsic_value_per_share:,.2f}"
            )
        if self.current_price is not None:
            lines.append(
                f"  Current Price    : {currency_symbol}{self.current_price:,.2f}"
            )
        if self.upside_pct is not None:
            sign = "+" if self.upside_pct >= 0 else ""
            verdict = "UNDERVALUED 📈" if self.upside_pct > 10 else ("OVERVALUED 📉" if self.upside_pct < -10 else "FAIRLY VALUED ⚖️")
            lines.append(
                f"  Upside / Downside: {sign}{self.upside_pct:.1f}%  → {verdict}"
            )
        lines.append("\n  Key Assumptions:")
        for k, v in self.key_assumptions.items():
            lines.append(f"    {k:<35}: {v}")
        lines.append("=" * 70)
        return lines


# ---------------------------------------------------------------------------
# Discount rate helpers
# ---------------------------------------------------------------------------

_RISK_FREE_INDIA = 0.068    # ~10-yr Gsec yield
_RISK_FREE_US = 0.043       # ~10-yr Treasury yield
_EQUITY_RISK_PREMIUM_INDIA = 0.075
_EQUITY_RISK_PREMIUM_US = 0.055


def _cost_of_equity(beta: float | None, is_indian: bool) -> float:
    """CAPM cost of equity."""
    b = beta if beta is not None else 1.0
    if is_indian:
        return _RISK_FREE_INDIA + b * _EQUITY_RISK_PREMIUM_INDIA
    return _RISK_FREE_US + b * _EQUITY_RISK_PREMIUM_US


def _wacc(
    cost_of_equity: float,
    cost_of_debt_pretax: float,
    tax_rate: float,
    debt: float,
    equity_market_cap: float,
) -> float:
    """Weighted average cost of capital."""
    total = debt + equity_market_cap
    if total <= 0:
        return cost_of_equity
    wd = debt / total
    we = equity_market_cap / total
    return we * cost_of_equity + wd * cost_of_debt_pretax * (1 - tax_rate)


# ---------------------------------------------------------------------------
# Terminal value helper
# ---------------------------------------------------------------------------

def _terminal_value(cash_flow_next: float, discount_rate: float, growth_rate: float) -> float:
    """Gordon Growth Model terminal value."""
    if discount_rate <= growth_rate:
        raise ValueError(
            f"Discount rate ({discount_rate:.3f}) must exceed terminal growth ({growth_rate:.3f})"
        )
    return cash_flow_next / (discount_rate - growth_rate)


# ---------------------------------------------------------------------------
# Valuation Engine
# ---------------------------------------------------------------------------

class ValuationEngine:
    """
    Runs the appropriate Damodaran model based on ModelSelectionResult.

    Usage
    -----
    engine = ValuationEngine()
    result = engine.value(model_result, audit_report)
    """

    def value(
        self,
        model_result: Any,   # ModelSelectionResult
        audit_report: Any,   # AuditReport
    ) -> ValuationResult:
        """Dispatch to the correct valuation model."""
        consensus = audit_report.as_dict()
        model = model_result.selected_model
        ticker = model_result.ticker
        company_name = model_result.company_name
        is_indian = model_result.is_indian
        currency = "INR" if is_indian else "USD"

        beta = consensus.get("beta")
        ke = _cost_of_equity(beta, is_indian)

        shares = consensus.get("shares_outstanding") or 1.0
        current_price = consensus.get("current_price")
        market_cap = consensus.get("market_cap") or (
            current_price * shares if current_price and shares else None
        )
        total_debt = consensus.get("total_debt") or 0.0
        fcf = consensus.get("free_cash_flow") or 0.0
        net_income = consensus.get("net_income") or 0.0
        dps = model_result.dividend_per_share
        high_g = model_result.high_growth_rate
        terminal_g = model_result.terminal_growth_rate
        high_g_yrs = model_result.high_growth_years

        base_assumptions: dict[str, Any] = {
            "Beta": f"{beta:.2f}" if beta else "1.00 (assumed)",
            "Cost of Equity (CAPM)": f"{ke * 100:.2f}%",
            "Terminal Growth Rate": f"{terminal_g * 100:.2f}%",
            "High-Growth Rate": f"{high_g * 100:.2f}%",
            "High-Growth Years": high_g_yrs,
            "Risk-Free Rate": f"{(_RISK_FREE_INDIA if is_indian else _RISK_FREE_US) * 100:.2f}%",
            "Equity Risk Premium": f"{(_EQUITY_RISK_PREMIUM_INDIA if is_indian else _EQUITY_RISK_PREMIUM_US) * 100:.2f}%",
        }

        try:
            if "FCFE" in model:
                return self._value_fcfe(
                    ticker, company_name, model, currency, ke, high_g, terminal_g,
                    high_g_yrs, fcf, net_income, shares, current_price,
                    model_result, base_assumptions,
                )
            elif "FCFF" in model:
                tax_rate = 0.25 if is_indian else 0.21
                cost_of_debt = 0.09 if is_indian else 0.05
                w = _wacc(ke, cost_of_debt, tax_rate, total_debt, market_cap or (shares * (current_price or 100)))
                ebitda = consensus.get("ebitda") or 0.0
                return self._value_fcff(
                    ticker, company_name, model, currency, w, high_g, terminal_g,
                    high_g_yrs, fcf, ebitda, shares, current_price,
                    model_result, base_assumptions, total_debt,
                )
            elif "DDM" in model:
                return self._value_ddm(
                    ticker, company_name, model, currency, ke, high_g, terminal_g,
                    high_g_yrs, dps, current_price, model_result, base_assumptions,
                )
            else:
                return ValuationResult(
                    ticker=ticker, company_name=company_name, model_used=model,
                    intrinsic_value_per_share=None, current_price=current_price,
                    upside_pct=None, currency=currency, key_assumptions=base_assumptions,
                    error=f"Unknown model: {model}",
                )
        except Exception as exc:
            logger.exception("Valuation failed for %s (%s): %s", ticker, model, exc)
            return ValuationResult(
                ticker=ticker, company_name=company_name, model_used=model,
                intrinsic_value_per_share=None, current_price=current_price,
                upside_pct=None, currency=currency, key_assumptions=base_assumptions,
                error=str(exc),
            )

    # ------------------------------------------------------------------ #
    # FCFE models
    # ------------------------------------------------------------------ #

    def _value_fcfe(
        self, ticker, company_name, model, currency, ke, high_g, terminal_g,
        high_g_yrs, fcf, net_income, shares, current_price,
        model_result, assumptions,
    ) -> ValuationResult:
        # FCFE ≈ free cash flow (already equity cash flow proxy)
        fcfe = fcf if fcf and fcf > 0 else net_income * 0.7  # fallback: 70% of NI
        if not fcfe or fcfe <= 0:
            return ValuationResult(
                ticker=ticker, company_name=company_name, model_used=model,
                intrinsic_value_per_share=None, current_price=current_price,
                upside_pct=None, currency=currency, key_assumptions=assumptions,
                error="Insufficient FCFE data (negative or zero free cash flow)",
            )
        fcfe_per_share = fcfe / shares

        pv = 0.0
        if "Three-Stage" in model and model_result.transition_growth_rate is not None:
            trans_g = model_result.transition_growth_rate
            trans_yrs = model_result.transition_years or 5
            cf = fcfe_per_share
            # Phase 1: high growth
            for t in range(1, high_g_yrs + 1):
                cf *= (1 + high_g)
                pv += cf / (1 + ke) ** t
            # Phase 2: transition (linearly declining growth)
            growth_step = (trans_g - high_g) / trans_yrs
            g = high_g
            for t in range(1, trans_yrs + 1):
                g += growth_step
                cf *= (1 + g)
                pv += cf / (1 + ke) ** (high_g_yrs + t)
            # Phase 3: terminal
            cf_terminal = cf * (1 + terminal_g)
            tv = _terminal_value(cf_terminal, ke, terminal_g)
            pv += tv / (1 + ke) ** (high_g_yrs + trans_yrs)
        elif "Two-Stage" in model and high_g_yrs > 0:
            cf = fcfe_per_share
            for t in range(1, high_g_yrs + 1):
                cf *= (1 + high_g)
                pv += cf / (1 + ke) ** t
            cf_terminal = cf * (1 + terminal_g)
            tv = _terminal_value(cf_terminal, ke, terminal_g)
            pv += tv / (1 + ke) ** high_g_yrs
        else:
            # Stable FCFE
            cf_next = fcfe_per_share * (1 + terminal_g)
            pv = _terminal_value(cf_next, ke, terminal_g)

        upside = ((pv - current_price) / current_price * 100) if current_price and pv else None
        assumptions.update({
            "FCFE per Share (base)": f"{fcfe_per_share:.4f}",
            "Discount Rate": f"{ke * 100:.2f}%",
        })
        return ValuationResult(
            ticker=ticker, company_name=company_name, model_used=model,
            intrinsic_value_per_share=pv, current_price=current_price,
            upside_pct=upside, currency=currency, key_assumptions=assumptions,
        )

    # ------------------------------------------------------------------ #
    # FCFF models
    # ------------------------------------------------------------------ #

    def _value_fcff(
        self, ticker, company_name, model, currency, wacc, high_g, terminal_g,
        high_g_yrs, fcf, ebitda, shares, current_price,
        model_result, assumptions, total_debt,
    ) -> ValuationResult:
        # FCFF approximated from FCF + after-tax interest (or 80% of EBITDA)
        fcff = fcf if fcf and fcf > 0 else (ebitda * 0.6 if ebitda and ebitda > 0 else None)
        if not fcff or fcff <= 0:
            return ValuationResult(
                ticker=ticker, company_name=company_name, model_used=model,
                intrinsic_value_per_share=None, current_price=current_price,
                upside_pct=None, currency=currency, key_assumptions=assumptions,
                error="Insufficient FCFF data",
            )

        pv_firm = 0.0
        if "Three-Stage" in model and model_result.transition_growth_rate is not None:
            trans_g = model_result.transition_growth_rate
            trans_yrs = model_result.transition_years or 5
            cf = fcff
            for t in range(1, high_g_yrs + 1):
                cf *= (1 + high_g)
                pv_firm += cf / (1 + wacc) ** t
            growth_step = (trans_g - high_g) / trans_yrs
            g = high_g
            for t in range(1, trans_yrs + 1):
                g += growth_step
                cf *= (1 + g)
                pv_firm += cf / (1 + wacc) ** (high_g_yrs + t)
            cf_terminal = cf * (1 + terminal_g)
            tv = _terminal_value(cf_terminal, wacc, terminal_g)
            pv_firm += tv / (1 + wacc) ** (high_g_yrs + trans_yrs)
        elif "Two-Stage" in model and high_g_yrs > 0:
            cf = fcff
            for t in range(1, high_g_yrs + 1):
                cf *= (1 + high_g)
                pv_firm += cf / (1 + wacc) ** t
            cf_terminal = cf * (1 + terminal_g)
            tv = _terminal_value(cf_terminal, wacc, terminal_g)
            pv_firm += tv / (1 + wacc) ** high_g_yrs
        else:
            cf_next = fcff * (1 + terminal_g)
            pv_firm = _terminal_value(cf_next, wacc, terminal_g)

        equity_value = pv_firm - total_debt
        intrinsic_per_share = equity_value / shares if shares and equity_value > 0 else None

        upside = (
            (intrinsic_per_share - current_price) / current_price * 100
            if current_price and intrinsic_per_share else None
        )
        assumptions.update({
            "WACC": f"{wacc * 100:.2f}%",
            "FCFF (base)": f"{fcff:,.0f}",
            "Total Debt (deducted)": f"{total_debt:,.0f}",
        })
        return ValuationResult(
            ticker=ticker, company_name=company_name, model_used=model,
            intrinsic_value_per_share=intrinsic_per_share, current_price=current_price,
            upside_pct=upside, currency=currency, key_assumptions=assumptions,
        )

    # ------------------------------------------------------------------ #
    # DDM models
    # ------------------------------------------------------------------ #

    def _value_ddm(
        self, ticker, company_name, model, currency, ke, high_g, terminal_g,
        high_g_yrs, dps, current_price, model_result, assumptions,
    ) -> ValuationResult:
        if not dps or dps <= 0:
            return ValuationResult(
                ticker=ticker, company_name=company_name, model_used=model,
                intrinsic_value_per_share=None, current_price=current_price,
                upside_pct=None, currency=currency, key_assumptions=assumptions,
                error="No dividend data — DDM cannot be applied",
            )

        pv = 0.0
        if "H-Model" in model:
            # H-Model: P = D0(1+g_L)/(r-g_L) + D0*H*(g_s-g_L)/(r-g_L)
            # where H = half-life of high-growth period
            h = high_g_yrs / 2.0
            if ke > terminal_g:
                pv = (dps * (1 + terminal_g) + dps * h * (high_g - terminal_g)) / (ke - terminal_g)
            else:
                pv = 0.0
        elif "Two-Stage" in model and high_g_yrs > 0:
            d = dps
            for t in range(1, high_g_yrs + 1):
                d *= (1 + high_g)
                pv += d / (1 + ke) ** t
            d_terminal = d * (1 + terminal_g)
            tv = _terminal_value(d_terminal, ke, terminal_g)
            pv += tv / (1 + ke) ** high_g_yrs
        else:
            # Stable DDM (Gordon Growth Model)
            d_next = dps * (1 + terminal_g)
            pv = _terminal_value(d_next, ke, terminal_g)

        upside = ((pv - current_price) / current_price * 100) if current_price and pv else None
        assumptions.update({
            "Dividend Per Share (base)": f"{dps:.4f}",
            "Discount Rate": f"{ke * 100:.2f}%",
        })
        return ValuationResult(
            ticker=ticker, company_name=company_name, model_used=model,
            intrinsic_value_per_share=pv, current_price=current_price,
            upside_pct=upside, currency=currency, key_assumptions=assumptions,
        )
