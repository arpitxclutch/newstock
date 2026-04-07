"""
tests/test_pipeline.py
----------------------
Unit tests for the NewStock multi-source data verification pipeline.
Tests cover data auditing, model selection, valuation, and cross-verification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from data_auditor import DataAuditor, MetricAudit, compute_revenue_cagr
from source_attribution import SourceAttribution, DataPoint
from enhanced_model_selector import EnhancedModelSelector
from valuation_engine import ValuationEngine, _cost_of_equity, _terminal_value
from cross_verify import cross_verify
from multi_source_fetcher import _is_indian_ticker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INDIAN_SOURCES = [
    {
        "source": "Yahoo Finance",
        "fetch_success": True,
        "company_name": "Tata Motors Limited",
        "currency": "INR",
        "revenue": 4.52e11,
        "net_income": 4.5e10,
        "total_debt": 1.2e11,
        "book_value": 2.8e11,
        "cash": 3e10,
        "free_cash_flow": 4.2e10,
        "shares_outstanding": 3.58e9,
        "current_price": 780.0,
        "beta": 1.2,
        "market_cap": 2.8e12,
        "dividend_per_share": 8.0,
        "return_on_equity": 0.18,
        "debt_to_equity": 0.43,
        "historical_revenues": [4.52e11, 3.96e11, 3.51e11, 2.95e11],
    },
    {
        "source": "Screener.in",
        "fetch_success": True,
        "company_name": "Tata Motors",
        "currency": "INR",
        "revenue": 4.515e11,
        "net_income": 4.48e10,
        "total_debt": 1.18e11,
        "book_value": 2.75e11,
    },
    {
        "source": "NSE/BSE",
        "fetch_success": True,
        "company_name": "TATA MOTORS LIMITED",
        "currency": "INR",
        "current_price": 782.0,
        "market_cap": 2.8e12,
    },
]

US_SOURCES = [
    {
        "source": "Yahoo Finance",
        "fetch_success": True,
        "company_name": "NVIDIA Corporation",
        "currency": "USD",
        "revenue": 60.9e9,
        "net_income": 29.7e9,
        "total_debt": 9.7e9,
        "book_value": 42.9e9,
        "cash": 25.9e9,
        "free_cash_flow": 27.0e9,
        "shares_outstanding": 24.4e9,
        "current_price": 118.0,
        "beta": 1.68,
        "market_cap": 2.88e12,
        "dividend_per_share": 0.04,
        "return_on_equity": 0.69,
        "debt_to_equity": 0.47,
        "historical_revenues": [60.9e9, 26.97e9, 16.68e9, 26.97e9],
    },
    {
        "source": "SEC EDGAR",
        "fetch_success": True,
        "company_name": "NVIDIA CORP",
        "currency": "USD",
        "revenue": 60.92e9,
        "net_income": 29.8e9,
        "total_debt": 9.7e9,
    },
    {
        "source": "Alpha Vantage",
        "fetch_success": True,
        "company_name": "NVIDIA Corporation",
        "currency": "USD",
        "revenue": 61.0e9,
        "ebitda": 35.0e9,
        "return_on_equity": 0.70,
        "beta": 1.70,
    },
]


# ---------------------------------------------------------------------------
# Tests: Ticker detection
# ---------------------------------------------------------------------------

class TestTickerDetection:
    def test_indian_ns(self):
        assert _is_indian_ticker("TATAMOTORS.NS") is True

    def test_indian_bo(self):
        assert _is_indian_ticker("RELIANCE.BO") is True

    def test_us_ticker(self):
        assert _is_indian_ticker("NVDA") is False

    def test_us_ticker_lowercase(self):
        assert _is_indian_ticker("aapl") is False

    def test_case_insensitive(self):
        assert _is_indian_ticker("tcs.ns") is True


# ---------------------------------------------------------------------------
# Tests: CAGR computation
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_3yr_cagr(self):
        # 100 -> 150 in 3 years = ~14.47% CAGR
        revs = [150, 130, 115, 100]
        cagr = compute_revenue_cagr(revs, years=3)
        assert cagr is not None
        assert abs(cagr - ((150 / 100) ** (1 / 3) - 1)) < 0.001

    def test_2yr_cagr(self):
        revs = [121, 110, 100]
        cagr = compute_revenue_cagr(revs, years=2)
        assert cagr is not None
        assert abs(cagr - ((121 / 100) ** (1 / 2) - 1)) < 0.001

    def test_insufficient_data_single(self):
        assert compute_revenue_cagr([100], years=3) is None

    def test_insufficient_data_empty(self):
        assert compute_revenue_cagr([], years=3) is None

    def test_negative_revenue(self):
        revs = [100, -50]  # negative base
        cagr = compute_revenue_cagr(revs, years=1)
        assert cagr is None

    def test_zero_revenue(self):
        revs = [100, 0]
        cagr = compute_revenue_cagr(revs, years=1)
        assert cagr is None


# ---------------------------------------------------------------------------
# Tests: Data Auditor
# ---------------------------------------------------------------------------

class TestDataAuditor:
    def test_audit_indian(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        assert report.ticker == "TATAMOTORS.NS"
        assert report.is_indian is True
        assert report.currency == "INR"
        assert report.company_name == "Tata Motors Limited"
        assert len(report.successful_sources) == 3

    def test_audit_us(self):
        auditor = DataAuditor("NVDA", is_indian=False)
        report = auditor.audit(US_SOURCES)
        assert report.is_indian is False
        assert report.currency == "USD"
        assert report.threshold_pct == 20.0

    def test_consensus_revenue(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        rev_consensus = report.get_consensus("revenue")
        assert rev_consensus is not None
        # Should be median of 4.52e11 and 4.515e11 = 4.5175e11
        assert abs(rev_consensus - 4.5175e11) < 1e9

    def test_no_discrepancy_flag_on_close_values(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        assert len(report.flagged_metrics) == 0

    def test_discrepancy_flag_on_wide_spread(self):
        sources = [
            {
                "source": "Source A",
                "fetch_success": True,
                "currency": "USD",
                "company_name": "Test Corp",
                "revenue": 100.0,
            },
            {
                "source": "Source B",
                "fetch_success": True,
                "currency": "USD",
                "company_name": "Test Corp",
                "revenue": 200.0,  # 100% spread — well above threshold
            },
        ]
        auditor = DataAuditor("TEST", is_indian=False)
        report = auditor.audit(sources)
        flagged = [m for m in report.metric_audits if m.metric == "revenue" and m.is_flagged]
        assert len(flagged) == 1

    def test_threshold_indian_15pct(self):
        auditor = DataAuditor("TCS.NS", is_indian=True)
        assert auditor.threshold_pct == 15.0

    def test_threshold_us_20pct(self):
        auditor = DataAuditor("AAPL", is_indian=False)
        assert auditor.threshold_pct == 20.0

    def test_missing_metric(self):
        sources = [{"source": "S1", "fetch_success": True, "currency": "INR", "company_name": "X"}]
        auditor = DataAuditor("X.NS", is_indian=True)
        report = auditor.audit(sources)
        revenue_audit = next(m for m in report.metric_audits if m.metric == "revenue")
        assert not revenue_audit.has_data

    def test_failed_source_excluded(self):
        sources = INDIAN_SOURCES + [{"source": "Failed", "fetch_success": False}]
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(sources)
        assert len(report.successful_sources) == 3

    def test_summary_lines_generated(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        lines = report.summary_lines()
        assert len(lines) > 5
        assert any("DATA AUDIT REPORT" in line for line in lines)

    def test_as_dict(self):
        auditor = DataAuditor("NVDA", is_indian=False)
        report = auditor.audit(US_SOURCES)
        d = report.as_dict()
        assert "revenue" in d
        assert d["revenue"] is not None


# ---------------------------------------------------------------------------
# Tests: Source Attribution
# ---------------------------------------------------------------------------

class TestSourceAttribution:
    def test_from_audit_report(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        attribution = SourceAttribution.from_audit_report(report)
        cite = attribution.cite("revenue")
        assert cite != "Unknown Source"

    def test_missing_metric_cite(self):
        attribution = SourceAttribution()
        assert attribution.cite("nonexistent") == "Unknown Source"

    def test_consensus_citation_multi_source(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        attribution = SourceAttribution.from_audit_report(report)
        # Revenue comes from multiple sources → is_consensus should be True
        point = attribution.get("revenue")
        assert point is not None
        assert point.is_consensus is True

    def test_provenance_lines(self):
        auditor = DataAuditor("NVDA", is_indian=False)
        report = auditor.audit(US_SOURCES)
        attribution = SourceAttribution.from_audit_report(report)
        lines = attribution.provenance_lines()
        assert len(lines) > 0
        assert "DATA PROVENANCE" in lines[0]


# ---------------------------------------------------------------------------
# Tests: Enhanced Model Selector
# ---------------------------------------------------------------------------

class TestEnhancedModelSelector:
    def setup_method(self):
        self.indian_auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        self.indian_report = self.indian_auditor.audit(INDIAN_SOURCES)
        self.indian_attr = SourceAttribution.from_audit_report(self.indian_report)

        self.us_auditor = DataAuditor("NVDA", is_indian=False)
        self.us_report = self.us_auditor.audit(US_SOURCES)
        self.us_attr = SourceAttribution.from_audit_report(self.us_report)

    def test_indian_model_selection(self):
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        assert result.selected_model in (
            "Stable FCFE", "Two-Stage FCFE", "Three-Stage FCFE",
            "Stable FCFF", "Two-Stage FCFF", "Three-Stage FCFF",
            "Stable DDM", "Two-Stage DDM", "H-Model DDM",
        )
        assert result.terminal_growth_rate == pytest.approx(0.065, rel=0.01)

    def test_us_model_selection(self):
        selector = EnhancedModelSelector("NVDA", is_indian=False)
        result = selector.select(self.us_report, self.us_attr)
        assert result.terminal_growth_rate == pytest.approx(0.025, rel=0.01)

    def test_economy_growth_india(self):
        selector = EnhancedModelSelector("TCS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        assert result.economy_growth == pytest.approx(0.065, rel=0.01)
        assert "RBI" in result.economy_growth_source

    def test_economy_growth_us(self):
        selector = EnhancedModelSelector("AAPL", is_indian=False)
        result = selector.select(self.us_report, self.us_attr)
        assert result.economy_growth == pytest.approx(0.025, rel=0.01)
        assert "Federal Reserve" in result.economy_growth_source

    def test_eligibility_matrix_populated(self):
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        assert len(result.model_eligibility) == 9  # all 9 Damodaran models

    def test_selected_model_is_eligible(self):
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        selected = next(
            m for m in result.model_eligibility if m.model_name == result.selected_model
        )
        assert selected.is_eligible is True

    def test_rationale_lines_generated(self):
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        lines = result.rationale_lines()
        assert len(lines) > 10
        assert any("MODEL SELECTION RATIONALE" in line for line in lines)
        assert any("FINAL MODEL SELECTION" in line for line in lines)

    def test_firm_growth_from_cagr(self):
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        # Historical revenues are provided in mock data, so CAGR should be used
        assert "CAGR" in result.firm_growth_source
        assert result.firm_growth > 0

    def test_growth_premium_sign(self):
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        result = selector.select(self.indian_report, self.indian_attr)
        assert result.growth_premium == pytest.approx(
            result.firm_growth - result.economy_growth, rel=0.01
        )


# ---------------------------------------------------------------------------
# Tests: Valuation Engine
# ---------------------------------------------------------------------------

class TestValuationEngine:
    def setup_method(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        self.report = auditor.audit(INDIAN_SOURCES)
        attr = SourceAttribution.from_audit_report(self.report)
        selector = EnhancedModelSelector("TATAMOTORS.NS", is_indian=True)
        self.model_result = selector.select(self.report, attr)

    def test_valuation_runs(self):
        engine = ValuationEngine()
        result = engine.value(self.model_result, self.report)
        assert result is not None
        # Either succeeds with a value or provides a useful error message
        assert result.intrinsic_value_per_share is not None or result.error is not None

    def test_valuation_currency(self):
        engine = ValuationEngine()
        result = engine.value(self.model_result, self.report)
        assert result.currency == "INR"

    def test_valuation_upside_computed(self):
        engine = ValuationEngine()
        result = engine.value(self.model_result, self.report)
        if result.intrinsic_value_per_share is not None and result.current_price:
            assert result.upside_pct is not None

    def test_cost_of_equity_india(self):
        ke = _cost_of_equity(1.0, is_indian=True)
        assert ke > 0.10  # Should be > 10% for India
        assert ke < 0.25

    def test_cost_of_equity_us(self):
        ke = _cost_of_equity(1.0, is_indian=False)
        assert ke > 0.05
        assert ke < 0.15

    def test_cost_of_equity_default_beta(self):
        ke_default = _cost_of_equity(None, is_indian=False)
        ke_one = _cost_of_equity(1.0, is_indian=False)
        assert ke_default == pytest.approx(ke_one, rel=0.001)

    def test_terminal_value(self):
        tv = _terminal_value(100.0, 0.10, 0.03)
        assert abs(tv - 100 / 0.07) < 0.01

    def test_terminal_value_error_when_rate_le_growth(self):
        with pytest.raises(ValueError):
            _terminal_value(100.0, 0.03, 0.03)

    def test_summary_lines(self):
        engine = ValuationEngine()
        result = engine.value(self.model_result, self.report)
        lines = result.summary_lines()
        assert any("VALUATION" in line for line in lines)

    def test_ddm_no_dividend_returns_error(self):
        """DDM should gracefully handle missing dividend data."""
        sources = [
            {
                "source": "Test",
                "fetch_success": True,
                "currency": "USD",
                "company_name": "No Div Corp",
                "free_cash_flow": 1e9,
                "shares_outstanding": 1e8,
                "current_price": 50.0,
            }
        ]
        auditor = DataAuditor("NODIV", is_indian=False)
        report = auditor.audit(sources)
        attr = SourceAttribution.from_audit_report(report)
        selector = EnhancedModelSelector("NODIV", is_indian=False)
        mr = selector.select(report, attr)
        # Force DDM
        mr.selected_model = "Stable DDM"
        mr.dividend_per_share = 0.0
        engine = ValuationEngine()
        result = engine.value(mr, report)
        assert result.error is not None

    def test_us_valuation(self):
        auditor = DataAuditor("NVDA", is_indian=False)
        report = auditor.audit(US_SOURCES)
        attr = SourceAttribution.from_audit_report(report)
        selector = EnhancedModelSelector("NVDA", is_indian=False)
        mr = selector.select(report, attr)
        engine = ValuationEngine()
        result = engine.value(mr, report)
        assert result.currency == "USD"
        assert result.ticker == "NVDA"


# ---------------------------------------------------------------------------
# Tests: Cross-Verify
# ---------------------------------------------------------------------------

class TestCrossVerify:
    def test_high_quality_output(self):
        auditor = DataAuditor("TATAMOTORS.NS", is_indian=True)
        report = auditor.audit(INDIAN_SOURCES)
        cv = cross_verify(report)
        assert cv["overall_quality"] == "HIGH"
        assert cv["source_count"] == 3
        assert len(cv["flagged_metrics"]) == 0

    def test_flagged_metrics_detected(self):
        sources = [
            {"source": "A", "fetch_success": True, "currency": "USD",
             "company_name": "X", "revenue": 100.0, "net_income": 10.0},
            {"source": "B", "fetch_success": True, "currency": "USD",
             "company_name": "X", "revenue": 250.0, "net_income": 25.0},
        ]
        auditor = DataAuditor("X", is_indian=False)
        report = auditor.audit(sources)
        cv = cross_verify(report)
        assert "revenue" in cv["flagged_metrics"]
        assert cv["overall_quality"] in ("MEDIUM", "LOW")

    def test_verification_lines_returned(self):
        auditor = DataAuditor("NVDA", is_indian=False)
        report = auditor.audit(US_SOURCES)
        cv = cross_verify(report)
        assert len(cv["verification_lines"]) > 0

    def test_consensus_data_populated(self):
        auditor = DataAuditor("NVDA", is_indian=False)
        report = auditor.audit(US_SOURCES)
        cv = cross_verify(report)
        assert "revenue" in cv["consensus_data"]
        assert cv["consensus_data"]["revenue"] is not None

    def test_single_source_quality(self):
        sources = [
            {"source": "Only", "fetch_success": True, "currency": "INR",
             "company_name": "Solo", "revenue": 1e9}
        ]
        auditor = DataAuditor("SOLO.NS", is_indian=True)
        report = auditor.audit(sources)
        cv = cross_verify(report)
        # 1 source is insufficient for HIGH quality
        assert cv["overall_quality"] in ("MEDIUM", "LOW")


# ---------------------------------------------------------------------------
# Tests: MetricAudit edge cases
# ---------------------------------------------------------------------------

class TestMetricAudit:
    def test_spread_zero_single_source(self):
        audit = MetricAudit("revenue", {"Source A": 100.0}, 15.0)
        assert audit.spread_pct == 0.0
        assert not audit.is_flagged

    def test_spread_calculated_correctly(self):
        audit = MetricAudit("revenue", {"A": 100.0, "B": 80.0}, 15.0)
        # median = 90, max deviation = 10/90 = 11.1%
        assert abs(audit.spread_pct - (10 / 90 * 100)) < 0.1
        assert not audit.is_flagged  # 11.1% < 15% threshold

    def test_flagged_when_above_threshold(self):
        audit = MetricAudit("revenue", {"A": 100.0, "B": 200.0}, 15.0)
        assert audit.is_flagged

    def test_no_data(self):
        audit = MetricAudit("revenue", {}, 15.0)
        assert not audit.has_data
        assert audit.consensus_value is None
        assert audit.status_icon == "⚠️"

    def test_status_icon_ok(self):
        audit = MetricAudit("revenue", {"A": 100.0, "B": 102.0}, 15.0)
        assert audit.status_icon == "✅"

    def test_status_icon_flagged(self):
        audit = MetricAudit("revenue", {"A": 100.0, "B": 500.0}, 15.0)
        assert audit.status_icon == "🚩"
