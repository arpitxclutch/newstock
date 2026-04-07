"""
multi_source_fetcher.py
-----------------------
Fetches financial data from multiple credible sources for both Indian and US companies.

Indian Companies (.NS / .BO):
  - Source 1: Yahoo Finance (yfinance) — primary market data
  - Source 2: Screener.in — financial metrics & ratios (public)
  - Source 3: BSE/NSE public endpoints — official exchange data

US Companies (no suffix or well-known US tickers):
  - Source 1: Yahoo Finance (yfinance)
  - Source 2: SEC EDGAR REST API — official 10-K/10-Q filings (free, no key)
  - Source 3: Alpha Vantage — alternative fundamentals (free tier, optional API key)
"""

from __future__ import annotations

import os
import re
import time
import logging
from typing import Any

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": "NewStock Research Bot research@newstock.example.com",
    "Accept": "application/json",
}

_REQUEST_TIMEOUT = 15  # seconds


def _safe_get(url: str, params: dict | None = None, headers: dict | None = None) -> dict | None:
    """Perform a GET request and return parsed JSON, or None on failure."""
    try:
        resp = requests.get(
            url,
            params=params,
            headers=headers or HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug("HTTP fetch failed for %s: %s", url, exc)
        return None


def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")


def _us_cik(ticker: str) -> str | None:
    """Resolve a US ticker to its SEC CIK number."""
    data = _safe_get(
        "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=2020-01-01&forms=10-K",
    )
    if data:
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_source", {}).get("entity_id")
    # Fallback: company_tickers.json
    ct = _safe_get("https://www.sec.gov/files/company_tickers.json")
    if ct:
        for entry in ct.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    return None


# ---------------------------------------------------------------------------
# Yahoo Finance fetcher (shared for both markets)
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str) -> dict[str, Any]:
    """Fetch key fundamentals via yfinance."""
    result: dict[str, Any] = {"source": "Yahoo Finance", "ticker": ticker}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result["company_name"] = info.get("longName") or info.get("shortName", ticker)
        result["currency"] = info.get("currency", "USD")
        result["market_cap"] = info.get("marketCap")
        result["revenue"] = info.get("totalRevenue")
        result["net_income"] = info.get("netIncomeToCommon")
        result["ebitda"] = info.get("ebitda")
        result["total_debt"] = info.get("totalDebt")
        result["cash"] = info.get("totalCash")
        result["free_cash_flow"] = info.get("freeCashflow")
        result["shares_outstanding"] = info.get("sharesOutstanding")
        result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        result["beta"] = info.get("beta")
        result["pe_ratio"] = info.get("trailingPE")
        result["dividend_per_share"] = info.get("dividendRate")
        result["book_value"] = info.get("bookValue")
        result["revenue_growth"] = info.get("revenueGrowth")
        result["earnings_growth"] = info.get("earningsGrowth")
        result["return_on_equity"] = info.get("returnOnEquity")
        result["return_on_assets"] = info.get("returnOnAssets")
        result["debt_to_equity"] = info.get("debtToEquity")
        result["profit_margin"] = info.get("profitMargins")
        result["sector"] = info.get("sector")
        result["industry"] = info.get("industry")
        result["country"] = info.get("country")

        # Historical revenue for CAGR
        try:
            fin = t.financials
            if fin is not None and not fin.empty:
                rev_row = fin.loc["Total Revenue"] if "Total Revenue" in fin.index else None
                if rev_row is not None:
                    revs = rev_row.dropna().tolist()
                    result["historical_revenues"] = revs[:4]  # up to 4 years
        except Exception:
            pass

        result["fetch_success"] = True
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
        result["fetch_success"] = False
    return result


# ---------------------------------------------------------------------------
# Screener.in fetcher (Indian companies)
# ---------------------------------------------------------------------------

def _fetch_screener(ticker: str) -> dict[str, Any]:
    """
    Fetch data from Screener.in public API.
    Screener uses NSE/BSE symbol without the exchange suffix.
    """
    result: dict[str, Any] = {"source": "Screener.in", "ticker": ticker}
    symbol = re.sub(r"\.(NS|BO)$", "", ticker.upper())
    url = f"https://www.screener.in/api/company/{symbol}/financial/"
    try:
        resp = requests.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 404:
            # Try lowercase
            url2 = f"https://www.screener.in/api/company/{symbol.lower()}/financial/"
            resp = requests.get(url2, headers={**HEADERS, "Accept": "application/json"}, timeout=_REQUEST_TIMEOUT)
        if resp.status_code not in (200, 201):
            result["fetch_success"] = False
            return result
        data = resp.json()
        result["currency"] = "INR"

        # Extract latest annual data from the income statement
        income = data.get("income_statement", {})
        latest = {}
        for key, series in income.items():
            if series:
                latest[key] = series[-1].get("value") if isinstance(series[-1], dict) else series[-1]

        result["revenue"] = latest.get("Net Sales") or latest.get("Revenue")
        result["net_income"] = latest.get("Net Profit") or latest.get("PAT")
        result["ebitda"] = latest.get("EBITDA") or latest.get("Operating Profit")

        balance = data.get("balance_sheet", {})
        bal_latest = {}
        for key, series in balance.items():
            if series:
                bal_latest[key] = series[-1].get("value") if isinstance(series[-1], dict) else series[-1]
        result["total_debt"] = bal_latest.get("Borrowings") or bal_latest.get("Total Debt")
        result["book_value"] = bal_latest.get("Total Equity") or bal_latest.get("Net Worth")

        # Revenue history for CAGR
        rev_series = income.get("Net Sales") or income.get("Revenue")
        if rev_series:
            result["historical_revenues"] = [
                (v["value"] if isinstance(v, dict) else v) for v in rev_series[-4:]
            ]

        result["fetch_success"] = True
    except Exception as exc:
        logger.debug("Screener.in fetch failed for %s: %s", ticker, exc)
        result["fetch_success"] = False
    return result


# ---------------------------------------------------------------------------
# BSE/NSE public data fetcher (Indian companies)
# ---------------------------------------------------------------------------

def _fetch_bse_nse(ticker: str) -> dict[str, Any]:
    """
    Fetch data from NSE India public API.
    NSE provides publicly accessible endpoints for company financials.
    """
    result: dict[str, Any] = {"source": "NSE/BSE", "ticker": ticker}
    symbol = re.sub(r"\.(NS|BO)$", "", ticker.upper())
    nse_headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewStock/1.0)",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com",
    }
    try:
        # NSE company info
        session = requests.Session()
        session.headers.update(nse_headers)
        # Prime the session cookie
        session.get("https://www.nseindia.com", timeout=_REQUEST_TIMEOUT)
        time.sleep(0.5)

        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        resp = session.get(url, timeout=_REQUEST_TIMEOUT)
        if resp.status_code != 200:
            result["fetch_success"] = False
            return result
        data = resp.json()

        result["currency"] = "INR"
        md = data.get("priceInfo", {})
        result["current_price"] = md.get("lastPrice")

        company_info = data.get("info", {})
        result["company_name"] = company_info.get("companyName", symbol)

        # Market cap from NSE
        result["market_cap"] = data.get("securityInfo", {}).get("issuedCap")

        result["fetch_success"] = True
    except Exception as exc:
        logger.debug("NSE fetch failed for %s: %s", ticker, exc)
        result["fetch_success"] = False
    return result


# ---------------------------------------------------------------------------
# SEC EDGAR fetcher (US companies)
# ---------------------------------------------------------------------------

def _fetch_sec_edgar(ticker: str) -> dict[str, Any]:
    """
    Fetch financial data from SEC EDGAR REST API (free, no API key needed).
    Uses the companyfacts endpoint for structured financial data.
    """
    result: dict[str, Any] = {"source": "SEC EDGAR", "ticker": ticker}
    try:
        cik = _us_cik(ticker)
        if not cik:
            result["fetch_success"] = False
            result["error"] = "CIK not found"
            return result

        cik_padded = str(cik).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        data = _safe_get(url, headers={"User-Agent": "NewStock research@newstock.example.com"})
        if not data:
            result["fetch_success"] = False
            return result

        result["company_name"] = data.get("entityName", ticker)
        result["currency"] = "USD"

        facts = data.get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        def _latest_annual(concept: str) -> float | None:
            """Return the most recent annual value for a GAAP concept."""
            concept_data = us_gaap.get(concept, {})
            units = concept_data.get("units", {})
            usd_entries = units.get("USD") or units.get("shares") or []
            annual = [
                e for e in usd_entries
                if e.get("form") in ("10-K", "20-F") and e.get("fp") == "FY"
            ]
            if not annual:
                return None
            annual.sort(key=lambda x: x.get("end", ""), reverse=True)
            return annual[0].get("val")

        def _revenue_history() -> list[float]:
            """Return up to 4 years of annual revenue."""
            for concept in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"):
                concept_data = us_gaap.get(concept, {})
                entries = concept_data.get("units", {}).get("USD", [])
                annual = sorted(
                    [e for e in entries if e.get("form") in ("10-K", "20-F") and e.get("fp") == "FY"],
                    key=lambda x: x.get("end", ""),
                    reverse=True,
                )
                if annual:
                    return [e["val"] for e in annual[:4]]
            return []

        result["revenue"] = (
            _latest_annual("Revenues")
            or _latest_annual("RevenueFromContractWithCustomerExcludingAssessedTax")
            or _latest_annual("SalesRevenueNet")
        )
        result["net_income"] = _latest_annual("NetIncomeLoss")
        result["total_debt"] = (
            _latest_annual("LongTermDebt")
            or _latest_annual("LongTermDebtAndCapitalLeaseObligations")
        )
        result["cash"] = _latest_annual("CashAndCashEquivalentsAtCarryingValue")
        result["ebitda"] = _latest_annual("OperatingIncomeLoss")
        result["shares_outstanding"] = _latest_annual("CommonStockSharesOutstanding")
        result["historical_revenues"] = _revenue_history()

        result["fetch_success"] = True
    except Exception as exc:
        logger.warning("SEC EDGAR fetch failed for %s: %s", ticker, exc)
        result["fetch_success"] = False
    return result


# ---------------------------------------------------------------------------
# Alpha Vantage fetcher (US companies) — API key optional
# ---------------------------------------------------------------------------

def _fetch_alpha_vantage(ticker: str, api_key: str | None = None) -> dict[str, Any]:
    """
    Fetch fundamentals from Alpha Vantage.
    Uses the OVERVIEW endpoint. Free tier API key available at alphavantage.co.
    The key can be set via the AV_API_KEY environment variable.
    """
    result: dict[str, Any] = {"source": "Alpha Vantage", "ticker": ticker}
    key = api_key or os.environ.get("AV_API_KEY", "demo")
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": ticker, "apikey": key}
        data = _safe_get(url, params=params)
        if not data or "Symbol" not in data:
            result["fetch_success"] = False
            result["error"] = data.get("Note") or data.get("Information") or "No data returned"
            return result

        result["company_name"] = data.get("Name", ticker)
        result["currency"] = data.get("Currency", "USD")
        result["sector"] = data.get("Sector")
        result["industry"] = data.get("Industry")

        def _f(key_name: str) -> float | None:
            v = data.get(key_name)
            if v and v not in ("None", "-"):
                try:
                    return float(v)
                except ValueError:
                    return None
            return None

        result["market_cap"] = _f("MarketCapitalization")
        result["revenue"] = _f("RevenueTTM")
        result["ebitda"] = _f("EBITDA")
        result["net_income"] = _f("NetIncomeTTM") or (_f("EPS") and _f("SharesOutstanding") and _f("EPS") * _f("SharesOutstanding"))
        result["pe_ratio"] = _f("PERatio")
        result["book_value"] = _f("BookValue")
        result["dividend_per_share"] = _f("DividendPerShare")
        result["return_on_equity"] = _f("ReturnOnEquityTTM")
        result["beta"] = _f("Beta")
        result["revenue_growth"] = _f("QuarterlyRevenueGrowthYOY")
        result["earnings_growth"] = _f("QuarterlyEarningsGrowthYOY")
        result["profit_margin"] = _f("ProfitMargin")
        result["debt_to_equity"] = _f("DebtToEquityRatio") if "DebtToEquityRatio" in data else None
        shares = _f("SharesOutstanding")
        result["shares_outstanding"] = shares

        result["fetch_success"] = True
    except Exception as exc:
        logger.warning("Alpha Vantage fetch failed for %s: %s", ticker, exc)
        result["fetch_success"] = False
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultiSourceFetcher:
    """
    Orchestrates data collection from multiple sources for a given ticker.

    Usage:
        fetcher = MultiSourceFetcher("TATAMOTORS.NS")
        sources = fetcher.fetch_all()
    """

    def __init__(self, ticker: str, av_api_key: str | None = None) -> None:
        self.ticker = ticker.upper().strip()
        self.av_api_key = av_api_key
        self.is_indian = _is_indian_ticker(self.ticker)

    def fetch_all(self) -> list[dict[str, Any]]:
        """
        Fetch data from all applicable sources and return a list of source dicts.
        Each dict contains the fetched values plus 'source' and 'fetch_success' keys.
        """
        sources: list[dict[str, Any]] = []

        if self.is_indian:
            logger.info("Fetching Indian company data for %s", self.ticker)
            # Source 1: Yahoo Finance
            yf_data = _fetch_yfinance(self.ticker)
            sources.append(yf_data)
            # Source 2: Screener.in
            screener_data = _fetch_screener(self.ticker)
            sources.append(screener_data)
            # Source 3: NSE/BSE
            bse_data = _fetch_bse_nse(self.ticker)
            sources.append(bse_data)
        else:
            logger.info("Fetching US company data for %s", self.ticker)
            # Source 1: Yahoo Finance
            yf_data = _fetch_yfinance(self.ticker)
            sources.append(yf_data)
            # Source 2: SEC EDGAR
            sec_data = _fetch_sec_edgar(self.ticker)
            sources.append(sec_data)
            # Source 3: Alpha Vantage
            av_data = _fetch_alpha_vantage(self.ticker, self.av_api_key)
            sources.append(av_data)

        successful = [s for s in sources if s.get("fetch_success")]
        logger.info(
            "Fetched %d/%d sources successfully for %s",
            len(successful), len(sources), self.ticker,
        )
        return sources
