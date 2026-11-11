"""Load the reviewed Person 2 snapshot. See data/raw/P2_README.md for provenance."""

import warnings
import numpy as np
import pandas as pd
from src import config


def load_prices(ticker, start=config.START_DATE, end=config.END_DATE) -> pd.Series:
    """Load positive daily closes for PLAN.md Section 6, sorted by date.

    The committed MAIN_TICKER cache contains official NSE price-index levels.
    Stocks contain Yahoo adjusted closes. A missing index cache fails explicitly
    to avoid silently reverting to the incomplete Yahoo history. For stocks,
    a missing cache downloads full history from config.START_DATE via yfinance.
    The end argument is exclusive, matching yfinance.
    """
    if ticker not in config.TICKERS:
        raise ValueError(f"Unsupported ticker {ticker!r}; use config.TICKERS")
    path = config.DATA_RAW / f"{ticker}.csv"
    downloaded = not path.exists()
    if downloaded:
        if ticker == config.MAIN_TICKER:
            raise FileNotFoundError("Restore the reviewed NSE index CSV from Git; see data/raw/P2_README.md")
        import yfinance as yf
        frame = yf.Ticker(ticker).history(start=config.START_DATE, auto_adjust=True)
        if frame.empty:
            raise RuntimeError(f"Yahoo returned no data for {ticker}")
        prices = frame["Close"].astype(float)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        warnings.warn("Fresh Yahoo download needs the documented calendar audit before final use", stacklevel=2)
    else:
        frame = pd.read_csv(path, parse_dates=["Date"])
        prices = frame.set_index("Date")["adjusted_close"].astype(float)
    prices = prices.sort_index().dropna().rename("adjusted_close")
    if prices.empty or prices.index.hasnans or prices.index.has_duplicates:
        raise ValueError(f"Empty or duplicate/invalid dates for {ticker}")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"Nonpositive or nonfinite prices for {ticker}")
    if downloaded:
        config.DATA_RAW.mkdir(parents=True, exist_ok=True)
        prices.rename_axis("Date").to_csv(path)
    selected = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        selected = selected.loc[selected.index < pd.Timestamp(end)]
    return selected


def load_risk_free() -> pd.Series:
    """Load annual continuous decimal rates for PLAN.md Section 6.

    Index dates are actual last NSE sessions in completed months. Each quote
    is available by that date. The CSV uses r=log1p(y*tau)/tau, tau=91/365,
    for an RBI simple annual ACT/365 quote y expressed as a decimal.
    See P2_README.md for the departure from the plan's conversion shortcut.
    """
    path = config.DATA_RAW / "tbill_91d.csv"
    if not path.exists():
        raise FileNotFoundError("Restore data/raw/tbill_91d.csv from Git; no observed monthly rates available")
    frame = pd.read_csv(path, parse_dates=["date"])
    rates = frame.set_index("date")["rate"].astype(float).sort_index()
    if rates.empty or rates.index.hasnans or rates.index.has_duplicates or not np.isfinite(rates).all():
        raise ValueError("Invalid monthly T-bill series")
    if "available_on" in frame:
        available_on = pd.to_datetime(frame["available_on"])
        if available_on.isna().any() or (available_on > frame["date"]).any():
            raise ValueError("A T-bill quote postdates its monthly decision date")
    return rates


def risk_free_constant() -> float:
    """Mean monthly continuous rate for full-sample summaries (PLAN.md 6).

    If the file is absent, warn and use config.R_PLACEHOLDER as permitted by
    the plan. A backtest must use dated rates rather than this full-sample mean.
    """
    if not (config.DATA_RAW / "tbill_91d.csv").exists():
        warnings.warn("No T-bill file; using config.R_PLACEHOLDER assumption", stacklevel=2)
        return float(config.R_PLACEHOLDER)
    return float(load_risk_free().mean())
