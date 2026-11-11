"""Downloading and loading market data (owner: Person 2). See docs/PLAN.md Section 6.

TODO (Person 2): implement the three functions below. Keep the signatures exactly as they are.
"""

import pandas as pd

from src import config


def load_prices(ticker, start=config.START_DATE, end=config.END_DATE) -> pd.Series:
    """Daily adjusted closing prices for one ticker, as a pd.Series with a DatetimeIndex.

    Plan:
      - If data/raw/<ticker>.csv exists, load it from there (no re-download).
      - Otherwise download with yfinance (auto_adjust=True, so dividends and splits are included
        for stocks), save to data/raw/<ticker>.csv, and return it.
      - Drop missing values; check every price is > 0; sort by date.
      - Note: ^NSEI (NIFTY 50) is a price index, so its returns exclude dividends.
    """
    raise NotImplementedError("Person 2: implement load_prices")


def load_risk_free() -> pd.Series:
    """Monthly 91-day Indian T-bill rate as an annual, continuously compounded decimal.

    Plan:
      - Source: RBI Database on Indian Economy (https://data.rbi.org.in), saved as
        data/raw/tbill_91d.csv.
      - A quoted yield y in percent becomes r = ln(1 + y / 100).
      - Return a pd.Series indexed by month-end dates.
    """
    raise NotImplementedError("Person 2: implement load_risk_free")


def risk_free_constant() -> float:
    """Average of load_risk_free() over the sample period, as one decimal number (e.g. 0.061).

    If the T-bill series cannot be obtained, return config.R_PLACEHOLDER and document this
    in the report.
    """
    raise NotImplementedError("Person 2: implement risk_free_constant")
