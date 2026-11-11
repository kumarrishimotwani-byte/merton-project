"""Shared constants for the whole project (owner: Person 1).

Every script imports its settings from here. Do not hard-code these values anywhere else.
Units: time in years; rates are annual, continuously compounded, decimals (6% = 0.06).
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# --- Assets (Yahoo Finance tickers) -------------------------------------------
MAIN_TICKER = "^NSEI"                       # NIFTY 50 index (price index, no dividends)
TICKERS = ["^NSEI", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"]   # index, bank, IT, FMCG
START_DATE = "2008-01-01"
END_DATE = None                             # None = latest available

# --- Time ----------------------------------------------------------------------
TRADING_DAYS = 252
DT = 1 / TRADING_DAYS                       # one trading day in years

# --- Preferences -----------------------------------------------------------------
GAMMA = 3.0                                 # main relative risk aversion
GAMMAS = [2.0, 3.0, 5.0]                    # robustness checks

# --- Placeholders until Person 2's real estimates exist --------------------------
R_PLACEHOLDER = 0.065                       # risk-free rate
PLACEHOLDER_MU = 0.12                       # drift
PLACEHOLDER_SIGMA = 0.18                    # volatility

# --- Reproducibility and outputs -------------------------------------------------
SEED = 42
FIG_DPI = 150
