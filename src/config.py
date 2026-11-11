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

# --- Person 3: accuracy experiment (PLAN.md Section 7, Test A) --------------------
P3_ACCURACY_S0 = 100.0
P3_ACCURACY_MU = 0.10
P3_ACCURACY_SIGMA = 0.20
P3_ACCURACY_T = 1.0
P3_ACCURACY_N_PATHS = 10_000
P3_ACCURACY_STEPS = (12, 52, TRADING_DAYS)
# The experiment derives the fine grid as lcm(12, 52, 252) = 3276.
# The plan's 1008-step fine grid cannot aggregate into equal weekly blocks.

# --- Person 3: optimality and sensitivity (PLAN.md Section 7, Tests B and C) -----
P3_WEALTH_X0 = 1.0
P3_WEALTH_T = 5.0
P3_WEALTH_STEPS_PER_YEAR = 52
P3_WEALTH_N_PATHS = 20_000
P3_PI_GRID = (-0.5, 2.5, 31)                 # np.linspace(start, stop, count)
P3_WEALTH_MULTIPLIERS = (0.5, 1.0, 2.0)     # multiples of pi* at the main gamma
P3_SENSITIVITY_GAMMA = (1.0, 10.0)
P3_SENSITIVITY_SIGMA = (0.1, 0.4)
P3_SENSITIVITY_EXCESS_RETURN = (-0.05, 0.15)
P3_SENSITIVITY_POINTS = 101
