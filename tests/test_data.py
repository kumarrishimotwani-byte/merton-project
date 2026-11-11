"""Acceptance tests for src/data.py (Person 2). Skipped until implemented and data/raw is filled."""

import pandas as pd
import pytest

from src import config, data
from tests.conftest import call_or_skip


def test_main_ticker_prices():
    if not (config.DATA_RAW / f"{config.MAIN_TICKER}.csv").exists():
        pytest.skip("data/raw/^NSEI.csv not downloaded yet")
    s = call_or_skip(data.load_prices, config.MAIN_TICKER)
    assert isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex)
    assert (s > 0).all() and s.index.is_monotonic_increasing and len(s) > 252 * 10


def test_risk_free_is_a_decimal_rate():
    r = call_or_skip(data.risk_free_constant)
    assert 0.0 < r < 0.15, "rates must be decimals, e.g. 0.061, not 6.1"
