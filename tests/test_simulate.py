"""Acceptance tests for src/simulate.py (Person 3). Skipped until implemented."""

import numpy as np
import pytest

from src import simulate
from tests.conftest import call_or_skip


@pytest.mark.parametrize("method", ["exact", "euler", "milstein"])
def test_gbm_shape_and_start(method):
    S = call_or_skip(simulate.simulate_gbm, 100.0, 0.1, 0.2, 1.0, 50, 7, method=method, seed=0)
    assert S.shape == (7, 51)
    assert np.all(S[:, 0] == 100.0)


def test_gbm_exact_mean():
    S = call_or_skip(simulate.simulate_gbm, 100.0, 0.1, 0.2, 1.0, 12, 200_000, method="exact", seed=0)
    # E[S_T] = S0 e^{mu T}; MC se is about 100 * 0.2 / sqrt(2e5) = 0.045
    assert S[:, -1].mean() == pytest.approx(100 * np.exp(0.1), abs=0.3)


def test_same_Z_gives_close_schemes_for_small_dt():
    Z = np.random.default_rng(0).standard_normal((2000, 1008))
    ex = call_or_skip(simulate.simulate_gbm, 100.0, 0.1, 0.2, 1.0, 1008, 2000, method="exact", Z=Z)
    eu = simulate.simulate_gbm(100.0, 0.1, 0.2, 1.0, 1008, 2000, method="euler", Z=Z)
    mi = simulate.simulate_gbm(100.0, 0.1, 0.2, 1.0, 1008, 2000, method="milstein", Z=Z)
    err_eu = np.mean(np.abs(eu[:, -1] - ex[:, -1]))
    err_mi = np.mean(np.abs(mi[:, -1] - ex[:, -1]))
    assert err_eu < 0.5 and err_mi < err_eu


def test_seed_reproducible():
    a = call_or_skip(simulate.simulate_gbm, 100.0, 0.1, 0.2, 1.0, 10, 5, seed=42)
    b = simulate.simulate_gbm(100.0, 0.1, 0.2, 1.0, 10, 5, seed=42)
    assert np.array_equal(a, b)


def test_wealth_all_cash_is_deterministic():
    X = call_or_skip(simulate.simulate_wealth, 1.0, 0.0, 0.12, 0.18, 0.06, 2.0, 100, 3, method="exact", seed=0)
    assert X.shape == (3, 101)
    assert np.allclose(X[:, -1], np.exp(0.06 * 2.0))


def test_wealth_all_stock_matches_gbm_with_same_Z():
    Z = np.random.default_rng(3).standard_normal((4, 20))
    X = call_or_skip(simulate.simulate_wealth, 1.0, 1.0, 0.12, 0.18, 0.06, 1.0, 20, 4, method="exact", Z=Z)
    S = simulate.simulate_gbm(1.0, 0.12, 0.18, 1.0, 20, 4, method="exact", Z=Z)
    assert np.allclose(X, S)
