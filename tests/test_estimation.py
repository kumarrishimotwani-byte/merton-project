"""Acceptance tests for src/estimation.py (Person 2). Skipped until implemented."""

import numpy as np
import pytest

from src import estimation, theory
from tests.conftest import call_or_skip

DT = 1 / 252


def _gbm_log_returns(mu, sigma, n, rng):
    return (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * rng.standard_normal(n)


def test_log_returns_simple_case():
    out = call_or_skip(estimation.log_returns, np.array([100.0, 110.0, 121.0]))
    assert np.allclose(out, [np.log(1.1), np.log(1.1)])


def test_sigma_recovered_precisely():
    rng = np.random.default_rng(0)
    R = _gbm_log_returns(0.12, 0.18, 252 * 30, rng)
    mu_hat, sigma_hat = call_or_skip(estimation.estimate_params, R, DT)
    assert sigma_hat == pytest.approx(0.18, abs=0.005)


def test_mu_hat_unbiased_with_ito_correction():
    # Average mu_hat over many 10-year histories; the +sigma^2/2 correction must be included.
    rng = np.random.default_rng(1)
    call_or_skip(estimation.estimate_params, _gbm_log_returns(0.12, 0.18, 10, rng), DT)
    mus = [estimation.estimate_params(_gbm_log_returns(0.12, 0.18, 2520, rng), DT)[0] for _ in range(400)]
    # se of the average = (0.18 / sqrt(10)) / sqrt(400) = 0.0028
    assert np.mean(mus) == pytest.approx(0.12, abs=0.01)


def test_estimate_pi_consistent_with_theory():
    rng = np.random.default_rng(2)
    R = _gbm_log_returns(0.12, 0.18, 2520, rng)
    pi_hat = call_or_skip(estimation.estimate_pi, R, 0.06, 3.0, DT)
    mu_hat, sigma_hat = estimation.estimate_params(R, DT)
    assert pi_hat == pytest.approx(theory.merton_pi(mu_hat, sigma_hat, 0.06, 3.0))
