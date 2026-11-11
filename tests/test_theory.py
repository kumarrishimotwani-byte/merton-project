"""Checks of the closed-form formulas in src/theory.py (Person 1)."""

import numpy as np
import pytest

from src import theory

MU, SIGMA, R, GAMMA = 0.12, 0.18, 0.06, 3.0


def test_merton_pi_value():
    # 0.06 / (3 * 0.0324) = 0.6173
    assert theory.merton_pi(MU, SIGMA, R, GAMMA) == pytest.approx(0.06 / 0.0972)


def test_ce_rate_peaks_at_merton_pi():
    grid = np.linspace(-1, 3, 40001)
    best = grid[np.argmax(theory.ce_rate(grid, MU, SIGMA, R, GAMMA))]
    assert best == pytest.approx(theory.merton_pi(MU, SIGMA, R, GAMMA), abs=1e-4)


def test_ce_at_zero_is_r_and_at_optimum_is_r_plus_benefit():
    pi_star = theory.merton_pi(MU, SIGMA, R, GAMMA)
    assert theory.ce_rate(0.0, MU, SIGMA, R, GAMMA) == pytest.approx(R)
    assert theory.ce_rate(pi_star, MU, SIGMA, R, GAMMA) == pytest.approx(R + theory.benefit(MU, SIGMA, R, GAMMA))


@pytest.mark.parametrize("pi_hat", [-0.5, 0.0, 0.3, 1.2, 2.0])
def test_loss_equals_ce_difference(pi_hat):
    pi_star = theory.merton_pi(MU, SIGMA, R, GAMMA)
    direct = theory.ce_rate(pi_star, MU, SIGMA, R, GAMMA) - theory.ce_rate(pi_hat, MU, SIGMA, R, GAMMA)
    assert theory.loss(pi_hat, pi_star, SIGMA, GAMMA) == pytest.approx(direct)


@pytest.mark.parametrize("mu", [0.12, 0.06, 0.02])   # mu > r, mu = r, mu < r
def test_worse_than_cash_matches_ce(mu):
    pi_star = theory.merton_pi(mu, SIGMA, R, GAMMA)
    grid = np.linspace(-3, 3, 601)
    expected = theory.ce_rate(grid, mu, SIGMA, R, GAMMA) < R - 1e-12
    assert np.array_equal(theory.worse_than_cash(grid, pi_star), expected)


def test_worse_than_cash_boundaries_when_mu_above_r():
    pi_star = theory.merton_pi(MU, SIGMA, R, GAMMA)
    flags = theory.worse_than_cash([-0.01, 0.01, 2 * pi_star - 0.01, 2 * pi_star + 0.01], pi_star)
    assert list(flags) == [True, False, False, True]


@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0, 5.0])
def test_relative_risk_aversion_is_gamma(gamma):
    x, h = 2.0, 1e-4
    u = lambda z: theory.utility(z, gamma)
    u1 = (u(x + h) - u(x - h)) / (2 * h)
    u2 = (u(x + h) - 2 * u(x) + u(x - h)) / h**2
    assert -x * u2 / u1 == pytest.approx(gamma, rel=1e-3)


@pytest.mark.parametrize("gamma", [1.0, 3.0])
def test_certainty_equivalent_inverts_expected_utility(gamma):
    x0, T, ce = 2.0, 5.0, 0.07
    mean_u = theory.utility(x0 * np.exp(ce * T), gamma)
    assert theory.certainty_equivalent_from_utility(mean_u, x0, T, gamma) == pytest.approx(ce)


def test_ce_formula_matches_lognormal_expectation():
    # E[U(X_T)] for X_T = x0 exp((a - b^2/2) T + b W_T), computed by numerical integration over W_T.
    pi, T, x0 = 0.8, 2.0, 1.0
    a, b = R + pi * (MU - R), pi * SIGMA
    w = np.linspace(-12, 12, 200001) * np.sqrt(T)
    density = np.exp(-w**2 / (2 * T)) / np.sqrt(2 * np.pi * T)
    xT = x0 * np.exp((a - 0.5 * b**2) * T + b * w)
    mean_u = np.trapezoid(theory.utility(xT, GAMMA) * density, w)
    ce = theory.certainty_equivalent_from_utility(mean_u, x0, T, GAMMA)
    assert ce == pytest.approx(theory.ce_rate(pi, MU, SIGMA, R, GAMMA), rel=1e-6)


def test_value_function_solves_hjb():
    # Finite-difference check that V(t, x) satisfies the HJB equation with pi = pi*.
    T, t, x, h = 5.0, 1.0, 1.3, 1e-4
    V = lambda tt, xx: theory.value_function(tt, xx, T, MU, SIGMA, R, GAMMA)
    Vt = (V(t + h, x) - V(t - h, x)) / (2 * h)
    Vx = (V(t, x + h) - V(t, x - h)) / (2 * h)
    Vxx = (V(t, x + h) - 2 * V(t, x) + V(t, x - h)) / h**2
    pi = theory.merton_pi(MU, SIGMA, R, GAMMA)
    hjb = Vt + (R + pi * (MU - R)) * x * Vx + 0.5 * pi**2 * SIGMA**2 * x**2 * Vxx
    assert abs(hjb) < 1e-5 * abs(Vt)
    assert theory.value_function(T, x, T, MU, SIGMA, R, GAMMA) == pytest.approx(theory.utility(x, GAMMA))


def test_benchmark_approximations():
    assert theory.sd_pihat_approx(0.18, 3, 10) == pytest.approx(1 / (3 * 0.18 * np.sqrt(10)))
    assert theory.expected_loss_approx(3, 10) == pytest.approx(1 / 60)
