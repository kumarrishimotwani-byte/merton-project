"""Closed-form results of the Merton model (owner: Person 1).

All formulas are derived in docs/PLAN.md, Section 2.
Model: dS = mu S dt + sigma S dW (GBM), bank grows at r, investor holds a constant
fraction pi of wealth in the stock and has CRRA utility with risk aversion gamma.

The four functions in the first block are a frozen interface used by everyone.
The helpers below them are extra conveniences for comparisons with simulations.
"""

import numpy as np


# =============================================================================
# Frozen interface
# =============================================================================

def merton_pi(mu, sigma, r, gamma):
    """Merton's optimal fraction of wealth in the stock (PLAN.md 2.3).

        pi* = (mu - r) / (gamma * sigma^2)

    Comes from maximising the HJB bracket [r + pi(mu - r)] x V_x + 0.5 pi^2 sigma^2 x^2 V_xx
    with the ansatz V = f(t) x^(1-gamma)/(1-gamma), which gives x V_xx / V_x = -gamma.
    Works elementwise on numpy arrays.
    """
    return (mu - r) / (gamma * sigma**2)


def ce_rate(pi, mu, sigma, r, gamma):
    """Certainty-equivalent rate of a constant-fraction strategy (PLAN.md 2.4, eq. CE).

        CE(pi) = r + pi (mu - r) - 0.5 * gamma * sigma^2 * pi^2

    Meaning: E[U(X_T)] = U(x0 * exp(CE(pi) * T)), so CE is the sure, continuously
    compounded annual rate the investor values exactly as much as the risky strategy.
    Derivation: for constant pi, X_T is lognormal; use E[exp(k W_T)] = exp(k^2 T / 2).
    """
    return r + pi * (mu - r) - 0.5 * gamma * sigma**2 * pi**2


def loss(pi_hat, pi_star, sigma, gamma):
    """Certainty-equivalent rate lost per year by using pi_hat instead of pi_star (PLAN.md 2.4).

        Loss = CE(pi*) - CE(pi_hat) = 0.5 * gamma * sigma^2 * (pi_hat - pi*)^2

    Obtained by completing the square in CE(pi), using mu - r = gamma sigma^2 pi*.
    sigma and pi_star must be the TRUE values, not estimates.
    """
    return 0.5 * gamma * sigma**2 * (pi_hat - pi_star) ** 2


def utility(x, gamma):
    """CRRA utility U(x) = x^(1-gamma) / (1-gamma), and ln(x) when gamma == 1 (PLAN.md 2.2).

    Relative risk aversion -x U''(x) / U'(x) = gamma for every x.
    Returns a numpy array; x must be positive.
    """
    x = np.asarray(x, dtype=float)
    if gamma == 1:
        return np.log(x)
    return x ** (1 - gamma) / (1 - gamma)


# =============================================================================
# Extra helpers (not part of the frozen interface)
# =============================================================================

def benefit(mu, sigma, r, gamma):
    """Gain of optimal investing over cash, CE(pi*) - r = (mu - r)^2 / (2 gamma sigma^2)."""
    return (mu - r) ** 2 / (2 * gamma * sigma**2)


def worse_than_cash(pi_hat, pi_star):
    """True where CE(pi_hat) < r, i.e. pi_hat * (2 pi* - pi_hat) < 0 (PLAN.md 2.4).

    Because CE(pi_hat) - r = 0.5 gamma sigma^2 pi_hat (2 pi* - pi_hat).
    When mu > r this is the same as pi_hat < 0 or pi_hat > 2 pi*.
    """
    pi_hat = np.asarray(pi_hat, dtype=float)
    return pi_hat * (2 * pi_star - pi_hat) < 0


def certainty_equivalent_from_utility(mean_utility, x0, T, gamma):
    """Convert an average utility back into a CE rate (inverse of E[U] = U(x0 e^{CE T})).

        gamma != 1:  CE = (1/T) * ln( ((1-gamma) * Ubar)^(1/(1-gamma)) / x0 )
        gamma == 1:  CE = (Ubar - ln x0) / T

    Use it to turn a Monte Carlo average of utility(X_T) into a rate comparable with ce_rate.
    """
    if gamma == 1:
        return (mean_utility - np.log(x0)) / T
    return np.log(((1 - gamma) * mean_utility) ** (1 / (1 - gamma)) / x0) / T


def se_mu_approx(sigma, L):
    """Approximate standard error of mu_hat from L years of data: sigma / sqrt(L) (PLAN.md 2.5)."""
    return sigma / np.sqrt(L)


def sd_pihat_approx(sigma, gamma, L):
    """Benchmark sd of pi_hat, neglecting volatility estimation error: 1 / (gamma sigma sqrt(L)).

    Derivation: pi_hat - pi* ~ (mu_hat - mu) / (gamma sigma^2) and se(mu_hat) = sigma / sqrt(L).
    """
    return 1 / (gamma * sigma * np.sqrt(L))


def expected_loss_approx(gamma, L):
    """Benchmark expected CE loss per year from estimation error: 1 / (2 gamma L).

    E[Loss] = 0.5 gamma sigma^2 Var(pi_hat) = 0.5 gamma sigma^2 / (gamma^2 sigma^2 L);
    mu and sigma cancel. Neglects volatility estimation error.
    """
    return 1 / (2 * gamma * L)


def value_function(t, x, T, mu, sigma, r, gamma):
    """Merton value function V(t, x) = U(x * exp(rho (T - t))), rho = r + (mu - r)^2/(2 gamma sigma^2).

    Equivalent to f(t) x^(1-gamma)/(1-gamma) with f(t) = exp((1-gamma) rho (T - t)) (PLAN.md 2.3).
    """
    rho = r + benefit(mu, sigma, r, gamma)
    return utility(np.asarray(x, dtype=float) * np.exp(rho * (T - t)), gamma)
