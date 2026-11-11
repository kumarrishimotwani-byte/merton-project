"""GBM moment estimates for Person 2, following PLAN.md Section 2.5."""

import numpy as np
from src import config, theory


def log_returns(prices) -> np.ndarray:
    """Return ln(S_i / S_{i-1}) after dropping missing prices (PLAN.md 2.5).

    Missing observations bridge adjacent observed prices. Unequal time
    intervals require additional handling by the caller.
    """
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1:
        raise ValueError("Prices must be one-dimensional")
    values = values[~np.isnan(values)]
    if values.size < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("At least two finite positive prices are required")
    return np.diff(np.log(values))


def estimate_params(log_rets, dt=config.DT) -> tuple:
    """Estimate sigma^2=s^2/dt and mu=m/dt+sigma^2/2 (PLAN.md 2.5).

    Input is one finite history. Sample variance uses ddof=1. The Ito
    correction converts the mean log growth rate to the price SDE drift.
    """
    returns = np.asarray(log_rets, dtype=float)
    if returns.ndim != 1 or returns.size < 2 or not np.isfinite(returns).all():
        raise ValueError("At least two finite one-dimensional returns are required")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    sigma_squared = np.var(returns, ddof=1) / dt
    mu = np.mean(returns) / dt + sigma_squared / 2
    return float(mu), float(np.sqrt(sigma_squared))


def estimate_pi(log_rets, r, gamma, dt=config.DT) -> float:
    """Use theory.merton_pi with estimates from PLAN.md Section 2.5."""
    mu, sigma = estimate_params(log_rets, dt)
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(r) or sigma <= 0:
        raise ValueError("Positive gamma and volatility and finite r are required")
    return float(theory.merton_pi(mu, sigma, r, gamma))
