"""Estimating mu, sigma and pi from prices (owner: Person 2). See docs/PLAN.md Section 2.5.

Under GBM, daily log returns R_i = ln(S_i / S_{i-1}) are i.i.d.
N((mu - sigma^2/2) dt, sigma^2 dt). Reverse these moments to estimate mu and sigma.

TODO (Person 2): implement the three functions below. Keep the signatures exactly as they are.
"""

import numpy as np

from src import config


def log_returns(prices) -> np.ndarray:
    """Log returns ln(S_i / S_{i-1}) of a price series, with NaNs dropped.

    `prices` may be a pd.Series or a 1-D numpy array. Returns a 1-D numpy array of length n - 1.
    """
    raise NotImplementedError("Person 2: implement log_returns")


def estimate_params(log_rets, dt=config.DT) -> tuple:
    """Return (mu_hat, sigma_hat) from log returns sampled every dt years.

        sigma_hat^2 = s^2 / dt                    (s^2: sample variance, ddof=1)
        mu_hat      = m / dt + sigma_hat^2 / 2     (m: sample mean)

    The + sigma_hat^2 / 2 undoes the -sigma^2/2 that Ito's lemma puts into the log-return mean.
    Must also work row-wise if Person 4 passes a 2-D array (one history per row) - optional,
    but useful for speed; if implemented, return two arrays.
    """
    raise NotImplementedError("Person 2: implement estimate_params")


def estimate_pi(log_rets, r, gamma, dt=config.DT) -> float:
    """Estimated Merton fraction: theory.merton_pi(*estimate_params(log_rets, dt), r, gamma)."""
    raise NotImplementedError("Person 2: implement estimate_pi")
