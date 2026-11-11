"""Simulating price and wealth paths (owner: Person 3). See docs/PLAN.md Section 7.

Brownian increments: dW = sqrt(dt) * Z with Z ~ N(0, 1), independent per path and per step.
Pass the same Z to compare schemes or strategies on identical randomness
(common random numbers). Use np.random.default_rng(seed) when Z is None.

TODO (Person 3): implement the two functions below. Keep the signatures exactly as they are.
"""

import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, method="exact", Z=None, seed=None) -> np.ndarray:
    """Simulate GBM price paths dS = mu S dt + sigma S dW on [0, T] with n_steps equal steps.

    method:
      "exact"    : S_{k+1} = S_k exp((mu - sigma^2/2) dt + sigma dW_k)
      "euler"    : S_{k+1} = S_k (1 + mu dt + sigma dW_k)
      "milstein" : S_{k+1} = S_k (1 + mu dt + sigma dW_k + 0.5 sigma^2 (dW_k^2 - dt))
    Z: optional pre-drawn standard normals, shape (n_paths, n_steps).
    Returns an array of shape (n_paths, n_steps + 1); column 0 equals S0.
    """
    raise NotImplementedError("Person 3: implement simulate_gbm")


def simulate_wealth(x0, pi, mu, sigma, r, T, n_steps, n_paths, method="euler", Z=None, seed=None) -> np.ndarray:
    """Simulate wealth dX = X[r + pi(mu - r)] dt + X pi sigma dW for a constant fraction pi.

    method:
      "euler" : X_{k+1} = X_k (1 + [r + pi(mu - r)] dt + pi sigma dW_k)
      "exact" : X_{k+1} = X_k exp((r + pi(mu - r) - 0.5 pi^2 sigma^2) dt + pi sigma dW_k)
    Z: optional pre-drawn standard normals, shape (n_paths, n_steps).
    Returns an array of shape (n_paths, n_steps + 1); column 0 equals x0.
    """
    raise NotImplementedError("Person 3: implement simulate_wealth")
