"""Simulating price and wealth paths (owner: Person 3). See docs/PLAN.md Section 7.

Brownian increments: dW = sqrt(dt) * Z with Z ~ N(0, 1), independent per path and per step.
Pass the same Z to compare schemes or strategies on identical randomness
(common random numbers). Use np.random.default_rng(seed) when Z is None.

Exact paths stay positive. Euler and Milstein values are not clipped: coarse steps
can produce nonpositive values, which must be checked before computing CRRA utility.
"""

import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, method="exact", Z=None, seed=None) -> np.ndarray:
    """Simulate GBM price paths dS = mu S dt + sigma S dW (PLAN.md Sections 2.1 and 7).

    method:
      "exact"    : S_{k+1} = S_k exp((mu - sigma^2/2) dt + sigma dW_k)
      "euler"    : S_{k+1} = S_k (1 + mu dt + sigma dW_k)
      "milstein" : S_{k+1} = S_k (1 + mu dt + sigma dW_k + 0.5 sigma^2 (dW_k^2 - dt))
    Z: optional pre-drawn standard normals, shape (n_paths, n_steps).
    If Z is provided, seed is ignored and Z is never modified. Otherwise draw with
    np.random.default_rng(seed); experiments should pass config.SEED.
    S0 must be positive, sigma and T nonnegative, and step/path counts positive integers.
    Returns an array of shape (n_paths, n_steps + 1); column 0 equals S0.
    """
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    return _simulate_paths(S0, mu, sigma, T, n_steps, n_paths, method, Z, seed)


def simulate_wealth(x0, pi, mu, sigma, r, T, n_steps, n_paths, method="euler", Z=None, seed=None) -> np.ndarray:
    """Simulate constant-fraction wealth (PLAN.md Section 2.1, eq. W, and Section 7).

    method:
      "euler" : X_{k+1} = X_k (1 + [r + pi(mu - r)] dt + pi sigma dW_k)
      "exact" : X_{k+1} = X_k exp((r + pi(mu - r) - 0.5 pi^2 sigma^2) dt + pi sigma dW_k)
    Z: optional pre-drawn standard normals, shape (n_paths, n_steps).
    Its handling and input restrictions are the same as in simulate_gbm.
    pi may be negative (shorting) or greater than one (borrowing); it is not clipped.
    Returns an array of shape (n_paths, n_steps + 1); column 0 equals x0.
    """
    if method not in ("euler", "exact"):
        raise ValueError("wealth method must be 'euler' or 'exact'")
    if not np.all(np.isfinite([pi, mu, sigma, r])) or sigma < 0:
        raise ValueError("pi, mu, sigma and r must be finite, with sigma nonnegative")

    # Wealth has the same multiplicative SDE as price. Retain the sign of pi*sigma
    # so short positions respond in the opposite direction to the same Brownian shock.
    drift = r + pi * (mu - r)
    diffusion = pi * sigma
    return _simulate_paths(x0, drift, diffusion, T, n_steps, n_paths, method, Z, seed)


def _simulate_paths(x0, drift, diffusion, T, n_steps, n_paths, method, Z, seed):
    """Step dY = drift*Y*dt + diffusion*Y*dW using PLAN.md Section 7's recurrences.

    Exact growth uses exp((drift - diffusion^2/2)*dt + diffusion*dW).
    Euler uses 1 + drift*dt + diffusion*dW; Milstein adds
    diffusion^2*(dW^2 - dt)/2. Vectorise paths, with a loop only over time.
    """
    if method not in ("exact", "euler", "milstein"):
        raise ValueError("price method must be 'exact', 'euler' or 'milstein'")
    if not np.all(np.isfinite([x0, drift, diffusion, T])) or x0 <= 0 or T < 0:
        raise ValueError("initial value must be positive, T nonnegative and all inputs finite")
    for name, count in (("n_steps", n_steps), ("n_paths", n_paths)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if Z is None:
        Z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    else:
        Z = np.asarray(Z, dtype=float)
        if Z.shape != (n_paths, n_steps) or not np.all(np.isfinite(Z)):
            raise ValueError("Z must be finite with shape (n_paths, n_steps)")

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = x0
    for k in range(n_steps):
        dW = sqrt_dt * Z[:, k]
        if method == "exact":
            growth = np.exp((drift - 0.5 * diffusion**2) * dt + diffusion * dW)
        elif method == "euler":
            growth = 1 + drift * dt + diffusion * dW
        else:
            growth = 1 + drift * dt + diffusion * dW + 0.5 * diffusion**2 * (dW**2 - dt)
        paths[:, k + 1] = paths[:, k] * growth
    return paths
