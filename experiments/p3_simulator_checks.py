"""Person 3: simulator accuracy (PLAN.md Section 7, Test A; Phase 2).

Run: python -m experiments.p3_simulator_checks

Produces:
  figures/p3_convergence.png          Strong and weak terminal errors vs dt
  results/p3_convergence.csv          Errors and Monte Carlo standard errors
  results/p3_convergence_orders.csv   Observed and expected strong orders

Fine-grid correction: 1008/52 is not an integer. Use the least common multiple
of the prescribed coarse grids (3276 steps), as flagged in data/raw/P2_README.md.
Coarse Brownian increments are sums of fine increments, following Higham (2001),
https://doi.org/10.1137/S0036144500378302, Sections 4-6.
Tests B and C (optimality and sensitivity) belong to Phase 3.
"""

from math import lcm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, simulate


def accuracy_errors():
    """Measure Test A errors on shared Brownian paths (PLAN.md Section 7).

    Strong error is mean(abs(S_T_scheme - S_T_exact)); weak error is
    abs(mean(S_T_scheme) - S0*exp(mu*T)). The exact reference uses Section 2.1's
    closed-form terminal price, driven by the sum of the same fine increments.
    Return the error table, strong-order fits, fine-grid size and exact-mean MC SE.
    """
    S0 = config.P3_ACCURACY_S0
    mu = config.P3_ACCURACY_MU
    sigma = config.P3_ACCURACY_SIGMA
    T = config.P3_ACCURACY_T
    n_paths = config.P3_ACCURACY_N_PATHS
    n_steps_fine = lcm(*config.P3_ACCURACY_STEPS)
    step_counts = sorted(set((*config.P3_ACCURACY_STEPS, n_steps_fine)))

    rng = np.random.default_rng(config.SEED)
    Z_fine = rng.standard_normal((n_paths, n_steps_fine))
    W_T = np.sqrt(T / n_steps_fine) * Z_fine.sum(axis=1)
    exact_terminal = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T)
    expected_mean = S0 * np.exp(mu * T)
    exact_mc_se = exact_terminal.std(ddof=1) / np.sqrt(n_paths)

    rows = []
    for n_steps in step_counts:
        dt = T / n_steps
        block = n_steps_fine // n_steps
        # sqrt(dt)*Z_coarse = sum(sqrt(dt_fine)*Z_fine) within each block.
        if block == 1:
            Z = Z_fine
        else:
            Z = Z_fine.reshape(n_paths, n_steps, block).sum(axis=2) / np.sqrt(block)

        for method in ("euler", "milstein"):
            # Copy the terminal column so the full path array can be released.
            terminal = simulate.simulate_gbm(
                S0, mu, sigma, T, n_steps, n_paths, method=method, Z=Z
            )[:, -1].copy()
            absolute_error = np.abs(terminal - exact_terminal)
            rows.append({
                "method": method,
                "n_steps": n_steps,
                "dt": dt,
                "mean_terminal": terminal.mean(),
                "strong_error": absolute_error.mean(),
                "strong_error_se": absolute_error.std(ddof=1) / np.sqrt(n_paths),
                "weak_error": abs(terminal.mean() - expected_mean),
                "weak_mc_se": terminal.std(ddof=1) / np.sqrt(n_paths),
            })

    errors = pd.DataFrame(rows)
    orders = []
    for method, expected_order in (("euler", 0.5), ("milstein", 1.0)):
        group = errors.loc[errors["method"] == method]
        # Error ~ C*dt**p, so the slope in log(error) vs log(dt) estimates p.
        order = np.polyfit(np.log(group["dt"]), np.log(group["strong_error"]), 1)[0]
        orders.append({
            "method": method,
            "observed_strong_order": order,
            "expected_strong_order": expected_order,
        })
    return errors, pd.DataFrame(orders), n_steps_fine, exact_mc_se


def plot_accuracy(errors, orders, n_steps_fine, exact_mc_se):
    """Plot Test A's log-log convergence curves and MC noise scale (PLAN.md Section 7).

    Dashed strong-error references scale as dt**(1/2) and dt**1. The weak panel
    shows sampling error as well as discretisation bias; no weak order is fitted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, label, color in (("euler", "Euler", "tab:blue"), ("milstein", "Milstein", "tab:orange")):
        group = errors.loc[errors["method"] == method].sort_values("dt")
        row = orders.loc[orders["method"] == method].iloc[0]
        axes[0].loglog(
            group["dt"], group["strong_error"], "o-", color=color,
            label=f"{label}: fitted order {row['observed_strong_order']:.3f}",
        )
        reference = group.iloc[0]["strong_error"] * (
            group["dt"] / group.iloc[0]["dt"]
        ) ** row["expected_strong_order"]
        axes[0].loglog(
            group["dt"], reference, "--", color=color, alpha=0.6,
            label=f"Reference order {row['expected_strong_order']:g}",
        )
        axes[1].loglog(group["dt"], group["weak_error"], "o-", color=color, label=label)

    axes[0].set_title("Strong error: path-by-path accuracy")
    axes[0].set_ylabel("Mean absolute terminal error (price units)")
    axes[1].set_title("Weak error: accuracy of the sample mean")
    axes[1].set_ylabel("Absolute mean-price error (price units)")
    axes[1].axhline(
        exact_mc_se, color="0.35", linestyle=":",
        label=f"Exact-mean MC standard error: {exact_mc_se:.3f}",
    )
    for ax in axes:
        ax.set_xlabel(r"Time step $\Delta t$ (years)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Simulator accuracy on shared Brownian paths\n"
        f"S0 = {config.P3_ACCURACY_S0:g}, mu = {config.P3_ACCURACY_MU:.0%}/year, "
        f"sigma = {config.P3_ACCURACY_SIGMA:.0%}/sqrt(year), "
        f"T = {config.P3_ACCURACY_T:g} year; "
        f"{config.P3_ACCURACY_N_PATHS:,} paths, seed {config.SEED}",
        fontsize=11,
    )
    fig.text(
        0.5, 0.025,
        f"Fine grid: {n_steps_fine:,} steps; coarser increments sum blocks of the same fine shocks.\n"
        "Weak errors include Monte Carlo noise; a weak convergence order cannot be inferred from this run.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(config.FIGURES / "p3_convergence.png", dpi=config.FIG_DPI)
    plt.close(fig)


def main():
    """Regenerate Phase 2's Test A figure and tables from PLAN.md Section 7."""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    errors, orders, n_steps_fine, exact_mc_se = accuracy_errors()
    errors.to_csv(config.RESULTS / "p3_convergence.csv", index=False)
    orders.to_csv(config.RESULTS / "p3_convergence_orders.csv", index=False)
    plot_accuracy(errors, orders, n_steps_fine, exact_mc_se)
    print(f"Test A: {config.P3_ACCURACY_N_PATHS:,} paths; {n_steps_fine:,}-step fine grid.")
    print(errors.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\nStrong convergence orders:")
    print(orders.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nExact-mean Monte Carlo standard error: {exact_mc_se:.6g} price units.")
    print("Weak errors include Monte Carlo noise; do not infer a weak order from this run.")


if __name__ == "__main__":
    main()
