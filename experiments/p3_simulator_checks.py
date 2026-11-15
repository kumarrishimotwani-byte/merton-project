"""Person 3: simulator accuracy, optimality and sensitivity (PLAN.md Section 7).

Run: python -m experiments.p3_simulator_checks

Produces:
  figures/p3_convergence.png          Strong and weak terminal errors vs dt
  results/p3_convergence.csv          Errors and Monte Carlo standard errors
  results/p3_convergence_orders.csv   Observed and expected strong orders
  figures/p3_ce_vs_pi.png             Test B: simulated CE against the Merton formula
  figures/p3_wealth_hist.png          Test B: terminal wealth at 0.5, 1 and 2 times pi*
  figures/p3_sensitivity.png          Test C: allocation sensitivity
  results/p3_ce_vs_pi.csv             Grid CE, MC errors and utility concentration
  results/p3_ce_summary.csv           Theoretical fractions and simulated grid maxima
  results/p3_wealth_summary.csv       Terminal wealth summaries
  results/p3_sensitivity.csv          Values behind the sensitivity curves
  results/p3_report_draft.md          Report contribution and viva notes for Person 1

Fine-grid correction: 1008/52 is not an integer. Use the least common multiple
of the prescribed coarse grids (3276 steps), as flagged in data/raw/P2_README.md.
Coarse Brownian increments are sums of fine increments, following Higham (2001),
https://doi.org/10.1137/S0036144500378302, Sections 4-6.
Test B uses the committed NIFTY calibration from results/p2_params.csv and Euler
wealth as specified in the plan. Nonpositive wealth fails explicitly before CRRA
utility is evaluated. Extreme leverage can make MC utility averages tail-dominated;
the sampled CE standard errors are approximate, not reliable tail confidence bounds.
"""

from math import lcm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, simulate, theory


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


def wealth_terminal(pi, mu, sigma, r, Z):
    """Simulate Test B's Euler wealth from eq. W, checking its positive-wealth domain.

    Every fraction receives the same Z. No paths are clipped or discarded: an
    invalid Euler path stops the comparison before Section 2.2's CRRA utility.
    """
    paths = simulate.simulate_wealth(
        config.P3_WEALTH_X0, pi, mu, sigma, r, config.P3_WEALTH_T,
        Z.shape[1], Z.shape[0], method="euler", Z=Z,
    )
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise ValueError(f"Invalid Euler wealth at pi={pi:g}; CRRA requires finite, positive paths.")
    return paths[:, -1].copy()


def utility_summary(terminal, gamma):
    """Convert mean CRRA utility into Test B's CE rate (PLAN.md Sections 2.2 and 2.4).

    The delta-method MC SE is se(Ubar)/abs((1-gamma)*T*Ubar), or se(Ubar)/T for
    log utility. For gamma>1, the largest absolute-utility share diagnoses whether
    one low-wealth path dominates the estimate; MC SEs can be unreliable there.
    """
    u = theory.utility(terminal, gamma)
    if not np.all(np.isfinite(u)):
        raise ValueError("Nonfinite CRRA utility; the Monte Carlo CE cannot be computed.")
    mean_u = u.mean()
    ce = theory.certainty_equivalent_from_utility(
        mean_u, config.P3_WEALTH_X0, config.P3_WEALTH_T, gamma
    )
    se_u = u.std(ddof=1) / np.sqrt(len(u))
    denominator = config.P3_WEALTH_T if gamma == 1 else abs((1 - gamma) * config.P3_WEALTH_T * mean_u)
    return {
        "mean_utility": mean_u,
        "ce_sim": ce,
        "ce_mc_se": se_u / denominator,
        "largest_utility_share": np.max(np.abs(u)) / np.sum(np.abs(u)),
    }


def optimality_checks(mu, sigma, r):
    """Run Test B with common random numbers over fractions and risk aversions.

    Compare simulated CE with eq. CE, and sample the three specified terminal
    wealth distributions. Extra evaluations at pi* measure agreement at the
    analytic optimum without pretending the coarse allocation grid locates it exactly.
    """
    n_steps = round(config.P3_WEALTH_T * config.P3_WEALTH_STEPS_PER_YEAR)
    Z = np.random.default_rng(config.SEED).standard_normal((config.P3_WEALTH_N_PATHS, n_steps))
    rows = []
    for pi in np.linspace(*config.P3_PI_GRID):
        terminal = wealth_terminal(pi, mu, sigma, r, Z)
        for gamma in config.GAMMAS:
            rows.append({
                "gamma": gamma, "pi": pi,
                **utility_summary(terminal, gamma),
                "ce_theory": theory.ce_rate(pi, mu, sigma, r, gamma),
            })
    curves = pd.DataFrame(rows)

    summary = []
    for gamma in config.GAMMAS:
        group = curves.loc[curves["gamma"] == gamma]
        best = group.loc[group["ce_sim"].idxmax()]
        pi_star = theory.merton_pi(mu, sigma, r, gamma)
        at_star = utility_summary(wealth_terminal(pi_star, mu, sigma, r, Z), gamma)
        summary.append({
            "gamma": gamma, "pi_star": pi_star,
            "pi_grid_best_sim": best["pi"],
            "pi_grid_best_theory": group.loc[group["ce_theory"].idxmax(), "pi"],
            "ce_grid_best_sim": best["ce_sim"],
            "ce_theory_at_star": theory.ce_rate(pi_star, mu, sigma, r, gamma),
            "ce_sim_at_star": at_star["ce_sim"],
            "ce_mc_se_at_star": at_star["ce_mc_se"],
        })

    pi_star = theory.merton_pi(mu, sigma, r, config.GAMMA)
    samples = {}
    wealth_rows = []
    for multiple in config.P3_WEALTH_MULTIPLIERS:
        pi = multiple * pi_star
        terminal = wealth_terminal(pi, mu, sigma, r, Z)
        samples[multiple] = terminal
        q05, median, q95 = np.quantile(terminal, [0.05, 0.5, 0.95])
        wealth_rows.append({
            "multiple": multiple, "pi": pi,
            "mean_wealth": terminal.mean(), "sd_wealth": terminal.std(ddof=1),
            "q05": q05, "median": median, "q95": q95,
            "ce_sim": utility_summary(terminal, config.GAMMA)["ce_sim"],
            "ce_theory": theory.ce_rate(pi, mu, sigma, r, config.GAMMA),
        })
    return curves, pd.DataFrame(summary), pd.DataFrame(wealth_rows), samples


def plot_optimality(curves, summary):
    """Plot Test B's Euler CE against eq. CE for every prescribed gamma.

    Error bars are one approximate sampled MC SE. They are not confidence bounds
    for the tail-dominated, highly leveraged end of the allocation range.
    """
    fig, axes = plt.subplots(1, len(config.GAMMAS), figsize=(13, 4.5), squeeze=False)
    for ax, gamma in zip(axes[0], config.GAMMAS):
        group = curves.loc[curves["gamma"] == gamma].sort_values("pi")
        row = summary.loc[summary["gamma"] == gamma].iloc[0]
        ax.plot(group["pi"], 100 * group["ce_theory"], color="0.2", label="Theory")
        ax.errorbar(
            group["pi"], 100 * group["ce_sim"], yerr=100 * group["ce_mc_se"],
            fmt="o", markersize=3, color="tab:blue", label="Euler +/- 1 MC SE",
        )
        ax.axvline(row["pi_star"], color="tab:green", linestyle="--", label=f"pi* = {row['pi_star']:.3f}")
        ax.set_title(f"Risk aversion gamma = {gamma:g}")
        ax.set_xlabel("Risky fraction pi (1 = all wealth)")
        ax.set_ylabel("Certainty-equivalent rate (%/year)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Known-parameter NIFTY model: {config.P3_WEALTH_T:g} years, "
        f"{config.P3_WEALTH_N_PATHS:,} shared paths, weekly Euler steps",
        fontsize=12,
    )
    fig.text(
        0.5, 0.025,
        "Near the optimum, simulation follows the theoretical peak. At high leverage, rare low-wealth paths can dominate utility.\n"
        "The sampled MC error bars can be unreliable in that tail-dominated region; they are not confidence intervals.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(config.FIGURES / "p3_ce_vs_pi.png", dpi=config.FIG_DPI)
    plt.close(fig)


def plot_wealth(samples, wealth_summary):
    """Plot Test B's terminal-wealth histograms at the three multiples of pi*."""
    fig, axes = plt.subplots(1, len(samples), figsize=(12, 4), sharex=True, sharey=True, squeeze=False)
    bins = np.histogram_bin_edges(np.concatenate(list(samples.values())), bins=40)
    for ax, (multiple, terminal) in zip(axes[0], samples.items()):
        row = wealth_summary.loc[wealth_summary["multiple"] == multiple].iloc[0]
        ax.hist(terminal, bins=bins, density=True, color="tab:blue", alpha=0.75)
        ax.axvline(row["median"], color="tab:orange", linestyle="--", label=f"Median {row['median']:.3f}")
        ax.set_title(f"{multiple:g} x pi*: risky fraction {row['pi']:.3f}")
        ax.set_xlabel(f"Terminal wealth (initial = {config.P3_WEALTH_X0:g})")
        ax.set_ylabel("Probability density (1/wealth unit)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Wealth after {config.P3_WEALTH_T:g} years: NIFTY calibration, gamma = {config.GAMMA:g}")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "p3_wealth_hist.png", dpi=config.FIG_DPI)
    plt.close(fig)


def sensitivity_values(mu, sigma, r):
    """Evaluate Test C's pi*=(mu-r)/(gamma*sigma^2), varying one input at a time."""
    tables = []
    for name, bounds in (
        ("gamma", config.P3_SENSITIVITY_GAMMA),
        ("sigma", config.P3_SENSITIVITY_SIGMA),
        ("excess_return", config.P3_SENSITIVITY_EXCESS_RETURN),
    ):
        values = np.linspace(*bounds, config.P3_SENSITIVITY_POINTS)
        gammas = values if name == "gamma" else np.full_like(values, config.GAMMA)
        sigmas = values if name == "sigma" else np.full_like(values, sigma)
        mus = r + values if name == "excess_return" else np.full_like(values, mu)
        tables.append(pd.DataFrame({
            "parameter": name, "value": values,
            "mu": mus, "sigma": sigmas, "r": r, "gamma": gammas,
            "pi_star": theory.merton_pi(mus, sigmas, r, gammas),
        }))
    return pd.concat(tables, ignore_index=True)


def plot_sensitivity(values, mu, sigma, r):
    """Plot Test C's inverse-gamma, inverse-square-volatility and linear-premium responses."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pi_star = theory.merton_pi(mu, sigma, r, config.GAMMA)
    panels = (
        ("gamma", 1, config.GAMMA, "Risk aversion gamma (dimensionless)", r"$\pi^* \propto 1/\gamma$"),
        ("sigma", 100, sigma, "Volatility sigma (%/sqrt(year))", r"$\pi^* \propto 1/\sigma^2$"),
        ("excess_return", 100, mu - r, "Expected excess return (%/year)", r"$\pi^* \propto \mu-r$"),
    )
    for ax, (name, scale, baseline, label, title) in zip(axes, panels):
        group = values.loc[values["parameter"] == name]
        ax.plot(scale * group["value"], 100 * group["pi_star"], color="tab:blue")
        ax.scatter(scale * baseline, 100 * pi_star, color="tab:orange", label="NIFTY baseline", zorder=3)
        ax.axhline(0, color="0.5", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(label)
        ax.set_ylabel("Optimal risky allocation (% of wealth)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Merton allocation sensitivity: vary one parameter, hold the others fixed")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "p3_sensitivity.png", dpi=config.FIG_DPI)
    plt.close(fig)


def write_report(orders, summary, curves, mu, sigma, r, n_steps_fine):
    """Write a short Section 7 report contribution and viva notes from the generated results."""
    fitted = orders.set_index("method")["observed_strong_order"]
    main = summary.loc[summary["gamma"] == config.GAMMA].iloc[0]
    maxima = "; ".join(
        f"gamma={row.gamma:g}: pi={row.pi_grid_best_sim:.1f}" for row in summary.itertuples()
    )
    tail = curves.sort_values("largest_utility_share").iloc[-1]
    text = f"""# Person 3 report contribution: simulation and verification

We implemented exact, Euler-Maruyama and Milstein price simulation and exact/Euler
constant-fraction wealth simulation. With {config.P3_ACCURACY_N_PATHS:,} shared Brownian paths,
the observed strong orders were {fitted['euler']:.3f} for Euler and {fitted['milstein']:.3f} for
Milstein, consistent with the expected 1/2 and 1. The fine grid uses {n_steps_fine:,} steps:
the plan's 1,008-step grid cannot be split into equal weekly blocks. Weak errors
were too affected by Monte Carlo sampling noise to establish a convergence order.

Using Person 2's NIFTY estimates (mu={mu:.6f}, sigma={sigma:.6f}, r={r:.6f}) as known
model parameters, we simulated {config.P3_WEALTH_N_PATHS:,} common wealth paths over
{config.P3_WEALTH_T:g} years with weekly Euler steps. For gamma={config.GAMMA:g},
pi*={main['pi_star']:.4f}, with theoretical CE {100 * main['ce_theory_at_star']:.4f}%/year
and simulated CE {100 * main['ce_sim_at_star']:.4f}%/year (approximate MC SE
{100 * main['ce_mc_se_at_star']:.4f} percentage points/year). Grid maxima were
{maxima}, consistent with the theoretical optima at the grid's resolution.
Larger risky fractions produced more dispersed terminal wealth. Sensitivity
curves show that pi* falls as 1/gamma and 1/sigma^2 and rises linearly with mu-r.

Agreement near the optimum does not extend reliably to extreme leverage: at
gamma={tail['gamma']:g}, pi={tail['pi']:g}, one path contributes
{100 * tail['largest_utility_share']:.2f}% of absolute sampled utility. The CE estimate
is then tail-dominated, and sampled MC error bars are unreliable. No nonpositive
Euler paths occurred in this run; the script rejects them before computing utility.
These are known-parameter GBM verification results, not evidence of out-of-sample
performance or of the accuracy of estimated investment fractions.

## Reproduce and integrate

- Run `python -m experiments.p3_simulator_checks` from the repository root.
- Run `python -m pytest -q` for the repository acceptance checks.
- Main report figures: `figures/p3_convergence.png` and `figures/p3_ce_vs_pi.png`.
- Supporting figures: `figures/p3_wealth_hist.png` and `figures/p3_sensitivity.png`.
- Raw results: the six `results/p3_*.csv` tables; all rates are annual decimals.
- Calibration: the committed `{config.MAIN_TICKER}` row of `results/p2_params.csv`.
  Follow `data/raw/P2_README.md` for provenance and model limitations.
- Person 4 can use the frozen simulator interfaces in `src/simulate.py`.
  The final report and shared plan remain for Person 1 to assemble/update.

## Viva reminders

- Brownian increments have variance dt, so dW=sqrt(dt)*Z for standard normal Z.
- Euler keeps the drift and Brownian terms. Milstein adds the second-order Ito
  correction sigma^2*(dW^2-dt)/2, improving strong order from 1/2 to 1.
- Common random numbers expose differences between strategies to the same shocks.
- CE is the guaranteed rate giving the same utility as the risky terminal wealth.
- The grid optimum is only resolved to the chosen grid spacing, not exactly pi*.

References: `docs/PLAN.md` Sections 2 and 7; Higham (2001), SIAM Review 43(3),
525-546, https://doi.org/10.1137/S0036144500378302.
"""
    (config.RESULTS / "p3_report_draft.md").write_text(text, encoding="utf-8")


def main():
    """Regenerate all Person 3 figures, tables and report text from PLAN.md Section 7."""
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

    parameters = pd.read_csv(config.RESULTS / "p2_params.csv").set_index("ticker")
    mu, sigma, r = parameters.loc[config.MAIN_TICKER, ["mu", "sigma", "r"]].to_numpy(dtype=float)
    if not np.all(np.isfinite([mu, sigma, r])) or sigma <= 0:
        raise ValueError("NIFTY calibration requires finite mu, sigma and r, with sigma positive.")
    curves, summary, wealth_summary, samples = optimality_checks(mu, sigma, r)
    curves.to_csv(config.RESULTS / "p3_ce_vs_pi.csv", index=False)
    summary.to_csv(config.RESULTS / "p3_ce_summary.csv", index=False)
    wealth_summary.to_csv(config.RESULTS / "p3_wealth_summary.csv", index=False)
    plot_optimality(curves, summary)
    plot_wealth(samples, wealth_summary)

    sensitivity = sensitivity_values(mu, sigma, r)
    sensitivity.to_csv(config.RESULTS / "p3_sensitivity.csv", index=False)
    plot_sensitivity(sensitivity, mu, sigma, r)
    write_report(orders, summary, curves, mu, sigma, r, n_steps_fine)
    print("\nTest B: known-parameter calibration, weekly Euler, common random numbers.")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("All sampled Euler wealth paths were finite and positive.")
    tail = curves.loc[curves["largest_utility_share"].idxmax()]
    print(
        f"Tail limitation: at gamma={tail['gamma']:g}, pi={tail['pi']:g}, the largest "
        f"absolute-utility share is {tail['largest_utility_share']:.2%}."
    )
    print("Test C: allocation sensitivity and report draft saved.")


if __name__ == "__main__":
    main()
