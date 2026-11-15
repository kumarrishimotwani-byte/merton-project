"""Run the estimation-risk experiments in PLAN.md Section 8.

Run from the repository root: python -m experiments.p4_estimation_risk
Produces two CSVs in results/ and ten figures in figures/.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from src import config, data, estimation, simulate, theory

# Fixed Section 8 settings specific to Person 4. Shared settings stay in src.config.
HISTORY_YEARS = (5, 10, 20)
MAIN_YEARS = 10
N_HISTORIES = 1_000
BLOCK_DAYS = 21
WEALTH_YEARS = 10
N_WEALTH_PATHS = 20_000
PI_PERCENTILES = (5, 50, 95)
INITIAL_VALUE = 1.0  # Normalization for prices and wealth; neither affects rates.

METHOD_LABELS = {
    "gbm": "Exact GBM",
    "bootstrap_iid": "Daily bootstrap",
    "bootstrap_block": f"{BLOCK_DAYS}-day block bootstrap",
}


def load_calibration():
    """Load the Section 8 benchmark and check the bootstrap uses the same NIFTY sample."""
    params = pd.read_csv(config.RESULTS / "p2_params.csv").set_index("ticker")
    row = params.loc[config.MAIN_TICKER]
    prices = data.load_prices(config.MAIN_TICKER)
    returns = estimation.log_returns(prices)
    mu, sigma, r = float(row["mu"]), float(row["sigma"]), float(row["r"])
    actual_params = estimation.estimate_params(returns, dt=config.DT)
    if returns.size != row["n"] or not np.allclose(actual_params, (mu, sigma)):
        raise ValueError("NIFTY prices and p2_params.csv differ. Regenerate Person 2's table.")
    return mu, sigma, r, returns


def gbm_returns(mu, sigma, L, rng):
    """Generate exact GBM prices and daily log returns (PLAN.md Sections 2.1 and 8A)."""
    n_steps = config.TRADING_DAYS * L
    Z = rng.standard_normal((N_HISTORIES, n_steps))
    prices = simulate.simulate_gbm(
        INITIAL_VALUE, mu, sigma, L, n_steps, N_HISTORIES, method="exact", Z=Z
    )
    return np.diff(np.log(prices), axis=1)


def bootstrap_returns(returns, L, block_days, rng):
    """Resample overlapping contiguous blocks with replacement (PLAN.md Section 8B).

    A one-day block is the i.i.d. bootstrap. Multi-day blocks preserve order
    within each block and never wrap across the endpoints of the source series.
    """
    n_steps = config.TRADING_DAYS * L
    n_blocks = (n_steps + block_days - 1) // block_days
    starts = rng.integers(returns.size - block_days + 1, size=(N_HISTORIES, n_blocks))
    indices = starts[:, :, None] + np.arange(block_days)
    return returns[indices.reshape(N_HISTORIES, -1)[:, :n_steps]]


def estimate_histories(returns):
    """Vectorize the single-history estimator across rows (PLAN.md Section 2.5).

    Use the same sample variance and Ito correction as src.estimation:
    sigma_hat^2 = var(R, ddof=1) / dt; mu_hat = mean(R) / dt + sigma_hat^2 / 2.
    """
    sigma_squared = returns.var(axis=1, ddof=1) / config.DT
    mu_hat = returns.mean(axis=1) / config.DT + sigma_squared / 2
    return mu_hat, np.sqrt(sigma_squared)


def summarize(estimates, mu, sigma, r, gamma, L, method):
    """Score the estimated allocations with the CE loss from PLAN.md Section 2.4.

    For bootstrap samples the full-sample GBM fit defines a common benchmark.
    Its CE loss is a model-based score for the resampled allocations.
    """
    mu_hat, sigma_hat = estimates
    pi_hat = theory.merton_pi(mu_hat, sigma_hat, r, gamma)
    pi_star = theory.merton_pi(mu, sigma, r, gamma)
    loss = theory.loss(pi_hat, pi_star, sigma, gamma)
    benefit = theory.benefit(mu, sigma, r, gamma)
    row = {
        "method": method,
        "L_years": L,
        "gamma": gamma,
        "n_histories": pi_hat.size,
        "benchmark_mu": mu,
        "benchmark_sigma": sigma,
        "r": r,
        "benchmark_pi": pi_star,
        "mu_hat_mean": mu_hat.mean(),
        "mu_hat_sd": mu_hat.std(ddof=1),
        "sigma_hat_mean": sigma_hat.mean(),
        "sigma_hat_sd": sigma_hat.std(ddof=1),
        "pi_hat_mean": pi_hat.mean(),
        "pi_hat_sd": pi_hat.std(ddof=1),
        "gbm_approx_pi_hat_sd": theory.sd_pihat_approx(sigma, gamma, L),
        "short_pct": 100 * (pi_hat < 0).mean(),
        "borrow_pct": 100 * (pi_hat > 1).mean(),
        "worse_than_cash_pct": 100 * theory.worse_than_cash(pi_hat, pi_star).mean(),
        "mean_loss_pct_per_year": 100 * loss.mean(),
        "mean_loss_mc_se_pct_per_year": 100 * loss.std(ddof=1) / np.sqrt(loss.size),
        "gbm_approx_loss_pct_per_year": 100 * theory.expected_loss_approx(gamma, L),
        "optimal_benefit_pct_per_year": 100 * benefit,
        "mean_ce_minus_cash_pct_per_year": 100 * (benefit - loss.mean()),
    }
    return row, pi_hat


def wealth_utility(pi, mu, sigma, r, gamma, Z):
    """Compute utility at T using exact wealth (PLAN.md Sections 2.2, 2.4 and 8A).

    A single exact step gives the exact terminal distribution under constant pi.
    Intermediate wealth is unused in the certainty-equivalent comparison.
    """
    wealth = simulate.simulate_wealth(
        INITIAL_VALUE, pi, mu, sigma, r, WEALTH_YEARS,
        Z.shape[1], Z.shape[0], method="exact", Z=Z,
    )
    return theory.utility(wealth[:, -1], gamma)


def verify_loss(pi_hat, mu, sigma, r, gamma, rng):
    """Compare paired Monte Carlo CE gaps with Section 2.4's loss (Section 8A).

    For gamma != 1, the derivative of CE with respect to mean utility is
    1 / ((1 - gamma) * T * mean_utility). Use it on paired utilities
    to estimate the gap's MC standard error, including covariance.
    """
    pi_star = theory.merton_pi(mu, sigma, r, gamma)
    Z = rng.standard_normal((N_WEALTH_PATHS, 1))
    optimal_utility = wealth_utility(pi_star, mu, sigma, r, gamma, Z)
    optimal_mean = optimal_utility.mean()
    ce_star = theory.certainty_equivalent_from_utility(
        optimal_mean, INITIAL_VALUE, WEALTH_YEARS, gamma
    )
    optimal_influence = (optimal_utility / optimal_mean - 1) / ((1 - gamma) * WEALTH_YEARS)

    rows = []
    for percentile, pi in zip(PI_PERCENTILES, np.percentile(pi_hat, PI_PERCENTILES)):
        utility = wealth_utility(pi, mu, sigma, r, gamma, Z)
        mean_utility = utility.mean()
        ce = theory.certainty_equivalent_from_utility(
            mean_utility, INITIAL_VALUE, WEALTH_YEARS, gamma
        )
        influence = (utility / mean_utility - 1) / ((1 - gamma) * WEALTH_YEARS)
        gap_mc_se = (optimal_influence - influence).std(ddof=1) / np.sqrt(N_WEALTH_PATHS)
        loss = theory.loss(pi, pi_star, sigma, gamma)
        rows.append({
            "percentile": percentile,
            "pi_hat": pi,
            "benchmark_pi": pi_star,
            "gamma": gamma,
            "T_years": WEALTH_YEARS,
            "n_paths": N_WEALTH_PATHS,
            "optimal_ce_sim_pct_per_year": 100 * ce_star,
            "estimated_ce_sim_pct_per_year": 100 * ce,
            "sim_loss_pct_per_year": 100 * (ce_star - ce),
            "formula_loss_pct_per_year": 100 * loss,
            "sim_minus_formula_pct_per_year": 100 * (ce_star - ce - loss),
            "paired_mc_se_pct_per_year": 100 * gap_mc_se,
        })
    return pd.DataFrame(rows)


def plot_histograms(distributions, pi_star):
    """Plot Section 8A/B allocation histograms on common bins for each L."""
    for L in HISTORY_YEARS:
        pooled = np.concatenate([distributions[method, L] for method in METHOD_LABELS])
        bins = np.histogram_bin_edges(pooled, bins="auto")
        for method, label in METHOD_LABELS.items():
            fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
            ax.hist(distributions[method, L], bins=bins, color="#28778c", edgecolor="white")
            ax.axvline(pi_star, color="#b23b2a", linewidth=2, label=f"Benchmark {pi_star:.1%}")
            ax.set(
                title=f"{label} · {L}-year histories",
                xlabel="Estimated stock allocation (% of wealth)",
                ylabel="Number of histories",
            )
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.legend(frameon=False)
            suffix = "" if method == "gbm" else f"_{method}"
            fig.savefig(config.FIGURES / f"p4_pihat_hist_L{L}{suffix}.png", dpi=config.FIG_DPI)
            plt.close(fig)


def plot_loss(summary):
    """Plot simulated GBM losses, 1/(2 gamma L), and the benefit (PLAN.md Section 8A)."""
    main = summary.loc[(summary["method"] == "gbm") & (summary["gamma"] == config.GAMMA)]
    years = main["L_years"].to_numpy()
    benefit = main["optimal_benefit_pct_per_year"].iloc[0]
    grid = np.linspace(years.min(), years.max())

    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    observed = ax.errorbar(
        years, main["mean_loss_pct_per_year"], yerr=main["mean_loss_mc_se_pct_per_year"],
        fmt="o", capsize=3, color="#28778c", label="Mean GBM loss ± 1 MC SE",
    )
    approximation, = ax.plot(
        grid, 100 * theory.expected_loss_approx(config.GAMMA, grid),
        color="#28778c", linestyle="--", label=r"Approximation $1/(2\gamma L)$",
    )
    optimal = ax.axhline(benefit, color="#b23b2a", linewidth=2, label="Benefit at optimum vs. cash")
    ax.set(
        title=f"Cost of estimating the Merton allocation (γ = {config.GAMMA:g})",
        xlabel="History length (years)",
        ylabel="Annual certainty equivalent (%)",
        xticks=years,
        ylim=(0, None),
    )
    ax.legend(handles=[observed, approximation, optimal], frameon=False)
    fig.savefig(config.FIGURES / "p4_loss_vs_L.png", dpi=config.FIG_DPI)
    plt.close(fig)


def main():
    """Run PLAN.md Section 8A–C and regenerate the two tables and ten figures."""
    mu, sigma, r, returns = load_calibration()
    pi_star = theory.merton_pi(mu, sigma, r, config.GAMMA)
    history_seed, wealth_seed = np.random.SeedSequence(config.SEED).spawn(2)
    rng = np.random.default_rng(history_seed)
    rows = []
    distributions = {}

    for L in HISTORY_YEARS:
        estimates = estimate_histories(gbm_returns(mu, sigma, L, rng))
        row, distributions["gbm", L] = summarize(estimates, mu, sigma, r, config.GAMMA, L, "gbm")
        rows.append(row)
        if L == MAIN_YEARS:
            # Keep the same histories so only gamma changes in Experiment C.
            for gamma in config.GAMMAS:
                if gamma != config.GAMMA:
                    row, _ = summarize(estimates, mu, sigma, r, gamma, L, "gbm")
                    rows.append(row)

    for method, block_days in (("bootstrap_iid", 1), ("bootstrap_block", BLOCK_DAYS)):
        for L in HISTORY_YEARS:
            estimates = estimate_histories(bootstrap_returns(returns, L, block_days, rng))
            row, distributions[method, L] = summarize(estimates, mu, sigma, r, config.GAMMA, L, method)
            rows.append(row)

    summary = pd.DataFrame(rows)
    check = verify_loss(
        distributions["gbm", MAIN_YEARS], mu, sigma, r, config.GAMMA,
        np.random.default_rng(wealth_seed),
    )
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.RESULTS / "p4_estimation_risk.csv", index=False)
    check.to_csv(config.RESULTS / "p4_loss_verification.csv", index=False)
    plot_histograms(distributions, pi_star)
    plot_loss(summary)


if __name__ == "__main__":
    main()
