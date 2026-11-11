"""Person 2 analysis for PLAN.md Section 6.

Run from the root with python -m experiments.p2_data_summary.
All outputs are regenerated from the committed, reviewed price/rate snapshot.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from src import config, data, estimation, theory


def main():
    """Produce the parameter table, five-year rolling plots and normality checks."""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    rates = data.load_risk_free()
    r = data.risk_free_constant()
    z = stats.norm.ppf((1 + 0.95) / 2)  # PLAN.md 6 requests 95% intervals.
    parameter_rows = []
    coverage_rows = []
    gap_rows = []
    calendar = data.load_prices(config.MAIN_TICKER).index
    completed_months = rates.loc[rates.index <= calendar[-1]].index.to_period("M")

    for ticker in config.TICKERS:
        prices = data.load_prices(ticker)
        returns = pd.Series(estimation.log_returns(prices), index=prices.index[1:])
        mu, sigma = estimation.estimate_params(returns)
        n = len(returns)
        L = n * config.DT
        se_mu = theory.se_mu_approx(sigma, L)
        row = {
            "ticker": ticker, "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()), "n": n, "years": L,
            "mu": mu, "mu_se": se_mu,
            "mu_ci_low": mu - z * se_mu, "mu_ci_high": mu + z * se_mu,
            "sigma": sigma, "sigma_se": sigma / np.sqrt(2 * n), "r": r,
        }
        for gamma in config.GAMMAS:
            row[f"pi_gamma_{gamma:g}"] = estimation.estimate_pi(returns, r, gamma)
        parameter_rows.append(row)

        # PLAN.md 6 fixes a five-calendar-year window moved by one month.
        # This experiment-specific window is not currently exposed in config.
        rolling_rows = []
        for end in returns.resample(pd.offsets.MonthEnd()).last().index:
            if end > returns.index[-1] and end.to_period("M") not in completed_months:
                # A dated monthly rate snapshot confirms a completed exchange month,
                # even when its last session precedes calendar month-end.
                continue
            start = end - pd.DateOffset(years=5)
            if start < prices.index[0]:
                continue
            window = returns.loc[(returns.index > start) & (returns.index <= end)]
            if len(window) < 2:
                continue
            current_r = rates.asof(end)
            if not np.isfinite(current_r):
                raise ValueError(f"No rate available for {end.date()}")
            rolling_mu, rolling_sigma = estimation.estimate_params(window)
            rolling_rows.append({
                "date": end, "last_price_date": window.index[-1], "r": current_r,
                "mu": rolling_mu, "sigma": rolling_sigma,
                "pi": estimation.estimate_pi(window, current_r, config.GAMMA),
            })
        rolling = pd.DataFrame(rolling_rows).set_index("date")
        rolling.to_csv(config.RESULTS / f"p2_rolling_{ticker}.csv")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, column, label in zip(
            axes, ("mu", "sigma", "pi"),
            ("Drift (annual decimal)", "Volatility (annual decimal)", "Stock fraction"),
        ):
            ax.plot(rolling.index, rolling[column])
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Month-end date")
        fig.suptitle(f"{ticker}: five-year rolling estimates, gamma={config.GAMMA:g}")
        fig.tight_layout()
        fig.savefig(config.FIGURES / f"p2_rolling_{ticker}.png", dpi=config.FIG_DPI)
        plt.close(fig)

        # Audit the gaps against the authoritative index session calendar.
        missing = calendar.difference(prices.index)
        month_ends = pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).last()
        for date in missing:
            coverage_rows.append({
                "ticker": ticker, "date": date, "weekday": date.day_name(),
                "is_month_last": date in month_ends.to_numpy(),
            })
        # Supplemental validation requested for the documented stock gaps.
        # Weighted least squares estimates log-drift with variance sigma^2 * dt_i.
        positions = calendar.get_indexer(prices.index)
        if np.any(positions < 0):
            raise ValueError(f"{ticker} has a close outside the reference calendar")
        intervals = np.diff(positions) * config.DT
        log_drift = returns.sum() / intervals.sum()
        gap_sigma = np.sqrt(np.sum((returns.to_numpy() - log_drift * intervals)**2 / intervals) / (n - 1))
        gap_mu = log_drift + gap_sigma**2 / 2
        gap_rows.append({
            "ticker": ticker, "missing_sessions": len(missing),
            "baseline_mu": mu, "gap_aware_mu": gap_mu,
            "baseline_sigma": sigma, "gap_aware_sigma": gap_sigma,
            "baseline_pi": theory.merton_pi(mu, sigma, r, config.GAMMA),
            "gap_aware_pi": theory.merton_pi(gap_mu, gap_sigma, r, config.GAMMA),
        })

        if ticker == config.MAIN_TICKER:
            jb = stats.jarque_bera(returns)
            diagnostics = {
                "ticker": ticker, "n": n,
                "skewness": stats.skew(returns, bias=False),
                "excess_kurtosis": stats.kurtosis(returns, fisher=True, bias=False),
                "jarque_bera_statistic": jb.statistic, "jarque_bera_pvalue": jb.pvalue,
            }
            pd.DataFrame([diagnostics]).to_csv(config.RESULTS / "p2_normality.csv", index=False)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(returns, bins="auto", density=True, alpha=0.6, label="Observed returns")
            x = np.linspace(returns.min(), returns.max(), len(returns))
            axes[0].plot(x, stats.norm.pdf(x, returns.mean(), returns.std(ddof=1)), label="Fitted normal")
            axes[0].set_title("NIFTY daily log returns")
            axes[0].set_xlabel("Daily log return (decimal)")
            axes[0].set_ylabel("Probability density")
            axes[0].legend()
            stats.probplot(returns, dist="norm", plot=axes[1])
            axes[1].set_title("Normal QQ plot")
            axes[1].set_xlabel("Standard normal quantile")
            axes[1].set_ylabel("Daily log return (decimal)")
            axes[1].legend(["Observed quantiles", "Fitted line"])
            fig.tight_layout()
            fig.savefig(config.FIGURES / "p2_nifty_normality.png", dpi=config.FIG_DPI)
            plt.close(fig)

    table = pd.DataFrame(parameter_rows)
    table.to_csv(config.RESULTS / "p2_params.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(config.RESULTS / "p2_missing_stock_sessions.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(config.RESULTS / "p2_gap_sensitivity.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
