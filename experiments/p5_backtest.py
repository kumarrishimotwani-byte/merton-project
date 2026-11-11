"""Person 5: out-of-sample backtest on real data. Spec: docs/PLAN.md Section 9.

Run: python -m experiments.p5_backtest

Produces:
  results/p5_backtest_summary.csv   metrics per asset, window W, gamma and strategy
  figures/p5_wealth_NSEI.png        wealth curves (log scale), 2008 and 2020 crashes marked
  figures/p5_pihat_path.png         pi_hat_m over time with 0 and 1 marked
  figures/p5_ce_bars.png            realised CE by strategy and asset

NO LOOK-AHEAD: at month m use only data up to month m (except the clearly labelled oracle).
"""

from src import config, data, estimation, theory  # noqa: F401

STRATEGIES = ["cash", "60/40", "all_stock", "merton", "merton_clipped", "merton_oracle"]
WINDOWS_YEARS = [3, 5, 10]


def main():
    # TODO (Person 5):
    # 1. Month-end prices and monthly returns; monthly risk-free r_f = exp(r_m / 12) - 1.
    # 2. Rolling loop: pi_hat_m from the past W years of DAILY log returns only; hold for one month.
    # 3. Metrics: mean, vol, Sharpe, max drawdown, final wealth, realised CE, mean |pi_hat|.
    raise NotImplementedError("Person 5: implement the backtest")


if __name__ == "__main__":
    main()
