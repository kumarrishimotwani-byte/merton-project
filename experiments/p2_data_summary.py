"""Person 2: data summary and parameter estimates. Spec: docs/PLAN.md Section 6.

Run: python -m experiments.p2_data_summary

Produces:
  results/p2_params.csv               Table 1: L, n, mu_hat (se, 95% CI), sigma_hat (se), pi_hat for gamma = 2, 3, 5
  figures/p2_rolling_<ticker>.png     5-year rolling mu_hat, sigma_hat, pi_hat (moved monthly)
  figures/p2_nifty_normality.png      histogram + fitted normal, QQ plot; skew, kurtosis, Jarque-Bera
"""

from src import config, data, estimation, theory  # noqa: F401


def main():
    # TODO (Person 2):
    # 1. For each ticker in config.TICKERS: prices = data.load_prices(ticker); R = estimation.log_returns(prices)
    # 2. Table 1 using estimation.estimate_params and theory.merton_pi with r = data.risk_free_constant()
    # 3. Rolling 5-year estimates
    # 4. Normality checks for config.MAIN_TICKER
    raise NotImplementedError("Person 2: implement the data summary")


if __name__ == "__main__":
    main()
