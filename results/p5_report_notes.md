# Person 5 — real-data backtest report handoff

## NIFTY main case: 5-year estimation window, gamma=3

The evaluation uses 165 completed monthly holding periods, from the 2013-01-31 rebalance through 2026-10-30. Every decision uses only the previous five calendar years of daily log returns and the 91-day T-bill rate known at that month's rebalance. Cash earns exp(r/12)-1 over the following month. November 2026 is a partial month and is excluded.

| Strategy | Annual mean | Annualized volatility | Annual realised CE | Final wealth from 1 | Max drawdown | Mean absolute stock weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cash | 6.21% | 0.47% annualized | 6.21% | 2.344 | 0.0% | 0.000 |
| 60/40 | 9.11% | 9.46% annualized | 7.77% | 3.278 | 17.6% | 0.600 |
| all_stock | 11.05% | 15.77% annualized | 7.32% | 3.820 | 29.3% | 1.000 |
| merton | 8.65% | 11.41% annualized | 6.70% | 2.993 | 18.7% | 0.765 |
| merton_clipped | 9.27% | 9.88% annualized | 7.81% | 3.333 | 14.8% | 0.665 |
| merton_oracle | 7.35% | 3.72% annualized | 7.14% | 2.713 | 6.3% | 0.236 |

Unconstrained estimated Merton achieved 6.699985% per year in realised mean-variance CE, compared with 7.771076% for fixed 60/40 stock/cash. Clipping weights to [0,1] raised CE to 7.806333%; the clipped advantage over 60/40 is only 0.03526 annual percentage points, not evidence of statistical superiority. The estimated NIFTY allocation exceeded [0,1] in 65 of 165 decisions (39.4%).

## Matched-endpoint sensitivity

Three stock datasets have earlier observed month-end dates than NIFTY: 2015-02-27, 2016-10-28 and 2019-03-28. To test the impact, the alternative NIFTY backtest uses NIFTY closes on those stock-observed dates, retains its entire daily price history for estimation, and looks up rates at the earlier decision. This **does not fill missing stock prices**. The complete comparisons are in `results/p5_endpoint_sensitivity.csv`.

At W=5 and gamma=3, the change in NIFTY CE is 0.01807 percentage points/year for unconstrained Merton, +0.01796 for clipped Merton, and +0.00220 for 60/40. These small changes do not alter the main comparison. Across W=3,5,10, the largest absolute CE change across strategies is 0.04309 percentage points/year.

## Interpretation and limitations

- The estimated allocation is noisy because the GBM drift is difficult to estimate from finite return histories. For 2016-02-29's five-year window, the start is 2011-02-28, correctly handling leap-day calendar offsets.
- The 60/40 rule is a 60% investment in the named asset plus 40% cash, rebalanced monthly; it is **not** a stock/bond portfolio.
- The oracle uses the entire price history and full-sample mean interest rate; it has look-ahead bias and is a reference, **not** an implementable strategy or an upper bound on ex-post CE.
- NIFTY 50 is a price index excluding dividends; stocks use adjusted closes. The assets therefore use different total-return conventions.
- The 2008 crash is within the training period, before any out-of-sample strategy can be plotted. The 2020 market crash is marked on the wealth chart.
- Mean and CE are annualized rates; volatility is an **annualized standard deviation** (not a return of 'percent per year'). Cash has an undefined excess-return Sharpe ratio.
- Realised CE is a mean-variance proxy `12*mean(R) - gamma/2*12*sample_variance(R)`, not exact observed CRRA utility. Sample variance uses ddof=1.
- We assume frictionless monthly rebalancing and borrowing/lending at the same T-bill proxy rate, with no trading costs or financing spreads. Leveraged portfolios can reach zero wealth; the code explicitly handles that case.
- The full CSV has 216 rows: four assets, three windows, three gamma values and six strategies. Window lengths have different start dates; comparison across those windows is not on a single common calendar span.

## Viva reminders

1. Why can we only use price and T-bill observations already known at the rebalance?
2. Why does clipping estimated Merton weights sometimes help, and why does it not guarantee a better outcome?
3. Why is the oracle not a fair prospective benchmark?
4. How does gamma enter the Merton weight and empirical mean-variance CE?
