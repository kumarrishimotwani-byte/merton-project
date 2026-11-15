# Estimation risk

We calibrated a GBM market to the full-sample NIFTY 50 estimates (annual drift
9.62%, volatility 20.19%, and risk-free rate 6.30%). At $\gamma=3$, the known
optimal stock allocation was 27.12% and its annual certainty-equivalent (CE)
benefit over cash was 0.45 percentage points (pp). We simulated 1,000 daily
histories per sample length, re-estimated the allocation, and evaluated its
CE loss under the calibrated market.

| History | Allocation SD, simulated / theory (pp of wealth) | Mean CE loss, simulated / theory (pp/year) | Shorting | Borrowing | CE below cash |
|---:|---:|---:|---:|---:|---:|
| 5 years | 71.9 / 73.8 | 3.16 / 3.33 | 33.8% | 16.0% | 69.3% |
| 10 years | 52.2 / 52.2 | 1.68 / 1.67 | 32.9% | 6.8% | 61.0% |
| 20 years | 34.8 / 36.9 | 0.74 / 0.83 | 20.9% | 2.0% | 42.0% |

The mean loss fell with history length and exceeded the 0.45 pp benefit
at every length. Even with 20 years of data, the estimated strategy averaged
0.29 pp/year below cash in CE. The benchmarks hold volatility fixed when
approximating estimation error. The observed loss at 20 years was about 11%
below the approximation in this run.

At 10 years, the i.i.d. and 21-day block bootstraps of observed NIFTY returns
gave mean losses of 1.67 and 1.95 pp/year under the same GBM scoring rule.
Resampling changes the source of estimation error, while the CE score remains
model-based. Using the same 10-year GBM histories at $\gamma=2,3,5$ gave losses
of 2.51, 1.68, and 1.01 pp/year, consistent with inverse scaling. Across the
5th, 50th, and 95th allocation percentiles, paired wealth simulations with
20,000 paths reproduced the formula to within 0.02 pp/year.

**Report figures.** [Loss versus history length](../figures/p4_loss_vs_L.png)
and [10-year allocation histogram](../figures/p4_pihat_hist_L10.png).
The full [results table](p4_estimation_risk.csv) includes all bootstrap lengths
and risk-aversion settings. The [wealth check](p4_loss_verification.csv)
reports simulated gaps and paired Monte Carlo standard errors.

## Viva notes

- **What does knowing the truth mean?** The full-sample estimates are fixed
  inputs to the simulated GBM market. They determine the market that produces
  the histories and the optimum used to grade the estimates. In the bootstrap,
  the observed full-sample fit supplies the reference allocation and scoring rule.
- **Why are the allocations so spread out?** Under GBM the drift estimator's
  standard error is approximately $\sigma/\sqrt L$. Its mean log-return term
  is $\log(S_L/S_0)/L$, since the daily log returns telescope. Sampling more
  frequently over the same span helps estimate volatility but barely changes
  drift precision. Dividing drift error by $\gamma\sigma^2$ gives
  an allocation SD of about $1/(\gamma\sigma\sqrt L)$.
- **Why does the expected loss simplify?** Completing the square in CE gives
  loss $\tfrac12\gamma\sigma^2(\hat\pi-\pi^*)^2$. Holding volatility fixed
  and using the approximate variance of $\hat\pi$ yields $1/(2\gamma L)$.
  Drift drops out of the allocation error, and the volatility factors cancel.
- **What did the paired wealth check do?** It applied each allocation to the
  same Brownian shocks over a ten-year investment horizon, then converted
  average terminal utility back into an annual CE. One exact step gives
  the exact terminal wealth distribution for a constant allocation. This
  investment horizon is separate from the length of the estimation history.
- **Why use two bootstraps?** Sampling single days preserves the empirical
  daily return distribution. Sampling 21 consecutive trading days at a time
  also retains short-run dependence within each block.
- **How do we derive the Merton fraction?** The self-financing wealth SDE
  gives the HJB control terms
  $[r+\pi(\mu-r)]xV_x+\tfrac12\pi^2\sigma^2x^2V_{xx}$.
  Differentiating with respect to $\pi$ gives
  $\pi^*=-(\mu-r)V_x/(\sigma^2xV_{xx})$. For the CRRA ansatz,
  $xV_{xx}/V_x=-\gamma$, which yields the constant optimum.

## Reproduce

From the repository root, run `python -m experiments.p4_estimation_risk`.
The run uses Person 2's committed NIFTY snapshot through 9 November 2026 and
requires Person 3's implemented `src/simulate.py` from `p3-simulator`.
See `docs/PLAN.md` Sections 2 and 8 and `data/raw/P2_README.md`.
