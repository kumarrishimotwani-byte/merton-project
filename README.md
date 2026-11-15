# The Merton Problem and the Cost of Estimation Risk

Group project for **MAC F244: Stochastic Calculus and Applications in Finance**, BITS Pilani K K Birla Goa Campus, Semester I 2026–27.

The Merton model gives the fraction of wealth to invest in a risky asset:

$$
\pi^* = \frac{\mu-r}{\gamma\sigma^2}
$$

Here, $\mu$ is the asset's expected return, $\sigma$ its volatility, $r$ the risk-free rate and $\gamma$ the investor's risk aversion. The formula assumes these parameters are known. In practice, they must be estimated.

This project asks **how much estimation error costs, and whether an estimated Merton strategy performs better than simple allocation rules on Indian market data.**

## What we do

1. Derive the optimal allocation and the certainty-equivalent loss from choosing a different allocation.
2. Estimate drift and volatility for NIFTY 50, HDFC Bank, Infosys and Hindustan Unilever.
3. Check exact, Euler–Maruyama and Milstein simulations against the model's formulas.
4. Measure estimation risk using simulated histories, daily bootstrap samples and 21-day block bootstrap samples.
5. Backtest monthly allocations against cash, 60/40 stock/cash and all-stock portfolios. We also test Merton allocations clipped to the range $[0,1]$.

## Main findings

We measure estimation cost using a certainty-equivalent return: the sure return an investor values as much as the risky strategy.

- For NIFTY, the estimated annual drift is **9.62%**, with an approximate 95% interval of **0.43%–18.81%**. Estimated volatility is **20.19%**. Drift uncertainty makes the optimal allocation difficult to estimate reliably.
- With ten-year simulated histories and $\gamma=3$, the average certainty-equivalent loss is **1.68 percentage points per year**. The known optimum's advantage over cash is only **0.45 percentage points**. Under the calibrated GBM model, **61%** of estimated allocations score below cash.
- In the NIFTY backtest, using five years of past data and $\gamma=3$:

| Strategy | Realised CE proxy (% per year) | Final wealth from 1 |
|---|---:|---:|
| Cash | 6.21 | 2.34 |
| 60/40 stock/cash | 7.77 | 3.28 |
| All stock | 7.32 | 3.82 |
| Estimated Merton | 6.70 | 2.99 |
| Merton clipped to $[0,1]$ | 7.81 | 3.33 |

The backtest covers **February 2013–October 2026**. Clipped Merton and 60/40 have similar results in this sample; the small difference does not establish general superiority. Results also vary across assets.

The simulations use certainty equivalents based on expected CRRA utility. The backtest uses a realised mean-variance proxy calculated from monthly returns. These are different measures. A full-sample oracle is included in the outputs only as a hindsight reference because it uses future information.

## Data and assumptions

The committed price snapshot covers **1 January 2008–9 November 2026**. NIFTY uses official NSE index closes and excludes dividends. The three stocks use Yahoo Finance prices adjusted for dividends and splits. The risk-free proxy comes from 91-day Indian T-bill auction yields, converted to continuous annual rates.

Use the committed CSVs to reproduce the results. [Data sources, corrections and assumptions](data/raw/P2_README.md) document the snapshot, missing stock sessions and rate availability.

The analysis assumes 252 trading sessions per year. Each implementable strategy uses only information available at its decision date. The backtest excludes the incomplete November 2026 holding month and omits transaction costs and borrowing spreads. The simulation results depend on the constant-parameter GBM model.

## Run the project

Use Python 3.10 or later.

```bash
git clone https://github.com/kumarrishimotwani-byte/merton-project.git
cd merton-project
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install the dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Run these commands from the repository root, in order:

```bash
python -m experiments.p2_data_summary
python -m experiments.p3_simulator_checks
python -m experiments.p4_estimation_risk
python -m experiments.p5_backtest
```

The scripts save tables in `results/` and plots in `figures/`. Use the supplied data and random seeds to reproduce the results.

## Repository contents

| Folder | Contents |
|---|---|
| `src/` | Model formulas, data loaders, parameter estimation and simulators |
| `experiments/` | Data analysis, simulation checks, estimation-risk experiments and backtest |
| `data/raw/` | Committed prices, T-bill rates and source documentation |
| `results/` | Numerical tables and report contribution notes |
| `figures/` | Generated plots |
| `tests/` | Checks for the shared code and backtest |
| `docs/` | Original proposal and project plan |

## Team

| Name | Student ID | Contribution |
|---|---|---|
| Rishi Kumar Motwani | 2024ADPS0838G | Theory, coordination and report |
| Devam Jatin Bhavsar | 2024ADPS0831G | Data and parameter estimation |
| Arnav Gupta | 2024ADPS0843G | Simulation and verification |
| Yash Saxena | 2024ADPS0868G | Estimation-risk experiments |
| Raghav Singhal | 2024ADPS0827G | Real-data backtest |
