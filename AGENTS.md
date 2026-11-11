# Instructions for AI coding agents

This file is for any AI assistant (Claude Code, Codex, Cursor, Copilot, ChatGPT...) working in this repository.
Read it fully before writing code. `CLAUDE.md` points here.

## 1. What this project is

MAC F244 (Stochastic Calculus and Applications in Finance) group project, BITS Pilani Goa, Sem I 2026-27.
**Topic:** the Merton portfolio problem and the cost of estimation risk.

The optimal fraction of wealth in a stock is `pi* = (mu - r) / (gamma * sigma^2)`, but `mu` and `sigma`
must be estimated from data. We measure how unreliable the estimate `pi_hat` is and how much
certainty-equivalent return that costs, in simulation and on real Indian market data.

**The full specification is `docs/PLAN.md`. Read Sections 1-4 and the section for the person you are helping.**
`docs/project_plan.pdf` is the same plan as a PDF; `docs/proposal.pdf` is what was submitted to the professor.

**Before using Person 2's inputs, including work for Persons 3, 4 and 5, also read
[`data/raw/P2_README.md`](data/raw/P2_README.md).** Treat it as the data handoff alongside this file and
`docs/PLAN.md`. It documents the CSV schemas and usage examples, source corrections, snapshot dates,
rate units and availability, stock-calendar gaps, and limits of the verification. Apply its as-of timing
and gap-handling instructions when consuming the data. Review the documented departures from the plan
with the coordinator before changing those choices.

## 2. Who owns what

| Person | Role | Owns these files |
|---|---|---|
| 1 | Theory and coordination | `src/config.py`, `src/theory.py`, `tests/test_theory.py`, `report/` |
| 2 | Data and parameter estimation | `src/data.py`, `src/estimation.py`, `experiments/p2_data_summary.py`, `data/raw/` |
| 3 | Simulator and verification | `src/simulate.py`, `experiments/p3_simulator_checks.py` |
| 4 | Estimation-risk experiments | `experiments/p4_estimation_risk.py` |
| 5 | Real-data backtest | `experiments/p5_backtest.py` |

**Ask the user which person they are before starting.** Only edit files that person owns.
If a file owned by someone else needs a change (a bug, a missing function), do not edit it:
tell the user what is needed so they can ask the owner.

## 3. Frozen interfaces (do not change)

Other people's code calls these functions. Names, arguments, defaults, return types and array shapes
must be exactly as written in the stubs in `src/`. Implement the body; never rename or reorder.
If an interface genuinely must change, stop and tell the user to agree it with Person 1 first.

```python
# src/theory.py      (Person 1, already implemented)
merton_pi(mu, sigma, r, gamma) -> float
ce_rate(pi, mu, sigma, r, gamma) -> float
loss(pi_hat, pi_star, sigma, gamma) -> float
utility(x, gamma) -> np.ndarray

# src/data.py        (Person 2)
load_prices(ticker, start="2008-01-01", end=None) -> pd.Series
load_risk_free() -> pd.Series
risk_free_constant() -> float

# src/estimation.py  (Person 2)
log_returns(prices) -> np.ndarray
estimate_params(log_rets, dt=1/252) -> tuple[float, float]      # (mu_hat, sigma_hat)
estimate_pi(log_rets, r, gamma, dt=1/252) -> float

# src/simulate.py    (Person 3)
simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, method="exact", Z=None, seed=None) -> np.ndarray   # (n_paths, n_steps+1)
simulate_wealth(x0, pi, mu, sigma, r, T, n_steps, n_paths, method="euler", Z=None, seed=None) -> np.ndarray  # (n_paths, n_steps+1)
```

Until a dependency is implemented, experiment scripts may use the placeholder parameters in
`src/config.py` (`PLACEHOLDER_MU`, `PLACEHOLDER_SIGMA`, `R_PLACEHOLDER`), but must switch to the real
functions before results are final.

## 4. Conventions (all code)

- **Units:** time in years; every rate is annual, continuously compounded, as a decimal (6% = `0.06`, never `6`).
  One trading day is `dt = 1/252`.
- **Notation:** use the symbols of `docs/PLAN.md` Section 2 in variable names: `mu`, `sigma`, `r`, `gamma`,
  `pi`, `pi_star`, `pi_hat`, `x0`, `T`, `L` (years of data), `dt`.
- **Constants** live only in `src/config.py`. Import them; never hard-code tickers, dates, seeds, gamma or r.
- **Randomness:** use `np.random.default_rng(seed)` with `config.SEED` (42). Never the global `np.random` state.
- **Vectorise** over paths with numpy. No Python loops over paths (a loop over time steps is fine).
- **Comment** every function: what it computes, the formula, and the equation it comes from in `docs/PLAN.md`.
- **Imports:** scripts are run from the repo root as `python -m experiments.p4_estimation_risk`, and import
  as `from src import theory, config`.
- **Outputs:** figures as PNG at 150 dpi in `figures/`, tables as CSV in `results/`, file names starting with
  the person number (`p4_pihat_hist_L10.png`). Each experiment script regenerates all its outputs with one command.
- **Plots:** axis labels with units, a title, a legend when there is more than one series.
- **Look-ahead bias (Person 5):** at month m, use only data up to month m. Any future data invalidates the backtest.
- **Dependencies:** only those in `requirements.txt`. Ask before adding a package.

## 5. Checking your work

- Run `pytest` from the repo root. Tests for unimplemented stubs are skipped automatically; once a
  function is implemented its tests must pass.
- Every number reported must come from running the code, not from memory or estimation.
- Compare simulation results with the theoretical formulas in `src/theory.py` where the plan says to.

## 6. Important: the student must understand the work

There is a 15-20 minute viva on the mathematical formulation and findings, and all five members sit it
together. **Explain each step and formula to the user as you implement it**, keep the code readable rather
than clever, and do not add features or experiments beyond `docs/PLAN.md` without asking.

## 7. Git workflow

- Work on your own branch: `p2-data`, `p3-simulator`, `p4-estimation-risk`, `p5-backtest`.
- Commit small, clear steps. Do not commit to `main` directly; Person 1 merges branches at the end.
- Commit the downloaded CSVs in `data/raw/` (small files) so everyone uses identical data.
- Never commit secrets, API keys, `.venv/` or large generated files.
