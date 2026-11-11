# Project plan: The Merton Problem and the Cost of Estimation Risk

MAC F244 Stochastic Calculus and Applications in Finance, Semester I, 2026-27.
Final submission (code + 3-4 page report): **15 Nov 2026**. Viva: 15-20 minutes, all members together.

This is the working specification for the code. `docs/project_plan.pdf` is the same plan as a PDF.

| # | Role |
|---|---|
| 1 | Theory and coordination |
| 2 | Data and parameter estimation |
| 3 | Simulator and verification |
| 4 | Estimation-risk experiments |
| 5 | Real-data backtest |

Everyone reads Sections 1-4, then their own section.

---

## 1. The project in one page

**Problem.** An investor splits wealth between a risky stock and a risk-free bank account and can rebalance
continuously. What fraction of wealth should be in the stock to maximise expected utility of final wealth?
This is the Merton problem. Its answer is $\pi^* = (\mu - r)/(\gamma\sigma^2)$, but the formula needs the
*true* drift $\mu$ and volatility $\sigma$, which nobody knows.

**Research question.** *Does Merton's optimal strategy still work when its parameters must be estimated
from real market data, and how much does the estimation error cost?*

**The flow.**
1. **Theory** (Person 1): model, HJB equation, $\pi^*$, and a formula for the cost of using a wrong fraction.
2. **Data** (Person 2): estimate $\mu, \sigma, r$ from NIFTY 50 and three NSE stocks. Show $\sigma$ is estimated precisely but $\mu$ is not.
3. **Simulator** (Person 3): Euler-Maruyama simulator; check accuracy; confirm numerically that $\pi^*$ gives the highest expected utility when $\mu, \sigma$ are known.
4. **Estimation risk** (Person 4): in a simulated market where we *choose* the true $\mu, \sigma$ (so the true $\pi^*$ is known), generate 1,000 fake histories, re-estimate $\hat\pi$ from each, and measure how wrong it is and what that costs.
5. **Backtest** (Person 5): on real prices, each month estimate $\hat\pi$ from past data only, invest with it for a month, roll forward, and compare with simple rules ($\pi = 0, 0.6, 1$).

**Expected outcome.** Inside the model $\pi^*$ is optimal. In practice $\mu$ is so noisy that $\hat\pi$ can range
from negative to well above 1, and the average loss may be comparable to the whole benefit of investing
optimally. We quantify this.

---

## 2. The mathematical model (shared notation)

Time in **years**. All rates **annual, continuously compounded, decimals** (6% = 0.06).

| Symbol | Meaning |
|---|---|
| $S_t$ | stock (or index) price |
| $B_t$ | bank account, grows at rate $r$ |
| $\mu, \sigma$ | drift and volatility of the stock |
| $W_t$ | standard Brownian motion |
| $\pi_t$ | fraction of wealth in the stock ($\pi > 1$ borrowing, $\pi < 0$ short selling) |
| $X_t$ | wealth |
| $\gamma > 0$ | relative risk aversion; main value $\gamma = 3$ |
| $T$ | horizon; $L$ = years of data |
| $\Delta t$ | time step; one trading day is $1/252$ |

### 2.1 Market and wealth
$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t, \qquad dB_t = r B_t\,dt.$$
By Ito's lemma $d\ln S_t = (\mu - \tfrac12\sigma^2)dt + \sigma dW_t$, so $S_t = S_0 \exp\big((\mu - \tfrac12\sigma^2)t + \sigma W_t\big)$.

Self-financing ($dX = \phi\,dS + \psi\,dB$) with $\pi = \phi S / X$ gives the **wealth SDE**
$$dX_t = X_t\big[r + \pi_t(\mu - r)\big]dt + X_t \pi_t \sigma\, dW_t. \tag{W}$$

### 2.2 Objective
CRRA utility $U(x) = x^{1-\gamma}/(1-\gamma)$ ($\gamma \ne 1$; $\ln x$ for $\gamma = 1$).
$$V(t,x) = \sup_\pi \mathbb{E}\big[U(X_T) \mid X_t = x\big].$$

### 2.3 HJB equation and solution
$$0 = V_t + \sup_\pi \Big\{ [r + \pi(\mu - r)]\,x V_x + \tfrac12 \pi^2\sigma^2 x^2 V_{xx} \Big\}, \qquad V(T,x) = U(x).$$
First-order condition: $\pi^* = -(\mu - r)V_x / (\sigma^2 x V_{xx})$. Ansatz $V = f(t)\,x^{1-\gamma}/(1-\gamma)$ gives $x V_{xx}/V_x = -\gamma$, so
$$\boxed{\pi^* = \frac{\mu - r}{\gamma\sigma^2}}, \qquad f(t) = \exp\Big((1-\gamma)\big(r + \tfrac{(\mu-r)^2}{2\gamma\sigma^2}\big)(T - t)\Big).$$

### 2.4 Certainty equivalent and the cost of a wrong $\pi$
For constant $\pi$, $X_T$ is lognormal and $\mathbb{E}[U(X_T)] = U\big(x_0 e^{\mathrm{CE}(\pi)T}\big)$, where the
certainty-equivalent rate (continuously compounded, per year) is
$$\mathrm{CE}(\pi) = r + \pi(\mu - r) - \tfrac12 \gamma\sigma^2\pi^2. \tag{CE}$$
It is a downward parabola with peak at $\pi^*$. Using $\hat\pi$ instead of $\pi^*$ costs
$$\mathrm{Loss}(\hat\pi) = \mathrm{CE}(\pi^*) - \mathrm{CE}(\hat\pi) = \tfrac12\gamma\sigma^2(\hat\pi - \pi^*)^2, \qquad
\mathrm{CE}(\pi^*) - r = \frac{(\mu - r)^2}{2\gamma\sigma^2}\ \text{(benefit of optimal investing)}.$$
**Worse than cash:** $\mathrm{CE}(\hat\pi) - r = \tfrac12\gamma\sigma^2\,\hat\pi(2\pi^* - \hat\pi)$, so $\mathrm{CE}(\hat\pi) < r$ exactly when
$\hat\pi(2\pi^* - \hat\pi) < 0$. When $\mu > r$ this is $\hat\pi < 0$ or $\hat\pi > 2\pi^*$.

### 2.5 Estimating $\mu$ and $\sigma$
Daily log returns $R_i = \ln(S_i / S_{i-1})$ are i.i.d. $N\big((\mu - \tfrac12\sigma^2)\Delta t,\ \sigma^2\Delta t\big)$ under GBM.
With sample mean $m$ and sample variance $s^2$ (ddof = 1):
$$\hat\sigma^2 = \frac{s^2}{\Delta t}, \qquad \hat\mu = \frac{m}{\Delta t} + \frac{\hat\sigma^2}{2}, \qquad \hat\pi = \frac{\hat\mu - r}{\gamma\hat\sigma^2}.$$
Precision with $n$ returns over $L = n\Delta t$ years: $\mathrm{se}(\hat\sigma) \approx \sigma/\sqrt{2n}$ (small), but
$\mathrm{se}(\hat\mu) \approx \sigma/\sqrt{L}$ (only more *years* help; $\sum R_i = \ln(S_n/S_0)$ telescopes).

**Benchmark approximations** (neglecting volatility estimation error; Person 4 tests them):
$$\mathrm{sd}(\hat\pi) \approx \frac{1}{\gamma\sigma\sqrt L}, \qquad \mathbb{E}[\mathrm{Loss}] \approx \frac{1}{2\gamma L}.$$
Example: $\gamma = 3$, $L = 10$ gives an expected loss of about $1/60 \approx 1.7\%$ per year.

---

## 3. Fixed settings (everyone)

All of these are in `src/config.py`.

- Python 3.10+, `numpy`, `pandas`, `scipy`, `matplotlib`, `yfinance`.
- Assets: `^NSEI` (NIFTY 50, main asset), `HDFCBANK.NS` (bank), `INFY.NS` (IT), `HINDUNILVR.NS` (FMCG).
- Period: 1 Jan 2008 to latest. Daily data, $\Delta t = 1/252$.
- Risk aversion: main $\gamma = 3$; robustness $\gamma \in \{2, 5\}$.
- Risk-free rate: 91-day Indian T-bill (Person 2). Until ready, `R_PLACEHOLDER = 0.065`.
- Seed 42 everywhere.
- Outputs: PNG (150 dpi) in `figures/`, CSV in `results/`, names start with the person number.

## 4. Repository structure and shared code

```
src/config.py        constants                         (Person 1)
src/theory.py        formulas of Section 2              (Person 1, implemented)
src/data.py          downloading / loading data         (Person 2)
src/estimation.py    mu, sigma, pi estimates            (Person 2)
src/simulate.py      Euler-Maruyama / Milstein / exact  (Person 3)
experiments/p2_data_summary.py      (Person 2)
experiments/p3_simulator_checks.py  (Person 3)
experiments/p4_estimation_risk.py   (Person 4)
experiments/p5_backtest.py          (Person 5)
tests/               pytest checks for every module
```
The function signatures are frozen; see `AGENTS.md` Section 3 and the stubs in `src/`.

---

## 5. Person 1: Theory and coordination

1. Derivation note covering Section 2 in full: self-financing to the wealth SDE; dynamic programming and Ito to HJB; first-order condition; ansatz; solving for $f(t)$; remark on verification.
2. Derive (CE), the loss formula, the worse-than-cash condition, and the two benchmark approximations.
3. `src/config.py`, `src/theory.py` and their tests. **(Done.)**
4. Compare Person 4's simulated sd($\hat\pi$) and mean loss with the benchmark approximations.
5. Assemble the 3-4 page report; write Introduction, Model and Conclusion; check all references.
6. Viva rehearsal: everyone explains the derivation once.

Report section: Model and theory (about 1 page).

## 6. Person 2: Data and parameter estimation

Basic code needed early: Persons 4 and 5 depend on it.

1. **Prices** (`src/data.py`): download daily prices with `yfinance` (`auto_adjust=True`) for the four tickers. Save each as `data/raw/<ticker>.csv` and load from there on later runs. Drop missing values; check prices are positive. Note in the report that NIFTY 50 is a *price* index (dividends excluded).
2. **Risk-free rate**: monthly 91-day T-bill yields from the RBI Database on Indian Economy (https://data.rbi.org.in), saved as `data/raw/tbill_91d.csv`. Convert a quoted yield $y$ (%) to a continuous decimal $r = \ln(1 + y/100)$. Implement `load_risk_free()` and `risk_free_constant()`. If the series cannot be obtained, use a documented constant and say so.
3. **Estimation** (`src/estimation.py`): `log_returns`, `estimate_params`, `estimate_pi` with the formulas of Section 2.5 (including the $+\hat\sigma^2/2$ correction).
4. **Table 1** (`results/p2_params.csv`): for each asset, $L$, $n$, $\hat\mu$ with se $\hat\sigma/\sqrt L$ and 95% CI, $\hat\sigma$ with se $\hat\sigma/\sqrt{2n}$, and $\hat\pi$ for $\gamma = 2, 3, 5$.
5. **Rolling estimates**: 5-year window moved monthly; plot $\hat\mu_t$, $\hat\sigma_t$, $\hat\pi_t$ (`figures/p2_rolling_<ticker>.png`).
6. **Is GBM realistic?** For NIFTY: histogram of daily log returns with fitted normal, normal QQ plot, skewness, excess kurtosis, Jarque-Bera test (`scipy.stats.jarque_bera`).

Report section: Data and estimation (about half a page).
Viva: Why add $\hat\sigma^2/2$? Why does more frequent data help $\sigma$ but not $\mu$?

## 7. Person 3: Simulator and verification

`src/simulate.py` needed early: Person 4 depends on it. Draw $Z \sim N(0,1)$ per path per step; $\Delta W = \sqrt{\Delta t}\,Z$.

- Price, exact: $S_{k+1} = S_k \exp\big((\mu - \tfrac12\sigma^2)\Delta t + \sigma\Delta W_k\big)$
- Price, Euler-Maruyama: $S_{k+1} = S_k(1 + \mu\Delta t + \sigma\Delta W_k)$
- Price, Milstein: $S_{k+1} = S_k\big(1 + \mu\Delta t + \sigma\Delta W_k + \tfrac12\sigma^2(\Delta W_k^2 - \Delta t)\big)$
- Wealth (constant $\pi$), Euler: $X_{k+1} = X_k\big(1 + [r + \pi(\mu - r)]\Delta t + \pi\sigma\Delta W_k\big)$; exact: $X_{k+1} = X_k\exp\big((r + \pi(\mu - r) - \tfrac12\pi^2\sigma^2)\Delta t + \pi\sigma\Delta W_k\big)$

**Test A: accuracy** ($\mu = 0.10$, $\sigma = 0.20$, $S_0 = 100$, $T = 1$, 10,000 paths).
Draw increments once on the finest grid ($\Delta t = 1/1008$); for coarser grids ($1/12, 1/52, 1/252$) sum blocks of the fine increments so every scheme uses the same Brownian path as the exact solution (Higham 2001).
Strong error $\mathbb{E}|S_T^{\text{scheme}} - S_T^{\text{exact}}|$; weak error $|\mathbb{E}[S_T^{\text{scheme}}] - S_0 e^{\mu T}|$.
Log-log plot against $\Delta t$ (`figures/p3_convergence.png`). Expected strong orders: Euler about 1/2, Milstein about 1. (Weak error may be hidden by Monte Carlo noise; say so.)

**Test B: is $\pi^*$ the best?** (NIFTY estimates from Person 2 or placeholders; $\gamma = 3$, $x_0 = 1$, $T = 5$, $\Delta t = 1/52$, 20,000 paths.)
For $\pi = -0.5, -0.4, \dots, 2.5$ simulate wealth with the **same** `Z` for every $\pi$ (common random numbers). Compute $\bar U = \frac1N\sum U(X_T)$ and $\mathrm{CE}_{\text{sim}} = \frac1T \ln\big(((1-\gamma)\bar U)^{1/(1-\gamma)}/x_0\big)$.
Plot against `theory.ce_rate`, mark $\pi^*$ (`figures/p3_ce_vs_pi.png`). It should peak near $\pi^*$ and agree within Monte Carlo and discretisation error. Repeat for $\gamma = 2, 5$. Histograms of $X_T$ for $\pi = 0.5\pi^*, \pi^*, 2\pi^*$.

**Test C: sensitivity.** Plot $\pi^*$ against $\gamma \in [1, 10]$, $\sigma \in [0.1, 0.4]$, $\mu - r \in [-0.05, 0.15]$ (`figures/p3_sensitivity.png`).

Report section: Simulation and verification (about half a page).
Viva: Where does $\sqrt{\Delta t}\,Z$ come from? Why Euler order 1/2 and Milstein 1? Why common random numbers?

## 8. Person 4: Estimation risk

Choose the true $(\mu_0, \sigma_0, r_0)$, so $\pi^*_0$ is known (the answer key); generate many fake histories, estimate $\hat\pi$ from each, grade against $\pi^*_0$.

**Experiment A: GBM world**
1. Truth: full-sample NIFTY estimates from `results/p2_params.csv`; $\pi_0^* =$ `merton_pi(mu0, sigma0, r0, gamma)`, $\gamma = 3$.
2. For $L \in \{5, 10, 20\}$: $M = 1000$ daily paths with `simulate_gbm(method="exact", n_steps=252*L)`.
3. Each path: `log_returns` then `estimate_params` then `estimate_pi`, then `loss(pi_hat, pi_star0, sigma0, gamma)`.
4. Table (`results/p4_estimation_risk.csv`), one row per $L$: mean/sd of $\hat\mu, \hat\sigma, \hat\pi$; % $\hat\pi < 0$; % $\hat\pi > 1$; mean loss (% per year); % worse than cash ($\hat\pi(2\pi_0^* - \hat\pi) < 0$); the benefit $\mathrm{CE}(\pi_0^*) - r_0$.
5. Compare observed sd($\hat\pi$) with $1/(\gamma\sigma_0\sqrt L)$ and mean loss with $1/(2\gamma L)$.
6. Figures: histogram of $\hat\pi$ per $L$ with a line at $\pi_0^*$ (`p4_pihat_hist_L<L>.png`); mean loss vs $L$ with the curve $1/(2\gamma L)$ and a horizontal line at the benefit (`p4_loss_vs_L.png`).
7. Check the loss formula by simulation: for $L = 10$ take the 5th, 50th, 95th percentile $\hat\pi$; in the true market run `simulate_wealth` for $\hat\pi$ and $\pi_0^*$ ($T = 10$, 20,000 paths, same `Z`), compute $\mathrm{CE}_{\text{sim}}$ as in Test B, confirm the gap matches $\tfrac12\gamma\sigma_0^2(\hat\pi - \pi_0^*)^2$.

**Experiment B: bootstrap.** Build each history by resampling real NIFTY daily log returns with replacement (length $252L$); truth = full-sample $\hat\pi$. Also a block bootstrap (blocks of 21 days). This is a robustness check against non-Gaussian, non-i.i.d. returns; the closed-form results remain specific to GBM.

**Experiment C:** repeat $L = 10$ for $\gamma = 2, 5$. Theory: loss scales like $1/\gamma$.

Report section: Estimation risk (about 3/4 page).
Viva: What does "knowing the truth" mean? Why is $\hat\pi$ so spread out? Why is the loss about $1/(2\gamma L)$, independent of $\mu, \sigma$?

## 9. Person 5: Real-data backtest

1. Month-end prices and monthly simple returns $R^S_m = S_m/S_{m-1} - 1$; monthly risk-free return $r^f_m = \exp(r_m/12) - 1$.
2. Start once $W = 5$ years of history exist. At each month-end $m$: estimate $\hat\pi_m$ from the past $W$ years of daily log returns **only**; hold it for the next month, $R^P_{m+1} = \hat\pi_m R^S_{m+1} + (1 - \hat\pi_m) r^f_{m+1}$; update $X_{m+1} = X_m(1 + R^P_{m+1})$ from $X = 1$. Assume borrowing and lending at $r$.
3. Strategies: cash ($\pi = 0$); 60/40 ($\pi = 0.6$, monthly rebalanced); all-stock ($\pi = 1$); Merton unconstrained ($\hat\pi_m$); Merton constrained ($\hat\pi_m$ clipped to $[0,1]$); reference only: Merton "oracle" with the full-sample $\hat\pi$ (uses future data; label it clearly).
4. Metrics (annualised from monthly): mean, volatility, Sharpe $\sqrt{12}\,\overline{(R^P - r^f)}/\mathrm{sd}(R^P - r^f)$, maximum drawdown, final wealth, realised CE $= 12\,\overline{R^P} - \tfrac{\gamma}{2}\cdot 12\,\mathrm{Var}(R^P)$ (DeMiguel et al. 2009), average $|\hat\pi_m|$.
5. Robustness: all four assets, $W \in \{3, 5, 10\}$, $\gamma \in \{2, 3, 5\}$ (`results/p5_backtest_summary.csv`).
6. Figures: wealth curves (log scale) for NIFTY (`p5_wealth_NSEI.png`); $\hat\pi_m$ over time with 0 and 1 marked (`p5_pihat_path.png`); realised CE by strategy and asset (`p5_ce_bars.png`). Mark the 2008 and 2020 crashes.

Report section: Real-data results (about 3/4 page).
Viva: Does estimated Merton beat 60/40? Why might clipping help? Why is the oracle unfair?

---

## 10. Report outline (3-4 pages)
1. Introduction and research question (P1)
2. Model, HJB solution, loss formula (P1)
3. Data and estimation, Table 1 (P2)
4. Simulation and verification (P3)
5. Estimation risk (P4)
6. Real-data backtest (P5)
7. Conclusion and limitations (P1): constant parameters, fat tails, no transaction costs, borrowing at $r$, NIFTY as a price index
8. References

About 5-6 figures in the report; extra figures stay in the repo.

## 11. Viva quick-check
1. The problem: choose $\pi$ to maximise $\mathbb{E}[U(X_T)]$.
2. Why CRRA: the optimal *fraction* does not depend on wealth ($U(cx) = c^{1-\gamma}U(x)$).
3. $\gamma$: relative risk aversion, $-xU''(x)/U'(x) = \gamma$.
4. HJB: dynamic programming principle plus Ito's lemma on $V(t, X_t)$.
5. Why $\pi^*$ is constant: constant $\mu, \sigma, r$ and CRRA; $xV_{xx}/V_x = -\gamma$.
6. Why $\mu$ is hard to estimate: se $= \sigma/\sqrt L$; only more years help.
7. Certainty equivalent: the sure rate worth the same as the risky strategy.
8. Worse than cash: $\hat\pi(2\pi^* - \hat\pi) < 0$.
9. Euler-Maruyama: SDE version of Euler's method; strong order 1/2 (Milstein 1).
10. Limitations: constant parameters, normal returns, no transaction costs, borrowing at $r$.

## References
- R. C. Merton (1969). Lifetime portfolio selection under uncertainty: the continuous-time case. *Review of Economics and Statistics* 51(3), 247-257.
- R. C. Merton (1971). Optimum consumption and portfolio rules in a continuous-time model. *Journal of Economic Theory* 3(4), 373-413.
- R. C. Merton (1980). On estimating the expected return on the market. *Journal of Financial Economics* 8(4), 323-361.
- S. E. Shreve (2004). *Stochastic Calculus for Finance II*. Springer.
- H. Pham (2009). *Continuous-time Stochastic Control and Optimization with Financial Applications*. Springer.
- P. E. Kloeden and E. Platen (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.
- D. J. Higham (2001). An algorithmic introduction to numerical simulation of SDEs. *SIAM Review* 43(3), 525-546.
- R. Kan and G. Zhou (2007). Optimal portfolio choice with parameter uncertainty. *JFQA* 42(3), 621-656.
- V. DeMiguel, L. Garlappi and R. Uppal (2009). Optimal versus naive diversification. *Review of Financial Studies* 22(5), 1915-1953.
- B. Efron and R. J. Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Data: Yahoo Finance via `yfinance`; Reserve Bank of India, Database on Indian Economy (91-day T-bill yields).
