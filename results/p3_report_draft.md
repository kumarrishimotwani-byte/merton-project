# Person 3 report contribution: simulation and verification

We implemented exact, Euler-Maruyama and Milstein price simulation and exact/Euler
constant-fraction wealth simulation. With 10,000 shared Brownian paths,
the observed strong orders were 0.497 for Euler and 0.998 for
Milstein, consistent with the expected 1/2 and 1. The fine grid uses 3,276 steps:
the plan's 1,008-step grid cannot be split into equal weekly blocks. Weak errors
were too affected by Monte Carlo sampling noise to establish a convergence order.

Using Person 2's NIFTY estimates (mu=0.096206, sigma=0.201907, r=0.063037) as known
model parameters, we simulated 20,000 common wealth paths over
5 years with weekly Euler steps. For gamma=3,
pi*=0.2712, with theoretical CE 6.7535%/year
and simulated CE 6.7785%/year (approximate MC SE
0.0177 percentage points/year). Grid maxima were
gamma=2: pi=0.4; gamma=3: pi=0.3; gamma=5: pi=0.2, consistent with the theoretical optima at the grid's resolution.
Larger risky fractions produced more dispersed terminal wealth. Sensitivity
curves show that pi* falls as 1/gamma and 1/sigma^2 and rises linearly with mu-r.

Agreement near the optimum does not extend reliably to extreme leverage: at
gamma=5, pi=2.5, one path contributes
99.83% of absolute sampled utility. The CE estimate
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
- Calibration: the committed `^NSEI` row of `results/p2_params.csv`.
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
