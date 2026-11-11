"""Person 4: estimation-risk experiments. Spec: docs/PLAN.md Section 8.

Run: python -m experiments.p4_estimation_risk

Produces:
  results/p4_estimation_risk.csv       one row per L (and gamma / method): pi_hat stats, mean loss, % worse than cash
  figures/p4_pihat_hist_L<L>.png       histogram of pi_hat with the true pi*
  figures/p4_loss_vs_L.png             mean loss vs L, theory 1/(2 gamma L), benefit line
"""

from src import config, estimation, simulate, theory  # noqa: F401


def main():
    # TODO (Person 4):
    # Experiment A: truth = full-sample NIFTY estimates (results/p2_params.csv) or config placeholders.
    #   For L in (5, 10, 20): 1000 exact GBM paths -> log returns -> pi_hat -> theory.loss.
    #   Compare with theory.sd_pihat_approx and theory.expected_loss_approx; use theory.worse_than_cash.
    #   Check the loss formula by simulation with simulate.simulate_wealth (same Z for pi_hat and pi*).
    # Experiment B: i.i.d. and block (21-day) bootstrap of real NIFTY log returns.
    # Experiment C: gamma in config.GAMMAS.
    raise NotImplementedError("Person 4: implement the estimation-risk experiments")


if __name__ == "__main__":
    main()
