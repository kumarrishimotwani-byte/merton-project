"""Person 3: simulator accuracy and verification of pi*. Spec: docs/PLAN.md Section 7.

Run: python -m experiments.p3_simulator_checks

Produces:
  figures/p3_convergence.png      Test A: strong and weak error vs dt (Euler, Milstein), log-log
  figures/p3_ce_vs_pi.png         Test B: simulated CE vs pi against theory.ce_rate, pi* marked
  figures/p3_wealth_hist.png      Test B: X_T histograms for pi = 0.5 pi*, pi*, 2 pi*
  figures/p3_sensitivity.png      Test C: pi* vs gamma, sigma, mu - r
"""

from src import config, simulate, theory  # noqa: F401


def main():
    # TODO (Person 3):
    # Test A: draw fine increments once (dt = 1/1008); coarser grids sum blocks of them.
    # Test B: common random numbers Z for every pi; CE_sim via theory.certainty_equivalent_from_utility.
    # Test C: theory.merton_pi over parameter grids.
    raise NotImplementedError("Person 3: implement the simulator checks")


if __name__ == "__main__":
    main()
