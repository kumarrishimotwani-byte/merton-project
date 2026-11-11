# The Merton Problem and the Cost of Estimation Risk

MAC F244 Stochastic Calculus and Applications in Finance, BITS Pilani K K Birla Goa Campus, Semester I 2026-27.

**Research question:** does Merton's optimal portfolio strategy still work when its parameters must be
estimated from real market data, and how much does the estimation error cost?

Merton's optimal fraction of wealth in the stock is $\pi^* = (\mu - r)/(\gamma\sigma^2)$. We derive it from the
HJB equation, verify it by SDE simulation, measure how unreliable its estimate $\hat\pi$ is in a simulated
market where the truth is known, and backtest it on NIFTY 50 and three NSE stocks.

## Team

| Name | ID |
|---|---|
| Rishi Kumar Motwani | 2024ADPS0838G |
| Devam Jatin Bhavsar | 2024ADPS0831G |
| Yash Saxena | 2024ADPS0868G |
| Raghav Singhal | 2024ADPS0827G |
| Arnav Gupta | 2024ADPS0843G |

## Where to start

1. Read **`docs/PLAN.md`**: Sections 1-4 (everyone), then your own section.
2. Find your files in the table below and work only on those.
3. **Using an AI assistant?** Point it at this repo and tell it which person you are. It must read
   `AGENTS.md` first (Claude Code reads it automatically through `CLAUDE.md`). The function signatures in
   `src/` are shared by all five of us and must not change.

| Role | Files | Status |
|---|---|---|
| 1 Theory and coordination | `src/config.py`, `src/theory.py`, `report/` | theory code done |
| 2 Data and estimation | `src/data.py`, `src/estimation.py`, `experiments/p2_data_summary.py` | to do |
| 3 Simulator | `src/simulate.py`, `experiments/p3_simulator_checks.py` | to do |
| 4 Estimation risk | `experiments/p4_estimation_risk.py` | to do (needs 2 and 3) |
| 5 Backtest | `experiments/p5_backtest.py` | to do (needs 2) |

## Setup

```bash
git clone <this repo>
cd merton-project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                 # theory tests pass; others skip until implemented
```

## Running

Always run from the repo root:

```bash
python -m experiments.p2_data_summary
python -m experiments.p3_simulator_checks
python -m experiments.p4_estimation_risk
python -m experiments.p5_backtest
```

Figures go to `figures/`, tables to `results/`.

## Repository layout

```
AGENTS.md            rules for AI coding agents (CLAUDE.md points here)
docs/PLAN.md         full specification: maths, settings, each person's tasks
docs/project_plan.pdf, docs/proposal.pdf
src/                 shared code (config, theory, data, estimation, simulate)
experiments/         one script per person
tests/               pytest checks; a stub's tests are skipped until it is implemented
data/raw/            downloaded CSVs (committed, so everyone uses identical data)
figures/  results/  report/
```

## Workflow

Each person works on their own branch (`p2-data`, `p3-simulator`, `p4-estimation-risk`, `p5-backtest`)
and runs `pytest` before pushing. Person 1 merges into `main` at the end.

Final submission (code + 3-4 page report) is due **15 Nov 2026**.
