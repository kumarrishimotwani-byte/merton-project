"""Person 5: real-data, monthly out-of-sample Merton backtest (PLAN.md Section 9).

Run from the repository root: python -m experiments.p5_backtest

Outputs:
  results/p5_backtest_summary.csv      all 4 assets x 3 windows x 3 gammas x 6 strategies
  results/p5_main_metrics.csv          NIFTY, 5 years of training, gamma=3
  results/p5_nifty_monthly_paths.csv   audit trail: signals, risk-free rate, returns, wealth
  results/p5_report_notes.md           report-ready findings and methodological caveats
  figures/p5_wealth_NSEI.png           log wealth (NIFTY; 2020 crash annotated)
  figures/p5_pihat_path.png            rolling unconstrained NIFTY allocation
  figures/p5_ce_bars.png               realised CE by asset and strategy

The oracle is an explicitly IN-SAMPLE, HINDSIGHT reference; never treat it as
an implementable strategy. Other strategies only use data available when the
portfolio is rebalanced. The cash leg assumes borrowing/lending at the dated
91-day T-bill yield, held constant over the following model month.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, data, estimation

# Experiment settings fixed in PLAN.md Section 9 (no shared interfaces changed).
STRATEGIES = ["cash", "60/40", "all_stock", "merton", "merton_clipped", "merton_oracle"]
WINDOWS_YEARS = [3, 5, 10]
MAIN_WINDOW_YEARS = 5

LABELS = {
    "cash": "Cash (0%)",
    "60/40": "60/40 stock/cash",
    "all_stock": "All stock (100%)",
    "merton": "Merton (unconstrained)",
    "merton_clipped": "Merton clipped [0, 1]",
    "merton_oracle": "Oracle (hindsight only)",
}


def completed_month_ends(prices, rates):
    """Return actual last *observed* close of each completed month (PLAN.md 9.1).

    Use timestamps of stock observations, not nominal calendar month-end dates:
    some Yahoo stock sessions are missing. Exclude the final partial month by
    requiring the month to have a completed Person 2 risk-free snapshot.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    last_completed = rates.index[-1].to_period("M")
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1)
    month_ends = month_ends[month_ends.index.to_period("M") <= last_completed]
    if len(month_ends) < 2:
        raise ValueError("Need at least two completed months of price observations")
    return month_ends


def decisions_for_window(prices, rates, W, gamma):
    """Generate no-look-ahead monthly decisions and returns (PLAN.md 9.1-9.2).

    At observed month-end m, estimate pi_hat from daily prices in (m-W, m].
    Invest during (m,m+1], using simple stock return and the *known at m*
    annual continuously compounded cash rate: exp(r_m/12)-1. Later rate
    snapshots are NEVER used for the preceding holding month.
    """
    month_ends = completed_month_ends(prices, rates)
    first_eligible = prices.index[0] + pd.DateOffset(years=W)
    rows = []
    for i in range(len(month_ends) - 1):
        decision_date = month_ends.index[i]
        holding_end = month_ends.index[i + 1]
        if decision_date < first_eligible:
            continue
        lookback_start = decision_date - pd.DateOffset(years=W)
        past_prices = prices.loc[(prices.index >= lookback_start) & (prices.index <= decision_date)]
        log_rets = estimation.log_returns(past_prices)
        r = rates.asof(decision_date)
        if not np.isfinite(r):
            raise ValueError(f"No risk-free quote known by {decision_date.date()}")
        pi_hat = estimation.estimate_pi(log_rets, r=float(r), gamma=gamma)
        rows.append({
            "decision_date": decision_date,
            "holding_end": holding_end,
            "train_start": past_prices.index[0],
            "train_n_returns": len(log_rets),
            "known_r_annual_cont": float(r),
            "stock_return": float(month_ends.iloc[i + 1] / month_ends.iloc[i] - 1),
            "rf_return": float(np.expm1(float(r) / 12)),
            "pi_hat": float(pi_hat),
        })
    if not rows:
        raise ValueError(f"No out-of-sample months for W={W}")
    return pd.DataFrame(rows)


def strategy_weights(decisions, oracle_pi):
    """Build six comparable stock-weight paths, with monthly rebalancing (PLAN.md 9.3)."""
    n = len(decisions)
    pihat = decisions["pi_hat"].to_numpy(dtype=float)
    return {
        "cash": np.zeros(n),
        "60/40": np.full(n, 0.6),
        "all_stock": np.ones(n),
        "merton": pihat,
        "merton_clipped": np.clip(pihat, 0.0, 1.0),
        "merton_oracle": np.full(n, float(oracle_pi)),
    }


def strategy_path(decisions, pi):
    """Compute R^P=pi*R^S+(1-pi)*R^f and X_next=X*(1+R^P) (PLAN.md 9.2).

    If any simple-return gross factor <=0 (possible with unconstrained
    leverage), wealth is ruined, remains zero, and post-ruin portfolio returns
    are undefined. We NEVER allow negative wealth or plot it on log axes.
    """
    stock = decisions["stock_return"].to_numpy(dtype=float)
    rf = decisions["rf_return"].to_numpy(dtype=float)
    pi = np.asarray(pi, dtype=float)
    returns = pi * stock + (1.0 - pi) * rf
    if not np.isfinite(returns).all():
        raise ValueError("Nonfinite portfolio returns")
    gross = 1.0 + returns
    ruined_at = np.flatnonzero(gross <= 0)
    ruin_idx = int(ruined_at[0]) if len(ruined_at) else None
    if ruin_idx is not None:
        gross = gross.copy()
        gross[ruin_idx:] = 0.0
        returns = returns.copy()
        returns[ruin_idx:] = np.nan
    wealth = np.r_[1.0, np.cumprod(gross)]
    return returns, wealth, ruin_idx


def performance_metrics(returns, wealth, pi, rf, gamma, ruin_idx):
    """Annualised monthly moments, drawdown and empirical CE (PLAN.md 9.4).

    Mean=12*mean(R); vol=sqrt(12)*sd(R); Sharpe=sqrt(12)*mean(R-rf)/
    sd(R-rf); CE=12*mean(R)-gamma/2*12*sample_var(R), using ddof=1.
    A strategy ruined by leverage has no well-defined whole-horizon monthly
    returns or CRRA utility, so report NaNs instead of misleading moments.
    """
    values = np.asarray(returns, dtype=float)
    ex = values - np.asarray(rf, dtype=float)
    bankrupt = ruin_idx is not None
    if bankrupt:
        avg = vol = sharpe = ce = float("nan")
    else:
        avg = 12.0 * float(np.mean(values))
        var = float(np.var(values, ddof=1))
        vol = np.sqrt(12.0 * var)
        ex_sd = float(np.std(ex, ddof=1))
        sharpe = float(np.sqrt(12.0) * np.mean(ex) / ex_sd) if ex_sd > 0 else float("nan")
        ce = float(avg - gamma / 2.0 * 12.0 * var)
    peak = np.maximum.accumulate(wealth)
    max_drawdown = float(-np.min(wealth / peak - 1.0))
    return {
        "mean_return_ann": avg,
        "volatility_ann": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_wealth": float(wealth[-1]),
        "realised_ce_ann": ce,
        "mean_abs_pi": float(np.mean(np.abs(pi[:ruin_idx + 1] if bankrupt else pi))),
        "bankrupt": bankrupt,
        "bankruptcy_month": str(ruin_idx + 1) if bankrupt else "",
    }


def run_asset(prices, rates, ticker, W, gamma):
    """Backtest all six strategies on identical out-of-sample months (PLAN.md 9.3-9.5)."""
    decisions = decisions_for_window(prices, rates, W, gamma)
    month_ends = completed_month_ends(prices, rates)
    history_through_last_complete_month = prices.loc[:month_ends.index[-1]]
    # Intentionally hindsight-biased. Never use in the rolling decisions.
    oracle_pi = estimation.estimate_pi(
        estimation.log_returns(history_through_last_complete_month),
        r=float(rates.mean()), gamma=gamma,
    )
    weights = strategy_weights(decisions, oracle_pi)
    all_rows = []
    path_rows = []
    for strategy in STRATEGIES:
        pi = weights[strategy]
        returns, wealth, ruin_idx = strategy_path(decisions, pi)
        row = {
            "ticker": ticker,
            "window_years": W,
            "gamma": gamma,
            "strategy": strategy,
            "n_months": len(decisions),
            "start_decision": decisions["decision_date"].iloc[0].date().isoformat(),
            "end_holding": decisions["holding_end"].iloc[-1].date().isoformat(),
            **performance_metrics(returns, wealth, pi, decisions["rf_return"], gamma, ruin_idx),
        }
        all_rows.append(row)
        for i, item in decisions.iterrows():
            path_rows.append({
                "ticker": ticker,
                "window_years": W,
                "gamma": gamma,
                "strategy": strategy,
                "decision_date": item["decision_date"],
                "holding_end": item["holding_end"],
                "train_start": item["train_start"],
                "train_n_returns": item["train_n_returns"],
                "r_known_annual_cont": item["known_r_annual_cont"],
                "pi": pi[i],
                "stock_return": item["stock_return"],
                "rf_return": item["rf_return"],
                "portfolio_return": returns[i],
                "wealth": wealth[i + 1],
            })
    return pd.DataFrame(all_rows), pd.DataFrame(path_rows)


def make_figures(paths, main_summary):
    """Save three 150-dpi audit figures requested by PLAN.md Section 9.6."""
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(10, 6))
    for strategy in STRATEGIES:
        sub = paths.loc[paths["strategy"] == strategy]
        w = sub["wealth"].to_numpy(dtype=float)
        ax.plot(sub["holding_end"], np.where(w > 0, w, np.nan),
                label=LABELS[strategy], linewidth=1.7,
                linestyle="--" if strategy == "merton_oracle" else "-")
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
               color="0.80", alpha=0.35, label="2020 COVID crash")
    ax.set_yscale("log")
    ax.set(xlabel="Holding month end", ylabel="Wealth (initial capital = 1, log scale)",
           title="NIFTY: out-of-sample monthly backtest (W=5 years, gamma=3)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.text(0.01, -0.15,
            "2008 crash is in the training period, not the out-of-sample curve. "
            "Oracle uses future information.",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "p5_wealth_NSEI.png", dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    p = paths.loc[paths["strategy"] == "merton"]
    ax.plot(p["decision_date"], p["pi"], lw=1.5, label="Estimated pi_hat")
    ax.axhline(0.0, ls="--", color="0.4", label="0 (cash)")
    ax.axhline(1.0, ls=":", color="0.3", label="1 (all stock)")
    ax.set(xlabel="Decision date", ylabel="Stock wealth fraction pi_hat",
           title="NIFTY: rolling estimated Merton allocation (W=5 years, gamma=3)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "p5_pihat_path.png", dpi=config.FIG_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ticks = list(config.TICKERS)
    x = np.arange(len(ticks))
    width = 0.13
    for j, strategy in enumerate(STRATEGIES):
        sub = main_summary.loc[main_summary["strategy"] == strategy].set_index("ticker")
        vals = 100.0 * sub.reindex(ticks)["realised_ce_ann"].to_numpy(dtype=float)
        ax.bar(x + (j - (len(STRATEGIES)-1)/2)*width, vals, width=width,
               label=LABELS[strategy])
    ax.axhline(0.0, color="0.4", linewidth=0.7)
    ax.set_xticks(x, ["NIFTY 50", "HDFC Bank", "Infosys", "Hindustan Unilever"])
    ax.set(ylabel="Realised certainty equivalent (% per year)",
           title="Realised CE by asset and strategy (W=5 years, gamma=3)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "p5_ce_bars.png", dpi=config.FIG_DPI)
    plt.close(fig)


def make_report_notes(main, paths, full):
    """Prepare a small numerical handoff to Person 1, without inventing results."""
    nifty = main.loc[main["ticker"] == config.MAIN_TICKER].set_index("strategy")
    def shown(x):
        return "undefined (ruin)" if not np.isfinite(x) else f"{100*x:.2f}%/year"
    lines = [
        "# Person 5 — real-data backtest report handoff",
        "",
        "## NIFTY main case (W=5 calendar years, gamma=3)",
        "",
        "The strategies are evaluated over the same out-of-sample months, from "
        f"{nifty.iloc[0]['start_decision']} (rebalance) to {nifty.iloc[0]['end_holding']}. "
        "Monthly stock returns are simple month-end-to-month-end returns. "
        "At each rebalance we estimate the Merton fraction from the preceding W years "
        "of daily log returns and invest during the following month. "
        "Cash uses the latest T-bill quote *known by that rebalance*, "
        "converted to a monthly simple return exp(r/12)-1.",
        "",
        "| Strategy | Mean return (annual) | Volatility (annual) | CE (annual) | "
        "Final wealth | Max drawdown | Mean abs. allocation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in STRATEGIES:
        row = nifty.loc[s]
        lines.append(
            f"| {LABELS[s]} | {shown(row['mean_return_ann'])} | "
            f"{shown(row['volatility_ann'])} | {shown(row['realised_ce_ann'])} | "
            f"{row['final_wealth']:.3f} | {100*row['max_drawdown']:.1f}% | "
            f"{row['mean_abs_pi']:.3f} |"
        )
    ce_fixed = nifty.loc["60/40", "realised_ce_ann"]
    for s in ("merton", "merton_clipped"):
        ce = nifty.loc[s, "realised_ce_ann"]
        if np.isfinite(ce) and np.isfinite(ce_fixed):
            verb = "above" if ce >= ce_fixed else "below"
            lines.append(
                f"The {LABELS[s]} strategy had a realised mean-variance CE "
                f"{100*abs(ce - ce_fixed):.2f} annual percentage points {verb} 60/40."
            )
        else:
            lines.append(f"{LABELS[s]} suffered bankruptcy; do not compare its undefined CE to 60/40.")
    n_unconstrained = paths[paths["strategy"] == "merton"]
    leverage = float(np.mean((n_unconstrained["pi"] < 0) | (n_unconstrained["pi"] > 1)))
    lines.append(f"The NIFTY unconstrained pi_hat lay outside [0,1] in {100*leverage:.1f}% of decisions.")
    lines += [
        "",
        "## Implementation, interpretation and limits",
        "",
        "- The 60/40 benchmark means 60% of wealth in a single stock (or NIFTY) "
        "and 40% in cash, rebalanced monthly, **not** 60% equity/40% bonds.",
        "- The full-sample oracle uses future prices and a full-sample average interest rate. "
        "It is an in-sample reference, never an investable comparator.",
        "- The 2008 crash belongs to the parameter-training period. With at least "
        "three years of training, no out-of-sample portfolio exists in 2008; the 2020 "
        "crash is annotated in the wealth graph.",
        "- NIFTY levels exclude dividends. Stocks use split/dividend-adjusted Yahoo closes. "
        "Cross-asset results therefore do not have identical total-return conventions.",
        "- For three documented stock-calendar gaps (Feb 2015, Oct 2016, Mar 2019), "
        "the last observed stock date precedes the official month-end. The code "
        "retains that earlier decision date and never uses a quote published later. "
        "Those monthly returns can bridge a missing last-session movement; comparing "
        "assets at perfectly matched calendar endpoints would require an additional sensitivity study.",
        "- T-bill yields are an investable-cash proxy, treated as continuously earning "
        "the rate known at the start of the holding month, and the model permits "
        "borrowing/lending at the same rate. Fees, financing spreads, and "
        "transaction costs are omitted.",
        "- Realised CE is a *sample mean-variance proxy*, not an empirical estimate "
        "of exact CRRA expected utility. The sample variance uses ddof=1. "
        "The cash strategy has undefined Sharpe because its excess return is zero.",
        "- A monthly unconstrained portfolio can have gross return <=0. "
        "The code records ruin and an absorbing zero wealth value. Its full-horizon "
        "mean/volatility/Sharpe/CE are then undefined (blank in CSV), not reported "
        "from a fictitious negative-wealth trajectory.",
        "- Rolling calibration uses calendar-year lookbacks, sample daily variance "
        "and 252-trading-day annualisation, as in the group's estimator. "
        "Final partial November 2026 observations are excluded.",
        "",
        "## Robustness",
        "",
        f"The full summary contains {len(full)} rows covering four assets, "
        "W in {3,5,10} years, gamma in {2,3,5} and six strategies. "
        "Each asset/window/gamma set evaluates all strategies on the same months "
        "(though starting dates differ across window lengths).",
        "",
        "## Viva prompts",
        "",
        "1. At time m, why must both pi_hat and the interest rate use only information "
        "available by m? What would go wrong if we used next month's RBI quote?",
        "2. Why might clipping pi_hat to [0,1] improve CE relative to the unconstrained strategy?",
        "3. Why is the oracle an invalid real-world benchmark?",
        "4. What is being measured by the realised CE proxy and what does gamma control?",
    ]
    return "\n".join(lines) + "\n"


def main():
    """Regenerate every Person 5 table/figure from the committed input files."""
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    rates = data.load_risk_free().sort_index()
    if rates.empty:
        raise ValueError("T-bill input is empty")
    all_summaries = []
    main_paths = None
    for ticker in config.TICKERS:
        prices = data.load_prices(ticker)
        for W in WINDOWS_YEARS:
            for gamma in config.GAMMAS:
                summary, paths = run_asset(prices, rates, ticker, W, gamma)
                all_summaries.append(summary)
                if ticker == config.MAIN_TICKER and W == MAIN_WINDOW_YEARS and gamma == config.GAMMA:
                    main_paths = paths
    full = pd.concat(all_summaries, ignore_index=True)
    full.to_csv(config.RESULTS / "p5_backtest_summary.csv", index=False)
    main = full[(full["window_years"] == MAIN_WINDOW_YEARS) & (full["gamma"] == config.GAMMA)]
    main.loc[main["ticker"] == config.MAIN_TICKER].to_csv(
        config.RESULTS / "p5_main_metrics.csv", index=False)
    main_paths.to_csv(config.RESULTS / "p5_nifty_monthly_paths.csv", index=False)
    make_figures(main_paths, main)
    (config.RESULTS / "p5_report_notes.md").write_text(
        make_report_notes(main, main_paths, full), encoding="utf-8")
    print("Saved Person 5's 3 figures, summary CSV, NIFTY audit paths and report notes.")
    print(main.loc[main["ticker"] == config.MAIN_TICKER,
                   ["strategy", "realised_ce_ann", "final_wealth", "bankrupt"]].to_string(index=False))


if __name__ == "__main__":
    main()