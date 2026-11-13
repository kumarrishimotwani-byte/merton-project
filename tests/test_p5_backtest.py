"""Person 5's focused tests; run with pytest -q tests/test_p5_backtest.py."""
import numpy as np
import pandas as pd
from experiments.p5_backtest import (
    completed_month_ends, decisions_for_window, strategy_weights,
    strategy_path, performance_metrics,
)


def sample_history():
    dates = pd.bdate_range('2015-01-01', '2025-07-18')
    vals = 100 * np.exp(np.cumsum(0.0003 + 0.004 * np.sin(np.arange(len(dates))*0.4)))
    prices = pd.Series(vals, index=dates)
    rate_dates = pd.Series(dates, index=dates).groupby(dates.to_period('M')).last()
    rates = pd.Series(0.06 + 0.002*np.cos(np.arange(len(rate_dates))),
                      index=pd.DatetimeIndex(rate_dates.values))
    # Rates end in June, so July 2025 is an incomplete holding month.
    return prices, rates.iloc[:-1]


def test_incomplete_month_is_not_backtested():
    prices, rates = sample_history()
    ends = completed_month_ends(prices, rates)
    assert ends.index[-1].to_period('M') == pd.Period('2025-06')


def test_decisions_cannot_see_future_prices_or_rates():
    prices, rates = sample_history()
    base = decisions_for_window(prices, rates, W=3, gamma=3)
    decision = base['decision_date'].iloc[-8]
    changed = prices.copy()
    changed.loc[changed.index > decision] *= 1.1
    rates_changed = rates.copy()
    rates_changed.loc[rates_changed.index > decision] *= 2
    other = decisions_for_window(changed, rates_changed, W=3, gamma=3)
    idx = base['decision_date'] <= decision
    np.testing.assert_allclose(other.loc[idx,'pi_hat'], base.loc[idx,'pi_hat'])
    np.testing.assert_allclose(other.loc[idx,'rf_return'], base.loc[idx,'rf_return'])


def test_cash_and_rebalanced_6040_formula():
    decisions = pd.DataFrame({'stock_return':[0.20,-0.10], 'rf_return':[0.01,0.02],
                              'pi_hat':[-0.3,1.4]})
    pi = strategy_weights(decisions, oracle_pi=0.7)
    r_cash, w_cash, fail = strategy_path(decisions, pi['cash'])
    np.testing.assert_allclose(r_cash,[0.01,0.02])
    np.testing.assert_allclose(w_cash,[1,1.01,1.0302])
    assert fail is None
    r_6040, _, _ = strategy_path(decisions, pi['60/40'])
    np.testing.assert_allclose(r_6040, [0.6*.2+.4*.01,0.6*(-.1)+.4*.02])
    np.testing.assert_allclose(pi['merton_clipped'],[0,1])


def test_bankruptcy_is_absorbing_and_ce_undefined():
    decisions=pd.DataFrame({'stock_return':[-0.3,0.5], 'rf_return':[0,0]})
    returns, wealth, at=strategy_path(decisions, np.array([4.,4.]))
    assert at == 0
    np.testing.assert_allclose(wealth,[1,0,0])
    assert np.isnan(returns).all()
    metrics=performance_metrics(returns,wealth,np.array([4.,4.]),
                                decisions['rf_return'],3,at)
    assert metrics['bankrupt']
    assert np.isnan(metrics['realised_ce_ann'])
    assert metrics['final_wealth']==0
    assert metrics['max_drawdown']==1


def test_end_to_end_strategies_share_months_and_cash_matches_product():
    """One integrated check of monthly alignment and compounded benchmark wealth."""
    from experiments.p5_backtest import run_asset

    prices, rates = sample_history()
    summary, paths = run_asset(prices, rates, '^NSEI', W=3, gamma=3)
    assert set(summary['strategy']) == {
        'cash', '60/40', 'all_stock', 'merton', 'merton_clipped', 'merton_oracle'
    }
    assert paths.groupby('strategy').size().nunique() == 1
    cash = paths.loc[paths['strategy'] == 'cash']
    stock = paths.loc[paths['strategy'] == 'all_stock']
    cash_wealth = np.prod(1 + cash['rf_return'].to_numpy())
    stock_wealth = np.prod(1 + stock['stock_return'].to_numpy())
    np.testing.assert_allclose(cash['wealth'].iloc[-1], cash_wealth)
    np.testing.assert_allclose(stock['wealth'].iloc[-1], stock_wealth)
    assert np.isnan(summary.set_index('strategy').loc['cash', 'sharpe'])