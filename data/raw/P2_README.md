# Person 2 handoff

Run `python -m experiments.p2_data_summary` from the repository root, then `python -m pytest -q`. These commands use the committed CSV snapshot and require no network access. Every figure and result table is regenerated. Read this note together with the PR description before using the inputs in an AI coding tool.

## Scope and model

Section 6 of `docs/PLAN.md` is implemented in `src/data.py`, `src/estimation.py` and `experiments/p2_data_summary.py`. Imports use the existing `src/config.py` and `src/theory.py`, and the shared signatures/defaults are preserved. `estimate_pi` delegates to `theory.merton_pi`. Estimation accepts a single one-dimensional history at a time. Person 4 may loop over histories as the specification permits, or later agree a vectorized extension. Daily log returns use sample variance with ddof=1, sigma squared equals variance divided by dt, and mu adds the Ito correction sigma squared/2. The drift is the instantaneous SDE coefficient. It differs from the mean log-return rate by sigma squared/2.

The main table uses the plan's dt=1/252 and L=n/252 convention. L is a trading-observation year count, which can differ from elapsed calendar years. Confidence intervals and standard errors are approximate under constant-parameter i.i.d. GBM. Short special sessions count as one session under the same convention.

## Reproducible inputs and source audit

Prices cover 1 January 2008 through 9 November 2026. This is a frozen snapshot, even though config.END_DATE is None. Keep the committed CSVs for comparable team results. Retrieving new prices changes the experiment's inputs and requires another calendar audit. All original Yahoo snapshots are retained in `p2_provenance/yahoo_*.csv`. The canonical CSVs have columns `Date,adjusted_close`.

- The NIFTY cache uses official [NSE index history](https://niftyindices.com/reports/historical-data). The CSV field name stays `adjusted_close` for compatibility, but its contents are NIFTY 50 price-index levels. Ordinary dividends are excluded. NSE's history was retrieved from the site's historical report endpoint, `POST https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString`, with a JSON body whose `cinfo` string specifies `name` and `indexName` as `NIFTY 50`, `startDate` as `01-Jan-2008` and `endDate` as `09-Nov-2026`. This response had 4,671 unique positive closes. All 4,625 overlapping Yahoo observations match within 0.001 index points. The 46 recovered sessions are recorded in `p2_provenance/nifty_yahoo_missing_sessions.csv`.
- Stocks use Yahoo Finance via yfinance with auto_adjust=True. All three original caches included the same six closed dates with adjusted closes exactly equal to the prior observation. These six rows were removed. Their dates, reasons and original stock values are in `p2_provenance/nse_closed_dates.csv`. A missing stock cache can be downloaded by load_prices, but a warning requires the new download to be audited. A missing NIFTY cache fails explicitly and asks for the reviewed file to be restored, so downstream work cannot silently switch back to incomplete Yahoo data.
- Monthly T-bill rates use 226 observations from January 2008 through October 2026. Each `date` is the last official NSE index session of that completed month. `available_on` conservatively means the calendar day after auction, including releases actually published the same day. The latest successful auction with available_on no later than the decision date is selected. The normalized underlying quotes through 4 November 2026 are in `p2_provenance/rbi_91d_auction_quotes.csv`. The November partial month is excluded from the monthly snapshot.

The [RBI Innovation Hub bulk table](https://data-api.dbie.rbihub.in/api/tables/public_finance/r307_auctions_of_91_day_government_of_india_treasury_bills/csv) contains rounded displayed yields, so precise quotes use the [Measure of India compilation of RBI Report 307](https://measure.thisindianlife.today/data/rates-tbill-91/). It is an intermediary source, and every exact historical quote has not been independently verified. Before the one correction below, 970 common valid quotes from December 2007 through 23 September 2026 matched the bulk export at its displayed integer precision. Exact comparisons against the [2019 RBI handbook table](https://www.rbi.org.in/scripts/PublicationsView.aspx?id=19946) and [2025-26 table](https://rbi.org.in/Scripts/PublicationsView.aspx?id=24043) matched 57 and 49 observations, respectively. A price/yield consistency check exposed a discrepant 6 January 2010 quote. The [RBI Bulletin](https://www.rbi.org.in/scripts/BS_ViewBulletin.aspx?Id=11799) gives 3.6835%, which replaces the intermediary's 3.3835%. Later quotes missing from that compilation were added from the RBI releases linked in the quote CSV. A cancelled/unaccepted auction supplies no rate.

The conversion intentionally differs from PLAN.md Section 6.2. An RBI quoted yield y is simple annual ACT/365, so with tau=91/365, the annual continuously compounded rate is `log1p((y/100)*tau)/tau`. This matches the bill's 91-day growth factor. The plan's `log1p(y/100)` assumes y is an effective one-year return. The average of the chosen monthly continuous rates is 0.0630372232, compared with 0.0614990024 using the plan's shortcut on the identical quotes. These yields are a cash-rate proxy in the model, rather than observed returns from a fully specified tradable T-bill strategy. [RBI yield convention, questions 25-26](https://www.rbi.org.in/scripts/FAQView.aspx?Id=79)

## The remaining stock gaps and their acceptability

All three stocks lack exactly the same 28 sessions, 18 of which fall on weekends. The pattern and the [Budget session circular](https://nsearchives.nseindia.com/content/circulars/CMPT72389.pdf) and [May 2024 special-session circular](https://nsearchives.nseindia.com/content/circulars/MSD61893.pdf) support a vendor coverage limitation. Yahoo's internal reason for every omitted row is unverified. The official daily equity archives confirm all three stocks traded on [30 October 2016](https://archives.nseindia.com/content/historical/EQUITIES/2016/OCT/cm30OCT2016bhav.csv.zip) and [29 March 2019](https://archives.nseindia.com/content/historical/EQUITIES/2019/MAR/cm29MAR2019bhav.csv.zip). Unadjusted NSE closes were kept separate from adjusted Yahoo values to avoid false split/dividend jumps.

`results/p2_missing_stock_sessions.csv` enumerates every missing stock/date and flags official month-ends. The only missing month-end sessions are 28 February 2015, 30 October 2016 and 29 March 2019. The other gaps bridge two actual sessions into a single observed return. `results/p2_gap_sensitivity.csv` compares the baseline with an unequal-interval GBM estimate. With t_i equal to the elapsed count of official trading sessions divided by 252, it computes a=sum(R_i)/sum(t_i), sigma squared=sum((R_i-a*t_i)^2/t_i)/(n-1), and mu=a+sigma squared/2. This is a sensitivity check using the observed endpoints, not a reconstruction of missing daily movements.

At gamma=3, the full-sample stock fractions are as follows. All rates use the same monthly average r.

| Asset | Baseline fraction | Gap-aware fraction |
|---|---:|---:|
| HDFC Bank | 0.473071 | 0.470208 |
| Infosys | 0.365210 | 0.363004 |
| Hindustan Unilever | 0.578870 | 0.578302 |

The maximum change is 0.2864 percentage points of wealth. Drift and volatility each move by at most 0.12 annual percentage points. This supports using the supplied data for a disclosed course-project comparison. It does not establish negligible effects on every rolling estimate, risk statistic or backtest outcome. NIFTY's primary series has no gaps relative to the retrieved official index report. All histories remain subject to model limitations and source revisions.

## Person 4 inputs

```python
import pandas as pd
from src import config, data, estimation, theory

parameters = pd.read_csv(config.RESULTS / 'p2_params.csv').set_index('ticker')
row = parameters.loc[config.MAIN_TICKER]
mu0, sigma0, r0 = row['mu'], row['sigma'], row['r']
pi_star0 = theory.merton_pi(mu0, sigma0, r0, config.GAMMA)
real_returns = estimation.log_returns(data.load_prices(config.MAIN_TICKER))
```

`mu`, `sigma`, `r` and all `pi_gamma_2`, `pi_gamma_3`, `pi_gamma_5` fields are decimals. The `r` column is a fixed full-sample average for simulation calibration. Bootstrap returns come from the corrected official NIFTY price history. Generate positive exact GBM wealth when checking CRRA utilities, and keep the plan's estimation-error formulas labelled as benchmarks neglecting volatility estimation error.

## Person 5 inputs and time alignment

Use `data.load_prices(ticker)` and `data.load_risk_free()`. The latter returns a dated Series of continuous annual rates. `rates.asof(decision_date)` selects a monthly snapshot available by the decision date. At month-end m, use the rate selected then for the following holding month when forming the model cash return `expm1(r_m/12)`. Using a next-month endpoint yield to credit the preceding holding period would use future information. `data.risk_free_constant()` uses the full sample and is for calibration/descriptive summaries.

A stock's last observed close precedes the actual last exchange close in three months. Retain that actual observed timestamp. A snapshot dated after it should not be looked up as though known on that earlier stock date. An alternative is an as-of lookup against the underlying auction quotes using `available_on`. In the existing Person 2 rolling summaries the displayed monthly quote was already available at each of those earlier closes; this was checked explicitly. For price-return comparisons on identical endpoints, align NIFTY to the same observed stock dates and report that sensitivity. Calendar-month labels in rolling CSVs describe descriptive windows and do not change the source observation timestamps in `last_price_date`.

Use completed holding months only and explain that a 60/40 strategy here means stock/cash. A labelled full-sample oracle remains a descriptive reference. Daily Sharpe/drawdown metrics on the stocks cannot recover missed intraperiod moves. An unconstrained monthly portfolio may suffer nonpositive wealth under large losses; the backtest owner must handle that explicitly before log-scale wealth charts or CRRA calculations.

## Findings and report contribution

`P2_REPORT_DRAFT.md` supplies text for Person 1 to incorporate in `report/`. The primary NIFTY estimates are mu=0.0962059359 and sigma=0.2019070199, with a gamma=3 fraction 0.2712092667. The approximate drift standard error is 0.0469022037. The normal-quantile 95% interval is about [0.0042793, 0.1881326]. The volatility standard error is 0.0020891902. These are model-conditional uncertainties rather than guarantees of future parameter stability. Normality diagnostics give skewness -0.3303, excess kurtosis 16.0745 and a Jarque-Bera statistic about 50,247.65. A displayed p-value of zero is numerical underflow and should be described as extremely small.

## Items for the coordinator

These are flagged for review, without editing other members' files.

- Correct PLAN.md 6.2's T-bill yield conversion for simple annual ACT/365 quotes, and update the source reference to include official NSE index data and the documented intermediary for RBI history.
- PLAN.md 7 Test A asks for blocks from a 1/1008-year fine grid to a 1/52-year grid. Since 1008/52 is not an integer, those increments cannot be aggregated by equal-sized blocks. A grid with 3276 increments per year is divisible by 12, 52 and 252, or Person 3 can agree another nested family.
- PLAN.md 9 asks to annotate the 2008 crash on an out-of-sample strategy that requires at least three years of data beginning in 2008. The out-of-sample curve starts in 2011 or later, so 2008 belongs in training context rather than on that curve.
- Config has no setting for the fixed five-year Person 2 rolling window or the CI confidence level. The script applies the literal specification locally. Person 1 can centralize these parameters later if the all-constants-in-config rule is intended to include fixed experiment/mathematical settings.
- The README's Person 2 status is still to do until the coordinator merges this PR. The source imports and shared config/theory stay under their existing owners. No new dependencies are introduced.
