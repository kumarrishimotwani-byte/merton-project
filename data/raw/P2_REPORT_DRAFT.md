# Data and estimation

The NIFTY 50 price index uses 4,671 official NSE Indices daily closes from 1 January 2008 through 9 November 2026. HDFC Bank, Infosys and Hindustan Unilever use Yahoo dividend- and split-adjusted closes. Comparing Yahoo with NSE recovered 46 index observations and identified six exchange closures with repeated Yahoo stock closes. We removed those rows. NIFTY excludes ordinary dividends, and the three stock series each lack 28 sessions in the official index calendar. We use 252 observations per year, so the stock estimates approximate some multi-session moves as daily returns.

The risk-free proxy uses the latest 91-day Government of India T-bill auction yield available by each last NSE session of the month. We treat the quote as available one calendar day after the auction and convert the simple ACT/365 quote to a continuously compounded rate. The mean of the 226 monthly observations from January 2008 to October 2026 was 6.30% annually. The rolling estimates use the rate available for each month.

NIFTY's estimated annual drift was 9.62% (approximate 95% GBM confidence interval 0.43% to 18.81%) and volatility was 20.19%. At risk aversion 3, the full-sample unconstrained Merton fraction was 27.12%. The wide drift interval illustrates the uncertainty in using historical returns to estimate the model input. Sample skewness was −0.33 and excess kurtosis was 16.07. The Jarque–Bera statistic was 50,247.65, with a p-value below numerical reporting precision, indicating substantial departures from normality.

Sources and verification details are recorded in data/raw/P2_README.md. See `results/p2_params.csv` for estimates for all four assets and `figures/p2_rolling_^NSEI.png` for the rolling NIFTY estimates.
