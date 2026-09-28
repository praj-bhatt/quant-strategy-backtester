# Moving Average Crossover Backtester

A Python backtester that tests a moving-average crossover strategy against buy-and-hold, and checks whether the result is distinguishable from luck. It has a command-line runner and an interactive Streamlit app.

The point of the project is not to find a profitable strategy. It is to build the evaluation carefully: no look-ahead bias, transaction costs, out-of-sample validation, and a significance test.

## Features

- Daily adjusted-close data from Yahoo Finance (`yfinance`), so dividends and splits are accounted for
- Configurable short and long SMA windows (default 50/200)
- Signals lagged by one trading day to prevent look-ahead bias
- Flat transaction cost applied on every position change (default 0.05%)
- Metrics: total return, CAGR, maximum drawdown, Sharpe ratio, trade count, market exposure, each compared with buy-and-hold
- Statistical significance test: the strategy's Sharpe ratio is compared with 1,000 randomly timed strategies that hold the same number of days (seeded, so results are reproducible)
- Multi-asset run: the same strategy across several tickers, with an equal-weight portfolio and a correlation matrix
- Walk-forward validation: the window pair is chosen on a chronological training period and reported only on the held-out test period
- Streamlit app with sliders for ticker, history, SMA windows and transaction cost

## Strategy

- **Long:** short SMA is above long SMA
- **Cash:** short SMA is at or below long SMA
- The signal from day *t* is traded on day *t+1*.

## How to run

```bash
pip install -r requirements.txt

python backtest.py        # command-line run: prints metrics and opens Plotly charts
streamlit run app.py      # interactive app
```

`backtest.py` runs three things in order: the single-ticker backtest with the significance test, the multi-asset portfolio (`^GSPC`, `GLD`, `XLK`, `XLF`, `XLE`), and the walk-forward validation.

## Design decisions

**No parameter grid on the full sample.** An early version compared several window pairs and ranked them by Sharpe ratio over the same 5 years it then reported. That is data snooping: the "best" pair is the one that fit the noise in that window. I removed it and replaced it with walk-forward validation, where the pair is selected on the first 60% of the data and evaluated only on the remaining 40%.

**Lookback handling in the test period.** The long SMA needs history before the test period begins. The test run borrows that history from the end of the training period to compute the signals, then filters down to test-period dates before computing returns, so no training-period returns leak into the out-of-sample numbers.

**CAGR uses actual elapsed calendar time** (first date to last date, in years of 365.25 days) rather than assuming the number of rows equals the number of trading days.

## Findings (^GSPC, 5-year window, September 2026)

Results change as the window moves forward, so treat these as a snapshot.

- The 50/200 strategy trailed buy-and-hold on total return, CAGR and Sharpe ratio. It made only 3 trades and was invested about 81% of the time.
- Its maximum drawdown was identical to buy-and-hold's, because it was fully invested through the worst decline.
- The significance test found the strategy's Sharpe ratio not distinguishable from randomly timed strategies with the same exposure.
- In walk-forward validation, the 50/200 pair was selected on the training period and also trailed buy-and-hold on the held-out test period.

A plausible explanation is that trend-following crossovers tend to lag in a sustained bull market. This project does not test that explanation.

## Known limitations

- One 5-year window that is mostly a rising market. Nothing here shows how the strategy behaves in other regimes.
- The walk-forward test is a single train/test split over three candidate window pairs, not a rolling re-optimization.
- The significance test compares gross (pre-cost) Sharpe ratios. That is why it can differ slightly from the net-of-cost Sharpe shown in the metrics.
- The correlation matrix in the portfolio run is computed on strategy returns, which are zero on days the strategy is out of the market. It reflects overlap in signal timing, not the correlation of the underlying assets.
- In the portfolio run, each ticker's printed metrics use that ticker's full history, while the combined chart uses the dates all tickers share. The portfolio is equal-weighted with no rebalancing costs beyond the per-trade cost.
- Sharpe ratios use a 3% annual risk-free rate and 252 trading days per year.
- Yahoo Finance data can have gaps or revisions. This is not investment advice.

## Project structure

```text
tradingproject/
├── app.py            # Streamlit app
├── backtest.py       # command-line runner: single, portfolio and walk-forward backtests
├── data.py           # data download
├── strategy.py       # SMA signals, positions and trades
├── metrics.py        # returns, CAGR, drawdown, Sharpe
├── significance.py   # random-exposure significance test
├── requirements.txt
└── README.md
```