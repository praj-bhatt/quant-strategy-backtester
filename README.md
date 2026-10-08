# Quantitative Strategy Backtester

A backtesting framework for trading strategies, built as a data science project. Market data is the dataset. The focus is the pipeline around it: ETL into SQL, time-series feature engineering, leakage-free validation, statistical testing, and a classifier evaluated with proper metrics. It is not meant to produce a strategy for real trading.

## What it does

- ETL pipeline from Yahoo Finance into SQLite (`ingestion.py`). Backtests read from the local database with SQL (`data.py`).
- Three strategies on a shared `BaseStrategy` interface: SMA crossover, RSI mean reversion, and a Random Forest predictor
- Long and short positions, with transaction costs and short borrow costs
- Metrics against buy-and-hold: return, CAGR, Sharpe ratio, max drawdown, exposure, trade count
- Significance test: the strategy's Sharpe ratio against 1,000 random strategies with the same number of long and short days
- Model evaluation tab for the ML strategy: accuracy against a majority-class baseline, precision, recall, F1, ROC-AUC, confusion matrix
- Streamlit dashboard
- CLI only (`backtest.py`): equal-weight multi-asset portfolio with a correlation matrix, and walk-forward validation of SMA window selection

## Architecture

```text
Yahoo Finance -> ingestion.py (ETL) -> market_data.db (SQLite) -> data.py (SQL)
    -> strategies -> metrics.py -> app.py (Streamlit) / backtest.py (CLI)
```

## Setup and usage

```bash
pip install -r requirements.txt

python ingestion.py       # builds market_data.db; rerun to pick up new trading days
streamlit run app.py      # dashboard
python backtest.py        # CLI: single-ticker, portfolio and walk-forward runs
pytest tests/             # tests
```

The app reads from the local database, so it only works for tickers that `ingestion.py` has loaded: `^GSPC`, `GLD`, `XLK`, `XLF`, `XLE`. To add one, put it in the `tickers` list in `ingestion.py` and rerun.

## How the strategies work

- **SMA crossover:** long when the 50-day average is above the 200-day average, short otherwise.
- **RSI mean reversion:** long when RSI(14) falls below 30, short when it rises above 70, otherwise hold the previous position.
- **ML predictor:** a Random Forest predicts next-day direction from 1, 3 and 5-day returns, 10 and 20-day volatility, RSI, and a volume spike ratio. It retrains every 60 days on all earlier data (500-day minimum). Predicted up means long, predicted down means short.

Every signal is traded one day later. Once warmed up, each strategy is always long or short, never flat. During warmup there is no position: about 200 days for SMA, until the first signal for RSI, and 500 days for the ML predictor. Over 5 years that makes exposure close to 100% for SMA but about 60% for the ML predictor, while buy-and-hold is invested the whole time.

## Avoiding leakage and overfitting

- **Walk-forward instead of a full-sample parameter grid.** An early version picked the best SMA window pair by Sharpe on the same data it then reported, which is data snooping. Pairs are now chosen on the first 60% of the data and judged only on the remaining 40%.
- **ML features use data through day t only.** The target is the move from day t to day t+1. Each retrain sees only rows before the block it predicts, and a test records exactly which rows every fit received.
- **Reproducible randomness.** The significance test is seeded, and so is the Random Forest.
- **CAGR uses elapsed calendar time**, not the row count.

## Findings (^GSPC, 5 years, October 2026)

| Strategy | Return | Buy and hold | Sharpe | Significance percentile |
|---|---|---|---|---|
| SMA crossover 50/200 | TODO | TODO | TODO | TODO |
| RSI mean reversion | TODO | TODO | TODO | TODO |
| ML predictor | -49.21% | 76.81% | -1.36 | 8.0% |

The ML predictor's out-of-sample accuracy was 0.482 against a majority-class baseline of 0.563, and its ROC-AUC was 0.461, below chance. It traded 631 times. A classifier this close to a coin flip changes its mind often, and every flip pays costs.

This is one asset over one window that is mostly a rising market. It shows that these simple strategies and a baseline classifier did not beat holding the index here. It does not show that markets cannot be beaten.

## Limitations

- Walk-forward validation covers SMA window selection only. The ML strategy has its own retraining loop, and RSI has no walk-forward step.
- Short borrow cost is a flat 3% a year, not a real variable broker rate. Transaction cost is a flat rate per trade, with no slippage model.
- The significance test compares gross (pre-cost) Sharpe ratios, so it can differ slightly from the net Sharpe shown elsewhere.
- The portfolio correlation matrix uses strategy returns, so it reflects overlap in signal timing, not how correlated the underlying assets are.
- Sharpe ratios assume a 3% risk-free rate and 252 trading days a year.
- `ingestion.py` runs manually. There is no scheduler.
- Yahoo Finance data can have gaps or revisions.

## Stack

Python, pandas, NumPy, SQLite, yfinance, scikit-learn, Plotly, Streamlit, pytest.

## Disclaimer

Educational and research use only. Not investment advice. Backtest results depend on assumptions, data quality, costs and parameter choices.
