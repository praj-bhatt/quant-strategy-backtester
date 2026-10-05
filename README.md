# Quantitative Strategy Backtester

A modular backtesting framework built as a Data Science / Data Analytics project — financial markets are the dataset, used to demonstrate ETL, SQL, time-series analysis, statistical validation, and machine learning, not to produce a trading strategy for real use.

## What it does

- Ingests historical price and volume data via a Yahoo Finance → SQLite ETL pipeline (`ingestion.py`), so backtests read locally instead of hitting the API every run
- Three pluggable strategies, sharing one `BaseStrategy` interface: `SMACrossover`, `RSIMeanReversion`, and `MLPredictor` (a walk-forward-retrained Random Forest)
- Long and short positions, transaction costs, and short borrow costs
- Metrics: total return, CAGR (from actual elapsed calendar time), Sharpe ratio, max drawdown, market exposure
- A randomized significance test comparing each strategy's Sharpe against 1,000 exposure-matched random strategies
- Multi-asset portfolio backtesting with a strategy-return correlation matrix
- Chronological walk-forward validation (parameters selected on a training slice, evaluated only on a held-out test slice)
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) for the ML strategy specifically
- An interactive Streamlit dashboard

## Architecture

```text
Yahoo Finance → ingestion.py (ETL) → market_data.db (SQLite) → data.py (SQL read)
    → strategy classes → metrics.py → backtest.py (CLI) / app.py (Streamlit)
```

## How to run it

```bash
pip install -r requirements.txt

python ingestion.py        # one-time setup: builds market_data.db (rerun periodically for new days)
python backtest.py         # CLI: single-ticker, portfolio, and walk-forward backtests
streamlit run app.py       # interactive app
pytest tests/              # test suite
```

## A note on the strategies

All three strategies are always in the market — long or short, never flat. "Market Exposure" reads near 100% by design, not as a bug.

## Key design decisions

- **Walk-forward validation, not a full-sample parameter grid** — an earlier version picked the best SMA window pair by Sharpe over the same data it then reported on, which is data snooping. Parameters are now selected on the first 60% of data and evaluated only on the held-out remaining 40%.
- **CAGR uses actual elapsed calendar time**, not row count.
- **The significance test compares gross (pre-cost) Sharpe**, so it can read slightly differently from the net-of-cost Sharpe shown elsewhere.
- **The ML strategy's features are leakage-free by construction**: each feature uses only data through day *t*, matched to a same-day target built by shifting tomorrow's return backward — no extra lag is added on top, and this is covered by a dedicated test that inspects what each retrained model actually saw.
- **Portfolio correlation is computed on strategy returns**, not raw asset returns — it reflects overlap in signal timing, not how correlated the underlying securities are.
- `ingestion.py` is run manually; there's no scheduler. Short borrow cost is a flat 3% annualized rate, not a real variable broker rate.

## Findings (5-year window, ^GSPC, as of testing)

None of the three strategies beat buy-and-hold or showed a statistically significant edge over randomly-timed exposure. The ML Predictor's accuracy came in below the majority-class baseline, and its ROC-AUC was below 0.5. Three independently-built strategies, three different methodologies, the same honest result — consistent with the well-documented difficulty of beating a trending market with simple rules or a baseline classifier.

## Stack

Python, pandas, NumPy, yfinance, SQLite, scikit-learn, Plotly, Streamlit, pytest.

## Disclaimer

Educational and research use only. Not investment advice. Backtest results are affected by assumptions, data quality, transaction costs, parameter selection, and other forms of model risk.
