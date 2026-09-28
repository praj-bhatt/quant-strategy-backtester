# Quantitative Strategy Backtester

A Python backtesting framework for evaluating trading strategies on historical market data. The project has evolved from a single moving-average strategy into a pluggable strategy engine with long/short positions, transaction costs, short borrow costs, statistical testing, multi-asset portfolio analysis, walk-forward validation, and a Streamlit interface.

The project uses a local SQLite data warehouse so that backtests read historical data locally instead of making live Yahoo Finance API calls every time the application runs.

---

## 1. What the Project Is

This project is a modular backtester built around a common `BaseStrategy` interface.

The current strategy engine includes:

- `SMACrossover`
- `RSIMeanReversion`
- `BaseStrategy` as the abstract base class

Both strategies generate trading signals that are converted into positions and trades by the base strategy layer.

The backtester supports:

- Long positions
- Short positions
- Transaction costs
- Short borrow costs
- Total return
- CAGR
- Maximum Drawdown
- Sharpe Ratio
- Buy-and-hold comparison
- Market exposure
- Trade counts
- Multi-asset portfolio analysis
- Strategy-return correlation analysis
- A randomized statistical significance test
- Chronological walk-forward validation
- Interactive Streamlit visualization

The project is primarily intended as a data science and data analytics project. The financial market is the dataset/domain used to demonstrate data ingestion, transformation, time-series analysis, statistical evaluation, validation, visualization, and software design.

Historical market data is stored locally in SQLite. Once the database has been populated, `data.py` retrieves the requested ticker and period using SQL rather than downloading the data again.

---

## 2. Architecture

The current data flow is:

```text
Yahoo Finance
      │
      ▼
ingestion.py
(Extract → Transform → Load)
      │
      ▼
market_data.db
(SQLite data warehouse)
      │
      ▼
data.py
(SQL read + period filtering)
      │
      ▼
Strategy Classes
(BaseStrategy
 ├── SMACrossover
 └── RSIMeanReversion)
      │
      ▼
metrics.py
(Returns, CAGR, Sharpe, MDD,
 transaction costs, borrow costs)
      │
      ├──────────────► backtest.py
      │
      └──────────────► app.py
                       (Streamlit UI)
