# Quantitative Strategy Backtester

A modular Python backtesting framework built as a **Data Science / Data Analytics project** using financial market data as the application domain.

## What It Does

- Loads historical market data through a **Yahoo Finance → SQLite ETL pipeline**
- Supports multiple pluggable strategies:
  - SMA Crossover
  - RSI Mean Reversion
  - ML Predictor (Random Forest)
- Supports **long/short positions**
- Calculates:
  - Returns
  - CAGR
  - Sharpe Ratio
  - Maximum Drawdown
  - Market Exposure
  - Transaction Costs
  - Short Borrow Costs
- Includes:
  - Statistical significance testing
  - Multi-asset portfolio analysis
  - Chronological walk-forward validation
  - Interactive Streamlit dashboard

## Architecture

```text
Yahoo Finance
      ↓
   ETL Pipeline
      ↓
 SQLite Database
      ↓
   SQL + Pandas
      ↓
 Strategy Engine
      ↓
 Metrics / Validation
      ↓
 Streamlit Dashboard

 # Disclaimer
 
 This project is intended for **educational and research purposes only.
 
 Historical backtest performance does not guarantee future results. Backtests can be affected by assumptions, transaction costs, data quality, parameter selection, survivorship bias, execution assumptions, and other forms of model risk.
 
 Nothing in this repository constitutes financial or investment advice.
