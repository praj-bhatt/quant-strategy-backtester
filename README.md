# Moving Average Crossover Backtester

A Python-based backtesting project that evaluates a moving average crossover trading strategy against a buy-and-hold benchmark.

## Features

- Historical market data using Yahoo Finance
- Configurable short and long moving averages
- Buy/sell signal generation
- Look-ahead bias prevention
- Transaction costs
- Total return calculation
- CAGR
- Maximum Drawdown (MDD)
- Sharpe Ratio
- Market exposure
- Interactive Plotly visualizations
- Buy and sell signal visualization

## Strategy

The default strategy uses a 50-day SMA and a 200-day SMA.

- **Buy/Long:** SMA 50 > SMA 200
- **Exit:** SMA 50 <= SMA 200
- Signals are shifted by one trading day to avoid look-ahead bias.

## Default Benchmark

The default ticker is the S&P 500 index (`^GSPC`) using approximately 5 years of historical data.

## Project Structure

```text
tradingproject/
├── backtest.py
├── data.py
├── strategy.py
├── metrics.py
├── requirements.txt
└── README.md
