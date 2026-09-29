# Quantitative Strategy Backtester

A modular Python backtesting framework for evaluating trading strategies on historical market data.

The project evolved from a single moving-average crossover strategy into a **pluggable strategy engine** supporting long/short positions, transaction costs, short borrow costs, statistical significance testing, multi-asset portfolio analysis, and chronological walk-forward validation.

Historical market data is stored in a local **SQLite data warehouse**, allowing backtests to run against local data without making Yahoo Finance API calls every time the application starts.

> **Note:** This project is primarily a **Data Science / Data Analytics project**. Financial markets are used as the dataset and application domain to demonstrate data ingestion, ETL, time-series analysis, statistical evaluation, validation, visualization, and modular software design. It is not investment advice.

---

## Features

### Strategy Engine

The backtester uses a common `BaseStrategy` interface, allowing strategies to be added without modifying the core backtesting pipeline.

Currently implemented:

* `BaseStrategy` — abstract strategy interface
* `SMACrossover` — moving-average trend-following strategy
* `RSIMeanReversion` — RSI-based mean-reversion strategy

The strategy layer handles:

* Signal generation
* Long positions
* Short positions
* Position changes
* Trade generation

### Performance Analysis

The framework calculates:

* Total Return
* CAGR
* Maximum Drawdown
* Sharpe Ratio
* Buy-and-Hold Return
* Market Exposure
* Trade Count
* Transaction Costs
* Short Borrow Costs

### Portfolio Analysis

The project also supports:

* Multi-asset backtesting
* Portfolio-level performance analysis
* Strategy-return correlation matrices
* Cross-ticker strategy comparison

### Statistical Testing

A randomized significance test compares the strategy's **gross, pre-cost Sharpe ratio** against 1,000 randomly timed strategies with matching long/short exposure.

This provides a statistical benchmark for determining whether the observed strategy performance is unusual relative to randomized exposure patterns.

### Walk-Forward Validation

Instead of selecting parameters using the entire dataset, the framework uses chronological walk-forward validation.

The current implementation:

1. Uses the first **60% of the data** for parameter selection.
2. Selects the strategy parameters on that training period.
3. Evaluates the selected parameters only on the remaining **40% held-out data**.

This helps reduce the risk of evaluating parameters on the same data used to select them.

### Interactive Visualization

A Streamlit application provides an interactive interface for:

* Selecting strategies
* Selecting tickers
* Running backtests
* Viewing performance metrics
* Comparing strategies
* Exploring portfolio results
* Visualizing cumulative returns and other results

---

# Architecture

```text
                         Yahoo Finance
                              │
                              ▼
                       ┌──────────────┐
                       │ ingestion.py │
                       │     ETL      │
                       └──────┬───────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ market_data.db  │
                     │     SQLite      │
                     └────────┬────────┘
                              │
                              ▼
                         ┌─────────┐
                         │ data.py │
                         │ SQL Read│
                         └────┬────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Strategy Engine  │
                    │                  │
                    │  BaseStrategy    │
                    │       │          │
                    │   ┌───┴────┐     │
                    │   │        │     │
                    │  SMA      RSI    │
                    │ Crossover  MR    │
                    └──────┬───────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  metrics.py  │
                    │              │
                    │ Returns      │
                    │ CAGR         │
                    │ Sharpe       │
                    │ Drawdown     │
                    │ Costs        │
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    ▼              ▼
              ┌───────────┐  ┌────────────┐
              │backtest.py│  │   app.py   │
              │    CLI    │  │  Streamlit │
              └───────────┘  └────────────┘
```

---

# Strategy Behavior

Both currently implemented strategies are **always in the market**.

There is no cash/flat position.

A strategy is either:

* `+1` → Long
* `-1` → Short

## SMA Crossover

The `SMACrossover` strategy compares a short-term and long-term simple moving average.

```text
SMA(short) > SMA(long)
        │
        ▼
      Long

SMA(short) <= SMA(long)
        │
        ▼
      Short
```

## RSI Mean Reversion

The `RSIMeanReversion` strategy uses RSI thresholds:

```text
RSI < Oversold
      │
      ▼
    Long


Oversold <= RSI <= Overbought
      │
      ▼
 Hold Previous Position


RSI > Overbought
      │
      ▼
    Short
```

### Why is Market Exposure close to 100%?

Because both strategies are always either long or short.

Therefore:

```text
Long exposure + Short exposure ≈ 100%
```

after the initial indicator lookback period.

This is **intentional behavior**, not a bug.

Market exposure in this implementation therefore represents the proportion of the backtest spent holding a directional position, rather than the proportion of time invested versus sitting in cash as in a traditional long-only strategy.

---

# Data Pipeline

The project uses an ETL pipeline to move historical market data from Yahoo Finance into a local relational database.

```text
Extract
   │
   ▼
Yahoo Finance
   │
   ▼
Transform
   │
   ▼
Normalize / Validate Data
   │
   ▼
Load
   │
   ▼
SQLite
```

## Why SQLite?

Instead of downloading historical data every time the application runs:

```text
Yahoo Finance API
       │
       ▼
   Every Run
```

the project uses:

```text
Yahoo Finance
       │
       ▼
 ingestion.py
       │
       ▼
 market_data.db
       │
       ▼
    data.py
       │
       ▼
   Backtester
```

Once the database has been populated, backtests read historical data locally using SQL.

This provides:

* Faster repeated backtests
* Reproducible datasets
* Reduced API dependency
* SQL-based filtering
* A realistic data engineering workflow

---

# ETL Pipeline

`ingestion.py` is responsible for:

1. Extracting historical market data from Yahoo Finance.
2. Transforming it into the project's normalized schema.
3. Loading it into SQLite.
4. Preventing duplicate records.
5. Updating previously stored prices when Yahoo Finance revises historical data.

The database uses a composite key to prevent duplicate ticker/date records.

Re-running ingestion is therefore safe.

```text
INSERT OR REPLACE
```

is used so previously stored prices can also be corrected when historical data is revised, for example after a stock split.

### Updating the Database

`ingestion.py` must currently be run manually to retrieve new trading days.

There is no automated scheduler yet.

---

# Performance Metrics

## Total Return

Measures the overall percentage return generated during the backtest.

## CAGR

Compound Annual Growth Rate is calculated using the **actual elapsed calendar time** between the first and last dates.

The implementation uses:

```text
Years = elapsed_days / 365.25
```

rather than estimating years from the number of rows.

## Maximum Drawdown

Measures the largest peak-to-trough decline in cumulative portfolio value.

```text
Drawdown = Current Value / Running Peak - 1
```

The minimum drawdown during the period is the Maximum Drawdown.

## Sharpe Ratio

Sharpe Ratio measures risk-adjusted return.

The implementation uses:

```text
Annual Risk-Free Rate = 3%
Trading Days / Year = 252
```

and calculates Sharpe using excess daily returns.

## Transaction Costs

Transaction costs are applied whenever the strategy changes position.

They affect the net strategy return.

## Short Borrow Costs

Short positions incur a daily borrow cost based on an annualized borrow rate.

The default is:

```text
3% annualized
```

The model assumes a constant borrow rate.

Real-world borrow costs can vary by security and over time, so this is a simplified model.

---

# Statistical Significance Test

The project includes a randomized significance test implemented in:

```text
significance.py
```

The test compares the strategy's **gross, pre-cost Sharpe ratio** against:

```text
1,000 randomized strategies
```

The randomized strategies maintain matching long/short exposure while randomizing the timing of positions.

This provides a reference distribution against which the observed strategy Sharpe can be compared.

> The significance test uses **gross, pre-cost returns**, while the main performance metrics display **net-of-cost returns**. Therefore, the Sharpe ratios shown by the two components can differ slightly.

---

# Walk-Forward Validation

An earlier version of the project evaluated multiple SMA parameter combinations across the entire dataset and then reported performance on that same dataset.

That approach can introduce **data snooping / in-sample parameter selection bias**.

The current implementation instead uses chronological walk-forward validation.

```text
Historical Data
│
├─────────────── 60% ───────────────┬────────── 40% ──────────┤
│                                   │                          │
│           Training                │          Test            │
│                                   │                          │
│ Parameter Selection               │ Final Evaluation         │
│                                   │                          │
└───────────────────────────────────┴──────────────────────────┘
```

The parameters are selected using only the first 60% of observations.

The selected parameters are then evaluated on the remaining 40%.

This mirrors the structure of many real-world time-series problems where future observations must not influence model development.

---

# Portfolio Analysis

The framework supports running strategies across multiple assets.

Portfolio analysis includes:

* Individual asset strategy returns
* Portfolio performance
* Cross-strategy correlation
* Strategy-return comparison

The correlation matrix is calculated using **strategy returns**, rather than raw asset returns.

This means the matrix measures the overlap between strategy return streams and signal timing.

It should therefore **not** be interpreted as a correlation matrix of the underlying securities themselves.

---

# Project Structure

```text
tradingproject/
│
├── app.py                 # Streamlit application
├── backtest.py            # Command-line backtest runner
│
├── base_strategy.py       # BaseStrategy abstract interface
├── strategy.py            # SMACrossover and RSIMeanReversion
│
├── metrics.py             # Returns, CAGR, Sharpe, drawdown and costs
├── significance.py        # Randomized statistical significance test
│
├── data.py                # SQL read layer
├── ingestion.py           # Yahoo Finance → SQLite ETL pipeline
│
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── market_data.db         # Local SQLite data warehouse
```

> `market_data.db` is generated locally by `ingestion.py` and may be excluded from Git depending on the project's data-management strategy.

---

# Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd tradingproject
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## 1. Populate the Database

Run:

```bash
python ingestion.py
```

This creates/populates:

```text
market_data.db
```

and stores the historical market data required by the backtester.

Run the ingestion script again whenever you want to update the database with new trading days.

---

## 2. Run the Command-Line Backtester

```bash
python backtest.py
```

The command-line runner supports:

* Single-ticker backtests
* Multi-asset portfolio analysis
* Walk-forward validation
* Performance metrics
* Statistical analysis

---

## 3. Launch the Streamlit Application

```bash
streamlit run app.py
```

The application provides an interactive interface for exploring strategies, assets, and performance results.

---

# Technology Stack

| Technology              | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| **Python**              | Core programming language                  |
| **Pandas**              | Data manipulation and time-series analysis |
| **NumPy**               | Numerical computation                      |
| **yfinance**            | Historical market data ingestion           |
| **SQLite**              | Local relational data warehouse            |
| **SQL**                 | Local data querying and filtering          |
| **Matplotlib / Plotly** | Data visualization                         |
| **Streamlit**           | Interactive analytics interface            |

---

# Data Science & Data Analytics Skills Demonstrated

Although the project uses financial data, the underlying skills are broadly applicable to Data Science and Data Analytics.

### Data Engineering

```text
API
 ↓
Extract
 ↓
Transform
 ↓
Validate
 ↓
Load
 ↓
Relational Database
```

The ETL pipeline demonstrates how external data can be collected, normalized, validated, and stored in a structured database.

The same architecture can be applied to datasets such as:

* Customer transactions
* E-commerce events
* IoT measurements
* Application logs
* Fraud records
* Customer activity data

### SQL & Relational Data

The project uses SQLite to:

* Store structured historical data
* Query specific tickers
* Filter by date ranges
* Prevent duplicate records
* Separate the data layer from the analysis layer

### Time-Series Analysis

The project works with:

* Chronological data
* Rolling windows
* Lagged variables
* Returns
* Drawdowns
* Time-based filtering
* Sequential validation

These techniques also apply to forecasting, demand prediction, anomaly detection, and other time-series problems.

### Statistical Analysis

The project demonstrates:

* Sharpe ratio calculation
* Randomized statistical testing
* Return distributions
* Correlation analysis
* Benchmark comparison

### Model Validation

Walk-forward validation demonstrates how chronological train/test splits can prevent future information from leaking into model development.

This is directly applicable to time-series forecasting and other sequential prediction problems.

### Data Visualization

The Streamlit application turns analytical outputs into an interactive interface, allowing users to explore:

* Performance
* Returns
* Drawdowns
* Strategy behavior
* Portfolio relationships

### Software Engineering

The project also demonstrates:

* Object-oriented design
* Abstract base classes
* Modular architecture
* Separation of concerns
* Reusable strategy interfaces
* Database abstraction
* CLI execution
* Interactive application development

---

# Design Decisions & Limitations

## No Full-Sample Parameter Grid

The project intentionally avoids selecting strategy parameters using the entire dataset.

Instead, parameter selection is performed on the training portion of the data and evaluated on a later held-out period.

This reduces, but does not completely eliminate, the risk of overfitting.

## Historical Data Revisions

Yahoo Finance data may contain gaps or revisions.

The ingestion process uses `INSERT OR REPLACE` so previously stored observations can be updated when historical prices are revised.

## Manual Data Updates

There is currently no scheduler.

The user must manually run:

```bash
python ingestion.py
```

to retrieve newly available trading days.

## Simplified Borrow Cost Model

Short borrow cost is modeled as a constant annualized rate.

```text
Default Borrow Rate = 3%
```

Actual securities can have security-specific and time-varying borrow rates.

## Simplified Transaction Costs

Transaction costs are modeled using a configurable fixed rate rather than a full market-impact or execution model.

## Sharpe Assumptions

Sharpe calculations use:

```text
Risk-Free Rate = 3% annually
Trading Days = 252 per year
```

These are configurable modeling assumptions rather than universal constants.

## Data Dependency

Historical prices ultimately originate from Yahoo Finance.

Data availability, missing observations, and historical revisions can affect results.

---

# Key Learning Outcomes

This project demonstrates a complete analytical workflow:

```text
External Data Source
        ↓
      ETL
        ↓
 Relational Storage
        ↓
    SQL Query
        ↓
Data Transformation
        ↓
 Strategy / Analysis
        ↓
 Statistical Evaluation
        ↓
 Validation
        ↓
 Visualization
        ↓
 Interactive Application
```

The core goal is not simply to build a trading strategy.

It is to demonstrate how a **real-world data project can move from raw external data to a reproducible analytical system**.

---

# Future Improvements

Potential extensions include:

* Automated database updates
* More strategy implementations
* Additional statistical tests
* Better portfolio allocation methods
* Risk-adjusted position sizing
* Parameter stability analysis
* More rigorous walk-forward evaluation
* Benchmark comparisons
* Data-quality monitoring
* Unit and integration tests
* Logging and error monitoring
* Configuration files for backtest parameters
* Dockerized deployment
* Cloud database support
* CI/CD pipeline
* Interactive experiment tracking

---

# Disclaimer

This project is intended for **educational and research purposes only**.

Historical backtest performance does not guarantee future results. Backtests can be affected by assumptions, transaction costs, data quality, parameter selection, survivorship bias, execution assumptions, and other forms of model risk.

Nothing in this repository constitutes financial or investment advice.

---

## License

Add your preferred license here, for example:

```text
MIT License
```

See `LICENSE` for details.
