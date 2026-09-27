import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"

from data import get_data
from metrics import (
    calculate_max_drawdown,
    calculate_metrics,
    calculate_sharpe_ratio,
)
from significance import run_significance_test
from strategy import run_strategy


def validate_parameters(short_window, long_window, transaction_cost):
    if short_window <= 0:
        raise ValueError("short_window must be greater than 0")

    if long_window <= 0:
        raise ValueError("long_window must be greater than 0")

    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")


def run_backtest(
    ticker="^GSPC",
    period="5y",
    short_window=50,
    long_window=200,
    transaction_cost=0.0005,
):
    validate_parameters(short_window, long_window, transaction_cost)

    print(f"Downloading data for {ticker}...")

    data = get_data(ticker, period)

    data = run_strategy(data, short_window, long_window)

    metrics, data = calculate_metrics(data, transaction_cost)

    significance = run_significance_test(data)

    print("\n--- Statistical Significance ---")
    print(f"Random Sharpe Mean:       {significance['random_sharpe_mean']:.2f}")
    print(f"Random Sharpe Std:        {significance['random_sharpe_std']:.2f}")
    print(
        f"Actual Sharpe Percentile: "
        f"{significance['actual_sharpe_percentile']:.2f}%"
    )
    print(f"Verdict:                  {significance['verdict']}")

    print("\n--- Strategy Performance ---")
    print(f"Total Return (Buy & Hold): {metrics['buy_hold_return'] * 100:.2f}%")
    print(f"Total Return (Strategy):   {metrics['strategy_return'] * 100:.2f}%")
    print(f"Annualized Return (CAGR):  {metrics['annualized_return'] * 100:.2f}%")
    print(f"Buy & Hold CAGR:           {metrics['buy_hold_cagr'] * 100:.2f}%")
    print(f"Strategy MDD:              {metrics['max_drawdown'] * 100:.2f}%")
    print(
        f"Buy & Hold MDD:            "
        f"{metrics['buy_hold_max_drawdown'] * 100:.2f}%"
    )
    print(f"Annualized Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
    print(
        f"Buy & Hold Sharpe Ratio:   "
        f"{metrics['buy_hold_sharpe_ratio']:.2f}"
    )
    print(f"Total Trades:              {metrics['total_trades']:.0f}")
    print(f"Invested Days:             {metrics['invested_days']:.0f}")
    print(f"Market Exposure:           {metrics['exposure'] * 100:.2f}%")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Cum_Asset_Return"],
            mode="lines",
            name="Buy & Hold",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Cum_Strategy_Return"],
            mode="lines",
            name=f"{short_window}/{long_window} MA Strategy",
        )
    )

    fig.update_layout(
        title=(
            f"{short_window}/{long_window} Moving Average "
            f"Crossover Strategy vs {ticker}"
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    fig.show()

    price_fig = go.Figure()

    price_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Adj Close"],
            mode="lines",
            name="Price",
        )
    )

    price_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[f"SMA_{short_window}"],
            mode="lines",
            name=f"SMA {short_window}",
        )
    )

    price_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[f"SMA_{long_window}"],
            mode="lines",
            name=f"SMA {long_window}",
        )
    )

    buy_signals = data[data["Buy"] == 1]
    sell_signals = data[data["Sell"] == 1]

    price_fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Adj Close"],
            mode="markers",
            name="Buy",
        )
    )

    price_fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Adj Close"],
            mode="markers",
            name="Sell",
        )
    )

    price_fig.update_layout(
        title=f"{ticker} Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    price_fig.show()

    return data, fig, price_fig


def run_portfolio_backtest(
    tickers,
    period="5y",
    short_window=50,
    long_window=200,
    transaction_cost=0.0005,
):
    validate_parameters(short_window, long_window, transaction_cost)

    strategy_returns = {}
    ticker_metrics = {}

    for ticker in tickers:
        print(f"\nDownloading data for {ticker}...")

        data = get_data(ticker, period)

        data = run_strategy(
            data,
            short_window,
            long_window,
        )

        metrics, data = calculate_metrics(
            data,
            transaction_cost,
        )

        strategy_returns[ticker] = data["Net_Strategy_Return"]
        ticker_metrics[ticker] = metrics

    returns_data = pd.concat(
        strategy_returns,
        axis=1,
        join="inner",
    )

    portfolio_returns = returns_data.mean(axis=1)

    portfolio_cumulative_return = (
        1 + portfolio_returns
    ).cumprod()

    portfolio_max_drawdown = calculate_max_drawdown(
        portfolio_cumulative_return
    )

    portfolio_sharpe = calculate_sharpe_ratio(
        portfolio_returns
    )

    correlation_matrix = returns_data.corr()

    print("\n--- Portfolio Performance ---")

    print("\n--- Individual Ticker Metrics ---")

    for ticker in tickers:
        metrics = ticker_metrics[ticker]

        print(f"\n{ticker}")
        print(
            f"Total Return: {metrics['strategy_return'] * 100:.2f}%"
        )
        print(
            f"CAGR:         {metrics['annualized_return'] * 100:.2f}%"
        )
        print(
            f"MDD:          {metrics['max_drawdown'] * 100:.2f}%"
        )
        print(
            f"Sharpe:       {metrics['sharpe_ratio']:.2f}"
        )
        print(
            f"Total Trades: {metrics['total_trades']:.0f}"
        )
        print(
            f"Exposure:     {metrics['exposure'] * 100:.2f}%"
        )

    print("\n--- Portfolio Metrics ---")

    portfolio_return = (
        portfolio_cumulative_return.iloc[-1] - 1
    )

    print(
        f"Portfolio Return: {portfolio_return * 100:.2f}%"
    )
    print(
        f"Portfolio MDD:    {portfolio_max_drawdown * 100:.2f}%"
    )
    print(
        f"Portfolio Sharpe: {portfolio_sharpe:.2f}"
    )

    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)

    portfolio_fig = go.Figure()

    for ticker in returns_data.columns:
        ticker_cumulative_return = (
            1 + returns_data[ticker]
        ).cumprod()

        portfolio_fig.add_trace(
            go.Scatter(
                x=ticker_cumulative_return.index,
                y=ticker_cumulative_return,
                mode="lines",
                name=ticker,
            )
        )

    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_cumulative_return.index,
            y=portfolio_cumulative_return,
            mode="lines",
            name="Equal-Weight Portfolio",
            line={"width": 4},
        )
    )

    portfolio_fig.update_layout(
        title="Individual Strategies vs Equal-Weight Portfolio",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    portfolio_fig.show()

    return (
        ticker_metrics,
        portfolio_returns,
        portfolio_cumulative_return,
        correlation_matrix,
        portfolio_fig,
    )


def run_walkforward_backtest(
    ticker="^GSPC",
    period="5y",
    candidate_windows=None,
    transaction_cost=0.0005,
    train_fraction=0.6,
):
    if candidate_windows is None:
        candidate_windows = [(20, 50), (50, 100), (50, 200)]

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not candidate_windows:
        raise ValueError("candidate_windows cannot be empty")

    for short_window, long_window in candidate_windows:
        validate_parameters(
            short_window,
            long_window,
            transaction_cost,
        )

    print(
        f"\nDownloading data for walk-forward validation: "
        f"{ticker}..."
    )

    full_data = get_data(ticker, period)

    split_index = int(len(full_data) * train_fraction)

    train_data = full_data.iloc[:split_index].copy()
    test_data = full_data.iloc[split_index:].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "train_fraction produced an empty training or test dataset"
        )

    print("\n--- Chronological Train/Test Split ---")
    print(
        f"Training period: "
        f"{train_data.index[0].date()} "
        f"to {train_data.index[-1].date()}"
    )
    print(
        f"Test period:     "
        f"{test_data.index[0].date()} "
        f"to {test_data.index[-1].date()}"
    )

    print(
        "\n--- In-Sample / Training Results --- "
        "(for selection only, not a performance claim)"
    )

    best_sharpe = -float("inf")
    selected_short_window = None
    selected_long_window = None

    for short_window, long_window in candidate_windows:
        train_strategy_data = run_strategy(
            train_data,
            short_window,
            long_window,
        )

        train_metrics, _ = calculate_metrics(
            train_strategy_data,
            transaction_cost,
        )

        print(
            f"{short_window}/{long_window} | "
            f"Training Sharpe: "
            f"{train_metrics['sharpe_ratio']:.2f} | "
            f"Training Return: "
            f"{train_metrics['strategy_return'] * 100:.2f}% | "
            f"Training MDD: "
            f"{train_metrics['max_drawdown'] * 100:.2f}%"
        )

        if train_metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = train_metrics["sharpe_ratio"]
            selected_short_window = short_window
            selected_long_window = long_window

    print(
        f"\nSelected window pair: "
        f"{selected_short_window}/{selected_long_window}"
    )
    print(
        f"Selection criterion: "
        f"highest training Sharpe ({best_sharpe:.2f})"
    )

    lookback_start = max(
        0,
        split_index - selected_long_window,
    )

    test_data_with_lookback = full_data.iloc[
        lookback_start:
    ].copy()

    test_strategy_data = run_strategy(
        test_data_with_lookback,
        selected_short_window,
        selected_long_window,
    )

    test_strategy_data = test_strategy_data[
        test_strategy_data.index >= test_data.index[0]
    ].copy()

    test_metrics, test_strategy_data = calculate_metrics(
        test_strategy_data,
        transaction_cost,
    )

    print("\n--- Out-of-Sample Test Results (the real result) ---")
    print(
        f"Selected Window Pair:    "
        f"{selected_short_window}/{selected_long_window}"
    )
    print(
        f"Test Total Return:       "
        f"{test_metrics['strategy_return'] * 100:.2f}%"
    )
    print(
        f"Test CAGR:               "
        f"{test_metrics['annualized_return'] * 100:.2f}%"
    )
    print(
        f"Test Maximum Drawdown:   "
        f"{test_metrics['max_drawdown'] * 100:.2f}%"
    )
    print(
        f"Test Sharpe Ratio:       "
        f"{test_metrics['sharpe_ratio']:.2f}"
    )
    print(
        f"Test Total Trades:       "
        f"{test_metrics['total_trades']:.0f}"
    )
    print(
        f"Test Market Exposure:    "
        f"{test_metrics['exposure'] * 100:.2f}%"
    )
    print(
        f"Test Buy & Hold Return:  "
        f"{test_metrics['buy_hold_return'] * 100:.2f}%"
    )
    print(
        f"Test Buy & Hold CAGR:    "
        f"{test_metrics['buy_hold_cagr'] * 100:.2f}%"
    )
    print(
        f"Test Buy & Hold MDD:     "
        f"{test_metrics['buy_hold_max_drawdown'] * 100:.2f}%"
    )
    print(
        f"Test Buy & Hold Sharpe:  "
        f"{test_metrics['buy_hold_sharpe_ratio']:.2f}"
    )

    test_fig = go.Figure()

    test_fig.add_trace(
        go.Scatter(
            x=test_strategy_data.index,
            y=test_strategy_data["Cum_Asset_Return"],
            mode="lines",
            name="Buy & Hold",
        )
    )

    test_fig.add_trace(
        go.Scatter(
            x=test_strategy_data.index,
            y=test_strategy_data["Cum_Strategy_Return"],
            mode="lines",
            name=(
                f"{selected_short_window}/"
                f"{selected_long_window} MA Strategy"
            ),
        )
    )

    test_fig.update_layout(
        title=(
            f"Out-of-Sample Test: "
            f"{selected_short_window}/{selected_long_window} "
            f"MA Strategy vs {ticker}"
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    test_fig.show()

    return (
        selected_short_window,
        selected_long_window,
        test_metrics,
        test_strategy_data,
        test_fig,
    )


if __name__ == "__main__":
    data, fig, price_fig = run_backtest(
        ticker="^GSPC",
        period="5y",
        short_window=50,
        long_window=200,
        transaction_cost=0.0005,
    )

    run_portfolio_backtest(
        tickers=["^GSPC", "GLD", "XLK", "XLF", "XLE"],
        period="5y",
        short_window=50,
        long_window=200,
        transaction_cost=0.0005,
    )

    run_walkforward_backtest(
        ticker="^GSPC",
        period="5y",
        candidate_windows=[
            (20, 50),
            (50, 100),
            (50, 200),
        ],
        transaction_cost=0.0005,
        train_fraction=0.6,
    )
