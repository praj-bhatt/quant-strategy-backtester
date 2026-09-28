import plotly.graph_objects as go
import streamlit as st

from data import get_data
from metrics import calculate_metrics
from significance import run_significance_test
from strategy import run_strategy

st.set_page_config(page_title="MA Crossover Backtester", layout="wide")

NUM_SIMULATIONS = 1000


@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def load_data(ticker, period):
    return get_data(ticker, period)


@st.cache_data(show_spinner="Running significance test...")
def cached_significance(data):
    return run_significance_test(data, NUM_SIMULATIONS)


st.title("Moving Average Crossover Backtester")
st.caption(
    "Long when the short SMA is above the long SMA, otherwise in cash. "
    "Signals are lagged one trading day to avoid look-ahead bias."
)

with st.sidebar:
    st.header("Parameters")
    ticker = (
        st.text_input(
            "Ticker",
            value="^GSPC",
            help="Any Yahoo Finance symbol, e.g. ^GSPC, ^NSEI, AAPL, GLD",
        )
        .strip()
        .upper()
    )
    period = st.selectbox("History", ["3y", "5y", "10y"], index=1)
    short_window = st.slider("Short SMA window (days)", 5, 100, 50)
    long_window = st.slider("Long SMA window (days)", 20, 300, 200)
    cost_pct = st.slider(
        "Transaction cost per trade (%)", 0.00, 0.50, 0.05, step=0.01
    )

transaction_cost = cost_pct / 100

if not ticker:
    st.info("Enter a ticker in the sidebar.")
    st.stop()

if short_window >= long_window:
    st.error("Short window must be smaller than the long window.")
    st.stop()

try:
    raw = load_data(ticker, period)
except RuntimeError as exc:
    st.error(f"Could not load data for {ticker}: {exc}")
    st.stop()

if len(raw) < long_window + 60:
    st.error(
        f"Only {len(raw)} trading days available for {ticker}. "
        "Choose a longer history or a smaller long window."
    )
    st.stop()

data = run_strategy(raw, short_window, long_window)
metrics, data = calculate_metrics(data, transaction_cost)
significance = cached_significance(data)

st.subheader(f"{ticker}: {short_window}/{long_window} SMA crossover")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Strategy return",
    f"{metrics['strategy_return'] * 100:.2f}%",
    f"{(metrics['strategy_return'] - metrics['buy_hold_return']) * 100:+.2f} pp vs buy & hold",
)
c2.metric(
    "CAGR",
    f"{metrics['annualized_return'] * 100:.2f}%",
    f"{(metrics['annualized_return'] - metrics['buy_hold_cagr']) * 100:+.2f} pp vs buy & hold",
)
c3.metric(
    "Sharpe ratio",
    f"{metrics['sharpe_ratio']:.2f}",
    f"{metrics['sharpe_ratio'] - metrics['buy_hold_sharpe_ratio']:+.2f} vs buy & hold",
)
c4.metric(
    "Max drawdown",
    f"{metrics['max_drawdown'] * 100:.2f}%",
    f"{(metrics['max_drawdown'] - metrics['buy_hold_max_drawdown']) * 100:+.2f} pp vs buy & hold",
)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Buy & hold return", f"{metrics['buy_hold_return'] * 100:.2f}%")
c6.metric("Buy & hold max drawdown", f"{metrics['buy_hold_max_drawdown'] * 100:.2f}%")
c7.metric("Total trades", f"{metrics['total_trades']:.0f}")
c8.metric("Market exposure", f"{metrics['exposure'] * 100:.1f}%")

tab_equity, tab_price = st.tabs(["Equity curve", "Price & signals"])

with tab_equity:
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
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )
    st.plotly_chart(fig)

with tab_price:
    buy_signals = data[data["Buy"] == 1]
    sell_signals = data[data["Sell"] == 1]

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(x=data.index, y=data["Adj Close"], mode="lines", name="Price")
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
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )
    st.plotly_chart(price_fig)

st.subheader("Statistical significance")

s1, s2 = st.columns(2)
s1.metric(
    "Actual Sharpe percentile",
    f"{significance['actual_sharpe_percentile']:.1f}%",
)
s2.metric(
    "Random Sharpe (mean ± std)",
    f"{significance['random_sharpe_mean']:.2f} ± {significance['random_sharpe_std']:.2f}",
)

if significance["verdict"] == "statistically significant":
    st.success(f"Verdict: {significance['verdict']}")
else:
    st.info(f"Verdict: {significance['verdict']}")

st.caption(
    f"Compares the strategy's gross (pre-cost) Sharpe against {NUM_SIMULATIONS} "
    "randomly timed strategies with the same number of invested days. "
    "This is why it can differ from the net-of-cost Sharpe shown above."
)
