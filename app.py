import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data import get_data
from metrics import calculate_metrics
from significance import run_significance_test
from strategy import MLPredictor, RSIMeanReversion, SMACrossover

st.set_page_config(
    page_title="Quantitative Backtester",
    layout="wide",
)

NUM_SIMULATIONS = 1000


@st.cache_data(ttl=3600, show_spinner="Downloading market data...")
def load_data(ticker, period):
    return get_data(ticker, period)


@st.cache_data(show_spinner="Running significance test...")
def cached_significance(data):
    return run_significance_test(data, NUM_SIMULATIONS)


st.title("Quantitative Strategy Backtester")

st.caption(
    "Signals are lagged one trading day to avoid look-ahead bias. "
    "Evaluates both long and short market exposures."
)


with st.sidebar:
    st.header("Parameters")

    ticker = (
        st.text_input(
            "Ticker",
            value="^GSPC",
            help="Any Yahoo Finance symbol, e.g. ^GSPC, AAPL, GLD",
        )
        .strip()
        .upper()
    )

    period = st.selectbox(
        "History",
        ["3y", "5y", "10y"],
        index=1,
    )

    cost_pct = st.slider(
        "Transaction cost per trade (%)",
        0.00,
        0.50,
        0.05,
        step=0.01,
    )

    st.header("Strategy Engine")

    strategy_type = st.selectbox(
        "Select Strategy",
        [
            "SMA Crossover",
            "RSI Mean Reversion",
            "ML Predictor",
        ],
    )

    if strategy_type == "SMA Crossover":
        short_window = st.slider(
            "Short SMA window",
            5,
            100,
            50,
        )

        long_window = st.slider(
            "Long SMA window",
            20,
            300,
            200,
        )

        if short_window >= long_window:
            st.error(
                "Short window must be smaller than the long window."
            )
            st.stop()

        strategy = SMACrossover(
            short_window,
            long_window,
        )

        st.caption(
            "Long when short SMA > long SMA, "
            "short otherwise."
        )

    elif strategy_type == "RSI Mean Reversion":
        rsi_period = st.slider(
            "RSI Period",
            5,
            50,
            14,
        )

        oversold = st.slider(
            "Oversold Threshold",
            10,
            45,
            30,
        )

        overbought = st.slider(
            "Overbought Threshold",
            55,
            90,
            70,
        )

        if oversold >= overbought:
            st.error(
                "Oversold must be strictly less than Overbought."
            )
            st.stop()

        strategy = RSIMeanReversion(
            rsi_period,
            oversold,
            overbought,
        )

        st.caption(
            "Long when RSI drops below oversold, "
            "short when above overbought."
        )
    else:
        retrain_every = st.slider(
            "Retrain Every (days)",
            20,
            120,
            60,
        )

        min_train_window = st.slider(
            "Minimum Training Window",
            300,
            1000,
            500,
        )

        n_estimators = st.slider(
            "Random Forest Trees",
            50,
            500,
            200,
            step=50,
        )

        strategy = MLPredictor(
            retrain_every,
            min_train_window,
            n_estimators,
            random_state=42,
        )

        st.caption(
            "Random Forest predicts next-day price direction "
            "using walk-forward training."
        )


transaction_cost = cost_pct / 100


if not ticker:
    st.info("Enter a ticker in the sidebar.")
    st.stop()


try:
    raw = load_data(ticker, period)
except RuntimeError as exc:
    st.error(
        f"Could not load data for {ticker}: {exc}"
    )
    st.stop()


if strategy_type == "SMA Crossover":
    min_length = long_window + 60
elif strategy_type == "RSI Mean Reversion":
    min_length = rsi_period + 60
else:
    min_length = min_train_window + 60


if len(raw) < min_length:
    st.error(
        f"Only {len(raw)} trading days available for "
        f"{ticker}. Choose a longer history."
    )
    st.stop()


data = strategy.apply_signals(raw)

metrics, data = calculate_metrics(
    data,
    transaction_cost,
)

significance = cached_significance(data)


st.subheader(
    f"{ticker}: {strategy_type} Performance"
)


# -------------------------------------------------------------------
# PERFORMANCE METRICS
# -------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Strategy return",
    f"{metrics['strategy_return'] * 100:.2f}%",
    (
        f"{(metrics['strategy_return'] - metrics['buy_hold_return']) * 100:+.2f} "
        "pp vs buy & hold"
    ),
)

c2.metric(
    "CAGR",
    f"{metrics['annualized_return'] * 100:.2f}%",
    (
        f"{(metrics['annualized_return'] - metrics['buy_hold_cagr']) * 100:+.2f} "
        "pp vs buy & hold"
    ),
)

c3.metric(
    "Sharpe ratio",
    f"{metrics['sharpe_ratio']:.2f}",
    (
        f"{metrics['sharpe_ratio'] - metrics['buy_hold_sharpe_ratio']:+.2f} "
        "vs buy & hold"
    ),
)

c4.metric(
    "Max drawdown",
    f"{metrics['max_drawdown'] * 100:.2f}%",
    (
        f"{(metrics['max_drawdown'] - metrics['buy_hold_max_drawdown']) * 100:+.2f} "
        "pp vs buy & hold"
    ),
)


c5, c6, c7, c8 = st.columns(4)

c5.metric(
    "Buy & hold return",
    f"{metrics['buy_hold_return'] * 100:.2f}%",
)

c6.metric(
    "Buy & hold max drawdown",
    f"{metrics['buy_hold_max_drawdown'] * 100:.2f}%",
)

c7.metric(
    "Total trades",
    f"{metrics['total_trades']:.0f}",
)

c8.metric(
    "Market exposure",
    f"{metrics['exposure'] * 100:.1f}%",
)


# -------------------------------------------------------------------
# ADDITIONAL SHORTING INFORMATION
# -------------------------------------------------------------------

c9, c10 = st.columns(2)

c9.metric(
    "Short days",
    f"{metrics['short_days']:.0f}",
)

c10.metric(
    "Borrowing cost",
    f"{metrics['total_borrow_cost'] * 100:.2f}%",
)


# -------------------------------------------------------------------
# CHART TABS
# -------------------------------------------------------------------

if strategy_type == "ML Predictor":
    tab_equity, tab_price, tab_eval = st.tabs(
        ["Equity curve", "Price & signals", "Model evaluation"]
    )
else:
    tab_equity, tab_price = st.tabs(["Equity curve", "Price & signals"])
    tab_eval = None


# -------------------------------------------------------------------
# EQUITY CURVE
# -------------------------------------------------------------------

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
            name=strategy_type,
        )
    )

    fig.update_layout(
        title=f"{strategy_type} vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# PRICE + SIGNALS
# -------------------------------------------------------------------

with tab_price:
    buy_signals = data[data["Buy"] == 1]
    sell_signals = data[data["Sell"] == 1]
    short_signals = data[data["Short"] == 1]
    cover_signals = data[data["Cover"] == 1]

    if strategy_type == "SMA Crossover":
        price_fig = go.Figure()

        price_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Adj Close"],
                mode="lines",
                name="Price",
            )
        )

        if f"SMA_{short_window}" in data.columns:
            price_fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[f"SMA_{short_window}"],
                    mode="lines",
                    name=f"SMA {short_window}",
                )
            )

        if f"SMA_{long_window}" in data.columns:
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
                marker={
                    "color": "green",
                    "symbol": "triangle-up",
                    "size": 12,
                },
                name="Buy",
            )
        )

        price_fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "red",
                    "symbol": "triangle-down",
                    "size": 12,
                },
                name="Sell",
            )
        )

        price_fig.add_trace(
            go.Scatter(
                x=short_signals.index,
                y=short_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "purple",
                    "symbol": "triangle-down",
                    "size": 12,
                },
                name="Short",
            )
        )

        price_fig.add_trace(
            go.Scatter(
                x=cover_signals.index,
                y=cover_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "orange",
                    "symbol": "triangle-up",
                    "size": 12,
                },
                name="Cover",
            )
        )

        price_fig.update_layout(
            title=f"{ticker} Price, Moving Averages & Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified",
            height=500,
        )

    elif strategy_type == "RSI Mean Reversion":
        price_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.08,
        )

        price_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Adj Close"],
                mode="lines",
                name="Price",
            ),
            row=1,
            col=1,
        )

        price_fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "green",
                    "symbol": "triangle-up",
                    "size": 12,
                },
                name="Buy",
            ),
            row=1,
            col=1,
        )

        price_fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "red",
                    "symbol": "triangle-down",
                    "size": 12,
                },
                name="Sell",
            ),
            row=1,
            col=1,
        )

        price_fig.add_trace(
            go.Scatter(
                x=short_signals.index,
                y=short_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "purple",
                    "symbol": "triangle-down",
                    "size": 12,
                },
                name="Short",
            ),
            row=1,
            col=1,
        )

        price_fig.add_trace(
            go.Scatter(
                x=cover_signals.index,
                y=cover_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "orange",
                    "symbol": "triangle-up",
                    "size": 12,
                },
                name="Cover",
            ),
            row=1,
            col=1,
        )

        if "RSI" in data.columns:
            price_fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["RSI"],
                    mode="lines",
                    name="RSI",
                ),
                row=2,
                col=1,
            )

            price_fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=[oversold] * len(data),
                    mode="lines",
                    name="Oversold",
                    line={"dash": "dash"},
                ),
                row=2,
                col=1,
            )

            price_fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=[overbought] * len(data),
                    mode="lines",
                    name="Overbought",
                    line={"dash": "dash"},
                ),
                row=2,
                col=1,
            )

        price_fig.update_yaxes(
            title_text="Price",
            row=1,
            col=1,
        )

        price_fig.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            row=2,
            col=1,
        )

        price_fig.update_layout(
            title=f"{ticker} Price, RSI & Signals",
            hovermode="x unified",
            height=650,
        )
    else:
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
                x=buy_signals.index,
                y=buy_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "green",
                    "symbol": "triangle-up",
                    "size": 12,
                },
                name="Buy",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "red",
                    "symbol": "triangle-down",
                    "size": 12,
                },
                name="Sell",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=short_signals.index,
                y=short_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "purple",
                    "symbol": "triangle-down",
                    "size": 12,
                },
                name="Short",
            )
        )
        price_fig.add_trace(
            go.Scatter(
                x=cover_signals.index,
                y=cover_signals["Adj Close"],
                mode="markers",
                marker={
                    "color": "orange",
                    "symbol": "triangle-up",
                    "size": 12,
                },
                name="Cover",
            )
        )
    st.plotly_chart(
        price_fig,
        use_container_width=True,
    )

    if tab_eval is not None:
        with tab_eval:
            mask = data["ML_Proba"].notna() & data["Target"].notna()
            y_true = data.loc[mask, "Target"].astype(int)
            prob = data.loc[mask, "ML_Proba"]
            y_pred = (prob >= 0.5).astype(int)

            baseline = max(y_true.mean(), 1 - y_true.mean())

            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Accuracy",
                f"{accuracy_score(y_true, y_pred):.3f}",
                f"{accuracy_score(y_true, y_pred) - baseline:+.3f} vs majority-class baseline",
            )
            m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
            m3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")

            m4, m5 = st.columns(2)
            m4.metric("F1 score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
            if y_true.nunique() == 2:
                m5.metric("ROC-AUC", f"{roc_auc_score(y_true, prob):.3f}")
            else:
                m5.metric("ROC-AUC", "N/A (one class only)")

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Down", "Actual Up"],
                columns=["Predicted Down", "Predicted Up"],
            )
            st.subheader("Confusion matrix")
            st.dataframe(cm_df)


# -------------------------------------------------------------------
# STATISTICAL SIGNIFICANCE
# -------------------------------------------------------------------

st.subheader("Statistical significance")

s1, s2 = st.columns(2)

s1.metric(
    "Actual Sharpe percentile",
    f"{significance['actual_sharpe_percentile']:.1f}%",
)

s2.metric(
    "Random Sharpe (mean ± std)",
    (
        f"{significance['random_sharpe_mean']:.2f} ± "
        f"{significance['random_sharpe_std']:.2f}"
    ),
)


if significance["verdict"] == "statistically significant":
    st.success(
        f"Verdict: {significance['verdict']}"
    )
else:
    st.info(
        f"Verdict: {significance['verdict']}"
    )


st.caption(
    f"Compares the strategy's gross (pre-cost) Sharpe against "
    f"{NUM_SIMULATIONS} randomly timed strategies with the same "
    "number of invested days. This is why it can differ from the "
    "net-of-cost Sharpe shown above."
)
