import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from base_strategy import BaseStrategy


def compute_rsi(prices, period=14):
    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))


class SMACrossover(BaseStrategy):
    def __init__(self, short_window=50, long_window=200):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive.")

        if short_window >= long_window:
            raise ValueError("Short window must be smaller than long window.")

        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        data[f"SMA_{self.short_window}"] = (
            data["Adj Close"].rolling(window=self.short_window).mean()
        )

        data[f"SMA_{self.long_window}"] = (
            data["Adj Close"].rolling(window=self.long_window).mean()
        )

        data["Signal"] = np.where(
            data[f"SMA_{self.short_window}"] > data[f"SMA_{self.long_window}"],
            1,
            -1,
        )
        data.loc[data[f"SMA_{self.long_window}"].isna(), "Signal"] = 0

        return data


class RSIMeanReversion(BaseStrategy):
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        if rsi_period <= 0:
            raise ValueError("RSI period must be positive.")

        if not 0 <= oversold < overbought <= 100:
            raise ValueError(
                "RSI thresholds must satisfy 0 <= oversold < overbought <= 100."
            )

        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        data["RSI"] = compute_rsi(data["Adj Close"], self.rsi_period)

        data["Signal"] = np.where(
            data["RSI"] < self.oversold,
            1,
            np.where(data["RSI"] > self.overbought, -1, np.nan),
        )

        data["Signal"] = data["Signal"].ffill().fillna(0)

        return data

class MLPredictor(BaseStrategy):
    def __init__(
        self,
        retrain_every=60,
        min_train_window=500,
        n_estimators=200,
        random_state=42,
    ):
        self.retrain_every = retrain_every
        self.min_train_window = min_train_window
        self.n_estimators = n_estimators
        self.random_state = random_state

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        daily_returns = data["Adj Close"].pct_change()

        data["Return_1d"] = data["Adj Close"].pct_change(1)
        data["Return_3d"] = data["Adj Close"].pct_change(3)
        data["Return_5d"] = data["Adj Close"].pct_change(5)

        data["Volatility_10d"] = daily_returns.rolling(10).std()
        data["Volatility_20d"] = daily_returns.rolling(20).std()

        data["RSI"] = compute_rsi(data["Adj Close"], 14)

        data["Volume_Spike"] = data["Volume"] / data["Volume"].rolling(20).mean()
        next_day_return = data["Adj Close"].pct_change().shift(-1)

        data["Target"] = next_day_return.where(
            next_day_return.isna(),
            next_day_return.gt(0).astype(int),
        )
        feature_columns = [
            "Return_1d",
            "Return_3d",
            "Return_5d",
            "Volatility_10d",
            "Volatility_20d",
            "RSI",
            "Volume_Spike",
        ]

        predictions = pd.Series(index=data.index, dtype="float64")
        probabilities = pd.Series(index=data.index, dtype="float64")
        for block_start in range(
            self.min_train_window,
            len(data),
            self.retrain_every,
        ):
            block_end = min(block_start + self.retrain_every, len(data))

            train_data = data.iloc[:block_start - 1].dropna(
                subset=feature_columns + ["Target"]
            )

            test_data = data.iloc[block_start:block_end].dropna(
                subset=feature_columns
            )

            if test_data.empty:
                continue

            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )

            model.fit(
                train_data[feature_columns],
                train_data["Target"],
            )

            predictions.loc[test_data.index] = model.predict(
                test_data[feature_columns]
            )
            probabilities.loc[test_data.index] = model.predict_proba(
                test_data[feature_columns]
            )[:, 1]


        data["Signal"] = predictions.map({1.0: 1, 0.0: -1}).fillna(0)
        data["ML_Proba"] = probabilities

        return data
