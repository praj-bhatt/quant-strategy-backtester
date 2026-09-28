import numpy as np
import pandas as pd

from base_strategy import BaseStrategy


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

        data = data.dropna(
            subset=[f"SMA_{self.short_window}", f"SMA_{self.long_window}"]
        ).copy()

        data["Signal"] = np.where(
            data[f"SMA_{self.short_window}"] > data[f"SMA_{self.long_window}"],
            1,
            -1,
        )

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

        delta = data["Adj Close"].diff()

        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        average_gain = gains.rolling(window=self.rsi_period).mean()
        average_loss = losses.rolling(window=self.rsi_period).mean()

        relative_strength = average_gain / average_loss

        data["RSI"] = 100 - (100 / (1 + relative_strength))

        data = data.dropna(subset=["RSI"]).copy()

        data["Signal"] = np.where(
            data["RSI"] < self.oversold,
            1,
            np.where(data["RSI"] > self.overbought, -1, np.nan),
        )

        data["Signal"] = data["Signal"].ffill().fillna(0)

        return data
