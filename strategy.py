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
            data["Adj Close"]
            .rolling(window=self.short_window)
            .mean()
        )

        data[f"SMA_{self.long_window}"] = (
            data["Adj Close"]
            .rolling(window=self.long_window)
            .mean()
        )

        data = data.dropna(
            subset=[
                f"SMA_{self.short_window}",
                f"SMA_{self.long_window}",
            ]
        ).copy()

        data["Signal"] = np.where(
            data[f"SMA_{self.short_window}"]
            > data[f"SMA_{self.long_window}"],
            1,
            0,
        )

        data["Position"] = data["Signal"].shift(1).fillna(0)

        data["Trade"] = data["Position"].diff().abs().fillna(0)

        data["Buy"] = (
            (data["Position"] == 1)
            & (data["Position"].shift(1) == 0)
        ).astype(int)

        data["Sell"] = (
            (data["Position"] == 0)
            & (data["Position"].shift(1) == 1)
        ).astype(int)

        return data
