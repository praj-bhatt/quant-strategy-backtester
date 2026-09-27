import numpy as np


def run_strategy(data, short_window=50, long_window=200):
    data = data.copy()

    data[f"SMA_{short_window}"] = data["Adj Close"].rolling(window=short_window).mean()

    data[f"SMA_{long_window}"] = data["Adj Close"].rolling(window=long_window).mean()

    data = data.dropna(subset=[f"SMA_{short_window}", f"SMA_{long_window}"]).copy()

    data["Signal"] = np.where(
        data[f"SMA_{short_window}"] > data[f"SMA_{long_window}"], 1, 0
    )

    data["Position"] = data["Signal"].shift(1).fillna(0)

    data["Trade"] = data["Position"].diff().abs().fillna(0)

    data["Buy"] = ((data["Position"] == 1) & (data["Position"].shift(1) == 0)).astype(
        int
    )

    data["Sell"] = ((data["Position"] == 0) & (data["Position"].shift(1) == 1)).astype(
        int
    )
    return data
