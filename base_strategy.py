from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate raw trading signals."""

    def apply_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply signals and create positions and trades."""
        data = self.generate_signals(data)

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
