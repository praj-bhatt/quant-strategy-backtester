import pandas as pd
import pytest

from strategy import RSIMeanReversion, SMACrossover


def test_sma_crossover_generates_signals():
    data = pd.DataFrame(
        {
            "Adj Close": [10, 11, 12, 13, 14, 15],
        }
    )

    strategy = SMACrossover(short_window=2, long_window=3)
    result = strategy.generate_signals(data)

    assert "Signal" in result.columns
    assert not result.empty


def test_sma_crossover_rejects_invalid_windows():
    with pytest.raises(ValueError):
        SMACrossover(short_window=200, long_window=50)


def test_rsi_mean_reversion_generates_signals():
    data = pd.DataFrame(
        {
            "Adj Close": [
                10, 11, 10, 12, 11,
                13, 12, 14, 13, 15,
                14, 16, 15, 17, 16,
            ],
        }
    )

    strategy = RSIMeanReversion(
        rsi_period=5,
        oversold=30,
        overbought=70,
    )

    result = strategy.generate_signals(data)

    assert "RSI" in result.columns
    assert "Signal" in result.columns
    assert not result.empty


def test_rsi_mean_reversion_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        RSIMeanReversion(
            rsi_period=14,
            oversold=70,
            overbought=30,
        )
