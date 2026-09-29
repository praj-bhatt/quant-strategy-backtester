import pandas as pd
import pytest

from strategy import MLPredictor, RSIMeanReversion, SMACrossover


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


def test_ml_predictor_generates_signals():
    data = pd.DataFrame(
        {
            "Adj Close": [
                100 + i * 0.5 + (i % 7) * 0.2
                for i in range(600)
            ],
            "Volume": [1000000 + i * 1000 for i in range(600)],
        }
    )

    strategy = MLPredictor(
        retrain_every=60,
        min_train_window=500,
        n_estimators=10,
        random_state=42,
    )

    result = strategy.generate_signals(data)

    assert "Signal" in result.columns
    assert "Target" in result.columns
    assert "RSI" in result.columns
    assert "Volume_Spike" in result.columns
    assert not result.empty

def test_ml_predictor_trains_only_on_past_data(monkeypatch):
    fit_max_indices = []

    class SpyRandomForest:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            fit_max_indices.append(X.index.max())
            return self

        def predict(self, X):
            return [0] * len(X)

    monkeypatch.setattr(
        "strategy.RandomForestClassifier",
        SpyRandomForest,
    )

    data = pd.DataFrame(
        {
            "Adj Close": [
                100 + i * 0.5 + (i % 7) * 0.2
                for i in range(620)
            ],
            "Volume": [1000000 + i * 1000 for i in range(620)],
        }
    )

    strategy = MLPredictor(
        retrain_every=60,
        min_train_window=500,
        n_estimators=10,
        random_state=42,
    )

    result = strategy.generate_signals(data)

    assert fit_max_indices == [499, 559]
    assert result.index.min() == 0
