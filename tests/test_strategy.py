import numpy as np
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

        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.5, 0.5]] * len(X))

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

    assert fit_max_indices == [498, 558]
    assert result.index.min() == 0
def test_ml_predictor_skips_single_class_training(monkeypatch):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    def fail_if_fit_called(self, X, y):
        raise AssertionError("Model should not train on a single class")

    monkeypatch.setattr(RandomForestClassifier, "fit", fail_if_fit_called)

    data = pd.DataFrame(
        {
            "Adj Close": [100 + i for i in range(550)],
            "Volume": [1000] * 550,
        }
    )

    result = MLPredictor().generate_signals(data)

    assert (result["Signal"] == 0).all()


def make_random_walk(n):
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    volume = rng.integers(900_000, 1_100_000, n).astype(float)
    return pd.DataFrame({"Adj Close": close, "Volume": volume})


def test_ml_predictor_returns_valid_probabilities():
    data = make_random_walk(600)

    result = MLPredictor(
        retrain_every=60,
        min_train_window=500,
        n_estimators=10,
        random_state=42,
    ).generate_signals(data)

    probabilities = result["ML_Proba"].dropna()

    assert len(probabilities) > 0
    assert probabilities.between(0, 1).all()


def test_ml_predictor_features_use_only_current_and_past_prices():
    data = make_random_walk(100)

    result = MLPredictor(
        retrain_every=60,
        min_train_window=500,
        n_estimators=10,
        random_state=42,
    ).generate_signals(data)

    expected = data["Adj Close"].pct_change()
    pd.testing.assert_series_equal(
        result["Return_1d"], expected, check_names=False
    )


def test_ml_predictor_target_is_next_day_direction():
    data = make_random_walk(100)

    result = MLPredictor(
        retrain_every=60,
        min_train_window=500,
        n_estimators=10,
        random_state=42,
    ).generate_signals(data)

    close = data["Adj Close"]
    expected = (close.shift(-1) > close).astype(float)
    expected.iloc[-1] = np.nan

    pd.testing.assert_series_equal(
        result["Target"].astype(float), expected, check_names=False
    )
