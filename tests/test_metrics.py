import numpy as np
import pandas as pd
import pytest

from metrics import calculate_metrics


def test_transaction_and_borrow_costs():
    data = pd.DataFrame(
        {
            "Adj Close": [100.0, 110.0, 100.0],
            "Position": [0, 1, -1],
            "Trade": [0, 1, 1],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    )

    transaction_cost = 0.001
    short_borrow_rate = 0.252

    _, result = calculate_metrics(
        data,
        transaction_cost=transaction_cost,
        short_borrow_rate=short_borrow_rate,
    )

    daily_borrow_rate = short_borrow_rate / 252

    assert np.isclose(result["Transaction_Cost"].iloc[1], 0.001)
    assert np.isclose(result["Short_Borrow_Cost"].iloc[1], 0.0)
    assert np.isclose(result["Net_Strategy_Return"].iloc[1], 0.099)

    assert np.isclose(result["Transaction_Cost"].iloc[2], 0.001)
    assert np.isclose(result["Short_Borrow_Cost"].iloc[2], daily_borrow_rate)
    expected_return = (110.0 - 100.0) / 110.0 - 0.001 - daily_borrow_rate

    assert np.isclose(result["Net_Strategy_Return"].iloc[2], expected_return)
    assert np.isclose(result["Short_Borrow_Cost"].sum(), daily_borrow_rate)


def test_flip_costs_two_trades_and_counts_are_correct():
    # long -> short flip is Trade=2, as BaseStrategy produces
    data = pd.DataFrame(
        {
            "Adj Close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "Position": [0, 1, -1, -1, 0],
            "Trade": [0, 1, 2, 0, 1],
        },
        index=pd.date_range("2025-01-01", periods=5),
    )

    metrics, result = calculate_metrics(
        data,
        transaction_cost=0.001,
        short_borrow_rate=0.252,
    )

    daily_borrow = 0.252 / 252

    assert np.isclose(result["Transaction_Cost"].iloc[2], 0.002)
    assert metrics["total_trades"] == 4
    assert metrics["invested_days"] == 3
    assert np.isclose(metrics["exposure"], 3 / 5)
    assert metrics["short_days"] == 2
    assert np.isclose(metrics["total_borrow_cost"], 2 * daily_borrow)
    assert np.isclose(metrics["short_borrow_rate"], 0.252)


def test_cagr_uses_calendar_time_not_row_count():
    # only 3 rows, but two full calendar years apart
    data = pd.DataFrame(
        {
            "Adj Close": [100.0, 110.0, 121.0],
            "Position": [1, 1, 1],
            "Trade": [0, 0, 0],
        },
        index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
    )

    metrics, _ = calculate_metrics(data, transaction_cost=0.0)

    years = (pd.Timestamp("2025-01-01") - pd.Timestamp("2023-01-01")).days / 365.25
    expected = 1.21 ** (1 / years) - 1

    assert np.isclose(metrics["annualized_return"], expected)
    assert np.isclose(metrics["buy_hold_cagr"], expected)


def test_cagr_rejects_zero_elapsed_time():
    data = pd.DataFrame(
        {
            "Adj Close": [100.0],
            "Position": [1],
            "Trade": [0],
        },
        index=pd.to_datetime(["2025-01-01"]),
    )

    with pytest.raises(ValueError):
        calculate_metrics(data)
