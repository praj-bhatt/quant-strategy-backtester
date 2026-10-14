import numpy as np
import pandas as pd

from metrics import calculate_metrics


def test_transaction_and_borrow_costs():
    data = pd.DataFrame(
        {
            "Adj Close": [100.0, 110.0, 100.0],
            "Position": [0, 1, -1],
            "Trade": [0, 1, 1],
        },
        index=pd.to_datetime(
            ["2025-01-01", "2025-01-02", "2025-01-03"]
        ),
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
    assert np.isclose(
        result["Short_Borrow_Cost"].iloc[2],
        daily_borrow_rate,
    )
    expected_return = (
        (110.0 - 100.0) / 110.0
        - 0.001
        - daily_borrow_rate
    )

    assert np.isclose(
        result["Net_Strategy_Return"].iloc[2],
        expected_return,
    )
    assert np.isclose(
        result["Short_Borrow_Cost"].sum(),
        daily_borrow_rate,
    )
