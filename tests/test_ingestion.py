import pandas as pd

from ingestion import transform_data


def test_transform_data_removes_invalid_prices():
    data = pd.DataFrame(
        {
            "Adj Close": [100.0, 0.0, -5.0, None, 120.0],
            "Volume": [1000.0, 1000.0, 1000.0, 1000.0, 1200.0],
        },
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
            ]
        ),
    )
    data.index.name = "Date"

    result = transform_data(data, "TEST")

    assert len(result) == 2
    assert (result["adj_close"] > 0).all()
    assert result["adj_close"].tolist() == [100.0, 120.0]
