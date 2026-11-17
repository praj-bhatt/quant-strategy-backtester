import sqlite3

import pandas as pd
import pytest

import ingestion
from ingestion import transform_data


def make_raw(adj_close, volume):
    index = pd.date_range("2025-01-01", periods=len(adj_close))
    data = pd.DataFrame(
        {"Adj Close": adj_close, "Volume": volume},
        index=index,
    )
    data.index.name = "Date"
    return data


def test_transform_data_removes_invalid_prices():
    data = make_raw(
        [100.0, 0.0, -5.0, None, 120.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1200.0],
    )

    result = transform_data(data, "TEST")

    assert len(result) == 2
    assert (result["adj_close"] > 0).all()
    assert result["adj_close"].tolist() == [100.0, 120.0]


def test_transform_data_removes_missing_volume():
    data = make_raw(
        [100.0, 110.0, 120.0],
        [1000.0, None, 1200.0],
    )

    result = transform_data(data, "TEST")

    assert result["adj_close"].tolist() == [100.0, 120.0]
    assert result["volume"].notna().all()


def test_transform_data_output_shape():
    data = make_raw([100.0, 110.0], [1000.0, 1100.0])

    result = transform_data(data, "ABC")

    assert list(result.columns) == ["ticker", "date", "adj_close", "volume"]
    assert (result["ticker"] == "ABC").all()
    assert result["date"].tolist() == ["2025-01-01", "2025-01-02"]


@pytest.fixture
def ingestion_db(tmp_path, monkeypatch):
    db_path = tmp_path / "ingestion_test.db"
    monkeypatch.setattr(ingestion, "DB_PATH", str(db_path))
    ingestion.create_database()
    return db_path


def test_load_data_is_idempotent(ingestion_db):
    data = make_raw([100.0, 110.0], [1000.0, 1100.0])
    transformed = transform_data(data, "TEST")

    ingestion.load_data(transformed)
    ingestion.load_data(transformed)

    conn = sqlite3.connect(ingestion_db)
    count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    conn.close()

    assert count == 2


def test_load_data_replaces_existing_row(ingestion_db):
    first = transform_data(make_raw([100.0], [1000.0]), "TEST")
    second = transform_data(make_raw([105.0], [2000.0]), "TEST")

    ingestion.load_data(first)
    ingestion.load_data(second)

    conn = sqlite3.connect(ingestion_db)
    rows = conn.execute("SELECT adj_close, volume FROM prices").fetchall()
    conn.close()

    assert rows == [(105.0, 2000.0)]
