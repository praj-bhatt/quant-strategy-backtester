import sqlite3

import pytest

import data


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_market_data.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE prices (ticker TEXT, date TEXT, adj_close REAL, "
        "volume REAL, PRIMARY KEY (ticker, date))"
    )

    from datetime import date, timedelta

    start = date.today() - timedelta(days=365 * 8)
    rows = []
    d = start
    while d <= date.today():
        rows.append(("TEST", d.isoformat(), 100.0, 1000.0))
        d += timedelta(days=1)

    conn.executemany("INSERT INTO prices VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()

    monkeypatch.setattr(data, "DB_PATH", str(db_path))
    return db_path


def test_get_data_returns_dataframe(seeded_db):
    result = data.get_data("TEST", "5y")
    assert not result.empty


def test_get_data_has_expected_columns(seeded_db):
    result = data.get_data("TEST", "5y")
    assert list(result.columns) == ["Adj Close", "Volume"]
    assert result["Adj Close"].dtype == "float64"
    assert result["Volume"].dtype == "float64"


def test_get_data_invalid_ticker(seeded_db):
    with pytest.raises(RuntimeError):
        data.get_data("NOPE", "5y")


def test_get_data_period_narrows_range(seeded_db):
    full = data.get_data("TEST", "10y")
    narrow = data.get_data("TEST", "3y")
    assert narrow.index.min() > full.index.min()
    assert len(narrow) < len(full)
