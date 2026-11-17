import sqlite3
from datetime import datetime, timedelta, timezone

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

    start = datetime.now(timezone.utc).date() - timedelta(days=365 * 8)
    rows = []
    d = start
    while d <= datetime.now(timezone.utc).date():
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


def recent_dates(count=3):
    today = datetime.now(timezone.utc).date()
    return [(today - timedelta(days=count - i)).isoformat() for i in range(count)]


def test_get_data_removes_missing_values(seeded_db):
    d1, d2, d3 = recent_dates()

    conn = sqlite3.connect(seeded_db)
    conn.executemany(
        "INSERT INTO prices VALUES (?, ?, ?, ?)",
        [
            ("DIRTY", d1, None, 1000.0),
            ("DIRTY", d2, 100.0, None),
            ("DIRTY", d3, 100.0, 1000.0),
        ],
    )
    conn.commit()
    conn.close()

    result = data.get_data("DIRTY", "5y")

    assert len(result) == 1
    assert result["Adj Close"].notna().all()
    assert result["Volume"].notna().all()


def test_get_data_raises_when_every_row_is_invalid(seeded_db):
    d1, d2, _ = recent_dates()

    conn = sqlite3.connect(seeded_db)
    conn.executemany(
        "INSERT INTO prices VALUES (?, ?, ?, ?)",
        [
            ("BAD", d1, None, 1000.0),
            ("BAD", d2, 100.0, None),
        ],
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="No valid data"):
        data.get_data("BAD", "5y")