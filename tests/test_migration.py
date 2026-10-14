import sqlite3

import pytest

import ingestion


@pytest.fixture
def migration_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_migration.db"
    monkeypatch.setattr(ingestion, "DB_PATH", str(db_path))
    return db_path


def test_migrate_adds_not_null_constraint_and_preserves_valid_rows(
    migration_db,
):
    conn = sqlite3.connect(migration_db)
    conn.execute(
        """
        CREATE TABLE prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            adj_close REAL NOT NULL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.executemany(
        "INSERT INTO prices VALUES (?, ?, ?, ?)",
        [
            ("TEST", "2025-01-01", 100.0, 1000.0),
            ("TEST", "2025-01-02", 110.0, 1200.0),
            ("TEST", "2025-01-03", 120.0, None),
        ],
    )
    conn.commit()
    conn.close()

    ingestion.migrate()

    conn = sqlite3.connect(migration_db)
    rows = conn.execute(
        "SELECT ticker, date, adj_close, volume "
        "FROM prices ORDER BY date"
    ).fetchall()
    volume_column = next(
        row
        for row in conn.execute("PRAGMA table_info(prices)")
        if row[1] == "volume"
    )
    conn.close()

    assert rows == [
        ("TEST", "2025-01-01", 100.0, 1000.0),
        ("TEST", "2025-01-02", 110.0, 1200.0),
    ]
    assert volume_column[3] == 1


def test_migrate_does_nothing_when_already_migrated(migration_db):
    ingestion.create_database()

    conn = sqlite3.connect(migration_db)
    conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?)",
        ("TEST", "2025-01-01", 100.0, 1000.0),
    )
    conn.commit()
    conn.close()

    ingestion.migrate()

    conn = sqlite3.connect(migration_db)
    rows = conn.execute("SELECT * FROM prices").fetchall()
    conn.close()

    assert rows == [("TEST", "2025-01-01", 100.0, 1000.0)]


def test_migrate_creates_table_when_missing(migration_db):
    ingestion.migrate()

    conn = sqlite3.connect(migration_db)
    columns = conn.execute("PRAGMA table_info(prices)").fetchall()
    conn.close()

    assert columns
    volume_column = next(row for row in columns if row[1] == "volume")
    assert volume_column[3] == 1


def test_migrate_raises_error_when_volume_column_missing(migration_db):
    conn = sqlite3.connect(migration_db)
    conn.execute(
        """
        CREATE TABLE prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            adj_close REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="no volume column"):
        ingestion.migrate()
