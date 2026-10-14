import sqlite3

DB_PATH = "market_data.db"


def create_database():
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            adj_close REAL NOT NULL,
           volume REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    connection.commit()
    connection.close()

def migrate():
    connection = sqlite3.connect(DB_PATH)

    columns = connection.execute(
        "PRAGMA table_info(prices)"
    ).fetchall()

    if not columns:
        connection.close()
        create_database()
        return

    column_info = {row[1]: row for row in columns}

    if (
        "volume" in column_info
        and column_info["volume"][3] == 1
    ):
        connection.close()
        return

    if "volume" not in column_info:
        connection.close()
        raise RuntimeError(
            "The database has no volume column. "
            "Re-ingest the market data to rebuild it."
        )

    connection.execute("BEGIN")

    connection.execute(
        """
        CREATE TABLE prices_new (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            adj_close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
        """
    )

    connection.execute(
        """
        INSERT INTO prices_new (ticker, date, adj_close, volume)
        SELECT ticker, date, adj_close, volume
        FROM prices
        WHERE adj_close IS NOT NULL AND volume IS NOT NULL
        """
    )

    connection.execute("DROP TABLE prices")
    connection.execute("ALTER TABLE prices_new RENAME TO prices")

    connection.commit()
    connection.close()


def extract_data(ticker="^GSPC", period="5y"):
    import yfinance as yf

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        multi_level_index=False,
    )

    if data is None or data.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    return data


def transform_data(data, ticker):
    transformed = data[["Adj Close", "Volume"]].copy()
    transformed = transformed.reset_index()
    transformed["ticker"] = ticker

    transformed = transformed.rename(
        columns={
            "Date": "date",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    transformed = transformed[
        ["ticker", "date", "adj_close", "volume"]
    ]

    transformed["date"] = transformed["date"].astype(str)
    transformed = transformed.dropna(
        subset=["adj_close", "volume"]
    )
    transformed = transformed[transformed["adj_close"] > 0]
    return transformed


def load_data(data):
    connection = sqlite3.connect(DB_PATH)

    connection.executemany(
        """
        INSERT OR REPLACE INTO prices
        (ticker, date, adj_close, volume)
        VALUES (?, ?, ?, ?)
        """,
        data.itertuples(index=False, name=None),
    )

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_database()
    migrate()

    tickers = ["^GSPC", "GLD", "XLK", "XLF", "XLE"]

    for ticker in tickers:
        data = extract_data(ticker, "5y")
        transformed = transform_data(data, ticker)
        load_data(transformed)
