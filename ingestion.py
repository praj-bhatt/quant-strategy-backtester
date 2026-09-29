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
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
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
    transformed = transformed.dropna(subset=["adj_close"])

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

    tickers = ["^GSPC", "GLD", "XLK", "XLF", "XLE"]

    for ticker in tickers:
        data = extract_data(ticker, "5y")
        transformed = transform_data(data, ticker)
        load_data(transformed)
