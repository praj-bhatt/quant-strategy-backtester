import sqlite3
from datetime import datetime, timezone

import pandas as pd
from dateutil.relativedelta import relativedelta

DB_PATH = "market_data.db"


def get_data(ticker="^GSPC", period="5y"):
    years = int(period.rstrip("y"))

    cutoff_date = datetime.now(timezone.utc).date() - relativedelta(years=years)
    cutoff_date = cutoff_date.isoformat()

    connection = sqlite3.connect(DB_PATH)

    query = """
        SELECT date, adj_close, volume
        FROM prices
        WHERE ticker = ? AND date >= ?
        ORDER BY date
    """

    data = pd.read_sql_query(
        query,
        connection,
        params=(ticker, cutoff_date),
        parse_dates=["date"],
    )

    connection.close()

    if data.empty:
        raise RuntimeError(f"No data found for {ticker}")

    data = data.set_index("date")
    data = data.rename(
        columns={
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
    )

    data = data.dropna(subset=["Adj Close", "Volume"])

    if data.empty:
        raise RuntimeError(f"No valid data found for {ticker}")

    return data
