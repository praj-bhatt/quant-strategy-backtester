import yfinance as yf


def get_data(ticker="^GSPC", period="5y"):
    data = yf.download(
        ticker, period=period, auto_adjust=False, multi_level_index=False
    )

    if data is None or data.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    data = data[["Adj Close"]].copy()
    data = data.dropna(subset=["Adj Close"])

    return data
