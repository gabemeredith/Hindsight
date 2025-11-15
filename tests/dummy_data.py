import polars as pl

def dummy_ohlcv():
    return pl.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "date": [pl.date(2024,1,1), pl.date(2024,1,2), pl.date(2024,1,1)],
        "open": [100, None, -5],
        "high": [101, None, 10],
        "low": [99, None, 8],
        "close": [100.5, None, 9],
        "volume": [1000, None, -20],
    })