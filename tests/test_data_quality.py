# tests/test_data_quality.py
import sys
from pathlib import Path

# Add src to path FIRST
ROOT = Path(__file__).resolve().parents[1]
src_path = str(ROOT / "src")
sys.path.insert(0, src_path)

# Debug prints
print(f"ROOT: {ROOT}")
print(f"src_path: {src_path}")
print(f"src exists: {(ROOT / 'src').exists()}")
print(f"factorlabs exists: {(ROOT / 'src' / 'factorlabs').exists()}")
print(f"sys.path[0]: {sys.path[0]}")

# THEN import
import polars as pl
from factorlabs.data.ingest_yf import data_quality_fixing

def dummy_ohlcv():
    return pl.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "date": [pl.date(2024,1,1), pl.date(2024,1,2), pl.date(2024,1,1)],
        "open": [100, None, -5],
        "high": [101, None, 10],
        "low": [99, None, 8],
        "close": [100.5, None, 9], 
        "volume": [None, None, None],
    })

def test_quality():
    df = dummy_ohlcv()
    df = data_quality_fixing(df)
    print(df)

if __name__ == "__main__":
    test_quality()