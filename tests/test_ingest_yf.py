"""
Tests for factorlabs/data/src/ingest_yf.py

Test Philosophy:
- Test each transformation function in isolation
- Use small, deterministic DataFrames (no actual API calls)
- Validate schema transformations step-by-step
- Test edge cases (single ticker, multi-ticker, missing data)

Run with: pytest tests/test_ingest_yf.py -v
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/factorlabs/data/src')

import polars as pl
import pytest
from datetime import date
from factorlabs.data.src import ingest_yf


# ========================== FIXTURES ==========================

@pytest.fixture
def raw_single_ticker_df():
    """
    Simulates yfinance output for single ticker (AAPL).
    yfinance returns simple columns when there's only one ticker.
    """
    return pl.DataFrame({
        "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Open": [100.0, 102.0, 101.0],
        "High": [105.0, 106.0, 104.0],
        "Low": [99.0, 101.0, 100.0],
        "Close": [103.0, 105.0, 102.0],
        "Volume": [1000000, 1100000, 950000],
    })


@pytest.fixture
def raw_multi_ticker_df():
    """
    Simulates yfinance output for multiple tickers.
    yfinance returns MultiIndex columns: ('Close', 'AAPL'), ('Close', 'MSFT'), etc.
    We represent these as strings since that's how Polars sees them.
    """
    return pl.DataFrame({
        "('Date', '')": ["2020-01-01", "2020-01-02"],
        "('Close', 'AAPL')": [100.0, 110.0],
        "('Close', 'MSFT')": [200.0, 210.0],
        "('High', 'AAPL')": [105.0, 115.0],
        "('High', 'MSFT')": [205.0, 215.0],
        "('Low', 'AAPL')": [98.0, 108.0],
        "('Low', 'MSFT')": [198.0, 208.0],
        "('Open', 'AAPL')": [99.0, 109.0],
        "('Open', 'MSFT')": [199.0, 209.0],
        "('Volume', 'AAPL')": [1000000.0, 1100000.0],
        "('Volume', 'MSFT')": [2000000.0, 2100000.0],
    })


# ========================== TEST COLUMN NORMALIZATION ==========================

def test_normalize_column_names_single_ticker(raw_single_ticker_df):
    """
    Test that simple column names get lowercased.
    
    'Date' -> 'date'
    'Close' -> 'close'
    'Volume' -> 'volume'
    """
    result = ingest_yf.normalize_column_names(raw_single_ticker_df)
    
    expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    assert result.columns == expected_cols


def test_normalize_column_names_multi_ticker(raw_multi_ticker_df):
    """
    Test that MultiIndex columns are flattened correctly.
    
    ('Close', 'AAPL') -> 'close_aapl'
    ('Date', '')      -> 'date'
    """
    result = ingest_yf.normalize_column_names(raw_multi_ticker_df)
    
    expected_cols = [
        'date',
        'close_aapl', 'close_msft',
        'high_aapl', 'high_msft',
        'low_aapl', 'low_msft',
        'open_aapl', 'open_msft',
        'volume_aapl', 'volume_msft'
    ]
    
    assert sorted(result.columns) == sorted(expected_cols)


def test_normalize_column_names_handles_edge_cases():
    """
    Test edge cases in column naming.
    """
    df = pl.DataFrame({
        "('Close', '')": [100.0],  # Empty ticker
        "('', 'AAPL')": [200.0],   # Empty field name
        "SimpleColumn": [300.0],    # Already simple
    })
    
    result = ingest_yf.normalize_column_names(df)
    
    # ('Close', '') -> 'close'
    # ('', 'AAPL') -> '_aapl' (or handle gracefully)
    # 'SimpleColumn' -> 'simplecolumn'
    assert 'close' in result.columns
    assert 'simplecolumn' in result.columns


# ========================== TEST DATE NORMALIZATION ==========================

def test_normalize_date_column_from_string():
    """
    Test that string dates are converted to pl.Date.
    """
    df = pl.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "close": [100.0, 110.0, 120.0],
    })
    
    result = ingest_yf.normalize_date_column(df)
    
    assert result["date"].dtype == pl.Date
    assert result["date"][0] == date(2020, 1, 1)


def test_normalize_date_column_from_datetime():
    """
    Test that datetime objects are converted to pl.Date (time stripped).
    """
    df = pl.DataFrame({
        "date": pl.datetime_range(
            start=pl.datetime(2020, 1, 1, 9, 30),  # 9:30 AM
            end=pl.datetime(2020, 1, 3, 9, 30),
            interval="1d",
            eager=True
        ),
        "close": [100.0, 110.0, 120.0],
    })
    
    result = ingest_yf.normalize_date_column(df)
    
    assert result["date"].dtype == pl.Date
    assert result["date"][0] == date(2020, 1, 1)


def test_normalize_date_column_already_date():
    """
    Test that pl.Date columns are left unchanged.
    """
    df = pl.DataFrame({
        "date": pl.date_range(
            start=date(2020, 1, 1),
            end=date(2020, 1, 3),
            interval="1d",
            eager=True
        ),
        "close": [100.0, 110.0, 120.0],
    })
    
    result = ingest_yf.normalize_date_column(df)
    
    assert result["date"].dtype == pl.Date
    assert len(result) == 3


# ========================== TEST DATA QUALITY FIXING ==========================

def test_data_quality_fixing_removes_all_null_rows():
    """
    Test that rows with all null OHLCV values are removed.
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "close": [100.0, None, 110.0],
        "open": [99.0, None, 109.0],
        "high": [105.0, None, 115.0],
        "low": [98.0, None, 108.0],
    })
    
    result = ingest_yf.data_quality_fixing(df)
    
    # Row 2 (all nulls) should be removed
    assert len(result) == 2
    assert result["date"].to_list() == [date(2020, 1, 1), date(2020, 1, 3)]


def test_data_quality_fixing_keeps_partial_nulls():
    """
    Test that rows with SOME null values are kept.
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "ticker": ["AAPL", "AAPL"],
        "close": [100.0, 110.0],
        "open": [None, 109.0],  # First row has null open
        "high": [105.0, 115.0],
    })
    
    result = ingest_yf.data_quality_fixing(df)
    
    # Both rows should remain (not all values are null)
    assert len(result) == 2


def test_data_quality_fixing_handles_negative_prices():
    """
    Test that negative prices are replaced with None.
    
    NOTE: Your current implementation has a bug - it doesn't reassign.
    This test will likely fail until you fix it.
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "ticker": ["AAPL", "AAPL"],
        "close": [100.0, -5.0],  # Negative price (data error)
        "open": [99.0, 95.0],
    })
    
    result = ingest_yf.data_quality_fixing(df)
    
    # Negative price should be None
    # NOTE: This test will fail with your current code!
    # You need to fix the reassignment issue in data_quality_fixing()
    assert result["close"][1] is None or result["close"][1] != -5.0


# ========================== TEST WIDE TO LONG ==========================

def test_wide_to_long_multi_ticker():
    """
    Test that wide format is converted to long (tidy) format.
    
    Input:  date, close_aapl, close_msft, open_aapl, open_msft
    Output: date, ticker, close, open
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "close_aapl": [100.0, 110.0],
        "close_msft": [200.0, 210.0],
        "open_aapl": [99.0, 109.0],
        "open_msft": [199.0, 209.0],
    })
    
    result = ingest_yf.wide_to_long(df)
    
    # Check shape: 2 dates × 2 tickers = 4 rows
    assert len(result) == 4
    
    # Check columns exist
    assert "date" in result.columns
    assert "ticker" in result.columns
    assert "close" in result.columns
    assert "open" in result.columns
    
    # Check tickers are extracted correctly
    tickers = sorted(result["ticker"].unique().to_list())
    assert tickers == ["aapl", "msft"]
    
    # Check a specific value
    aapl_day1 = result.filter(
        (pl.col("ticker") == "aapl") & (pl.col("date") == date(2020, 1, 1))
    )
    assert aapl_day1["close"][0] == 100.0
    assert aapl_day1["open"][0] == 99.0


def test_wide_to_long_preserves_all_fields():
    """
    Test that all OHLCV fields are preserved in the pivot.
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1)],
        "open_aapl": [99.0],
        "high_aapl": [105.0],
        "low_aapl": [98.0],
        "close_aapl": [103.0],
        "volume_aapl": [1000000.0],
    })
    
    result = ingest_yf.wide_to_long(df)
    
    # All OHLCV columns should exist
    expected_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


# ========================== TEST FULL NORMALIZE_PRICES PIPELINE ==========================

def test_normalize_prices_end_to_end_single_ticker(raw_single_ticker_df):
    """
    Test the full normalization pipeline with single ticker data.
    
    Input: Raw yfinance DataFrame (simple columns)
    Output: Canonical schema [date, ticker, open, high, low, close, volume]
    
    NOTE: This will fail because single-ticker data doesn't have a ticker column.
    You may need to handle this case in your code.
    """
    result = ingest_yf.normalize_prices(raw_single_ticker_df)
    
    # Check essential columns exist
    assert "date" in result.columns
    assert "close" in result.columns
    
    # Check date is correct type
    assert result["date"].dtype == pl.Date
    
    # Check data is sorted by date
    dates = result["date"].to_list()
    assert dates == sorted(dates)


def test_normalize_prices_end_to_end_multi_ticker(raw_multi_ticker_df):
    """
    Test the full normalization pipeline with multi-ticker data.
    """
    result = ingest_yf.normalize_prices(raw_multi_ticker_df)
    
    # Check canonical schema
    assert "date" in result.columns
    assert "ticker" in result.columns
    assert "close" in result.columns
    
    # Check date type
    assert result["date"].dtype == pl.Date
    
    # Check we have both tickers
    tickers = sorted(result["ticker"].unique().to_list())
    assert tickers == ["aapl", "msft"]
    
    # Check shape: 2 dates × 2 tickers = 4 rows
    assert len(result) == 4
    
    # Spot check: AAPL on 2020-01-01 should have close=100
    aapl_data = result.filter(
        (pl.col("ticker") == "aapl") & (pl.col("date") == date(2020, 1, 1))
    )
    assert len(aapl_data) == 1
    assert aapl_data["close"][0] == 100.0


# ========================== TEST CONFIG & VALIDATION ==========================

def test_yf_ingest_config_defaults():
    """
    Test that YFIngestConfig has sensible defaults.
    """
    cfg = ingest_yf.YFIngestConfig(
        tickers=["AAPL"],
        start=date(2020, 1, 1),
        end=date(2020, 12, 31),
        interval="1d"
    )
    
    assert cfg.adjust is True  # Should default to adjusted prices
    assert cfg.out_path == "data/yf_prices.parquet"


def test_yf_ingest_config_validates_tickers():
    """
    Test that config accepts various ticker formats.
    """
    # Single ticker
    cfg1 = ingest_yf.YFIngestConfig(
        tickers=["AAPL"],
        start=date(2020, 1, 1),
        end=date(2020, 12, 31),
        interval="1d"
    )
    assert len(cfg1.tickers) == 1
    
    # Multiple tickers
    cfg2 = ingest_yf.YFIngestConfig(
        tickers=["AAPL", "MSFT", "GOOGL"],
        start=date(2020, 1, 1),
        end=date(2020, 12, 31),
        interval="1d"
    )
    assert len(cfg2.tickers) == 3


# ========================== EDGE CASE TESTS ==========================

def test_handles_empty_ticker_suffix():
    """
    Test that columns like ('Close', '') are handled correctly.
    """
    df = pl.DataFrame({
        "('Date', '')": ["2020-01-01"],
        "('Close', '')": [100.0],
    })
    
    result = ingest_yf.normalize_column_names(df)
    
    assert "date" in result.columns
    assert "close" in result.columns


def test_handles_mixed_case_tickers():
    """
    Test that ticker names are normalized to lowercase.
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1)],
        "close_AAPL": [100.0],
        "close_MsFt": [200.0],
    })
    
    result = ingest_yf.wide_to_long(df)
    
    # Tickers should be lowercase
    tickers = result["ticker"].unique().to_list()
    assert all(t.islower() for t in tickers)


def test_handles_single_date():
    """
    Test that pipeline works with just one date.
    """
    df = pl.DataFrame({
        "date": [date(2020, 1, 1)],
        "close_aapl": [100.0],
        "open_aapl": [99.0],
    })
    
    result = ingest_yf.wide_to_long(df)
    
    assert len(result) == 1
    assert result["date"][0] == date(2020, 1, 1)


# ========================== INTEGRATION TEST (Optional) ==========================

@pytest.mark.skip(reason="Requires actual API call - use only for manual testing")
def test_fetch_yf_data_real_api():
    """
    Integration test that actually calls Yahoo Finance.
    
    ONLY RUN THIS MANUALLY to verify API integration works.
    Skip in CI/CD to avoid rate limits and flakiness.
    
    Usage: pytest tests/test_ingest_yf.py::test_fetch_yf_data_real_api -v
    """
    cfg = ingest_yf.YFIngestConfig(
        tickers=["AAPL"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 5),
        interval="1d"
    )
    
    result = ingest_yf.fetch_yf_data(cfg)
    
    # Should return non-empty DataFrame
    assert len(result) > 0
    
    # Should have expected columns
    assert "Date" in result.columns or "date" in result.columns


# ========================== PERFORMANCE / SMOKE TESTS ==========================

def test_normalize_prices_doesnt_crash_on_large_mock_data():
    """
    Smoke test: Ensure pipeline handles larger datasets without crashing.
    """
    # Create 100 days × 5 tickers = 500 rows
    dates = pl.date_range(
        start=date(2020, 1, 1),
        end=date(2020, 4, 10),
        interval="1d",
        eager=True
    )
    
    df_dict = {"date": dates}
    for ticker in ["aapl", "msft", "googl", "amzn", "tsla"]:
        df_dict[f"close_{ticker}"] = [100.0 + i for i in range(len(dates))]
        df_dict[f"open_{ticker}"] = [99.0 + i for i in range(len(dates))]
    
    df = pl.DataFrame(df_dict)
    
    result = ingest_yf.wide_to_long(df)
    
    # Should have 100 days × 5 tickers = 500 rows
    assert len(result) == len(dates) * 5
    assert "ticker" in result.columns
    assert "date" in result.columns