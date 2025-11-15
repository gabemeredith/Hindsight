# tests/test_ingest_yf.py
"""
Comprehensive test suite for YFinance data ingestion pipeline.

Test Categories:
1. Unit tests - individual function behavior
2. Integration tests - full pipeline flow
3. Data quality tests - schema and value validation
4. Edge case tests - handle malformed/missing data
"""

import sys
from pathlib import Path
from datetime import date
from typing import List
import ast

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
import polars as pl
from factorlabs.data.ingest_yf import (
    YFIngestConfig,
    normalize_column_names,
    normalize_date_column,
    data_quality_fixing,
    normalize_prices,
)


# ============================================================================
# FIXTURES - Reusable test data
# ============================================================================

@pytest.fixture
def sample_raw_yf_single_ticker():
    """Raw yfinance data for single ticker (no MultiIndex)"""
    return pl.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Open": [100.0, 101.0, 102.0],
        "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [100.5, 101.5, 102.5],
        "Adj Close": [100.5, 101.5, 102.5],
        "Volume": [1000000, 1100000, 1200000],
    })


@pytest.fixture
def sample_raw_yf_multi_ticker():
    """Raw yfinance data with MultiIndex columns (multiple tickers)"""
    return pl.DataFrame({
        "('date', '')": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "('open', 'aapl')": [100.0, 101.0, None],
        "('high', 'aapl')": [101.0, 102.0, None],
        "('low', 'aapl')": [99.0, 100.0, None],
        "('close', 'aapl')": [100.5, 101.5, None],
        "('volume', 'aapl')": [1000000, 1100000, None],
        "('open', 'msft')": [200.0, None, 202.0],
        "('high', 'msft')": [201.0, None, 203.0],
        "('low', 'msft')": [199.0, None, 201.0],
        "('close', 'msft')": [200.5, None, 202.5],
        "('volume', 'msft')": [2000000, None, 2200000],
    })


@pytest.fixture
def sample_dirty_data():
    """Data with quality issues for testing data_quality_fixing"""
    return pl.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT", "TSLA"],
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)],
        "open": [100.0, None, -5.0, 200.0, 150.0],  # Row 2 all null, Row 3 negative
        "high": [101.0, None, 10.0, 201.0, 151.0],
        "low": [99.0, None, 8.0, 199.0, 149.0],
        "close": [100.5, None, 9.0, 200.5, 150.5],
        "volume": [1000000, None, 500000, 2000000, 1500000],
    })


# ============================================================================
# UNIT TESTS - Test individual functions
# ============================================================================

class TestNormalizeColumnNames:
    """Test column name normalization logic"""
    
    def test_multiindex_column_parsing(self):
        """Test that MultiIndex columns are properly flattened"""
        df = pl.DataFrame({
            "('close', 'aapl')": [100.0, 101.0],
            "('volume', 'msft')": [1000, 2000],
            "('date', '')": ["2024-01-01", "2024-01-02"],
        })
        
        result = normalize_column_names(df)
        
        assert "close_aapl" in result.columns
        assert "volume_msft" in result.columns
        assert "date" in result.columns
        assert len(result.columns) == 3
    
    def test_simple_column_lowercasing(self):
        """Test that simple columns are lowercased"""
        df = pl.DataFrame({
            "Date": ["2024-01-01"],
            "Open": [100.0],
            "Close": [101.0],
        })
        
        result = normalize_column_names(df)
        
        assert result.columns == ["date", "open", "close"]
    
    def test_preserves_data(self):
        """Ensure no data loss during renaming"""
        df = pl.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [100.0, 101.0],
        })
        
        result = normalize_column_names(df)
        
        assert len(result) == 2
        assert result["close"].to_list() == [100.0, 101.0]


class TestNormalizeDateColumn:
    """Test date column type handling"""
    
    def test_string_dates_converted(self):
        """String dates should be parsed to Date dtype"""
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"]
        })
        
        result = normalize_date_column(df)
        
        assert result["date"].dtype == pl.Date
        assert result["date"][0] == date(2024, 1, 1)
    
    def test_datetime_converted_to_date(self):
        """Datetime columns should be stripped to Date"""
        df = pl.DataFrame({
            "date": [
                pl.datetime(2024, 1, 1, 12, 30),
                pl.datetime(2024, 1, 2, 14, 45),
            ]
        })
        
        result = normalize_date_column(df)
        
        assert result["date"].dtype == pl.Date
        assert result["date"][0] == date(2024, 1, 1)
    
    def test_date_unchanged(self):
        """Date columns should pass through unchanged"""
        df = pl.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)]
        })
        
        result = normalize_date_column(df)
        
        assert result["date"].dtype == pl.Date
        assert len(result) == 2


class TestDataQualityFixing:
    """Test data quality validation and cleaning"""
    
    def test_removes_all_null_rows(self, sample_dirty_data):
        """Rows with all null OHLCV should be removed"""
        result = data_quality_fixing(sample_dirty_data)
        
        # Row 2 (AAPL 2024-01-02) has all nulls except ticker/date
        assert len(result) == 4  # Started with 5, removed 1
        
        # Verify the all-null row is gone
        aapl_dates = result.filter(pl.col("ticker") == "AAPL")["date"].to_list()
        assert date(2024, 1, 2) not in aapl_dates
    
    def test_keeps_partial_null_rows(self):
        """Rows with some nulls but valid data should be kept"""
        df = pl.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "open": [100.0, None],  # MSFT has null open
            "high": [101.0, 201.0],  # but valid high
            "close": [100.5, 200.5],
            "volume": [1000, 2000],
        })
        
        result = data_quality_fixing(df)
        
        assert len(result) == 2  # Both rows kept
    
    def test_empty_dataframe_handling(self):
        """Should handle empty DataFrames gracefully"""
        df = pl.DataFrame({
            "ticker": [],
            "date": [],
            "open": [],
            "close": [],
        })
        
        result = data_quality_fixing(df)
        
        assert len(result) == 0
        assert result.columns == df.columns


# ============================================================================
# INTEGRATION TESTS - Test full pipeline
# ============================================================================

class TestNormalizePrices:
    """Test the complete normalization pipeline"""
    
    def test_full_pipeline_single_ticker(self, sample_raw_yf_single_ticker):
        """Test end-to-end normalization for single ticker"""
        result = normalize_prices(sample_raw_yf_single_ticker)
        
        # Check schema
        assert "date" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns
        assert result["date"].dtype == pl.Date
        
        # Check data integrity
        assert len(result) == 3
        assert result["open"][0] == 100.0
    
    def test_full_pipeline_multi_ticker(self, sample_raw_yf_multi_ticker):
        """Test end-to-end normalization for multiple tickers"""
        result = normalize_prices(sample_raw_yf_multi_ticker)
        
        # Check columns normalized
        assert "date" in result.columns
        assert "open_aapl" in result.columns or "open" in result.columns
        
        # Check nulls removed
        # Original had rows with all nulls for each ticker
        assert len(result) < len(sample_raw_yf_multi_ticker)


# ============================================================================
# PROPERTY-BASED TESTS - Test data invariants
# ============================================================================

class TestDataInvariants:
    """Test that data quality properties hold after processing"""
    
    def test_no_fully_null_rows(self, sample_dirty_data):
        """After cleaning, no row should have all OHLCV columns null"""
        result = data_quality_fixing(sample_dirty_data)
        
        # Get OHLCV columns (exclude ticker, date)
        ohlcv_cols = [c for c in result.columns if c not in ["ticker", "date"]]
        
        # Check no row has all nulls
        for i in range(len(result)):
            row_values = [result[col][i] for col in ohlcv_cols]
            assert not all(v is None for v in row_values), f"Row {i} has all nulls"
    
    def test_date_monotonicity_per_ticker(self):
        """Dates should be sorted within each ticker"""
        df = pl.DataFrame({
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
            "close": [100.0, 101.0, 102.0],
        })
        
        # Note: normalize_prices should sort by ticker, date
        # This test documents expected behavior
        # You may need to add sorting to normalize_prices if not present
        pass  # Placeholder for sorting test


# ============================================================================
# SNAPSHOT TESTS - Visual inspection helpers
# ============================================================================

class TestDataVisualization:
    """Tests that help visualize data transformations"""
    
    def test_show_before_after_normalization(self, sample_raw_yf_multi_ticker, capsys):
        """Print before/after for visual inspection"""
        print("\n" + "="*80)
        print("BEFORE NORMALIZATION:")
        print("="*80)
        print(sample_raw_yf_multi_ticker)
        
        result = normalize_prices(sample_raw_yf_multi_ticker)
        
        print("\n" + "="*80)
        print("AFTER NORMALIZATION:")
        print("="*80)
        print(result)
        
        # This test always passes - it's for visual inspection
        assert True
    
    def test_show_data_quality_changes(self, sample_dirty_data, capsys):
        """Show what data_quality_fixing removes"""
        print("\n" + "="*80)
        print("BEFORE DATA QUALITY FIXING:")
        print("="*80)
        print(sample_dirty_data)
        print(f"Row count: {len(sample_dirty_data)}")
        
        result = data_quality_fixing(sample_dirty_data)
        
        print("\n" + "="*80)
        print("AFTER DATA QUALITY FIXING:")
        print("="*80)
        print(result)
        print(f"Row count: {len(result)}")
        print(f"Removed {len(sample_dirty_data) - len(result)} rows")
        
        assert True


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test handling of unusual or boundary conditions"""
    
    def test_all_rows_null(self):
        """DataFrame with all null OHLCV rows"""
        df = pl.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "open": [None, None],
            "close": [None, None],
        })
        
        result = data_quality_fixing(df)
        
        assert len(result) == 0  # All rows removed
    
    def test_single_row_dataframe(self):
        """Single row should be handled correctly"""
        df = pl.DataFrame({
            "ticker": ["AAPL"],
            "date": [date(2024, 1, 1)],
            "open": [100.0],
            "close": [101.0],
        })
        
        result = data_quality_fixing(df)
        
        assert len(result) == 1


# ============================================================================
# PARAMETRIZED TESTS - Test multiple scenarios efficiently
# ============================================================================

@pytest.mark.parametrize("date_input,expected_type", [
    (["2024-01-01", "2024-01-02"], pl.Date),
    ([pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 2)], pl.Date),
    ([date(2024, 1, 1), date(2024, 1, 2)], pl.Date),
])
def test_date_normalization_parametrized(date_input, expected_type):
    """Test various date input formats"""
    df = pl.DataFrame({"date": date_input})
    result = normalize_date_column(df)
    assert result["date"].dtype == expected_type


if __name__ == "__main__":
    # Run tests with verbose output and print statements
    pytest.main([__file__, "-v", "-s"])