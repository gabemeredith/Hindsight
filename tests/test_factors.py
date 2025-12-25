"""
Tests for factorlabs/financialfeatures/factors.py

Test Philosophy:
- Use small, deterministic DataFrames (3-5 rows)
- Validate exact numerical outputs
- Test each function in isolation
- Cover edge cases (nulls, insufficient data, multi-ticker)

Run with: pytest tests/test_factors.py -v
"""

import polars as pl
import pytest
from factorlabs.financialfeatures import factors


# ========================== FIXTURES ==========================

@pytest.fixture
def simple_prices():
    """
    Single ticker, 5 days, prices designed for easy mental math.
    Day 1: $100
    Day 2: $110 (+10%)
    Day 3: $121 (+10%)
    Day 4: $108.90 (-10%)
    Day 5: $119.79 (+10%)
    """
    return pl.DataFrame({
        "date": pl.date_range(
            start=pl.date(2020, 1, 1),
            end=pl.date(2020, 1, 5),
            interval="1d",
            eager=True
        ),
        "ticker": ["AAPL"] * 5,
        "close": [100.0, 110.0, 121.0, 108.9, 119.79],
    })


@pytest.fixture
def multi_ticker_prices():
    """
    Two tickers, 3 days each.
    AAPL: $100 -> $110 -> $121
    MSFT: $200 -> $210 -> $220.5
    """
    return pl.DataFrame({
        "date": pl.date_range(
            start=pl.date(2020, 1, 1),
            end=pl.date(2020, 1, 3),
            interval="1d",
            eager=True
        ).to_list() * 2,
        "ticker": ["AAPL"] * 3 + ["MSFT"] * 3,
        "close": [100.0, 110.0, 121.0, 200.0, 210.0, 220.5],
    }).sort("date", "ticker")


# ========================== TEST RETURNS ==========================

def test_calculate_returns_single_day(simple_prices):
    """
    Test 1-day returns are calculated correctly.
    
    Expected:
    Day 1: null (no prior day)
    Day 2: (110/100) - 1 = 0.10 (10%)
    Day 3: (121/110) - 1 = 0.10 (10%)
    Day 4: (108.9/121) - 1 = -0.10 (-10%)
    Day 5: (119.79/108.9) - 1 = 0.10 (10%)
    """
    result = factors.calculate_returns(simple_prices, delay=1)
    
    expected_returns = [None, 0.10, 0.10, -0.10, 0.10]
    actual_returns = result["ret_1d"].to_list()
    
    # First value should be null
    assert actual_returns[0] is None
    
    # Rest should match within floating point tolerance
    for actual, expected in zip(actual_returns[1:], expected_returns[1:]):
        assert abs(actual - expected) < 1e-6, f"Expected {expected}, got {actual}"


def test_calculate_returns_multi_day(simple_prices):
    """
    Test 2-day returns.
    
    Expected:
    Day 1: null
    Day 2: null
    Day 3: (121/100) - 1 = 0.21 (21%)
    Day 4: (108.9/110) - 1 = -0.01 (-1%)
    Day 5: (119.79/121) - 1 = -0.01 (-1%)
    """
    result = factors.calculate_returns(simple_prices, delay=2)
    
    actual_returns = result["ret_1d"].to_list()
    
    # First two should be null
    assert actual_returns[0] is None
    assert actual_returns[1] is None
    
    # Check remaining values
    assert abs(actual_returns[2] - 0.21) < 1e-6
    assert abs(actual_returns[3] - (-0.01)) < 1e-6


def test_calculate_returns_respects_ticker_groups(multi_ticker_prices):
    """
    Ensure returns are calculated separately per ticker.
    
    AAPL day 2: (110/100) - 1 = 0.10
    MSFT day 2: (210/200) - 1 = 0.05
    """
    result = factors.calculate_returns(multi_ticker_prices, delay=1)
    
    aapl_returns = result.filter(pl.col("ticker") == "AAPL")["ret_1d"].to_list()
    msft_returns = result.filter(pl.col("ticker") == "MSFT")["ret_1d"].to_list()
    
    # AAPL: [null, 0.10, 0.10]
    assert aapl_returns[0] is None
    assert abs(aapl_returns[1] - 0.10) < 1e-6
    
    # MSFT: [null, 0.05, 0.05]
    assert msft_returns[0] is None
    assert abs(msft_returns[1] - 0.05) < 1e-6


# ========================== TEST MOMENTUM ==========================

def test_calculate_momentum(simple_prices):
    """
    Momentum is (close / close.shift(delay)) - 1
    Same as returns, but different column name.
    
    10-day momentum on day 5 (delay=4):
    (119.79 / 110) - 1 = 0.089 (8.9%)
    """
    result = factors.calculate_momentum(simple_prices, delay=1, title="mom_1d")
    
    expected = [None, 0.10, 0.10, -0.10, 0.10]
    actual = result["mom_1d"].to_list()
    
    assert actual[0] is None
    for a, e in zip(actual[1:], expected[1:]):
        assert abs(a - e) < 1e-6


def test_calculate_momentum_longer_window(simple_prices):
    """
    Test momentum with delay=4 (almost entire window).
    Only last day should have a value.
    """
    result = factors.calculate_momentum(simple_prices, delay=4, title="mom_4d")
    
    actual = result["mom_4d"].to_list()
    
    # First 4 should be null
    assert all(v is None for v in actual[:4])
    
    # Last value: (119.79/100) - 1 = 0.1979
    assert abs(actual[4] - 0.1979) < 1e-4


# ========================== TEST SMA ==========================

def test_calculate_sma(simple_prices):
    """
    Test simple moving average.
    
    SMA(3) on day 3: (100 + 110 + 121) / 3 = 110.33
    SMA(3) on day 4: (110 + 121 + 108.9) / 3 = 113.30
    SMA(3) on day 5: (121 + 108.9 + 119.79) / 3 = 116.56
    """
    result = factors.calculate_sma(simple_prices, delay=3, title="sma_3d")
    
    actual = result["sma_3d"].to_list()
    
    # First 2 should be null (insufficient window)
    assert actual[0] is None
    assert actual[1] is None
    
    # Check calculated values
    assert abs(actual[2] - 110.33) < 0.01
    assert abs(actual[3] - 113.30) < 0.01
    assert abs(actual[4] - 116.56) < 0.01


def test_calculate_sma_single_day():
    """
    SMA(1) should just equal close price.
    """
    df = pl.DataFrame({
        "date": [pl.date(2020, 1, 1), pl.date(2020, 1, 2)],
        "ticker": ["AAPL", "AAPL"],
        "close": [100.0, 110.0],
    })
    
    result = factors.calculate_sma(df, delay=1, title="sma_1d")
    
    assert result["sma_1d"].to_list() == [100.0, 110.0]


# ========================== TEST VOLATILITY ==========================

def test_calculate_volatility_requires_returns(simple_prices):
    """
    Volatility calculation depends on ret_1d existing.
    This test ensures the function fails gracefully if ret_1d is missing.
    """
    with pytest.raises(Exception):  # Polars will raise ColumnNotFoundError
        factors.calculate_volitility(simple_prices, delay=3, title="vol_3d")


def test_calculate_volatility_with_returns(simple_prices):
    """
    Volatility is std dev of returns over window.
    
    Given returns: [null, 0.10, 0.10, -0.10, 0.10]
    Vol(3) on day 4: std([0.10, 0.10, -0.10]) = 0.1155
    Vol(3) on day 5: std([0.10, -0.10, 0.10]) = 0.1155
    """
    # First calculate returns
    df_with_returns = factors.calculate_returns(simple_prices, delay=1)
    
    result = factors.calculate_volitility(df_with_returns, delay=3, title="vol_3d")
    
    actual = result["vol_3d"].to_list()
    
    # First 3 should be null (insufficient window)
    assert all(v is None for v in actual[:3])
    
    # Check day 4 and 5 (both should be ~0.1155)
    assert abs(actual[3] - 0.1155) < 0.001
    assert abs(actual[4] - 0.1155) < 0.001


# ========================== TEST EDGE CASES ==========================

def test_empty_dataframe():
    """
    Ensure functions don't crash on empty input.
    """
    empty_df = pl.DataFrame({
        "date": [],
        "ticker": [],
        "close": [],
    })
    
    result = factors.calculate_returns(empty_df, delay=1)
    assert len(result) == 0


def test_insufficient_data_for_window():
    """
    If window is longer than data, all values should be null.
    """
    df = pl.DataFrame({
        "date": [pl.date(2020, 1, 1), pl.date(2020, 1, 2)],
        "ticker": ["AAPL", "AAPL"],
        "close": [100.0, 110.0],
    })
    
    result = factors.calculate_momentum(df, delay=10, title="mom_10d")
    
    # Both values should be null (window too large)
    assert all(v is None for v in result["mom_10d"].to_list())


def test_null_prices():
    """
    Test behavior when close prices contain nulls.
    """
    df = pl.DataFrame({
        "date": pl.date_range(
            start=pl.date(2020, 1, 1),
            end=pl.date(2020, 1, 4),
            interval="1d",
            eager=True
        ),
        "ticker": ["AAPL"] * 4,
        "close": [100.0, None, 121.0, 108.9],
    })
    
    result = factors.calculate_returns(df, delay=1)
    
    # Returns should handle nulls gracefully
    returns = result["ret_1d"].to_list()
    assert returns[0] is None  # No prior day
    assert returns[1] is None  # Prior day was 100, but this day is null
    assert returns[2] is None  # Prior day was null
    # Day 4: (108.9 / 121) - 1 = -0.10
    assert abs(returns[3] - (-0.10)) < 1e-6


# ========================== TEST LOG RETURNS ==========================

def test_calculate_log_return(simple_prices):
    """
    Log return is ln(close / close.shift(delay))
    
    Day 2: ln(110/100) = ln(1.10) = 0.0953
    Day 3: ln(121/110) = ln(1.10) = 0.0953
    """
    result = factors.calculate_log_return(simple_prices, delay=1)
    
    actual = result["log_return"].to_list()
    
    assert actual[0] is None
    assert abs(actual[1] - 0.0953) < 0.001
    assert abs(actual[2] - 0.0953) < 0.001