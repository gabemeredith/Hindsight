"""
Tests for Backtester class

Test Philosophy:
- Use small, deterministic price data (3-5 days)
- Hand-calculate expected equity curves and trades
- Test one behavior per test
- Start simple (buy and hold) → complex (rebalancing)

Run with: pytest tests/test_backtester.py -v
"""

import pytest
import polars as pl
from datetime import date
from hindsightpy.backtest.backtester import Backtester, BacktestConfig, BacktestResult


# ========================== FIXTURES ==========================

@pytest.fixture
def simple_prices():
    """
    3 days, single ticker, simple round numbers for easy calculation.

    Day 1: AAPL = $100
    Day 2: AAPL = $110 (+10%)
    Day 3: AAPL = $121 (+10%)
    """
    return pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "close": [100.0, 110.0, 121.0]
    })


@pytest.fixture
def two_stock_prices():
    """
    3 days, two tickers.

    AAPL: $100 → $110 → $121
    MSFT: $50 → $55 → $60.50
    """
    return pl.DataFrame({
        "date": [
            date(2020, 1, 1), date(2020, 1, 1),
            date(2020, 1, 2), date(2020, 1, 2),
            date(2020, 1, 3), date(2020, 1, 3)
        ],
        "ticker": ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL", "MSFT"],
        "close": [100.0, 50.0, 110.0, 55.0, 121.0, 60.50]
    })


# ========================== BASIC TESTS ==========================

def test_backtester_buy_and_hold_single_stock(simple_prices):
    """
    Test simplest backtest: buy once on day 1, hold to end.

    Setup:
    - Initial cash: $10,000
    - Strategy: 100% AAPL (never rebalance)
    - Prices: $100 → $110 → $121

    Expected sequence:

    Day 1 (2020-01-01):
    - Start: Cash = $10,000
    - Target: 100% AAPL @ $100
    - Trade: Buy 100 shares (calculate: $10,000 / $100)
    - End: Cash = 0, Shares = 100, Portfolio Value = 10,000

    Day 2 (2020-01-02):
    - Price = $110
    - Target: 100% AAPL (no rebalance)
    - Trade: None (already at target)
    - Portfolio Value = 11,000 (calculate: cash + shares * price)

    Day 3 (2020-01-03):
    - Price = $121
    - Target: 100% AAPL (no rebalance)
    - Trade: None
    - Portfolio Value = 12,100 (calculate: cash + shares * price)

    Calculate and fill in the ??? values!
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        initial_cash=10000.0,
        rebalance_frequency="daily"  # Check every day, but won't trade if at target
    )

    backtester = Backtester()

    # Static target: 100% AAPL, never changes
    result = backtester.run(
        prices=simple_prices,
        strategy={"AAPL": 1.0},
        config=config
    )

    # Check that we have a result object
    assert isinstance(result, BacktestResult)

    # Check equity curve has 3 rows (one per day)
    assert len(result.equity_curve) == 3  # How many days?

    # Day 1: Should buy ??? shares
    day1_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 1))["portfolio_value"][0]
    assert day1_value == 10000  # What's the portfolio value after buying?

    # Day 2: No trades, but value changes due to price
    day2_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 2))["portfolio_value"][0]
    assert day2_value == 11000  # Calculate: shares * $110

    # Day 3: Final value
    day3_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 3))["portfolio_value"][0]
    assert day3_value == 12100  # Calculate: shares * $121

    # Check trades: Should only have 1 trade (initial purchase)
    assert len(result.trades) == 1  # How many trades total?

    # First trade should be buy AAPL
    first_trade = result.trades.row(0,named=True)
    assert first_trade["ticker"] == "AAPL"
    assert first_trade["side"] == "buy"
    assert first_trade["shares"] == 100  # How many shares bought?
    assert first_trade["date"] == date(2020, 1, 1)


def test_backtester_daily_rebalance_single_stock(simple_prices):
    """
    Test rebalancing back to target weight each day.

    Setup:
    - Initial cash: $10,000
    - Strategy: 50% AAPL, 50% cash (rebalance daily)
    - Prices: $100 → $110 → $121

    Expected sequence:

    Day 1 (2020-01-01):
    - Start: Cash = $10,000
    - Target: 50% AAPL ($5,000 worth)
    - Trade: Buy 50 shares (calculate: $5,000 / $100)
    - End: Cash = 5000, Shares = 50, Portfolio Value = 5000

    Day 2 (2020-01-02):
    - Price = $110
    - Current: Shares worth 5500 (calculate: shares * $110)
    - Current allocation: ~52% AAPL (has grown above 50%)
    - Target: 50% AAPL (need to sell some to get back to 50%)
    - Total value = 10500
    - Target AAPL value = 5250 (50% of total)
    - Target shares = 48
    - Trade: Sell 2.272727 shares
    - End: Cash = 5250, Shares = 47.727272, Portfolio Value = 10500

    Day 3 (2020-01-03):
    - Price = $121
    - (Similar calculation - AAPL has grown again)
    - 5808$ of AAPL, 11028 total, 26 shares
    - Trade: Sell 22 shares to rebalance to 50%

    Calculate and fill in the ??? values!
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        initial_cash=10000.0,
        rebalance_frequency="daily"
    )

    backtester = Backtester()

    # Target: 50% AAPL, 50% cash (rebalance every day)
    result = backtester.run(
        prices=simple_prices,
        strategy={"AAPL": 0.5},
        config=config
    )

    # Check equity curve
    assert len(result.equity_curve) == 3

    # Day 1: Initial purchase
    day1_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 1))["portfolio_value"][0]
    assert day1_value == 10000  # Should be $10,000 (just bought 50%)

    # Day 2: After first rebalance
    day2_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 2))["portfolio_value"][0]
    assert day2_value == 10500  # Calculate total value after AAPL grew and we rebalanced

    # Day 3: After second rebalance
    day3_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 3))["portfolio_value"][0]
    assert day3_value == 11025

    # Check trades: Should have 3 trades (1 buy on day 1, then 2 sells for rebalancing)
    assert len(result.trades) == 3  # How many trades?

    # First trade: Buy
    trade_0 = result.trades.row(0, named=True)
    assert trade_0["side"] == "buy"
    assert trade_0["shares"] == 50

    # Second trade: Sell (rebalance on day 2)
    trade_1 = result.trades.row(1, named=True)
    assert trade_1["side"] == "sell"
    assert abs(trade_1["shares"] - 2.272727) < 0.001  # Fractional shares

    # Third trade: Sell (rebalance on day 3)
    trade_2 = result.trades.row(2, named=True)
    assert trade_2["side"] == "sell"


def test_backtester_two_stocks_60_40(two_stock_prices):
    """
    Test classic 60/40 portfolio (no rebalancing for simplicity).

    Setup:
    - Initial cash: $10,000
    - Strategy: 60% AAPL, 40% MSFT (buy once, no rebalance)
    - Day 1: AAPL=$100, MSFT=$50

    Day 1 Trades:
    - Target AAPL value = 6000 (60% of $10,000)
    - Target AAPL shares = 60 (value / price)
    - Target MSFT value = 4000 (40% of $10,000)
    - Target MSFT shares = 80 (value / price)

    Day 2: AAPL=$110, MSFT=$55
    - Portfolio value = 6600 + 4400 = 11000 

    Day 3: AAPL=$121, MSFT=$60.50
    - Portfolio value = 12100
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        initial_cash=10000.0,
        rebalance_frequency="never"  # Buy and hold
    )

    backtester = Backtester()

    result = backtester.run(
        prices=two_stock_prices,
        strategy={"AAPL": 0.6, "MSFT": 0.4},
        config=config
    )

    # Should have 3 days in equity curve
    assert len(result.equity_curve) == 3

    # Day 1: Initial purchase
    day1_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 1))["portfolio_value"][0]
    assert day1_value == 10000

    # Day 3: Final value
    day3_value = result.equity_curve.filter(pl.col("date") == date(2020, 1, 3))["portfolio_value"][0]
    assert day3_value == 12100  # Calculate: (aapl_shares * $121) + (msft_shares * $60.50)

    # Should have exactly 2 trades (initial purchases only)
    assert len(result.trades) == 2

    # Check AAPL purchase
    trades_list = list(result.trades.iter_rows(named=True))
    aapl_trade = [t for t in trades_list if t["ticker"] == "AAPL"][0]
    assert aapl_trade["shares"] == 60

    # Check MSFT purchase
    msft_trade = [t for t in trades_list if t["ticker"] == "MSFT"][0]
    assert msft_trade["shares"] == 80


def test_backtester_equity_curve_structure(simple_prices):
    """
    Test that equity curve has correct columns and structure.

    Expected columns:
    - date: Date of the snapshot
    - portfolio_value: Total value (cash + positions)
    - cash: Cash balance
    - positions_value: Sum of all position values
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        initial_cash=10000.0,
        rebalance_frequency="never"
    )

    backtester = Backtester()
    result = backtester.run(
        prices=simple_prices,
        strategy={"AAPL": 1.0},
        config=config
    )

    # Check columns exist
    equity_curve = result.equity_curve
    assert "date" in equity_curve.columns
    assert "portfolio_value" in equity_curve.columns
    assert "cash" in equity_curve.columns
    assert "positions_value" in equity_curve.columns

    # Check that portfolio_value = cash + positions_value (accounting identity)
    for row in equity_curve.iter_rows(named=True):
        calculated_value = row["cash"] + row["positions_value"]
        assert abs(calculated_value - row["portfolio_value"]) < 0.01  # Floating point tolerance


def test_backtester_trades_structure(simple_prices):
    """
    Test that trades DataFrame has correct columns.

    Expected columns:
    - date: Trade date
    - ticker: Stock symbol
    - shares: Number of shares
    - price: Execution price
    - side: "buy" or "sell"
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        initial_cash=10000.0,
        rebalance_frequency="daily"
    )

    backtester = Backtester()
    result = backtester.run(
        prices=simple_prices,
        strategy={"AAPL": 0.5},
        config=config
    )

    # Check columns exist
    trades = result.trades
    assert "date" in trades.columns
    assert "ticker" in trades.columns
    assert "shares" in trades.columns
    assert "price" in trades.columns
    assert "side" in trades.columns

    # Check that all trades have valid sides
    for row in trades.iter_rows(named=True):
        assert row["side"] in ["buy", "sell"]


# ========================== EDGE CASES ==========================

def test_backtester_no_trades_when_at_target(simple_prices):
    """
    Test that no trades are generated when already at target.

    If we start with cash and target is 0% stocks (100% cash),
    should have no trades.
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        initial_cash=10000.0,
        rebalance_frequency="daily"
    )

    backtester = Backtester()

    # Target: 100% cash (empty dict)
    result = backtester.run(
        prices=simple_prices,
        strategy={},
        config=config
    )

    # Should have no trades
    assert len(result.trades) == 0  # How many trades expected?

    # Equity curve should be flat at $10,000
    for row in result.equity_curve.iter_rows(named=True):
        assert row["portfolio_value"] == 10000
        assert row["cash"] == 10000
        assert row["positions_value"] == 0


def test_backtester_single_day(simple_prices):
    """
    Test backtest with only 1 day (start_date == end_date).

    Should execute initial trades and return 1-row equity curve.
    """
    config = BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 1),  # Same day
        initial_cash=10000.0,
        rebalance_frequency="daily"
    )

    backtester = Backtester()
    result = backtester.run(
        prices=simple_prices,
        strategy={"AAPL": 1.0},
        config=config
    )

    # Should have exactly 1 day in equity curve
    assert len(result.equity_curve) == 1

    # Should have 1 trade (initial purchase)
    assert len(result.trades) == 1