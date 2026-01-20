"""
Tests for Portfolio class.

Portfolio tracks:
- Cash
- Positions (ticker, shares, entry price)
- Total value
"""

import sys
sys.path.insert(0, 'src')

import pytest
from datetime import date
from hindsightpy.backtest.portfolio import Portfolio


def test_portfolio_starts_with_cash():
    """
    Test that a new portfolio has initial cash and no positions.
    
    Given: Create portfolio with $100,000
    When: Check initial state
    Then: Cash = $100,000, positions = empty, total value = $100,000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    assert portfolio.cash == 100000.0
    assert len(portfolio.positions) == 0
    assert portfolio.get_total_value() == 100000.0
    
def test_portfolio_buy_single_position():
    """
    Test buying a single position.
    
    Given: Portfolio with $100,000
    When: Buy 100 shares of AAPL at $150
    Then: 
        - Cash = $100,000 - $15,000 = $85,000
        - Positions = {AAPL: 100 shares @ $150}
        - Total value = $85,000 + 100*$150 = $100,000 (unchanged)
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Execute buy
    portfolio.buy(
        ticker="AAPL",
        shares=100.0,
        price=150.0,
        date=date(2020, 1, 1)
    )
    
    # Check cash decreased
    assert portfolio.cash == 85000.0
    
    # Check position exists
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].shares == 100.0
    assert portfolio.positions["AAPL"].entry_price == 150.0
    
    # Check total value unchanged (cash out, stock in)
    current_prices = {"AAPL": 150.0}
    assert portfolio.get_total_value(current_prices) == 100000.0
    
def test_portfolio_value_changes_with_price():
    """
    Test that portfolio value updates when stock prices change.
    
    Given: Portfolio with 100 shares of AAPL bought at $150
    When: Price moves to $160
    Then: Total value = $85,000 cash + 100*$160 = $101,000
          Unrealized P&L = $1,000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Price goes up $10
    new_prices = {"AAPL": 160.0}
    new_value = portfolio.get_total_value(new_prices)
    
    assert new_value == 101000.0
    
    # Calculate unrealized P&L
    cost_basis = 100.0 * 150.0  # What we paid
    market_value = 100.0 * 160.0  # What it's worth now
    unrealized_pnl = market_value - cost_basis
    
    assert unrealized_pnl == 1000.0

# ==================== SELL ====================
def test_portfolio_sell_entire_position():
    """
    Test selling entire position.
    
    Given: Portfolio with 100 shares AAPL bought at $150
    When: Sell all 100 shares at $160
    Then:
        - Cash increases by 100 * $160 = $16,000
        - Final cash = $85,000 + $16,000 = $101,000
        - Position removed from portfolio
        - Total value = $101,000 (all cash, no positions)
    
    Math check:
    - Started with $100,000
    - Bought 100 @ $150 = -$15,000 → cash = $85,000
    - Sold 100 @ $160 = +$16,000 → cash = $101,000
    - Profit = $1,000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Buy 100 shares at $150
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    assert portfolio.cash == 85000.0  # Sanity check
    
    # Sell all 100 shares at $160
    portfolio.sell("AAPL", 100.0, 160.0, date(2020, 1, 10))
    
    # Check cash increased
    assert portfolio.cash == 101000.0
    
    # Check position removed
    assert "AAPL" not in portfolio.positions
    
    # Check total value is all cash now
    assert portfolio.get_total_value() == 101000.0


def test_portfolio_sell_partial_position():
    """
    Test selling part of a position.
    
    Given: Portfolio with 100 shares AAPL bought at $150
    When: Sell 40 shares at $160
    Then:
        - Cash increases by 40 * $160 = $6,400
        - Final cash = $85,000 + $6,400 = $91,400
        - Position reduced to 60 shares
        - Entry price stays $150 (original cost basis)
    
    Math check:
    - Started with $100,000
    - Bought 100 @ $150 = -$15,000 → cash = $85,000
    - Sold 40 @ $160 = +$6,400 → cash = $91,400
    - Remaining: 60 shares @ $150 cost basis
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Buy 100 shares
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Sell 40 shares
    portfolio.sell("AAPL", 40.0, 160.0, date(2020, 1, 10))
    
    # Check cash
    assert portfolio.cash == 91400.0
    
    # Check position reduced
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].shares == 60.0
    
    # Check entry price unchanged (cost basis stays same)
    assert portfolio.positions["AAPL"].entry_price == 150.0


def test_portfolio_sell_at_loss():
    """
    Test selling at a loss.
    
    Given: Bought 100 shares at $150
    When: Sell all at $140 (loss of $10/share)
    Then:
        - Cash increases by 100 * $140 = $14,000
        - Final cash = $85,000 + $14,000 = $99,000
        - Position removed
        - Realized loss = $1,000
    
    Math check:
    - Started with $100,000
    - Bought 100 @ $150 = -$15,000 → cash = $85,000
    - Sold 100 @ $140 = +$14,000 → cash = $99,000
    - Loss = $1,000 (ended with less than we started)
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Buy at $150
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Sell at $140 (loss)
    portfolio.sell("AAPL", 100.0, 140.0, date(2020, 1, 10))
    
    # Check cash
    assert portfolio.cash == 99000.0
    
    # Check position gone
    assert "AAPL" not in portfolio.positions
    
    # Check total value (took a $1,000 loss)
    assert portfolio.get_total_value() == 99000.0


def test_portfolio_sell_multiple_partial_sales():
    """
    Test selling in multiple chunks.
    
    Given: 100 shares bought at $150
    When: Sell 30 shares, then sell 50 more shares
    Then: Left with 20 shares
    
    Math check:
    - Start: $100,000 cash
    - Buy 100 @ $150 = -$15,000 → cash = $85,000
    - Sell 30 @ $155 = +$4,650 → cash = $89,650
    - Sell 50 @ $160 = +$8,000 → cash = $97,650
    - Remaining: 20 shares
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Buy 100 shares
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    assert portfolio.positions["AAPL"].shares == 100.0
    
    # First sale: 30 shares
    portfolio.sell("AAPL", 30.0, 155.0, date(2020, 1, 5))
    assert portfolio.positions["AAPL"].shares == 70.0
    assert portfolio.cash == 89650.0
    
    # Second sale: 50 shares
    portfolio.sell("AAPL", 50.0, 160.0, date(2020, 1, 10))
    assert portfolio.positions["AAPL"].shares == 20.0
    assert portfolio.cash == 97650.0
    
# ========================== ERROR HANDLING TESTS ==========================

def test_portfolio_sell_more_than_owned():
    """
    Test error when selling more shares than owned.
    
    Given: Own 100 shares
    When: Try to sell 150 shares
    Then: Raises ValueError
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Should raise error
    with pytest.raises(ValueError, match="Insufficient shares"):
        portfolio.sell("AAPL", 150.0, 160.0, date(2020, 1, 10))
    
    # Portfolio should be unchanged (transaction didn't happen)
    assert portfolio.positions["AAPL"].shares == 100.0
    assert portfolio.cash == 85000.0


def test_portfolio_sell_stock_not_owned():
    """
    Test error when selling stock you don't own.
    
    Given: No MSFT position
    When: Try to sell MSFT
    Then: Raises ValueError
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Should raise error
    with pytest.raises(ValueError, match="No position in MSFT"):
        portfolio.sell("MSFT", 100.0, 200.0, date(2020, 1, 1))
    
    # Cash unchanged
    assert portfolio.cash == 100000.0


def test_portfolio_sell_negative_shares():
    """
    Test error when selling negative shares.
    
    Given: Own 100 shares
    When: Try to sell -50 shares
    Then: Raises ValueError
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Should raise error
    with pytest.raises(ValueError, match="Shares must be positive"):
        portfolio.sell("AAPL", -50.0, 160.0, date(2020, 1, 10))


def test_portfolio_sell_zero_shares():
    """
    Test error when selling zero shares.
    
    Given: Own 100 shares
    When: Try to sell 0 shares
    Then: Raises ValueError
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Should raise error
    with pytest.raises(ValueError, match="Shares must be positive"):
        portfolio.sell("AAPL", 0.0, 160.0, date(2020, 1, 10))


# ========================== EDGE CASE TESTS ==========================

def test_portfolio_sell_with_fractional_shares():
    """
    Test selling fractional shares (some brokers allow this).
    
    Given: Own 100.5 shares
    When: Sell 50.25 shares
    Then: Left with 50.25 shares
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Buy fractional shares
    portfolio.buy("AAPL", 100.5, 150.0, date(2020, 1, 1))
    
    # Sell fractional shares
    portfolio.sell("AAPL", 50.25, 160.0, date(2020, 1, 10))
    
    # Check remaining shares
    
    assert abs(portfolio.positions["AAPL"].shares - 50.25) < 0.0001
    
    # Check cash (50.25 * 160 = 8040) #NOTE: CHECK THIS TEST CASE!
    # expected_cash = 85000.0 - (100.5 * 150.0) + (50.25 * 160.0)
    # assert abs(portfolio.cash - expected_cash) < 0.01


def test_portfolio_sell_exactly_all_shares():
    """
    Test that selling exactly all shares removes the position.
    
    Edge case: Ensure 100.0 shares sold = 100.0 shares owned
    (no floating point rounding issues)
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    
    # Sell exactly what we own
    portfolio.sell("AAPL", 100.0, 160.0, date(2020, 1, 10))
    
    # Position should be completely removed
    assert "AAPL" not in portfolio.positions


# ========================== INTEGRATION TEST ==========================

def test_portfolio_multiple_positions_sell_one():
    """
    Test selling one position while holding others.
    
    Given: Hold AAPL and MSFT
    When: Sell AAPL
    Then: MSFT position unchanged
    """
    portfolio = Portfolio(initial_cash=100000.0)
    
    # Buy two different stocks
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.buy("MSFT", 50.0, 200.0, date(2020, 1, 1))
    
    # Cash after purchases: 100000 - 15000 - 10000 = 75000
    assert portfolio.cash == 75000.0
    
    # Sell AAPL
    portfolio.sell("AAPL", 100.0, 160.0, date(2020, 1, 10))
    
    # AAPL gone, MSFT still there
    assert "AAPL" not in portfolio.positions
    assert "MSFT" in portfolio.positions
    assert portfolio.positions["MSFT"].shares == 50.0
    
    # Cash: 75000 + 16000 = 91000
    assert portfolio.cash == 91000.0