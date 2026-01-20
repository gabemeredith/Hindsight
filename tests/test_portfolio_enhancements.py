"""
Tests for Portfolio enhancements (Phase 2).

Testing:
- get_unrealized_pnl() - paper gains/losses on open positions
- get_realized_pnl() - actual gains/losses from sales
- get_positions_df() - export positions as DataFrame
- get_holdings_value() - current market value per ticker
"""

import sys
sys.path.insert(0, 'src')

import pytest
from datetime import date
from hindsightpy.backtest.portfolio import Portfolio


# ========================== UNREALIZED P&L TESTS ==========================

def test_unrealized_pnl_single_position_gain():
    """
    Test unrealized P&L for a single position with gain.

    Given: Buy 100 shares AAPL at $150
    When: Current price is $160
    Then: Unrealized P&L = ???

    YOUR CALCULATION:
    - Cost basis = 100 × $150 = $15,000
    - Current value = 100 × $160 = $16,000
    - Unrealized P&L = $16,000 - $15,000 = $1,000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))

    current_prices = {"AAPL": 160.0}
    unrealized = portfolio.get_unrealized_pnl(current_prices)

    assert unrealized == 1000.0


def test_unrealized_pnl_single_position_loss():
    """
    Test unrealized P&L for a single position with loss.

    Given: Buy 100 shares AAPL at $150
    When: Current price is $140
    Then: Unrealized P&L = ???

    YOUR TURN: Calculate the expected value!
    - Cost basis = ?
    - Current value = ?
    - Unrealized P&L = ?
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))

    current_prices = {"AAPL": 140.0}
    unrealized = portfolio.get_unrealized_pnl(current_prices)


    assert unrealized == -1000


def test_unrealized_pnl_multiple_positions():
    """
    Test unrealized P&L across multiple positions.

    Given:
    - Buy 100 AAPL at $150
    - Buy 50 MSFT at $200
    When:
    - AAPL price = $160 (gain)
    - MSFT price = $190 (loss)
    Then: Total unrealized P&L = ???

    YOUR TURN: Calculate!
    - AAPL: cost = ?, value = ?, P&L = ?
    - MSFT: cost = ?, value = ?, P&L = ?
    - Total P&L = ?
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.buy("MSFT", 50.0, 200.0, date(2020, 1, 1))

    current_prices = {"AAPL": 160.0, "MSFT": 190.0}
    unrealized = portfolio.get_unrealized_pnl(current_prices)


    assert unrealized == 500


def test_unrealized_pnl_no_positions():
    """
    Test unrealized P&L when portfolio has no positions.

    Given: Empty portfolio (only cash)
    When: No positions held
    Then: Unrealized P&L = ???

    YOUR TURN: What should this be?
    """
    portfolio = Portfolio(initial_cash=100000.0)

    current_prices = {}
    unrealized = portfolio.get_unrealized_pnl(current_prices)


    assert unrealized == 0


def test_unrealized_pnl_after_averaging_up():
    """
    Test unrealized P&L after buying more shares (averaging up).

    Given:
    - Buy 100 AAPL at $150 (cost = $15,000)
    - Buy 100 more AAPL at $160 (cost = $16,000)
    - Average cost basis = $31,000 / 200 = $155
    When: Current price = $170
    Then: Unrealized P&L = ???

    YOUR TURN: Calculate!
    - Total shares = ?
    - Average cost basis = ?
    - Total cost = ?
    - Current value = ?
    - Unrealized P&L = ?
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.buy("AAPL", 100.0, 160.0, date(2020, 1, 5))

    current_prices = {"AAPL": 170.0}
    unrealized = portfolio.get_unrealized_pnl(current_prices)


    assert unrealized == 3000


def test_unrealized_pnl_missing_price():
    """
    Test unrealized P&L when current price is missing.

    Given: Hold 100 AAPL
    When: Price data doesn't include AAPL
    Then: Unrealized P&L = ???

    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))

    current_prices = {}  # No AAPL price!
    unrealized = portfolio.get_unrealized_pnl(current_prices)

    # Option B: Skip position if price missing (forgiving approach)
    # If we can't get the price, we can't calculate P&L for that position
    # So we treat it as 0 contribution to total unrealized P&L
    assert unrealized == 0.0


# ========================== REALIZED P&L TESTS ==========================

def test_realized_pnl_starts_at_zero():
    """
    Test that realized P&L starts at zero for new portfolio.

    Given: New portfolio
    When: No trades executed
    Then: Realized P&L = 0
    """
    portfolio = Portfolio(initial_cash=100000.0)

    realized = portfolio.get_realized_pnl()

    assert realized == 0.0


def test_realized_pnl_after_profitable_sale():
    """
    Test realized P&L after selling at a gain.

    Given: Buy 100 AAPL at $150
    When: Sell all 100 at $160
    Then: Realized P&L = ???

    YOUR TURN: Calculate!
    - Cost basis = 100 × $150 = 1,5000
    - Sale proceeds = 100 × $160 = 1,6000
    - Realized P&L = proceeds - cost = 1000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.sell("AAPL", 100.0, 160.0, date(2020, 1, 10))

    realized = portfolio.get_realized_pnl()


    assert realized == 1000


def test_realized_pnl_after_loss_sale():
    """
    Test realized P&L after selling at a loss.

    Given: Buy 100 AAPL at $150
    When: Sell all 100 at $140
    Then: Realized P&L = -100

    YOUR TURN: Calculate!
    - Cost basis = 1,5000
    - Sale proceeds = 1,4000
    - Realized P&L = -1000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.sell("AAPL", 100.0, 140.0, date(2020, 1, 10))

    realized = portfolio.get_realized_pnl()


    assert realized == -1000


def test_realized_pnl_accumulates_across_trades():
    """
    Test that realized P&L accumulates across multiple sales.

    Given:
    - Buy 100 AAPL at $150
    - Sell 50 at $160 (first sale: $500 gain)
    - Sell 50 at $155 (second sale: $250 gain)
    Then: Total realized P&L = 750

    YOUR TURN: Calculate!
    - First sale P&L = 50 × ($160 - $150) = 500
    - Second sale P&L = 50 × ($155 - $150) = 250
    - Total = 750
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))

    # First sale
    portfolio.sell("AAPL", 50.0, 160.0, date(2020, 1, 5))

    # Second sale
    portfolio.sell("AAPL", 50.0, 155.0, date(2020, 1, 10))

    realized = portfolio.get_realized_pnl()


    assert realized == 750


def test_realized_pnl_with_averaging():
    """
    Test realized P&L when cost basis was averaged.

    Given:
    - Buy 100 AAPL at $150 (cost = $15,000)
    - Buy 100 more at $160 (cost = $16,000)
    - Average cost basis = $155 per share
    When: Sell all 200 at $170
    Then: Realized P&L = 3000

    YOUR TURN: Calculate!
    - Total cost basis = $15,000 + $16,000 = 31,000
    - Sale proceeds = 200 × $170 = 34,000
    - Realized P&L = 3,000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.buy("AAPL", 100.0, 160.0, date(2020, 1, 5))

    # Average cost basis is now $155
    # Sell everything at $170
    portfolio.sell("AAPL", 200.0, 170.0, date(2020, 1, 10))

    realized = portfolio.get_realized_pnl()


    assert realized == 3000


def test_realized_vs_unrealized_pnl():
    """
    Test the difference between realized and unrealized P&L.

    Given:
    - Buy 200 AAPL at $150
    - Sell 100 at $160 (realized $1,000 gain)
    - Current price is $170
    When:
    - Realized P&L = gain from the 100 sold = 1,000
    - Unrealized P&L = paper gain on 100 still held = 1,000

    YOUR TURN: Calculate both!
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 200.0, 150.0, date(2020, 1, 1))
    portfolio.sell("AAPL", 100.0, 160.0, date(2020, 1, 5))

    current_prices = {"AAPL": 170.0}

    realized = portfolio.get_realized_pnl()
    unrealized = portfolio.get_unrealized_pnl(current_prices)


    assert realized == 1000  # From the 100 shares sold
    assert unrealized == 2000  # From the 100 shares still held


# ========================== HOLDINGS VALUE TESTS ==========================

def test_holdings_value_single_position():
    """
    Test holdings value for a single position.

    Given: Buy 100 AAPL at $150
    When: Current price is $160
    Then: Holdings value = {"AAPL": 15,000}

    YOUR TURN: Calculate!
    - 100 shares × $160 = 16,000
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))

    current_prices = {"AAPL": 160.0}
    holdings = portfolio.get_holdings_value(current_prices)


    assert holdings == {"AAPL": 16000}


def test_holdings_value_multiple_positions():
    """
    Test holdings value for multiple positions.

    Given:
    - Buy 100 AAPL at $150
    - Buy 50 MSFT at $200
    When:
    - AAPL current price = $160
    - MSFT current price = $190
    Then: Holdings = {"AAPL": ???, "MSFT": ???}

    YOUR TURN: Calculate both!
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))
    portfolio.buy("MSFT", 50.0, 200.0, date(2020, 1, 1))

    current_prices = {"AAPL": 160.0, "MSFT": 190.0}
    holdings = portfolio.get_holdings_value(current_prices)


    assert holdings == {"AAPL": 16000, "MSFT": 9500}


def test_holdings_value_no_positions():
    """
    Test holdings value when portfolio has no positions.

    Given: Empty portfolio
    When: No positions held
    Then: Holdings = 0

    YOUR TURN: What should this be?
    """
    portfolio = Portfolio(initial_cash=100000.0)

    current_prices = {}
    holdings = portfolio.get_holdings_value(current_prices)


    assert holdings == {}


def test_holdings_value_missing_price():
    """
    Test holdings value when price is missing.

    Given: Hold 100 AAPL
    When: Price data doesn't include AAPL
    Then: Should skip that position (like unrealized P&L)

    YOUR DECISION: Should this be:
    - {} (empty dict, position skipped)
    - {"AAPL": 0.0} (position included with 0 value)
    """
    portfolio = Portfolio(initial_cash=100000.0)
    portfolio.buy("AAPL", 100.0, 150.0, date(2020, 1, 1))

    current_prices = {}  # No AAPL price!
    holdings = portfolio.get_holdings_value(current_prices)

    assert holdings == {} #empty, I dont wanna skew the calculations too drastically 