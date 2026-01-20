"""
Tests for Rebalancer class

Test Philosophy:
- Use small, deterministic scenarios
- Hand-calculate expected trades
- Test one behavior per test

Run with: pytest tests/test_rebalancer.py -v
"""

import pytest
from datetime import date
from hindsightpy.backtest.portfolio import Portfolio
from hindsightpy.backtest.rebalancer import Rebalancer, Trade


# ========================== FIXTURES ==========================

@pytest.fixture
def empty_portfolio():
    """Portfolio with cash only, no positions"""
    return Portfolio(initial_cash=10000.0)


@pytest.fixture
def portfolio_with_aapl():
    """
    Portfolio with one AAPL position.

    Initial: $10,000 cash
    Bought: 50 AAPL @ $100 = $5,000
    Result: $5,000 cash + $5,000 in AAPL = $10,000 total
    """
    p = Portfolio(initial_cash=10000.0)
    p.buy(ticker="AAPL", shares=50, price=100.0, date=date(2020, 1, 1))
    return p


@pytest.fixture
def portfolio_with_two_stocks():
    """
    Portfolio with two positions.

    Initial: $10,000 cash
    Bought: 30 AAPL @ $100 = $3,000
    Bought: 50 MSFT @ $50 = $2,500
    Result: $4,500 cash + $3,000 AAPL + $2,500 MSFT = $10,000 total
    """
    p = Portfolio(initial_cash=10000.0)
    p.buy(ticker="AAPL", shares=30, price=100.0, date=date(2020, 1, 1))
    p.buy(ticker="MSFT", shares=50, price=50.0, date=date(2020, 1, 1))
    return p


# ========================== BASIC TESTS ==========================

def test_rebalance_from_cash_to_single_stock(empty_portfolio):
    """
    Test simplest case: all cash → buy one stock

    Given:
    - Cash: $10,000
    - Holdings: None
    - Total value: $10,000

    Target:
    - 100% AAPL @ $100/share

    Calculate:
    - Target AAPL value = 10,000 (what's 100% of $10,000?)
    - Target AAPL shares = 100 (how many shares at $100?)

    Expected:
    - 1 trade (buy)
    """
    rebalancer = Rebalancer()

    target_weights = {"AAPL": 1.0}
    prices = {"AAPL": 100.0}

    trades = rebalancer.calculate_trades(
        current_portfolio=empty_portfolio,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    # Should have exactly 1 trade
    assert len(trades) == 1  # How many trades expected?

    # Should be a buy
    trade = trades[0]
    assert trade.ticker == "AAPL"  # Which ticker?
    assert trade.side == "buy"  # "buy" or "sell"?
    assert trade.shares == 100  # How many shares? ($10,000 / $100)
    assert trade.price == 100  # What price?
    assert trade.date == date(2020, 1, 2)  # What date?


def test_rebalance_sell_entire_position(portfolio_with_aapl):
    """
    Test selling everything

    Given:
    - Cash: $5,000
    - Holdings: 50 AAPL @ $100 (current price) = $5,000
    - Total value: $10,000

    Target:
    - 0% AAPL (sell everything)
    - 100% cash

    Calculate:
    - Target AAPL value = 0 (what's 0% of $10,000?)
    - Current AAPL shares = 50 (how many do we own?)
    - Shares to sell = 50 (how many to reach 0?)

    Expected:
    - 1 trade (sell all 50 shares)
    """
    rebalancer = Rebalancer()

    target_weights = {}  # Empty dict = 0% everything = all cash
    prices = {"AAPL": 100.0}

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_aapl,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 1  # How many trades?

    trade = trades[0]
    assert trade.ticker == "AAPL"
    assert trade.side == "sell"  # "buy" or "sell"?
    assert trade.shares == 50  # How many shares to sell?
    assert trade.price == 100


def test_rebalance_reduce_position(portfolio_with_aapl):
    """
    Test partial sell

    Given:
    - Cash: $5,000
    - Holdings: 50 AAPL @ $100 = $5,000
    - Total value: $10,000

    Target:
    - 25% AAPL

    Calculate:
    - Target AAPL value = 2,500 (what's 25% of $10,000?)
    - Target AAPL shares = 25 (how many shares at $100?)
    - Current AAPL shares = 50
    - Shares to sell = 50 - 25 = 25 (50 - target shares)

    Expected:
    - Sell 25 shares (keeping 25)
    """
    rebalancer = Rebalancer()

    target_weights = {"AAPL": 0.25}  # 25% AAPL
    prices = {"AAPL": 100.0}

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_aapl,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 1

    trade = trades[0]
    assert trade.ticker == "AAPL"
    assert trade.side == "sell"
    assert trade.shares == 25  

def test_rebalance_increase_position(portfolio_with_aapl):
    """
    Test buying more of existing position

    Given:
    - Cash: $5,000
    - Holdings: 50 AAPL @ $100 = $5,000
    - Total value: $10,000

    Target:
    - 75% AAPL

    Calculate:
    - Target AAPL value = 7,500 (what's 75% of $10,000?)
    - Target AAPL shares = 75 (how many shares at $100?)
    - Current AAPL shares = 50
    - Shares to buy = 25 (target - current)

    Expected:
    - Buy 25 more shares (to reach 75)
    """
    rebalancer = Rebalancer()

    target_weights = {"AAPL": 0.75}  # 75% AAPL
    prices = {"AAPL": 100.0}

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_aapl,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 1

    trade = trades[0]
    assert trade.ticker == "AAPL"
    assert trade.side == "buy"  # "buy" or "sell"?
    assert trade.shares == 25  # How many to buy?


# ========================== MULTI-STOCK TESTS ==========================

def test_rebalance_multiple_stocks_from_cash(empty_portfolio):
    """
    Test buying multiple stocks from cash

    Given:
    - Cash: $10,000
    - Total value: $10,000

    Target:
    - 60% AAPL @ $100/share
    - 40% MSFT @ $50/share

    Calculate:
    - Target AAPL value = 6,000 (60% of $10,000)
    - Target AAPL shares = 60 (value / price)
    - Target MSFT value = 4,000 (40% of $10,000)
    - Target MSFT shares = 4,000 / 50 = 80 (value / price)

    Expected:
    - 2 trades (buy AAPL, buy MSFT)
    """
    rebalancer = Rebalancer()

    target_weights = {
        "AAPL": 0.6,  # 60%
        "MSFT": 0.4   # 40%
    }
    prices = {
        "AAPL": 100.0,
        "MSFT": 50.0
    }

    trades = rebalancer.calculate_trades(
        current_portfolio=empty_portfolio,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 2 # How many trades?

    # Find AAPL trade
    aapl_trade = [t for t in trades if t.ticker == "AAPL"][0]
    assert aapl_trade.side == "buy"
    assert aapl_trade.shares == 60  # $6,000 / $100 = ?

    # Find MSFT trade
    msft_trade = [t for t in trades if t.ticker == "MSFT"][0]
    assert msft_trade.side == "buy"
    assert msft_trade.shares == 80  # $4,000 / $50 = ?


def test_rebalance_swap_positions(portfolio_with_aapl):
    """
    Test selling one stock to buy another

    Given:
    - Cash: $5,000
    - Holdings: 50 AAPL @ $100 = $5,000
    - Total value: $10,000

    Target:
    - 0% AAPL
    - 100% MSFT @ $50/share

    Calculate:
    - Sell AAPL: 50 shares (all of them)
    - After sell, cash = 10,000 ($5,000 + $5,000 from sale)
    - Buy MSFT: 200 shares ($10,000 / $50)

    Expected:
    - 2 trades: sell AAPL, then buy MSFT
    - Sells should come before buys!
    """
    rebalancer = Rebalancer()

    target_weights = {"MSFT": 1.0}  # 100% MSFT, 0% AAPL
    prices = {
        "AAPL": 100.0,
        "MSFT": 50.0
    }

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_aapl,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 2  # How many trades?

    # First trade should be SELL (to free up cash)
    assert trades[0].side == "sell"  # Which comes first: "buy" or "sell"?
    assert trades[0].ticker == "AAPL" # Which ticker to sell?
    assert trades[0].shares == 50

    # Second trade should be BUY
    assert trades[1].side == "buy"
    assert trades[1].ticker == "MSFT"
    assert trades[1].shares == 200  # $10,000 / $50 = ?


def test_rebalance_complex_multi_stock(portfolio_with_two_stocks):
    """
    Test rebalancing between multiple existing positions

    Given:
    - Cash: $4,500
    - Holdings: 30 AAPL @ $100 = $3,000
    - Holdings: 50 MSFT @ $50 = $2,500
    - Total value: $10,000

    Target:
    - 50% AAPL (currently 30%)
    - 30% MSFT (currently 25%)
    - 20% cash (currently 45%)

    Calculate:
    - Target AAPL: 50 shares (50% of $10,000 = $5,000 / $100)
    - Current AAPL: 30 shares
    - AAPL trade: BUY 20 shares

    - Target MSFT: 60 shares (30% of $10,000 = $3,000 / $50)
    - Current MSFT: 50 shares
    - MSFT trade: BUY 10 shares

    Expected:
    - Buy 10 MSFT (to go from 25% to 30%)
    - Buy 20 AAPL (to go from 30% to 50%)
    - Both are buys (using the excess cash)
    """
    rebalancer = Rebalancer()

    target_weights = {
        "AAPL": 0.5,   # 50%
        "MSFT": 0.3    # 30% (20% will be cash automatically)
    }
    prices = {
        "AAPL": 100.0,
        "MSFT": 50.0
    }

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_two_stocks,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    # Should have 2 trades
    assert len(trades) == 2

    # Find MSFT trade
    msft_trade = [t for t in trades if t.ticker == "MSFT"][0]
    assert msft_trade.side == "buy"  # buy or sell?
    assert msft_trade.shares == 10  # Current 50, target 60, so trade = buy 10

    # Find AAPL trade
    aapl_trade = [t for t in trades if t.ticker == "AAPL"][0]
    assert aapl_trade.side == "buy"  # buy or sell?
    assert aapl_trade.shares == 20  # Current 30, target 50, so trade = buy 20


# ========================== EDGE CASES ==========================

def test_rebalance_no_changes_needed(portfolio_with_aapl):
    """
    Test when portfolio already matches target weights

    Given:
    - Cash: $5,000 (50%)
    - Holdings: 50 AAPL @ $100 = $5,000 (50%)

    Target:
    - 50% AAPL

    Expected:
    - No trades needed!
    """
    rebalancer = Rebalancer()

    target_weights = {"AAPL": 0.5}  # Already at 50%
    prices = {"AAPL": 100.0}

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_aapl,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 0  # Should be 0 (no trades needed)


def test_rebalance_with_price_change(portfolio_with_aapl):
    """
    Test when current price differs from entry price

    Given:
    - Cash: $5,000
    - Holdings: 50 AAPL bought @ $100, but now worth $200/share
    - Current holdings value: 50 * $200 = $10,000
    - Total value: $15,000

    Target:
    - 50% AAPL

    Calculate:
    - Target AAPL value = $7,500 (50% of $15,000)
    - Target AAPL shares = 37.5 ($7,500 / $200)
    - Current shares = 50
    - Trade = sell 12.5 shares (50 - 37.5)

    Expected:
    - Sell some AAPL (it's grown beyond 50%)
    """
    rebalancer = Rebalancer()

    target_weights = {"AAPL": 0.5}  # 50% AAPL
    prices = {"AAPL": 200.0}  # Price has doubled!

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_aapl,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 1

    trade = trades[0]
    assert trade.side == "sell"  # buy or sell?
    assert trade.shares == 12.5  # Current 50, target 37.5, trade = sell 12.5
    assert trade.price == 200.0  # Use current price, not entry price!


def test_rebalance_empty_target_weights(portfolio_with_two_stocks):
    """
    Test selling everything (go to 100% cash)

    Given:
    - Some stocks

    Target:
    - {} (empty dict = all cash)

    Expected:
    - Sell everything
    """
    rebalancer = Rebalancer()

    target_weights = {}  # Sell everything
    prices = {
        "AAPL": 100.0,
        "MSFT": 50.0
    }

    trades = rebalancer.calculate_trades(
        current_portfolio=portfolio_with_two_stocks,
        target_weights=target_weights,
        prices=prices,
        trade_date=date(2020, 1, 2)
    )

    assert len(trades) == 2  # How many positions to close?

    # Both should be sells
    assert all(t.side == "sell" for t in trades)

    # Find each ticker
    aapl_trade = [t for t in trades if t.ticker == "AAPL"][0]
    assert aapl_trade.shares == 30  # Sell all 30 shares

    msft_trade = [t for t in trades if t.ticker == "MSFT"][0]
    assert msft_trade.shares == 50  # Sell all 50 shares