"""
Tests for Strategy classes

Test Philosophy:
- Use small, deterministic scenarios
- Hand-calculate expected weights
- Test one behavior per test

Run with: pytest tests/test_strategy.py -v

"""

import pytest
from datetime import date
import polars as pl
from hindsightpy.backtest.portfolio import Portfolio
from hindsightpy.backtest.strategy import (
    Strategy,
    StaticWeightStrategy,
    MomentumStrategy,
)


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
def simple_prices():
    """Current prices for 3 stocks"""
    return {
        "AAPL": 100.0,
        "MSFT": 200.0,
        "GOOGL": 150.0,
    }


@pytest.fixture
def factors_single_day():
    """
    Factors for a single day with 3 stocks.

    Date: 2020-01-15
    Tickers: AAPL, MSFT, GOOGL
    Momentum (mom_10d): AAPL=0.15, MSFT=0.08, GOOGL=0.22
    RSI: AAPL=45, MSFT=72, GOOGL=28

    Momentum ranking (highest to lowest):
    1. GOOGL (0.22) - best momentum
    2. AAPL (0.15)
    3. MSFT (0.08) - worst momentum
    """
    return pl.DataFrame({
        "date": [date(2020, 1, 15)] * 3,
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "close": [100.0, 200.0, 150.0],
        "mom_10d": [0.15, 0.08, 0.22],
        "rsi_14": [45.0, 72.0, 28.0],
    })


@pytest.fixture
def factors_multi_day():
    """
    Factors for multiple days (for testing point-in-time lookups).

    Day 1 (2020-01-14): GOOGL has highest momentum
    Day 2 (2020-01-15): AAPL has highest momentum (momentum shifted!)

    This tests that strategy uses CURRENT day's factors, not future.
    """
    return pl.DataFrame({
        "date": [
            # Day 1
            date(2020, 1, 14), date(2020, 1, 14), date(2020, 1, 14),
            # Day 2
            date(2020, 1, 15), date(2020, 1, 15), date(2020, 1, 15),
        ],
        "ticker": [
            "AAPL", "MSFT", "GOOGL",
            "AAPL", "MSFT", "GOOGL",
        ],
        "close": [
            100.0, 200.0, 150.0,
            105.0, 198.0, 148.0,
        ],
        "mom_10d": [
            # Day 1: GOOGL best
            0.10, 0.05, 0.20,
            # Day 2: AAPL best
            0.25, 0.08, 0.12,
        ],
    })


# ========================== STATIC WEIGHT STRATEGY TESTS ==========================

class TestStaticWeightStrategy:
    """Tests for StaticWeightStrategy - the simplest strategy."""

    def test_static_returns_fixed_weights(self, empty_portfolio, simple_prices):
        """
        StaticWeightStrategy always returns the same weights.

        Given:
        - Strategy initialized with {"AAPL": 0.6, "MSFT": 0.4}

        Expected:
        - get_target_weights() returns {"AAPL": 0.6, "MSFT": 0.4}
        - Same result regardless of portfolio state or prices

        """
        strategy = StaticWeightStrategy(weights={"AAPL": 0.6, "MSFT": 0.4})

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=None,
        )

        assert weights["AAPL"] == .6  
        assert weights["MSFT"] == .4  
        assert "GOOGL" not in weights  # GOOGL wasn't in the initial weights

    def test_static_ignores_portfolio_state(self, portfolio_with_aapl, simple_prices):
        """
        StaticWeightStrategy doesn't care about current holdings.

        Even though portfolio already holds AAPL, strategy returns same weights.

        Given:
        - Portfolio: 50 AAPL shares
        - Strategy: {"AAPL": 0.3, "MSFT": 0.7}

        Expected:
        - Returns {"AAPL": 0.3, "MSFT": 0.7} regardless of holdings
        """
        strategy = StaticWeightStrategy(weights={"AAPL": 0.3, "MSFT": 0.7})

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=portfolio_with_aapl,
            prices=simple_prices,
            factors=None,
        )

        assert weights["AAPL"] == .3 # Still 0.3, not affected by holdings
        assert weights["MSFT"] == .7 # Still 0.7

    def test_static_ignores_factors(self, empty_portfolio, simple_prices, factors_single_day):
        """
        StaticWeightStrategy ignores factors completely.

        Even if factors are passed, they don't change the weights.
        """
        strategy = StaticWeightStrategy(weights={"GOOGL": 1.0})

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_single_day,  # Passed, but should be ignored
        )

        assert weights == {"GOOGL": 1.0}

    def test_static_empty_weights(self, empty_portfolio, simple_prices):
        """
        StaticWeightStrategy with empty weights = 100% cash.

        Given:
        - Strategy initialized with {}

        Expected:
        - Returns {} (meaning: hold no stocks, keep all cash)
        """
        strategy = StaticWeightStrategy(weights={})

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=None,
        )

        assert weights == {}


# ========================== MOMENTUM STRATEGY TESTS ==========================

class TestMomentumStrategy:
    """Tests for MomentumStrategy - picks top N stocks by momentum."""

    def test_momentum_selects_top_n(self, empty_portfolio, simple_prices, factors_single_day):
        """
        MomentumStrategy selects top N stocks by momentum factor.

        Given (from factors_single_day fixture):
        - AAPL mom_10d = 0.15
        - MSFT mom_10d = 0.08
        - GOOGL mom_10d = 0.22

        Ranking (highest to lowest):
        1. GOOGL (0.22)
        2. AAPL (0.15)
        3. MSFT (0.08)

        With n_positions=2:
        - Select top 2: GOOGL and AAPL
        - Equal weight each: 1/2 = 0.5

        """
        strategy = MomentumStrategy(n_positions=2, max_allocation=1.0)

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_single_day,
        )

        # selected tickers
        assert "GOOGL" in weights
        assert "AAPL" in weights
        assert "MSFT" not in weights


        assert weights["GOOGL"] == .5
        assert weights["AAPL"] == .5   

    def test_momentum_selects_top_1(self, empty_portfolio, simple_prices, factors_single_day):
        """
        MomentumStrategy with n_positions=1 selects only the best stock.

        Given:
        - GOOGL has highest momentum (0.22)

        With n_positions=1:
        - Only GOOGL selected
        - Weight = 1.0 (100% in one stock)
        """
        strategy = MomentumStrategy(n_positions=1, max_allocation=1.0)

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_single_day,
        )

        assert len(weights) == 1
        assert "GOOGL" in weights
        assert weights["GOOGL"] == 1.0

    def test_momentum_selects_all_when_n_exceeds_universe(
        self, empty_portfolio, simple_prices, factors_single_day
    ):
        """
        If n_positions > number of stocks, select all stocks.

        Given:
        - 3 stocks in universe
        - n_positions = 5 (more than available)

        Expected:
        - All 3 stocks selected
        - Equal weight: 1/3 each
        """
        strategy = MomentumStrategy(n_positions=5, max_allocation=1.0)  # More than 3 stocks available

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_single_day,
        )

        assert len(weights) == 3

        # Each should have equal weight
        expected_weight = 1.0 / 3
        assert weights["AAPL"] == pytest.approx(expected_weight)
        assert weights["MSFT"] == pytest.approx(expected_weight)
        assert weights["GOOGL"] == pytest.approx(expected_weight)

    def test_momentum_uses_current_date_factors(
        self, empty_portfolio, simple_prices, factors_multi_day
    ):
        """
        Strategy must use factors for CURRENT date only (no lookahead).

        Given (from factors_multi_day fixture):
        - Day 1 (2020-01-14): GOOGL has highest momentum (0.20)
        - Day 2 (2020-01-15): AAPL has highest momentum (0.25)

        When current_date = 2020-01-14:
        - Should select GOOGL (best on that day)
        - NOT AAPL (even though it's best on day 2)

        YOUR TASK: What stock is selected on day 1?
        """
        strategy = MomentumStrategy(n_positions=1)

        # Test for Day 1
        weights_day1 = strategy.get_target_weights(
            current_date=date(2020, 1, 14),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_multi_day,
        )

        # On 2020-01-14, which stock has highest momentum?
        # Look at factors_multi_day: mom_10d on day 1 is [0.10, 0.05, 0.20]
        # That's AAPL=0.10, MSFT=0.05, GOOGL=0.20
        assert "GOOGL" in weights_day1  # Which ticker? (highest on day 1)

        # Test for Day 2
        weights_day2 = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_multi_day,
        )

        # On 2020-01-15, which stock has highest momentum?
        # Look at factors_multi_day: mom_10d on day 2 is [0.25, 0.08, 0.12]
        # That's AAPL=0.25, MSFT=0.08, GOOGL=0.12
        assert "AAPL" in weights_day2  # Which ticker? (highest on day 2)

    def test_momentum_returns_empty_without_factors(self, empty_portfolio, simple_prices):
        """
        MomentumStrategy requires factors. Without them, returns empty dict.

        Given:
        - factors=None

        Expected:
        - Returns {} (can't rank without momentum data)
        """
        strategy = MomentumStrategy(n_positions=2)

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=None,  # No factors!
        )

        assert weights == {}

    def test_momentum_handles_missing_date(
        self, empty_portfolio, simple_prices, factors_single_day
    ):
        """
        If current_date not in factors, return empty weights.

        Given:
        - factors only has data for 2020-01-15
        - current_date = 2020-01-20 (not in factors)

        Expected:
        - Returns {} (no data for that date)
        """
        strategy = MomentumStrategy(n_positions=2)

        weights = strategy.get_target_weights(
            current_date=date(2020, 1, 20),  # Not in factors_single_day!
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_single_day,
        )

        assert weights == {}  # What when date not found?


# ========================== STRATEGY INTERFACE TESTS ==========================

class TestStrategyInterface:
    """Tests for the Strategy base class interface."""

    def test_strategy_is_abstract(self):
        """
        Cannot instantiate Strategy directly - must subclass.
        """
        with pytest.raises(TypeError):
            Strategy()  # Should raise TypeError (can't instantiate abstract class)

    def test_weights_sum_to_one_or_less(self, empty_portfolio, simple_prices, factors_single_day):
        """
        All strategy weights should sum to <= 1.0.

        This is a constraint test - weights > 1.0 would mean using leverage.
        """
        # Test StaticWeightStrategy
        static = StaticWeightStrategy(weights={"AAPL": 0.5, "MSFT": 0.3})
        static_weights = static.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=None,
        )
        assert sum(static_weights.values()) <= 1.0  #max sum allowed?

        # Test MomentumStrategy
        momentum = MomentumStrategy(n_positions=2)
        momentum_weights = momentum.get_target_weights(
            current_date=date(2020, 1, 15),
            portfolio=empty_portfolio,
            prices=simple_prices,
            factors=factors_single_day,
        )
        assert sum(momentum_weights.values()) <= 1.0  