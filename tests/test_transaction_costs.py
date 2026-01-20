"""
Tests for transaction costs in backtester.

Transaction costs have two components:
- Slippage: Price impact (buy higher, sell lower than quoted)
- Commission: Broker fee as percentage of trade value
"""

import pytest
import polars as pl
from datetime import date

from hindsightpy.backtest.backtester import Backtester, BacktestConfig
from hindsightpy.backtest.portfolio import Portfolio


@pytest.fixture
def simple_prices():
    """Two days of price data for one stock at $100."""
    return pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "ticker": ["aapl", "aapl"],
        "close": [100.0, 100.0],
    })


@pytest.fixture
def two_stock_prices():
    """Two days, two stocks, flat prices."""
    return pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 2)],
        "ticker": ["aapl", "msft", "aapl", "msft"],
        "close": [100.0, 200.0, 100.0, 200.0],
    })


# ========================== SLIPPAGE TESTS ==========================

class TestSlippage:
    """Slippage adjusts execution price."""

    def test_buy_slippage_increases_cost(self, simple_prices):
        """
        Buy 100 shares at $100 with 1% slippage.
        Effective price = $101, cost = $10,100.
        Final cash = $100,000 - $10,100 = $89,900.
        """
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.01,
            commission_pct=0.0,
        )
        strategy = {"aapl": 0.10}  # 10% = $10,000 target

        backtester = Backtester()
        result = backtester.run(simple_prices, strategy, config)

        # With slippage, we pay more per share
        # $10,000 / $101 = ~99 shares at $101 = $9,999
        final_cash = result.equity_curve["cash"][-1]

        # Cash should be less than if no slippage
        # Without slippage: 100 shares * $100 = $10,000 spent
        # With 1% slippage: effective price $101, fewer shares or more spent
        assert final_cash < 90000.0

    def test_zero_slippage_no_impact(self, simple_prices):
        """Zero slippage means execute at quoted price."""
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.0,
            commission_pct=0.0,
        )
        strategy = {"aapl": 0.10}

        backtester = Backtester()
        result = backtester.run(simple_prices, strategy, config)

        # $10,000 spent on shares at $100 = 100 shares
        # Cash = $90,000
        final_cash = result.equity_curve["cash"][-1]
        assert final_cash == pytest.approx(90000.0, rel=0.01)


# ========================== COMMISSION TESTS ==========================

class TestCommission:
    """Commission deducted from cash after trade."""

    def test_commission_deducted_on_buy(self, simple_prices):
        """
        Buy $10,000 of stock with 0.1% commission.
        Commission = $10,000 * 0.001 = $10.
        Final cash = $100,000 - $10,000 - $10 = $89,990.
        """
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.0,
            commission_pct=0.001,
        )
        strategy = {"aapl": 0.10}

        backtester = Backtester()
        result = backtester.run(simple_prices, strategy, config)

        final_cash = result.equity_curve["cash"][-1]
        # Should be ~$89,990 (spent $10,000 + $10 commission)
        assert final_cash == pytest.approx(89990.0, rel=0.01)

    def test_zero_commission_no_deduction(self, simple_prices):
        """Zero commission means no extra fees."""
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.0,
            commission_pct=0.0,
        )
        strategy = {"aapl": 0.10}

        backtester = Backtester()
        result = backtester.run(simple_prices, strategy, config)

        final_cash = result.equity_curve["cash"][-1]
        assert final_cash == pytest.approx(90000.0, rel=0.01)

    def test_commission_on_multiple_trades(self, two_stock_prices):
        """Commission charged on each trade."""
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.0,
            commission_pct=0.001,
        )
        # 40% each = $40,000 per stock (leaves room for commission)
        strategy = {"aapl": 0.40, "msft": 0.40}

        backtester = Backtester()
        result = backtester.run(two_stock_prices, strategy, config)

        # Two trades: $40,000 each = $80,000 total
        # Commission: 2 * $40,000 * 0.001 = $80
        # Final cash: $100,000 - $80,000 - $80 = $19,920
        total_trades = len(result.trades)
        assert total_trades == 2

        final_cash = result.equity_curve["cash"][-1]
        assert final_cash == pytest.approx(19920.0, rel=0.01)


# ========================== COMBINED COSTS TESTS ==========================

class TestCombinedCosts:
    """Slippage and commission together."""

    def test_both_costs_applied(self, simple_prices):
        """
        Both slippage and commission reduce portfolio value.
        """
        config_with_costs = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.01,
            commission_pct=0.001,
        )
        config_no_costs = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.0,
            commission_pct=0.0,
        )
        strategy = {"aapl": 0.50}

        backtester = Backtester()
        result_costs = backtester.run(simple_prices, strategy, config_with_costs)
        result_free = backtester.run(simple_prices, strategy, config_no_costs)

        value_costs = result_costs.equity_curve["portfolio_value"][-1]
        value_free = result_free.equity_curve["portfolio_value"][-1]

        # Portfolio with costs should be worth less
        assert value_costs < value_free

    def test_high_costs_significantly_reduce_value(self, simple_prices):
        """High transaction costs (5% total) have major impact."""
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.025,   # 2.5%
            commission_pct=0.025, # 2.5%
        )
        # 90% allocation to leave room for slippage
        strategy = {"aapl": 0.90}

        backtester = Backtester()
        result = backtester.run(simple_prices, strategy, config)

        final_value = result.equity_curve["portfolio_value"][-1]

        # With 5% costs on ~$90k trade, should lose ~$4.5k from $100k
        assert final_value < 96000.0


# ========================== EDGE CASES ==========================

class TestEdgeCases:
    """Edge cases for transaction costs."""

    def test_costs_recorded_in_trades(self, simple_prices):
        """Trade records should reflect effective price (with slippage)."""
        config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 2),
            initial_cash=100000.0,
            rebalance_frequency="never",
            slippage_pct=0.01,
            commission_pct=0.0,
        )
        strategy = {"aapl": 0.10}

        backtester = Backtester()
        result = backtester.run(simple_prices, strategy, config)

        if len(result.trades) > 0:
            trade_price = result.trades["price"][0]
            # Price should reflect slippage: $100 * 1.01 = $101
            assert trade_price == pytest.approx(101.0, rel=0.01)