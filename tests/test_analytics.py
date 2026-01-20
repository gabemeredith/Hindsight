"""
Tests for Analytics module - Performance metrics calculations.

Run with: pytest tests/test_analytics.py -v
"""

import pytest
import polars as pl
import math
from datetime import date
from hindsightpy.analytics.metrics import (
    total_return,
    cagr,
    sharpe_ratio,
    max_drawdown,
    annualized_volatility,
    sortino_ratio,
)


# ========================== FIXTURES ==========================

@pytest.fixture
def simple_equity_curve():
    """
    Simple equity curve: $10,000 → $11,000 → $12,100

    Day 1: $10,000 (start)
    Day 2: $11,000 (+10%)
    Day 3: $12,100 (+10%)

    Total return: (12100 / 10000) - 1 = 0.21 = 21%
    """
    return pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        "portfolio_value": [10000.0, 11000.0, 12100.0],
    })


@pytest.fixture
def one_year_equity_curve():
    """
    One year equity curve for annualized metrics.

    Start: $10,000
    End: $12,000
    Days: 252 trading days (1 year)

    Total return: 20%
    CAGR: 20% (since exactly 1 year)
    """
    dates = pl.date_range(date(2020, 1, 1), date(2020, 12, 31), "1d", eager=True)
    n_days = len(dates)

    # Linear growth from 10000 to 12000
    start_val = 10000.0
    end_val = 12000.0
    values = [start_val + (end_val - start_val) * i / (n_days - 1) for i in range(n_days)]

    return pl.DataFrame({
        "date": dates,
        "portfolio_value": values,
    })


@pytest.fixture
def two_year_equity_curve():
    """
    Two year equity curve for CAGR calculation.

    Start: $10,000
    End: $14,400
    Years: 2

    Total return: 44%
    CAGR: 20% (because 1.20^2 = 1.44)
    """
    return pl.DataFrame({
        "date": [date(2020, 1, 1), date(2022, 1, 1)],
        "portfolio_value": [10000.0, 14400.0],
    })


@pytest.fixture
def drawdown_equity_curve():
    """
    Equity curve with a drawdown for max drawdown calculation.

    Day 1: $10,000 (peak)
    Day 2: $11,000 (new peak)
    Day 3: $9,900 (drawdown: 9900/11000 - 1 = -10%)
    Day 4: $8,800 (deeper drawdown: 8800/11000 - 1 = -20%)
    Day 5: $10,000 (recovery, but still below peak)
    Day 6: $12,000 (new peak)

    Max drawdown: -20% (from 11000 to 8800)
    """
    return pl.DataFrame({
        "date": [
            date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3),
            date(2020, 1, 4), date(2020, 1, 5), date(2020, 1, 6),
        ],
        "portfolio_value": [10000.0, 11000.0, 9900.0, 8800.0, 10000.0, 12000.0],
    })


@pytest.fixture
def returns_series():
    """
    Daily returns for Sharpe/Sortino calculation.

    Returns: [0.01, 0.02, -0.01, 0.015, -0.005]
    Mean: 0.006 (0.6% per day)
    Std: ~0.0114

    For Sharpe with 0% risk-free:
    Daily Sharpe = mean / std = 0.006 / 0.0114 ≈ 0.526
    Annualized Sharpe = 0.526 * sqrt(252) ≈ 8.35
    """
    return pl.Series("returns", [0.01, 0.02, -0.01, 0.015, -0.005])


@pytest.fixture
def negative_returns_series():
    """
    Returns with more downside for Sortino calculation.

    Returns: [0.02, -0.03, 0.01, -0.02, 0.015]
    Negative returns only: [-0.03, -0.02]
    Downside deviation = std of negative returns
    """
    return pl.Series("returns", [0.02, -0.03, 0.01, -0.02, 0.015])


# ========================== TOTAL RETURN TESTS ==========================

class TestTotalReturn:
    """Tests for total_return calculation."""

    def test_total_return_simple(self, simple_equity_curve):
        """
        Given: $10,000 → $12,100
        Expected: (12100 / 10000) - 1 = 0.21 = 21%
        """
        result = total_return(simple_equity_curve)
        assert result == pytest.approx(0.21)

    def test_total_return_no_change(self):
        """Portfolio that doesn't change should return 0%."""
        flat = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "portfolio_value": [10000.0, 10000.0],
        })
        result = total_return(flat)
        assert result == pytest.approx(0)  # the return if no change?

    def test_total_return_loss(self):
        """Portfolio with loss should return negative."""
        loss = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "portfolio_value": [10000.0, 8000.0],
        })
        result = total_return(loss)
        assert result == pytest.approx(-0.2)


# ========================== CAGR TESTS ==========================

class TestCAGR:
    """Tests for Compound Annual Growth Rate."""

    def test_cagr_one_year(self, one_year_equity_curve):
        """
        Given: 20% return over 1 year
        Expected CAGR: 20% (same as total return for 1 year)
        """
        result = cagr(one_year_equity_curve)
        assert result == pytest.approx(.2, rel=0.01)  # 20% = 0.20

    def test_cagr_two_years(self, two_year_equity_curve):
        """
        Given: $10,000 → $14,400 over 2 years
        Total return: 44%
        CAGR: (14400/10000)^(1/2) - 1 = 1.44^0.5 - 1 = 0.20 = 20%

        Verify: $10,000 * 1.20 * 1.20 = $14,400
        """
        result = cagr(two_year_equity_curve)
        assert result == pytest.approx(.2,rel=0.01)  # 1.44^0.5 - 1?

    def test_cagr_half_year(self):
        """
        Given: 10% return over 6 months (0.5 years)
        CAGR: (1.10)^(1/0.5) - 1 = 1.10^2 - 1 = 0.21 = 21%
        """
        half_year = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 7, 1)],
            "portfolio_value": [10000.0, 11000.0],
        })
        result = cagr(half_year)
        assert result == pytest.approx(.21,rel=0.01)  # 1.10^2 - 1?


# ========================== MAX DRAWDOWN TESTS ==========================

class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_simple(self, drawdown_equity_curve):
        """
        Peak: $11,000
        Trough: $8,800
        Max drawdown: (8800 - 11000) / 11000 = -0.20 = -20%
        """
        result = max_drawdown(drawdown_equity_curve)
        assert result == pytest.approx(-.2)  # (8800-11000)/11000?

    def test_max_drawdown_no_drawdown(self):
        """Always increasing portfolio has 0% drawdown."""
        always_up = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "portfolio_value": [10000.0, 11000.0, 12000.0],
        })
        result = max_drawdown(always_up)
        assert result == pytest.approx(0)

    def test_max_drawdown_total_loss(self):
        """50% drawdown."""
        big_loss = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "portfolio_value": [10000.0, 5000.0],
        })
        result = max_drawdown(big_loss)
        assert result == pytest.approx(-.5)  #(5000-10000)/10000?


# ========================== VOLATILITY TESTS ==========================

class TestAnnualizedVolatility:
    """Tests for annualized volatility calculation."""

    def test_volatility_constant_returns(self):
        """
        Constant returns = 0 volatility.
        All returns are 1%: [0.01, 0.01, 0.01, 0.01]
        Std = 0
        """
        constant = pl.Series("returns", [0.01, 0.01, 0.01, 0.01])
        result = annualized_volatility(constant)
        assert result == pytest.approx(0) 

    def test_volatility_calculation(self, returns_series):
        """
        Returns: [0.01, 0.02, -0.01, 0.015, -0.005]
        Daily std ≈ 0.0114
        Annualized = 0.0114 * sqrt(252) ≈ 0.181 = 18.1%
        """
        result = annualized_volatility(returns_series)
        # Should be around 18% annualized
        assert result > 0.10  # Sanity check: positive
        assert result < 0.30  # Sanity check: reasonable range


# ========================== SHARPE RATIO TESTS ==========================

class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_zero_risk_free(self, returns_series):
        """
        Returns: [0.01, 0.02, -0.01, 0.015, -0.005]
        Mean daily return: 0.006
        Daily std: ~0.0114

        Daily Sharpe = 0.006 / 0.0114 ≈ 0.526
        Annualized = 0.526 * sqrt(252) ≈ 8.35
        """
        result = sharpe_ratio(returns_series, risk_free_rate=0.0)
        # High Sharpe because we have positive mean with low vol
        assert result > 5.0  # Sanity check
        assert result < 15.0  # Sanity check

    def test_sharpe_with_risk_free(self, returns_series):
        """
        Same returns but with 2% annual risk-free rate.
        Daily risk-free = 0.02 / 252 ≈ 0.000079
        Excess return = 0.006 - 0.000079 ≈ 0.00592
        Sharpe should be slightly lower.
        """
        result_with_rf = sharpe_ratio(returns_series, risk_free_rate=0.02)
        result_without_rf = sharpe_ratio(returns_series, risk_free_rate=0.0)

        # Sharpe with risk-free should be lower
        assert result_with_rf < result_without_rf

    def test_sharpe_negative_returns(self):
        """Negative average returns = negative Sharpe."""
        bad_returns = pl.Series("returns", [-0.01, -0.02, -0.01, -0.015])
        result = sharpe_ratio(bad_returns, risk_free_rate=0.0)
        assert result < 0.0  


# ========================== SORTINO RATIO TESTS ==========================

class TestSortinoRatio:
    """Tests for Sortino ratio (uses downside deviation only)."""

    def test_sortino_vs_sharpe(self, negative_returns_series):
        """
        Sortino only penalizes downside volatility.
        For asymmetric returns, Sortino != Sharpe.
        """
        sharpe = sharpe_ratio(negative_returns_series, risk_free_rate=0.0)
        sortino = sortino_ratio(negative_returns_series, risk_free_rate=0.0)

        # They should be different (Sortino ignores upside vol)
        assert sharpe != pytest.approx(sortino, rel=0.1)

    def test_sortino_no_downside(self):
        """
        All positive returns = no downside deviation.
        Sortino should be very high (or inf).
        """
        all_positive = pl.Series("returns", [0.01, 0.02, 0.015, 0.01])
        result = sortino_ratio(all_positive, risk_free_rate=0.0)

        # With no downside, Sortino approaches infinity
        # Implementation should handle this gracefully
        assert result > 10.0 or result == float('inf')