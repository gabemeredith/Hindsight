"""
Tests for visualization module.

These tests verify that chart functions:
1. Return valid matplotlib Figure objects
2. Contain expected axes and data
3. Handle edge cases gracefully

Test Philosophy:
- We test that the chart STRUCTURE is correct (axes, labels, data points)
- We don't test visual appearance (colors, fonts) - that's subjective
- Small fixtures with known values for predictable outputs
"""

import sys
sys.path.insert(0, 'src')

import pytest
import polars as pl
from datetime import date
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from factorlabs.visualization.charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_weights_over_time,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_equity_curve():
    """
    5-day equity curve with known values.

    Portfolio starts at $100,000 and grows 10% total:
    Day 1: $100,000 (start)
    Day 2: $102,000 (+2%)
    Day 3: $105,000 (+2.94%)
    Day 4: $103,000 (-1.9%)
    Day 5: $110,000 (+6.8%)

    Peak: $105,000 on Day 3
    Trough after peak: $103,000 on Day 4
    Max drawdown: (105,000 - 103,000) / 105,000 = 1.9%
    """
    return pl.DataFrame({
        "date": [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ],
        "portfolio_value": [100000.0, 102000.0, 105000.0, 103000.0, 110000.0],
        "cash": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        "positions_value": [99000.0, 101000.0, 104000.0, 102000.0, 109000.0],
    })


@pytest.fixture
def simple_benchmark():
    """
    Benchmark returns for comparison (e.g., SPY).
    Same dates as equity curve, different performance.

    Benchmark: +5% total (underperforms portfolio's +10%)
    """
    return pl.DataFrame({
        "date": [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ],
        "portfolio_value": [100000.0, 101000.0, 103000.0, 102000.0, 105000.0],
    })


@pytest.fixture
def simple_returns():
    """
    Daily returns series for histogram.

    Returns: [+2%, +2.94%, -1.9%, +6.8%]
    Mean: ~2.46%
    """
    return pl.Series("returns", [0.02, 0.0294, -0.019, 0.068])


@pytest.fixture
def position_weights_history():
    """
    Position weights over time for 3 tickers.
    Shows allocation drift from initial 40/40/20 split.
    """
    return pl.DataFrame({
        "date": [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
        ],
        "aapl": [0.40, 0.42, 0.45, 0.43, 0.44],
        "msft": [0.40, 0.38, 0.35, 0.37, 0.36],
        "googl": [0.20, 0.20, 0.20, 0.20, 0.20],
    })


# =============================================================================
# EQUITY CURVE TESTS
# =============================================================================

class TestPlotEquityCurve:
    """Tests for plot_equity_curve() function."""

    def test_returns_figure(self, simple_equity_curve):
        """Function should return a matplotlib Figure object."""
        fig = plot_equity_curve(simple_equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_single_axes(self, simple_equity_curve):
        """Figure should have exactly one Axes."""
        fig = plot_equity_curve(simple_equity_curve)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_axes_has_line(self, simple_equity_curve):
        """Axes should contain at least one line (the equity curve)."""
        fig = plot_equity_curve(simple_equity_curve)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close(fig)

    def test_line_has_correct_length(self, simple_equity_curve):
        """Line should have same number of points as data rows."""
        fig = plot_equity_curve(simple_equity_curve)
        ax = fig.axes[0]
        line = ax.get_lines()[0]
        assert len(line.get_ydata()) == 5  # 5 days of data
        plt.close(fig)

    def test_with_benchmark_has_two_lines(self, simple_equity_curve, simple_benchmark):
        """When benchmark provided, should have two lines."""
        fig = plot_equity_curve(simple_equity_curve, benchmark=simple_benchmark)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)

    def test_has_title(self, simple_equity_curve):
        """Chart should have a title."""
        fig = plot_equity_curve(simple_equity_curve)
        ax = fig.axes[0]
        assert ax.get_title() != ""
        plt.close(fig)

    def test_has_axis_labels(self, simple_equity_curve):
        """Chart should have x and y axis labels."""
        fig = plot_equity_curve(simple_equity_curve)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


# =============================================================================
# DRAWDOWN TESTS
# =============================================================================

class TestPlotDrawdown:
    """Tests for plot_drawdown() function."""

    def test_returns_figure(self, simple_equity_curve):
        """Function should return a matplotlib Figure object."""
        fig = plot_drawdown(simple_equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_drawdown_is_negative_or_zero(self, simple_equity_curve):
        """Drawdown values should always be <= 0."""
        fig = plot_drawdown(simple_equity_curve)
        ax = fig.axes[0]
        line = ax.get_lines()[0]
        ydata = line.get_ydata()
        assert all(y <= 0 for y in ydata)
        plt.close(fig)

    def test_max_drawdown_value(self, simple_equity_curve):
        """
        Max drawdown should match hand-calculated value.

        Peak at Day 3: $105,000
        Trough at Day 4: $103,000
        Drawdown: (103,000 - 105,000) / 105,000 = -0.019 = -1.9%
        """
        fig = plot_drawdown(simple_equity_curve)
        ax = fig.axes[0]
        line = ax.get_lines()[0]
        ydata = line.get_ydata()
        min_drawdown = min(ydata)
        # Allow small tolerance for floating point
        assert abs(min_drawdown - (-0.019047619)) < 0.001
        plt.close(fig)

    def test_uses_fill_between(self, simple_equity_curve):
        """Drawdown chart should use filled area (PolyCollection)."""
        fig = plot_drawdown(simple_equity_curve)
        ax = fig.axes[0]
        # fill_between creates PolyCollection
        collections = ax.collections
        assert len(collections) >= 1
        plt.close(fig)


# =============================================================================
# RETURNS DISTRIBUTION TESTS
# =============================================================================

class TestPlotReturnsDistribution:
    """Tests for plot_returns_distribution() function."""

    def test_returns_figure(self, simple_returns):
        """Function should return a matplotlib Figure object."""
        fig = plot_returns_distribution(simple_returns)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_histogram_bars(self, simple_returns):
        """Chart should contain histogram bars (patches)."""
        fig = plot_returns_distribution(simple_returns)
        ax = fig.axes[0]
        patches = ax.patches
        assert len(patches) > 0
        plt.close(fig)

    def test_accepts_bins_parameter(self, simple_returns):
        """Should accept custom number of bins."""
        fig = plot_returns_distribution(simple_returns, bins=10)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_accepts_equity_curve_df(self, simple_equity_curve):
        """Should also accept equity curve DataFrame and compute returns."""
        fig = plot_returns_distribution(simple_equity_curve)
        assert isinstance(fig, Figure)
        plt.close(fig)


# =============================================================================
# WEIGHTS OVER TIME TESTS
# =============================================================================

class TestPlotWeightsOverTime:
    """Tests for plot_weights_over_time() function."""

    def test_returns_figure(self, position_weights_history):
        """Function should return a matplotlib Figure object."""
        fig = plot_weights_over_time(position_weights_history)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_stacked_areas(self, position_weights_history):
        """Chart should have stacked area collections for each ticker."""
        fig = plot_weights_over_time(position_weights_history)
        ax = fig.axes[0]
        # stackplot creates PolyCollections
        collections = ax.collections
        assert len(collections) == 3  # 3 tickers
        plt.close(fig)

    def test_has_legend(self, position_weights_history):
        """Chart should have a legend showing ticker names."""
        fig = plot_weights_over_time(position_weights_history)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_weights_sum_to_one(self, position_weights_history):
        """Visual check: y-axis should max at 1.0 for proper weights."""
        fig = plot_weights_over_time(position_weights_history)
        ax = fig.axes[0]
        ylim = ax.get_ylim()
        # Upper limit should be around 1.0 (with small padding)
        assert ylim[1] <= 1.1
        plt.close(fig)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_day_equity_curve(self):
        """Should handle single-day data gracefully."""
        single_day = pl.DataFrame({
            "date": [date(2024, 1, 1)],
            "portfolio_value": [100000.0],
            "cash": [1000.0],
            "positions_value": [99000.0],
        })
        fig = plot_equity_curve(single_day)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_constant_equity_no_drawdown(self):
        """When equity is constant, drawdown should be all zeros."""
        constant = pl.DataFrame({
            "date": [date(2024, 1, i) for i in range(1, 6)],
            "portfolio_value": [100000.0] * 5,
            "cash": [1000.0] * 5,
            "positions_value": [99000.0] * 5,
        })
        fig = plot_drawdown(constant)
        ax = fig.axes[0]
        line = ax.get_lines()[0]
        ydata = line.get_ydata()
        assert all(y == 0.0 for y in ydata)
        plt.close(fig)

    def test_empty_returns_histogram(self):
        """Should handle empty returns gracefully."""
        empty = pl.Series("returns", [])
        # Should either return a figure or raise a clear error
        try:
            fig = plot_returns_distribution(empty)
            assert isinstance(fig, Figure)
            plt.close(fig)
        except ValueError as e:
            # Acceptable to raise ValueError for empty data
            assert "empty" in str(e).lower()