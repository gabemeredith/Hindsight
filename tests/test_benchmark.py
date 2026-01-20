"""
Tests for benchmark comparison metrics.

These metrics compare portfolio performance against a benchmark (e.g., SPY).
"""

import pytest
import polars as pl
import math
from datetime import date

from hindsightpy.analytics.benchmark import (
    calculate_alpha,
    calculate_beta,
    calculate_correlation,
    compare_to_benchmark,
)


# ========================== FIXTURES ==========================

@pytest.fixture
def portfolio_returns():
    """
    Portfolio daily returns for 5 days.
    Returns: [0.01, 0.02, -0.01, 0.015, 0.005]
    """
    return pl.Series("portfolio", [0.01, 0.02, -0.01, 0.015, 0.005])


@pytest.fixture
def benchmark_returns():
    """
    Benchmark (SPY) daily returns for 5 days.
    Returns: [0.008, 0.015, -0.005, 0.01, 0.003]
    """
    return pl.Series("benchmark", [0.008, 0.015, -0.005, 0.01, 0.003])


@pytest.fixture
def identical_returns():
    """Portfolio that exactly matches benchmark."""
    returns = [0.01, -0.02, 0.015, 0.005, -0.01]
    return (
        pl.Series("portfolio", returns),
        pl.Series("benchmark", returns),
    )


@pytest.fixture
def uncorrelated_returns():
    """Portfolio uncorrelated with benchmark."""
    return (
        pl.Series("portfolio", [0.01, -0.01, 0.01, -0.01, 0.01]),
        pl.Series("benchmark", [0.01, 0.01, -0.01, -0.01, 0.01]),
    )


# ========================== ALPHA TESTS ==========================

class TestAlpha:
    """
    Alpha = annualized excess return over benchmark.

    Formula: (portfolio_mean - benchmark_mean) * 252
    """

    def test_positive_alpha(self, portfolio_returns, benchmark_returns):
        """
        Portfolio outperforms benchmark.

        Portfolio: [0.01, 0.02, -0.01, 0.015, 0.005]
        Portfolio sum: 0.01 + 0.02 + (-0.01) + 0.015 + 0.005 = 0.04
        Portfolio mean: 0.04 / 5 = 0.008

        Benchmark: [0.008, 0.015, -0.005, 0.01, 0.003]
        Benchmark sum: 0.008 + 0.015 + (-0.005) + 0.01 + 0.003 = 0.031
        Benchmark mean: 0.031 / 5 = 0.0062

        Daily alpha: 0.008 - 0.0062 = 0.0018
        Annualized alpha: 0.0018 * 252 = .4536
        """
        result = calculate_alpha(portfolio_returns, benchmark_returns)

        expected = .4536
        assert result == pytest.approx(expected, rel=0.01)

    def test_zero_alpha_identical_returns(self, identical_returns):
        """Portfolio matching benchmark has zero alpha."""
        portfolio, benchmark = identical_returns
        result = calculate_alpha(portfolio, benchmark)

        assert result == pytest.approx(0.0, abs=0.001)

    def test_negative_alpha(self):
        """Portfolio underperforms benchmark."""
        portfolio = pl.Series("portfolio", [0.005, 0.01, -0.02, 0.005, 0.0])
        benchmark = pl.Series("benchmark", [0.01, 0.015, -0.01, 0.01, 0.005])

        result = calculate_alpha(portfolio, benchmark)

        # Portfolio mean < benchmark mean, so alpha < 0
        assert result < 0.0


# ========================== BETA TESTS ==========================

class TestBeta:
    """
    Beta = covariance(portfolio, benchmark) / variance(benchmark)

    Measures how much portfolio moves relative to benchmark.
    Beta = 1.0 means moves exactly with market.
    Beta > 1.0 means more volatile than market.
    Beta < 1.0 means less volatile than market.
    """

    def test_beta_identical_returns(self, identical_returns):
        """Portfolio identical to benchmark has beta = 1.0."""
        portfolio, benchmark = identical_returns
        result = calculate_beta(portfolio, benchmark)

        assert result == pytest.approx(1.0, rel=0.01)

    def test_beta_calculation(self, portfolio_returns, benchmark_returns):
        """
        Calculate beta from covariance and variance.

        Portfolio: [0.01, 0.02, -0.01, 0.015, 0.005], mean = 0.008
        Benchmark: [0.008, 0.015, -0.005, 0.01, 0.003], mean = 0.0062

        Portfolio deviations: [0.002, 0.012, -0.018, 0.007, -0.003]
        Benchmark deviations: [0.0018, 0.0088, -0.0112, 0.0038, -0.0032]

        Products of deviations:
          0.002 * 0.0018 = 0.0000036
          0.012 * 0.0088 = 0.0001056
          -0.018 * -0.0112 = 0.0002016
          0.007 * 0.0038 = 0.0000266
          -0.003 * -0.0032 = 0.0000096
          Sum = 0.000347

        Covariance = 0.00008675

        Benchmark deviations squared:
          0.0018^2 + 0.0088^2 + 0.0112^2 + 0.0038^2 + 0.0032^2 = 0.0002308

        Variance= 0.0000577

        Beta = cov / var = 0.00008675 / 0.0000577 = 1.503
        """
        result = calculate_beta(portfolio_returns, benchmark_returns)

        expected = 1.5034  
        assert result == pytest.approx(expected, rel=0.05)

    def test_beta_double_leverage(self):
        """Portfolio with 2x leverage has beta ≈ 2.0."""
        benchmark = pl.Series("benchmark", [0.01, -0.01, 0.02, -0.02, 0.01])
        portfolio = pl.Series("portfolio", [0.02, -0.02, 0.04, -0.04, 0.02])

        result = calculate_beta(portfolio, benchmark)

        assert result == pytest.approx(2.0, rel=0.01)


# ========================== CORRELATION TESTS ==========================

class TestCorrelation:
    """
    Correlation measures how closely portfolio tracks benchmark.
    Range: -1.0 to 1.0
    """

    def test_perfect_correlation(self, identical_returns):
        """Identical returns have correlation = 1.0."""
        portfolio, benchmark = identical_returns
        result = calculate_correlation(portfolio, benchmark)

        assert result == pytest.approx(1.0, rel=0.01)

    def test_correlation_range(self, portfolio_returns, benchmark_returns):
        """Correlation should be between -1 and 1."""
        result = calculate_correlation(portfolio_returns, benchmark_returns)

        assert -1.0 <= result <= 1.0

    def test_low_correlation(self, uncorrelated_returns):
        """Uncorrelated returns have correlation near 0."""
        portfolio, benchmark = uncorrelated_returns
        result = calculate_correlation(portfolio, benchmark)

        # Should be close to 0 (not perfectly 0 due to small sample)
        assert abs(result) < 0.5


# ========================== COMPARE TO BENCHMARK TESTS ==========================

class TestCompareToBenchmark:
    """
    Convenience function returning all comparison metrics.
    """

    def test_returns_dict_with_all_metrics(self, portfolio_returns, benchmark_returns):
        """Should return dict with alpha, beta, correlation."""
        result = compare_to_benchmark(portfolio_returns, benchmark_returns)

        assert "alpha" in result
        assert "beta" in result
        assert "correlation" in result

    def test_metrics_are_floats(self, portfolio_returns, benchmark_returns):
        """All metrics should be numeric."""
        result = compare_to_benchmark(portfolio_returns, benchmark_returns)

        assert isinstance(result["alpha"], float)
        assert isinstance(result["beta"], float)
        assert isinstance(result["correlation"], float)