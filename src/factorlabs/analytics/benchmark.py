"""
Benchmark comparison metrics.

Compare portfolio performance against a market benchmark (e.g., SPY).
"""

import polars as pl
import math


def calculate_alpha(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    trading_days: int = 252
) -> float:
    """
    Calculate annualized alpha (excess return over benchmark).

    Parameters
    ----------
    portfolio_returns : pl.Series
        Daily portfolio returns
    benchmark_returns : pl.Series
        Daily benchmark returns
    trading_days : int
        Trading days per year for annualization

    Returns
    -------
    float
        Annualized alpha as decimal (0.05 = 5% annual excess return)

    Formula
    -------
    daily_alpha = mean(portfolio_returns) - mean(benchmark_returns)
    annualized_alpha = daily_alpha * trading_days
    """
    daily_alpha = portfolio_returns.mean() - benchmark_returns.mean()
    annualized_alpha = daily_alpha * trading_days
    return annualized_alpha


def calculate_beta(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series
) -> float:
    """
    Calculate portfolio beta relative to benchmark.

    Parameters
    ----------
    portfolio_returns : pl.Series
        Daily portfolio returns
    benchmark_returns : pl.Series
        Daily benchmark returns

    Returns
    -------
    float
        Beta coefficient (1.0 = moves with market, >1 = more volatile)

    Formula
    -------
    beta = covariance(portfolio, benchmark) / variance(benchmark)

    Hints
    -----
    - Polars doesn't have built-in covariance, but you can compute it:
      cov(X, Y) = mean((X - mean(X)) * (Y - mean(Y)))
    - Or convert to Python lists/numpy and use standard formulas
    """
    n = len(portfolio_returns)
    covariance = ((portfolio_returns - portfolio_returns.mean()) * (benchmark_returns - benchmark_returns.mean())).sum() / (n - 1)
    beta = covariance / benchmark_returns.var()
    return beta


def calculate_correlation(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series
) -> float:
    """
    Calculate correlation between portfolio and benchmark.

    Parameters
    ----------
    portfolio_returns : pl.Series
        Daily portfolio returns
    benchmark_returns : pl.Series
        Daily benchmark returns

    Returns
    -------
    float
        Pearson correlation coefficient (-1 to 1)

    Formula
    -------
    correlation = covariance(portfolio, benchmark) / (std(portfolio) * std(benchmark))

    Or equivalently:
    correlation = beta * std(benchmark) / std(portfolio)
    """
    n = len(portfolio_returns)
    covariance = ((portfolio_returns - portfolio_returns.mean()) * (benchmark_returns - benchmark_returns.mean())).sum() / (n - 1)
    beta = covariance / benchmark_returns.var()
    
    correlation = beta * benchmark_returns.std() / portfolio_returns.std()
    return correlation


def compare_to_benchmark(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    trading_days: int = 252
) -> dict:
    """
    Calculate all benchmark comparison metrics.

    Parameters
    ----------
    portfolio_returns : pl.Series
        Daily portfolio returns
    benchmark_returns : pl.Series
        Daily benchmark returns
    trading_days : int
        Trading days per year

    Returns
    -------
    dict
        Contains: alpha, beta, correlation
    """
    return {
        "alpha": calculate_alpha(portfolio_returns, benchmark_returns, trading_days),
        "beta": calculate_beta(portfolio_returns, benchmark_returns),
        "correlation": calculate_correlation(portfolio_returns, benchmark_returns),
    }