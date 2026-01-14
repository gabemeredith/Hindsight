"""
Analytics module - Performance metrics for backtesting results.
"""

from .metrics import (
    total_return,
    cagr,
    sharpe_ratio,
    max_drawdown,
    annualized_volatility,
    sortino_ratio,
)

from .benchmark import (
    calculate_alpha,
    calculate_beta,
    calculate_correlation,
    compare_to_benchmark,
)

__all__ = [
    "total_return",
    "cagr",
    "sharpe_ratio",
    "max_drawdown",
    "annualized_volatility",
    "sortino_ratio",
    "calculate_alpha",
    "calculate_beta",
    "calculate_correlation",
    "compare_to_benchmark",
]