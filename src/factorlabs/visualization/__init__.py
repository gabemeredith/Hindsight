"""
FactorLab Visualization Module

Provides matplotlib-based charts for backtest analysis:
- Equity curves with benchmark comparison
- Drawdown (underwater) charts
- Returns distribution histograms
- Position weights over time
"""

from .charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_weights_over_time,
)

__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_weights_over_time",
]