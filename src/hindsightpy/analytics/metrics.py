"""
Performance metrics for backtesting results.

All functions operate on Polars DataFrames or Series.
"""

import polars as pl
import math
from datetime import date

def total_return(equity_curve: pl.DataFrame) -> float:
    """
    Calculate total return from equity curve.

    Parameters
    ----------
    equity_curve : pl.DataFrame
        Must contain 'portfolio_value' column.
        First row is starting value, last row is ending value.

    Returns
    -------
    float
        Total return as decimal (0.21 = 21%)

    Formula
    -------
    (ending_value / starting_value) - 1
    """
    ending_value = equity_curve["portfolio_value"][-1]
    starting_value = equity_curve["portfolio_value"][0]
    return ending_value / starting_value - 1

def cagr(equity_curve: pl.DataFrame) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Parameters
    ----------
    equity_curve : pl.DataFrame
        Must contain 'date' and 'portfolio_value' columns.

    Returns
    -------
    float
        CAGR as decimal (0.20 = 20%)

    Formula
    -------
    (ending_value / starting_value) ^ (1 / years) - 1

    Where years = (end_date - start_date) / 365
    """
    starting_value = equity_curve["portfolio_value"][0]
    ending_value = equity_curve["portfolio_value"][-1]
    start_date = equity_curve["date"][0]
    end_date = equity_curve["date"][-1]
    
    years = (end_date - start_date).days / 365
    
    base = (ending_value / starting_value)
    return math.pow(base,(1 / years)) - 1



# def max_drawdown(equity_curve: pl.DataFrame) -> float:
#     """
#     Calculate maximum drawdown (peak to trough decline).

#     Parameters
#     ----------
#     equity_curve : pl.DataFrame
#         Must contain 'portfolio_value' column.

#     Returns
#     -------
#     float
#         Max drawdown as negative decimal (-0.20 = -20% drawdown)

#     Formula
#     -------
#     For each point: (value - running_max) / running_max
#     Return the minimum (most negative) value.
#     """
#     equity_curve = equity_curve.with_columns(
#         pl.col("portfolio_value").cum_max().alias("running_max")
#     )
#     equity_curve =equity_curve.with_columns(
#          ((pl.col("portfolio_value") - pl.col("running_max")) / pl.col("running_max")).alias("difference")
#     )
#     return equity_curve.select(pl.col("difference").min()).item()

def calculate_drawdown_series(equity_curve: pl.DataFrame) -> pl.DataFrame:
    """
    calculates drawdown at each point in time.
    
    Returns df with added 'drawdown' column.
    """
    equity_curve = equity_curve.with_columns(
        pl.col("portfolio_value").cum_max().alias("running_max")
    )
    equity_curve =equity_curve.with_columns(
         ((pl.col("portfolio_value") - pl.col("running_max")) / pl.col("running_max")).alias("drawdown")
    )
    return equity_curve 

def max_drawdown(equity_curve: pl.DataFrame) -> float:                                                                     
      """Return the worst (minimum) drawdown."""                                                                             
      result = calculate_drawdown_series(equity_curve)                                                                       
      return result["drawdown"].min()    
  
def annualized_volatility(returns: pl.Series, trading_days: int = 252) -> float:
    """
    Calculate annualized volatility from daily returns.

    Parameters
    ----------
    returns : pl.Series
        Daily returns as decimals (0.01 = 1%)
    trading_days : int
        Trading days per year (default 252)

    Returns
    -------
    float
        Annualized volatility as decimal (0.20 = 20%)

    Formula
    -------
    daily_std * sqrt(trading_days)
    """
    daily_std = returns.std()
    return daily_std * math.sqrt(trading_days)


def sharpe_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Parameters
    ----------
    returns : pl.Series
        Daily returns as decimals
    risk_free_rate : float
        Annual risk-free rate as decimal (0.02 = 2%)
    trading_days : int
        Trading days per year

    Returns
    -------
    float
        Annualized Sharpe ratio

    Formula
    -------
    Daily Sharpe = (mean_return - daily_rf) / std_return
    Annualized Sharpe = Daily Sharpe * sqrt(trading_days)

    Where daily_rf = risk_free_rate / trading_days
    """
    std_ret = returns.std()
    mean_ret = returns.mean()
    daily_rf = risk_free_rate / trading_days
    daily_sharpe = (mean_ret - daily_rf) / std_ret
    
    ann_sharpe = daily_sharpe * math.sqrt(trading_days)
    return ann_sharpe


def sortino_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    trading_days: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation only).

    Parameters
    ----------
    returns : pl.Series
        Daily returns as decimals
    risk_free_rate : float
        Annual risk-free rate as decimal
    trading_days : int
        Trading days per year

    Returns
    -------
    float
        Annualized Sortino ratio

    Formula
    -------
    Like Sharpe, but uses downside deviation instead of std.
    Downside deviation = std of returns below target (usually 0 or rf).
    """
    returns_below_target = returns.filter(returns < risk_free_rate)
    
    down_dev = returns_below_target.std()   
    if down_dev is None or down_dev == 0:
      return float('inf')
    daily_rf = risk_free_rate / trading_days
    mean_ret = returns.mean()
    daily_sharpe = (mean_ret - daily_rf) / down_dev
    
    sortino_ratio = daily_sharpe * math.sqrt(trading_days)
    return sortino_ratio 