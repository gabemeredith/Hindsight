"""
Matplotlib-based visualization charts for backtest analysis.

All functions return matplotlib Figure objects that can be:
- Displayed interactively: plt.show()
- Saved to file: fig.savefig("chart.png")
- Embedded in dashboards
"""

from datetime import date
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
from factorlabs.analytics.metrics import calculate_drawdown_series                                                         

def plot_equity_curve(
    equity_curve: pl.DataFrame,
    benchmark: pl.DataFrame | None = None,
    title: str = "Portfolio Equity Curve",
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot portfolio value over time.

    Parameters
    ----------
    equity_curve : pl.DataFrame
        Must contain columns: date, portfolio_value
    benchmark : pl.DataFrame, optional
        Benchmark data with columns: date, portfolio_value
        Will be normalized to start at same value as portfolio
    title : str
        Chart title
    figsize : tuple
        Figure dimensions (width, height)

    Returns
    -------
    Figure
        Matplotlib figure object

    """
    fig, ax = plt.subplots(figsize=figsize)
    dates = equity_curve["date"].to_list()
    values = equity_curve["portfolio_value"].to_list()
    ax.plot(dates,values,label="Portfolio")
    if benchmark is not None:
        # portfolio_start_value / benchmark_start_value to normalize
        bm_values = benchmark["portfolio_value"].to_list() 
        bm_dates = benchmark["date"].to_list()
        scale_factor = values[0] / bm_values[0]
        scaled_values = [v * scale_factor for v in bm_values]
        ax.plot(bm_dates,scaled_values,label = "Benchmark")
    
    ax.set_title(title)
    ax.set_xlabel("portfolio vs. benchmark")
    ax.set_ylabel("money")
    return fig


def plot_drawdown(
    equity_curve: pl.DataFrame,
    title: str = "Portfolio Drawdown",
    figsize: tuple[int, int] = (12, 4),
) -> Figure:
    """
    Plot underwater chart showing drawdown from peak.

    Drawdown is always negative or zero, representing the percentage
    decline from the highest value seen so far.

    Parameters
    ----------
    equity_curve : pl.DataFrame
        Must contain columns: date, portfolio_value
    title : str
        Chart title
    figsize : tuple
        Figure dimensions (width, height)

    Returns
    -------
    Figure
        Matplotlib figure object

    """
    df_with_drawdown = calculate_drawdown_series(equity_curve)
    
    fig,ax = plt.subplots(figsize=figsize)
    
    dates = df_with_drawdown["date"].to_list()
    drawdowns = df_with_drawdown["drawdown"].to_list()
    
    ax.plot(dates,drawdowns)
    ax.fill_between(dates,drawdowns,0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    
    return fig


def plot_returns_distribution(
    returns_or_equity: pl.Series | pl.DataFrame,
    bins: int = 30,
    title: str = "Returns Distribution",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot histogram of returns.

    Parameters
    ----------
    returns_or_equity : pl.Series | pl.DataFrame
        Either:
        - pl.Series of return values
        - pl.DataFrame with 'portfolio_value' column (returns will be computed)
    bins : int
        Number of histogram bins
    title : str
        Chart title
    figsize : tuple
        Figure dimensions (width, height)

    Returns
    -------
    Figure
        Matplotlib figure object
 
    """
    fig,ax = plt.subplots(figsize=figsize)
    
    if isinstance(returns_or_equity,pl.DataFrame):
        returns = returns_or_equity["portfolio_value"].pct_change().drop_nulls()
    elif isinstance(returns_or_equity,pl.Series):
        returns = returns_or_equity
    if returns is not None:
        ax.hist(returns,bins=bins)
        ax.set_title(title)
        if len(returns) > 0:
            ax.axvline(returns.mean())
    else:
        raise ValueError("returns are None")
    return fig


def plot_weights_over_time(
    weights_df: pl.DataFrame,
    title: str = "Position Weights Over Time",
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot stacked area chart showing portfolio allocation over time.

    Parameters
    ----------
    weights_df : pl.DataFrame
        Must contain 'date' column and one column per ticker with weight values.
        Example: date | aapl | msft | googl
                 ...  | 0.4  | 0.4  | 0.2
    title : str
        Chart title
    figsize : tuple
        Figure dimensions (width, height)

    Returns
    -------
    Figure
        Matplotlib figure object

    """
    fig,ax = plt.subplots(figsize=figsize)

    dates = weights_df["date"]
    tickers = [col for col in weights_df.columns if col != "date"]
    weight_data = [weights_df[ticker].to_list() for ticker in tickers]
    
    ax.stackplot(dates,*weight_data,labels=tickers)
    ax.legend()
    ax.set_xlabel("Weights")
    ax.set_ylabel("Area")
    ax.set_title(title)
    return fig