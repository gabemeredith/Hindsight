"""
Strategy classes for portfolio allocation.

A Strategy converts market data and portfolio state into target weights.
It is called once per rebalancing decision by the Backtester.

"""

from abc import ABC, abstractmethod
from datetime import date
import polars as pl

from .portfolio import Portfolio


class Strategy(ABC):
    """
    Abstract base class for portfolio strategies.

    All strategies must implement get_target_weights(), which takes
    current market state and returns target portfolio weights.

    Weights are expressed as decimals (0.5 = 50%) and must sum to <= 1.0.
    The remainder (1.0 - sum(weights)) is held as cash.
    """

    @abstractmethod
    def get_target_weights(
        self,
        current_date: date,
        portfolio: Portfolio,
        prices: dict[str, float],
        factors: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Calculate target portfolio weights for current rebalancing decision.

        Parameters
        ----------
        current_date : date
            Today's date in the backtest
        portfolio : Portfolio
            Current portfolio state (positions, cash)
        prices : dict[str, float]
            Current prices: {ticker -> price}
        factors : pl.DataFrame, optional
            Historical factors DataFrame with columns:
            [date, ticker, close, mom_10d, rsi_14, ...]
            If provided, filter to current_date for signals.

        Returns
        -------
        dict[str, float]
            Target allocation: {ticker -> weight}
            Weights should sum to <= 1.0 (remainder is cash)
            Return empty dict {} for 100% cash.

        Examples
        --------
        >>> weights = strategy.get_target_weights(
        ...     current_date=date(2024, 1, 15),
        ...     portfolio=portfolio,
        ...     prices={"AAPL": 150.0, "MSFT": 330.0},
        ...     factors=factors_df
        ... )
        >>> weights
        {"AAPL": 0.6, "MSFT": 0.4}
        """
        pass


class StaticWeightStrategy(Strategy):
    """
    Simple buy-and-hold strategy: same weights every rebalance.

    This strategy ignores all market data and always returns
    the same fixed weights it was initialized with.

    Parameters
    ----------
    weights : dict[str, float]
        Fixed target weights, e.g. {"AAPL": 0.6, "MSFT": 0.4}

    Examples
    --------
    >>> strategy = StaticWeightStrategy({"AAPL": 0.6, "MSFT": 0.4})
    >>> strategy.get_target_weights(...)
    {"AAPL": 0.6, "MSFT": 0.4}  # Always the same!
    """

    def __init__(self, weights: dict[str, float]):
        """
        Initialize with fixed weights.
        """
        if weights is not None:
            self.weights = weights
        else:
            self.weights = {}


    def get_target_weights(
        self,
        current_date: date,
        portfolio: Portfolio,
        prices: dict[str, float],
        factors: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Return fixed weights (ignores all inputs).
        """
        return self.weights


class MomentumStrategy(Strategy):
    """
    Momentum strategy: buy top N stocks by momentum factor.

    Ranks all stocks by their momentum (mom_10d column) and
    equal-weights the top N performers.

    Parameters
    ----------
    n_positions : int
        Number of top momentum stocks to hold (default: 3)
    momentum_col : str
        Name of momentum column in factors DataFrame (default: "mom_10d")

    Examples
    --------
    >>> strategy = MomentumStrategy(n_positions=2)
    >>> # If GOOGL has highest momentum, AAPL second highest:
    >>> strategy.get_target_weights(...)
    {"GOOGL": 0.5, "AAPL": 0.5}  # Equal weight top 2
    """

    def __init__(self, n_positions: int = 3, momentum_col: str = "mom_10d", max_allocation: float = 0.97):
        """
        Initialize momentum strategy.

        Parameters
        ----------
        n_positions : int
            Number of top momentum stocks to hold
        momentum_col : str
            Column name for momentum factor
        max_allocation : float
            Maximum portfolio allocation (default 0.97 = 97%, leaves 3% buffer for costs)
        """
        self.n_positions = n_positions
        self.momentum_col = momentum_col
        self.max_allocation = max_allocation


    def get_target_weights(
        self,
        current_date: date,
        portfolio: Portfolio,
        prices: dict[str, float],
        factors: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """
        Select top N stocks by momentum, equal-weight them.

        Algorithm:
        1. If factors is None, return {} (can't rank without data)
        2. Filter factors to current_date only
        3. If no data for current_date, return {}
        4. Sort by momentum column (descending - highest first)
        5. Take top N stocks
        6. Equal-weight them: each gets 1/N weight
        7. Return as dict
        """
        # Step 1: Handle missing factors
        if factors is None:
            return {}
        todays_factors = factors.filter(pl.col("date") == current_date)
        if len(todays_factors) == 0:
            return {}
        # Step 2: Filter to current date, momentum
        #n_to_select is there to make sure if desired # of positions > our # of positions to select all positions
        n_to_select = min(self.n_positions,len(todays_factors))
        
        tickers = todays_factors.sort(
        by=self.momentum_col,descending=True).head(n_to_select)
        # Step 6: Calculate equal weight (using max_allocation to leave buffer for costs)
        weight = self.max_allocation / n_to_select
        # Step 7: Build and return weights dict
        weights = {}
        for row in tickers.iter_rows(named=True):
            weights[row["ticker"]] = weight
        
        return weights