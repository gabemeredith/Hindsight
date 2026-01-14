"""
Backtester - Orchestrates portfolio simulation over time

This module ties together the Portfolio, Rebalancer, and price data
to simulate a trading strategy over a historical period.

Key responsibilities:
- Run time loop (day by day)
- Get prices for each date
- Calculate required trades (using Rebalancer)
- Execute trades (using Portfolio)
- Record state (equity curve, trades, positions)
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal
import polars as pl

from .portfolio import Portfolio
from .rebalancer import Rebalancer
from .strategy import Strategy, StaticWeightStrategy


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    start_date: date
    end_date: date
    initial_cash: float
    rebalance_frequency: Literal["daily", "weekly", "monthly", "never"] = "daily"
    slippage_pct: float = 0.0 #e.g 0.001 = 0.1 %
    commission_pct: float = 0.0 #e.g 0.001 = 0.1 %
    


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    equity_curve: pl.DataFrame      # Columns: date, portfolio_value, cash, positions_value
    trades: pl.DataFrame            # Columns: date, ticker, shares, price, side
    positions_history: pl.DataFrame = None  # Optional: date, ticker, shares, value
    metrics: dict = field(default_factory=dict)  # Summary metrics (total return, etc.)


class Backtester:
    """
    Runs portfolio simulations over historical price data.

    The backtester implements an explicit time loop:
    1. For each date in the simulation period:
       a. Get prices for that date
       b. Calculate target trades (using Rebalancer)
       c. Execute trades (using Portfolio)
       d. Record portfolio state
    2. Return results as DataFrames

    This design prioritizes:
    - Clarity: Explicit day-by-day execution (no hidden vectorization)
    - Correctness: Realistic trade sequencing
    - Auditability: Every state change is recorded
    """

    def run(
        self,
        prices: pl.DataFrame,
        strategy: Strategy | dict[str, float],
        config: BacktestConfig,
        factors: pl.DataFrame | None = None,
    ) -> BacktestResult:
        """
        Run a backtest simulation.

        Parameters
        ----------
        prices : pl.DataFrame
            Historical prices with columns: date, ticker, close
        strategy : Strategy | dict[str, float]
            Either a Strategy object or a dict of static weights.
            If dict, will be wrapped in StaticWeightStrategy.
        config : BacktestConfig
            Backtest parameters (dates, cash, frequency, etc.)
        factors : pl.DataFrame, optional
            Pre-computed factors for dynamic strategies.
            Columns: [date, ticker, close, mom_10d, rsi_14, ...]

        Returns
        -------
        BacktestResult
            Contains equity_curve, trades, and other results
        """
        # Wrap dict in StaticWeightStrategy for backward compatibility
        if isinstance(strategy, dict):
            strategy = StaticWeightStrategy(strategy)

        current_portfolio = Portfolio(initial_cash=config.initial_cash)
        current_rebalancer = Rebalancer()
        
        #data - only loop through dates that exist in price data (skips weekends/holidays)
        list_of_dates = (
            prices
            .filter(pl.col("date") >= config.start_date)
            .filter(pl.col("date") <= config.end_date)
            .select("date")
            .unique()
            .sort("date")["date"]
            .to_list()
        )
        last_rebalance_date = None
        equity_records = []
        trade_records = []
        for date in list_of_dates:
            daily_prices = prices.filter(pl.col("date") == date)
            prices_dict = {
                row["ticker"]: row["close"] for row in daily_prices.iter_rows(named=True)
            }
            rebalancing = self._should_rebalance(date,last_rebalance_date,config.rebalance_frequency)
            if rebalancing:
                target_weights = strategy.get_target_weights(
                    current_date=date,
                    portfolio=current_portfolio,
                    prices=prices_dict,
                    factors=factors,
                )
                rebalancing_trades = current_rebalancer.calculate_trades(current_portfolio=current_portfolio,
                                            target_weights=target_weights,
                                            prices=prices_dict,trade_date=date)
                for trade in rebalancing_trades:
                    if trade.side == "buy":
                        effective_price = trade.price * (1 + config.slippage_pct)
                    else:
                        effective_price = trade.price * (1 - config.slippage_pct)
                    if trade.side == "sell":
                        current_portfolio.sell(trade.ticker,trade.shares,
                                               effective_price,trade.date)
                    elif trade.side == "buy":
                        
                        current_portfolio.buy(trade.ticker,trade.shares,
                                               effective_price,trade.date)
                    trade_value = trade.shares * effective_price
                    commission = trade_value * config.commission_pct
                    current_portfolio.cash -= commission
                    trade_records.append({
                        "date": trade.date,
                        "ticker": trade.ticker,
                        "shares": trade.shares,
                        "price": effective_price,
                        "side": trade.side
                    })
                last_rebalance_date = date
            total_value = current_portfolio.get_total_value(prices_dict)
            positions_value = current_portfolio.get_holdings_value(prices_dict)
            positions_value_sum = sum(positions_value.values())
            
            equity_records.append({
                "date":date,
                "portfolio_value":total_value,
                "cash":current_portfolio.cash,
                "positions_value":positions_value_sum
            })
            
        equity_curve = pl.DataFrame(equity_records)

        if trade_records:
            trades_df = pl.DataFrame(trade_records)
        else:
            # Empty DataFrame with correct schema
            trades_df = pl.DataFrame({
                "date": [],
                "ticker": [],
                "shares": [],
                "price": [],
                "side": []
            })

        return BacktestResult(equity_curve=equity_curve, trades=trades_df)

    def _should_rebalance(
        self,
        current_date: date,
        last_rebalance_date: date | None,
        frequency: str
    ) -> bool:
        """
        Check if should rebalance on this date.

        Parameters
        ----------
        current_date : date
            Today's date
        last_rebalance_date : date | None
            Last time we rebalanced (None if never)
        frequency : str
            "daily", "weekly", "monthly", or "never"

        Returns
        -------
        bool
            True if should rebalance today

        Examples
        --------
        - "daily": Always True
        - "never": Always False (except first day)
        - "weekly": True if 7+ days since last rebalance
        - "monthly": True if different month than last rebalance
        """
        # TODO: Implement this helper method
        # For now, you can start simple:
        # - "daily" → return True
        # - "never" → return True only if last_rebalance_date is None
        # - "weekly" and "monthly" → implement later if needed
        if frequency == "never":
            #unless sfirst day
            if last_rebalance_date == None:
                return True
        elif frequency == "daily":
            return True
        elif frequency == "weekly":
            if last_rebalance_date is None:
                return True
            if (current_date - last_rebalance_date).days >= 7:
                return True
        elif frequency == "monthly":
            if last_rebalance_date is None:
                return True
            return (last_rebalance_date.month != current_date.month or
                    last_rebalance_date.year != current_date.year)
        return False
