"""
Rebalancer - Converts target portfolio weights into trades

This module handles the logic of translating target allocations
into actual buy/sell orders that the Portfolio can execute.
"""

from dataclasses import dataclass
from typing import Literal
from datetime import date
from .portfolio import Portfolio


@dataclass
class Trade:
    """Represents a single trade to execute."""
    ticker: str
    shares: float
    price: float
    side: Literal["buy", "sell"]
    date: date


class Rebalancer:
    """
    Calculates trades needed to reach target portfolio weights.

    Key responsibilities:
    - Calculate current vs target positions
    - Generate buy/sell orders to reach targets
    - Order trades correctly (sells before buys)
    - Handle edge cases (no trades needed, etc.)
    """

    def calculate_trades(
        self,
        current_portfolio: Portfolio,
        target_weights: dict[str, float],
        prices: dict[str, float],
        trade_date: date
    ) -> list[Trade]:
        """
        Calculate trades needed to reach target weights.

        Algorithm:
        1. Calculate total portfolio value (cash + sum of all positions)
        2. For each ticker:
            a. Calculate target dollar value (weight * total_value)
            b. Calculate target shares (target_value / price)
            c. Get current shares (from portfolio.positions)
            d. Calculate delta (target - current)
            e. Create Trade if delta != 0
        3. Return sells first, then buys (to free up cash)

        Parameters
        ----------
        current_portfolio : Portfolio
            The current portfolio state
        target_weights : dict[str, float]
            Ticker → weight mapping (weights should sum to ≤ 1.0)
            Example: {"AAPL": 0.6, "MSFT": 0.4} = 60% AAPL, 40% MSFT
        prices : dict[str, float]
            Ticker → current price mapping
        trade_date : date
            Date to execute trades

        Returns
        -------
        list[Trade]
            List of trades to execute (sells first, then buys)

        Examples
        --------
        >>> portfolio = Portfolio(initial_cash=10000)
        >>> rebalancer = Rebalancer()
        >>> trades = rebalancer.calculate_trades(
        ...     portfolio,
        ...     target_weights={"AAPL": 1.0},
        ...     prices={"AAPL": 100.0},
        ...     trade_date=date(2020, 1, 1)
        ... )
        >>> trades[0].shares  # Should be 100 (10000 / 100)
        100.0
        """

        total_value = current_portfolio.get_total_value(prices)
        trade_buys = []
        trade_sells = []
        for ticker,weight in target_weights.items():
            target_value = weight * total_value
            target_shares = target_value / prices[ticker]
            current_shares = current_portfolio.positions.get(ticker)
            if current_shares is not None:
                current_shares = current_shares.shares
            else:
                current_shares = 0
            delta = target_shares - current_shares
            if delta > 0: #buy
                trade = Trade(ticker=ticker,shares=delta,price=prices[ticker],
                              side="buy",date=trade_date)
                trade_buys.append(trade)
            elif delta < 0: #sell
                trade = Trade(ticker=ticker,shares=abs(delta),price=prices[ticker],
                              side="sell",date=trade_date)
                trade_sells.append(trade)
        for ticker in current_portfolio.positions.keys():
            if ticker not in target_weights:
                shares_to_sell = current_portfolio.positions[ticker].shares
                trade = Trade(ticker=ticker,shares=shares_to_sell,price=prices[ticker],
                              side="sell",date=trade_date)
                trade_sells.append(trade)
        return trade_sells + trade_buys