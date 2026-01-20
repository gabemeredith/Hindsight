"""
Portfolio class for tracking positions and cash.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Position:
    """Represents a position in a single security."""
    ticker: str
    shares: float
    entry_price: float
    entry_date: date


class Portfolio:
    """
    Tracks cash and positions over time.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
    
    def get_total_value(self, prices: dict[str, float] = None) -> float:
        """
        Calculate total portfolio value.
        
        Parameters
        ----------
        prices : dict[str, float], optional
            Current prices for each ticker. If None, assumes only cash.
        
        Returns
        -------
        float
            Total value = cash + sum(position values)
        """
        if prices is None:
            # No positions valued yet, return just cash
            return self.cash
        
        positions_value = sum(
            pos.shares * prices.get(pos.ticker, 0.0)
            for pos in self.positions.values()
        )
        
        return self.cash + positions_value
    
    def buy(self, ticker: str, shares: float, price: float, date: date) -> None:
        """
        Execute a buy order.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        shares : float
            Number of shares to buy
        price : float
            Price per share
        date : date
            Trade date
            
        Raises
        ------
        ValueError
            If insufficient cash
        """
        cost = shares * price
        
        if cost > self.cash:
            raise ValueError(
                f"Insufficient cash: need ${cost:,.2f}, have ${self.cash:,.2f}"
            )
        
        # Deduct cash
        self.cash -= cost
        
        # Add or update position
        if ticker in self.positions:
            # Already have position - update it (average cost basis)
            existing = self.positions[ticker]
            total_shares = existing.shares + shares
            avg_price = (
                (existing.shares * existing.entry_price + shares * price)
                / total_shares
            )
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=total_shares,
                entry_price=avg_price,
                entry_date=existing.entry_date  # Keep original date
            )
        else:
            # New position
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=price,
                entry_date=date
            )
        
    def sell(self,ticker: str, shares: float,price: float, date: date) -> None:
        """
        executes a sell order

         Parameters
        ----------
        ticker : str
            Stock ticker symbol
        shares : float
            Number of shares to buy
        price : float
            Price per share
        date : date
            Trade date
            
        """
        if shares <= 0:
            raise ValueError("Shares must be positive")
        
        if ticker not in self.positions:
            raise ValueError(f"No position in {ticker}")
        
        if shares > self.positions[ticker].shares:
            raise ValueError(
                f"Insufficient shares: trying to sell {shares},"
                f"only own {self.positions[ticker].shares}"
            )
            
        proceeds = shares * price 
        self.cash += proceeds
        #tracking the realized pnl
        self.realized_pnl += proceeds - shares * self.positions[ticker].entry_price
        if shares == self.positions[ticker].shares:
            del self.positions[ticker]
        else:
            existing = self.positions[ticker]
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=existing.shares - shares,
                entry_price=existing.entry_price,
                entry_date=existing.entry_date)
    
    def get_unrealized_pnl(self,prices: dict[str,float]) -> float:
        """
        Calculate unrealized P&L for all open positions.
        
        Parameters
        ----------
        prices : dict[str, float]
            Current market prices (ticker → price)
        
        Returns
        -------
        float
            Total unrealized P&L across all positions
        """
        total_pnl = 0
        for pos in self.positions.values():
            if prices.get(pos.ticker) is not None:
                current_price = prices.get(pos.ticker) * pos.shares
                total_pnl += current_price - (pos.entry_price * pos.shares)
        return total_pnl
        
    def get_realized_pnl(self) -> float:
        """
        Calculate realized P&L for all open positions.
        
        Parameters
        ----------
        prices : dict[str, float]
            Current market prices (ticker → price)
        
        Returns
        -------
        float
            Total unrealized P&L across all positions
        """
        return self.realized_pnl
    
    def get_holdings_value(self,prices: dict[str,float]) -> dict:
        """
        returns dict of the value of all open positions
        """
        res = {}
        for pos in self.positions.values():
            if prices.get(pos.ticker) is not None:
                res[pos.ticker] = pos.shares * prices.get(pos.ticker)
        return res        
