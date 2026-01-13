# FactorLab

**A from-scratch quantitative backtesting engine built with test-driven development.**

No black-box libraries. No hidden magic. Every trade, every calculation, fully auditable.

[![Tests](https://img.shields.io/badge/tests-96%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![Polars](https://img.shields.io/badge/polars-1.34+-orange)]()

---

## Why I Built This

Most quant tutorials teach you to `import backtrader` and call it a day. You learn the API, not the concepts.

I wanted to understand:
- How does a rebalancer convert target weights into actual trades?
- What happens when you sell before you buy vs. buy before you sell?
- How do you prevent lookahead bias in factor calculations?

**So I built it from scratch.** 96 tests. Every expected value hand-calculated. Every edge case covered.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ingest_yf.py → Yahoo Finance API → Normalized Parquet         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FACTOR ENGINE                              │
│  factors.py → Returns, Momentum, RSI, SMA, Volatility          │
│  All calculations use .over("ticker") for multi-stock support  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STRATEGY LAYER                              │
│  strategy.py → StaticWeightStrategy | MomentumStrategy         │
│  Abstract interface: get_target_weights(date, prices, factors) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION ENGINE                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Portfolio  │───▶│  Rebalancer │───▶│  Backtester │        │
│  │ tracks cash │    │ weights →   │    │ time loop   │        │
│  │ & positions │    │ trades      │    │ simulation  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUTS                                   │
│  Equity Curve │ Trade History │ Performance Metrics            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Explicit Time Loop (No Vectorized Shortcuts)

```python
for date in trading_days:
    prices = get_prices(date)
    weights = strategy.get_target_weights(date, portfolio, prices, factors)
    trades = rebalancer.calculate_trades(portfolio, weights, prices)
    execute_trades(portfolio, trades)  # sells first, then buys
    record_state(equity_curve, portfolio)
```

Why? Because vectorized backtests hide execution order. In production, you can't buy with money you haven't freed up yet. This engine enforces realistic sequencing.

### Sells Before Buys

```python
# Rebalancer returns trades in this order:
[Trade(AAPL, sell, 50), Trade(MSFT, buy, 100)]
#      ↑ free up cash    ↑ then deploy it
```

### Strategy Pattern for Extensibility

```python
class Strategy(ABC):
    @abstractmethod
    def get_target_weights(self, date, portfolio, prices, factors) -> dict[str, float]:
        pass

# Implementations
class StaticWeightStrategy(Strategy):    # Buy-and-hold
class MomentumStrategy(Strategy):        # Rank by momentum, equal-weight top N
```

---

## Test Coverage

```
tests/test_factors.py                 21 passed
tests/test_ingest_yf.py               19 passed, 1 skipped (API)
tests/test_portfolio.py               14 passed
tests/test_portfolio_enhancements.py  14 passed
tests/test_rebalancer.py              10 passed
tests/test_backtester.py               7 passed
tests/test_strategy.py                12 passed
───────────────────────────────────────────────
TOTAL                                 96 passed
```

Every test uses **hand-calculated expected values**:

```python
def test_calculate_returns_single_day(simple_prices):
    """
    Prices: [100, 110, 121]
    Expected returns:
      Day 2: (110/100) - 1 = 0.10  ← calculated by hand
      Day 3: (121/110) - 1 = 0.10
    """
    result = calculate_returns(prices)
    assert result["ret_1d"][1] == pytest.approx(0.10)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/gabemeredith/FactorLab.git
cd FactorLab

# Install
pip install -e .

# Run tests
pytest tests/ -v

# Run demo pipeline
python demo_pipeline.py
```

---

## Demo Output

```
============================================================
  STEP 4: Run Backtest Simulation
============================================================

Initial capital: $100,000
Rebalance frequency: monthly
Trading days: 252

Running backtest...

✅ Backtest complete!
   - Simulated 252 trading days
   - Executed 3 trades

============================================================
  STEP 5: Performance Analysis
============================================================

 Performance Summary:
   Initial Value:    $  100,000.00
   Final Value:      $  130,300.00
   Total Return:            30.30%
   Volatility (ann):        18.20%
   Max Drawdown:           -13.25%
```

---

## Project Structure

```
FactorLab/
├── src/factorlabs/
│   ├── data/
│   │   ├── ingest_yf.py        # Yahoo Finance → normalized DataFrame
│   │   └── io_utils.py         # Parquet I/O
│   ├── financialfeatures/
│   │   └── factors.py          # Technical indicators (RSI, SMA, momentum)
│   └── backtest/
│       ├── portfolio.py        # Position & cash tracking
│       ├── rebalancer.py       # Weights → trades conversion
│       ├── strategy.py         # Strategy interface + implementations
│       └── backtester.py       # Time-loop simulation engine
├── tests/                      # 96 tests, 100% hand-calculated values
├── demo_pipeline.py            # End-to-end working example
└── README.md
```

---

## Implemented Factors

| Factor | Formula | Window |
|--------|---------|--------|
| `ret_1d` | `(close / close.shift(1)) - 1` | 1 day |
| `log_ret` | `ln(close / close.shift(1))` | 1 day |
| `mom_10d` | `(close / close.shift(10)) - 1` | 10 days |
| `sma_20d` | `close.rolling(20).mean()` | 20 days |
| `vol_10d` | `returns.rolling(10).std()` | 10 days |
| `rsi_14` | `100 - (100 / (1 + RS))` | 14 days |

All factors use Polars `.over("ticker")` for correct multi-stock calculation.

---

## Strategies

### StaticWeightStrategy
```python
strategy = StaticWeightStrategy({"AAPL": 0.6, "MSFT": 0.4})
# Returns same weights every rebalance
```

### MomentumStrategy
```python
strategy = MomentumStrategy(n_positions=3)
# Ranks stocks by 10-day momentum
# Equal-weights top N performers
# Recomputes on each rebalance date
```

---

## What I Learned

1. **TDD catches math bugs immediately.** My first returns calculation was `close/close.shift(1)` without the `- 1`. Test failed. Fixed in 30 seconds. Would have silently corrupted every backtest otherwise.

2. **Execution order matters.** Selling before buying isn't just good practice—it's required when your cash is fully deployed.

3. **Polars > Pandas for this use case.** Native multi-column operations with `.over()`, better type safety, and significantly faster on large datasets.

4. **Abstractions should be earned.** I started with a static weights dict. Only added the Strategy pattern when I needed dynamic momentum selection. No premature abstraction.

---

## Tech Stack

- **Python 3.11** — Type hints, pattern matching
- **Polars** — Fast DataFrame operations, native lazy evaluation
- **pytest** — Test framework with fixtures and parametrization
- **yfinance** — Market data API

---

## Roadmap

- [x] Data ingestion pipeline
- [x] Factor calculations (6 indicators)
- [x] Portfolio state management
- [x] Rebalancer (weights → trades)
- [x] Strategy interface (static + momentum)
- [x] Backtester with explicit time loop
- [ ] Analytics module (Sharpe, Sortino, Calmar)
- [ ] Visualization (equity curves, drawdown charts)
- [ ] Additional strategies (mean reversion, multi-factor)

---

## License

MIT

---

*Built to understand quantitative finance from first principles. Not financial advice. Not production-ready for real trading.*