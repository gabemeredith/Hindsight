# FactorLab

**A from-scratch quantitative backtesting engine built with test-driven development.**

No black-box libraries. No hidden magic. Every trade, every calculation, fully auditable.

[![Tests](https://img.shields.io/badge/tests-153%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![Polars](https://img.shields.io/badge/polars-1.34+-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
🌐 **[Try the Live Demo](https://factorlab.streamlit.app)

---

## Quick Start

```bash
# Install
pip install -e .

# Run a backtest (one command does everything)
factorlab run AAPL MSFT GOOGL --start 2024-01-01 --end 2024-06-01
```

**Output:**
```
==================================================
  FactorLab - Running Complete Backtest Pipeline
==================================================

📥 Step 1: Downloading price data...
   Tickers: AAPL, MSFT, GOOGL
   ✅ Downloaded 315 rows

⚖️  Step 2: Setting up equal-weight strategy...
   AAPL: 32.3%
   MSFT: 32.3%
   GOOGL: 32.3%

🚀 Step 3: Running backtest...
   ✅ Results saved to results/

📊 Step 4: Performance Summary
========================================
   Initial Value:  $   99,805.90
   Final Value:    $  113,323.04
   Total Return:          13.54%
   CAGR:                  36.22%
   Max Drawdown:          -9.12%
   Sharpe Ratio:           1.50

📈 Step 5: Charts
                       Portfolio Equity Curve
        ┌──────────────────────────────────────────────────┐
 115000 ┤                                    ⡀⡠⠺⡀          │
 110000 ┤                               ⢀⠎⠈⠁ ⢱            │
 105000 ┤     ⢀⠔⠊⠉⠈⠒⠙⡄     ⢀⠎⠈⠁ ⢱         ⣠  │
 100000 ┤⢀⠤⠎                                           │
        └──────────────────────────────────────────────────┘
            Jan      Feb      Mar      Apr      May

==================================================
  ✅ Pipeline complete!
==================================================
```

---

## Why I Built This

Most quant tutorials teach you to `import backtrader` and call it a day. You learn the API, not the concepts.

I wanted to understand:
- How does a rebalancer convert target weights into actual trades?
- What happens when you sell before you buy vs. buy before you sell?
- How do transaction costs (slippage, commission) compound over time?

**So I built it from scratch.** 153 tests. Every expected value hand-calculated. Every edge case covered.

---

## Features

| Feature | Description |
|---------|-------------|
| **CLI Interface** | One command runs entire pipeline: `factorlab run AAPL MSFT` |
| **Terminal Charts** | ASCII charts display directly in terminal (no GUI needed) |
| **Transaction Costs** | Realistic slippage (0.1%) and commission modeling |
| **Multiple Strategies** | Static weights, momentum ranking, extensible interface |
| **Full Analytics** | Sharpe, Sortino, CAGR, max drawdown, volatility |
| **153 Tests** | Every calculation hand-verified with TDD |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  ingest_yf.py → Yahoo Finance API → Normalized Parquet          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FACTOR ENGINE                              │
│  factors.py → Returns, Momentum, RSI, SMA, Volatility           │
│  All calculations use .over("ticker") for multi-stock support   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STRATEGY LAYER                              │
│  strategy.py → StaticWeightStrategy | MomentumStrategy          │
│  Abstract interface: get_target_weights(date, prices, factors)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION ENGINE                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Portfolio  │───▶│  Rebalancer │───▶│  Backtester │          │
│  │ tracks cash │    │ weights →   │    │ time loop   │          │
│  │ & positions │    │ trades      │    │ + costs     │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYTICS & OUTPUT                           │
│  metrics.py → Sharpe, Sortino, CAGR, Drawdown, Volatility       │
│  charts.py → Equity curves, drawdown, returns distribution      │
└─────────────────────────────────────────────────────────────────┘
```

---

## CLI Commands

```bash
# Simple - one command does everything
factorlab run AAPL MSFT GOOGL

# With options
factorlab run AAPL MSFT --start 2024-01-01 --end 2024-06-01 --cash 50000

# Save PNG charts
factorlab run AAPL MSFT GOOGL --save-charts

# Advanced - individual commands
factorlab ingest AAPL MSFT --start 2024-01-01 --end 2024-12-31
factorlab backtest data/prices.parquet --strategy static --weights "aapl:0.5,msft:0.5"
factorlab metrics results/equity_curve.parquet
factorlab plot equity results/equity_curve.parquet --output chart.png
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

Why? Because vectorized backtests hide execution order. In production, you can't buy with money you haven't freed up yet.

### Realistic Transaction Costs

```python
# Slippage: worse price on execution
if trade.side == "buy":
    effective_price = price * (1 + slippage_pct)  # pay more
else:
    effective_price = price * (1 - slippage_pct)  # receive less

# Commission: percentage of trade value
commission = trade_value * commission_pct
portfolio.cash -= commission
```

### Sells Before Buys

```python
# Rebalancer returns trades in this order:
[Trade(AAPL, sell, 50), Trade(MSFT, buy, 100)]
#      ↑ free up cash    ↑ then deploy it
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
tests/test_analytics.py               16 passed
tests/test_visualization.py           21 passed
───────────────────────────────────────────────
TOTAL                                153 passed
```

Every test uses **hand-calculated expected values**:

```python
def test_sharpe_ratio_basic():
    """
    Returns: [1%, 2%, 1%, 2%, 1%] → mean=1.4%, std=0.55%
    Risk-free: 5% annual → 0.0137% daily
    Sharpe = (1.4 - 0.0137) / 0.55 * sqrt(252) ≈ 4.0
    """
    result = sharpe_ratio(returns, risk_free_rate=0.05)
    assert result == pytest.approx(4.0, rel=0.1)
```

---

## Project Structure

```
FactorLab/
├── src/factorlabs/
│   ├── data/
│   │   └── ingest_yf.py        # Yahoo Finance → normalized DataFrame
│   ├── financialfeatures/
│   │   └── factors.py          # Technical indicators (RSI, SMA, momentum)
│   ├── backtest/
│   │   ├── portfolio.py        # Position & cash tracking
│   │   ├── rebalancer.py       # Weights → trades conversion
│   │   ├── strategy.py         # Strategy interface + implementations
│   │   └── backtester.py       # Time-loop simulation with costs
│   ├── analytics/
│   │   └── metrics.py          # Sharpe, Sortino, CAGR, drawdown
│   ├── visualization/
│   │   └── charts.py           # Matplotlib charts
│   └── cli/
│       └── main.py             # Typer CLI application
├── tests/                      # 153 tests, hand-calculated values
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

---

## Analytics

| Metric | Description |
|--------|-------------|
| **Total Return** | `(final - initial) / initial` |
| **CAGR** | Compound annual growth rate |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Sharpe Ratio** | Risk-adjusted return vs. risk-free rate |
| **Sortino Ratio** | Like Sharpe, but only penalizes downside volatility |
| **Volatility** | Annualized standard deviation of returns |

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
```

---

## What I Learned

1. **TDD catches math bugs immediately.** My first returns calculation was `close/close.shift(1)` without the `- 1`. Test failed. Fixed in 30 seconds.

2. **Execution order matters.** Selling before buying isn't just good practice—it's required when cash is fully deployed.

3. **Transaction costs compound.** A "small" 0.1% slippage on monthly rebalancing can reduce returns by 1-2% annually.

4. **Polars > Pandas for this use case.** Native multi-column operations with `.over()`, better type safety, 10x faster.

5. **CLI UX matters.** Nobody wants to write 4 commands. `factorlab run AAPL MSFT` is the right abstraction.

---

## Tech Stack

- **Python 3.11** — Type hints, pattern matching
- **Polars** — Fast DataFrame operations
- **Typer** — CLI framework
- **plotext** — Terminal ASCII charts
- **Matplotlib** — PNG chart export
- **pytest** — Test framework (153 tests)
- **yfinance** — Market data API

---

## Installation

```bash
# Clone
git clone https://github.com/gabemeredith/FactorLab.git
cd FactorLab

# Install (editable mode)
pip install -e .

# Verify
factorlab --help
pytest tests/ -v
```

---

## License

MIT

---

*Built to understand quantitative finance from first principles. Not financial advice.*