"""
FactorLab Streamlit Dashboard

Run with: streamlit run app.py

Streamlit Basics:
- The script runs top-to-bottom on every user interaction
- st.sidebar.* puts widgets in the left sidebar
- st.* puts content in the main area
- @st.cache_data caches expensive computations (like API calls)
"""

import streamlit as st
import polars as pl
from datetime import date, timedelta

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="FactorLab",
    page_icon="📈",
    layout="wide"  # Use full screen width
)

# ============================================================
# IMPORTS - Your existing code
# ============================================================
from factorlabs.data.ingest_yf import fetch_yf_data, YFIngestConfig, normalize_prices
from factorlabs.backtest.backtester import BacktestConfig, Backtester
from factorlabs.backtest.strategy import StaticWeightStrategy, MomentumStrategy
from factorlabs.financialfeatures.factors import calculate_momentum, calculate_returns
from factorlabs.analytics.metrics import (
    total_return, cagr, max_drawdown, sharpe_ratio,
    sortino_ratio, annualized_volatility
)
from factorlabs.analytics.benchmark import compare_to_benchmark


# ============================================================
# CACHED FUNCTIONS
# ============================================================
# @st.cache_data tells Streamlit: "If I call this function with
# the same arguments, return the cached result instead of re-running"
#
# ttl=3600 means the cache expires after 1 hour (time-to-live in seconds)
# This ensures you eventually get fresh data, but don't hammer the API

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(tickers: tuple, start: date, end: date) -> pl.DataFrame:
    """
    Fetch and normalize price data. Cached for 1 hour.

    Note: tickers must be a tuple (not list) because cache keys
    must be hashable, and lists aren't hashable.
    """
    config = YFIngestConfig(
        tickers=list(tickers),
        start=start,
        end=end,
        interval="1d"
    )
    raw_data = fetch_yf_data(config)
    return normalize_prices(raw_data)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark_data(ticker: str, start: date, end: date) -> pl.DataFrame:
    """Fetch benchmark data. Cached separately from main tickers."""
    config = YFIngestConfig(
        tickers=[ticker],
        start=start,
        end=end,
        interval="1d"
    )
    raw_data = fetch_yf_data(config)
    return normalize_prices(raw_data)

# ============================================================
# SIDEBAR - User Inputs
# ============================================================
# Sidebar is good for inputs because it stays visible while
# the main area shows results

st.sidebar.title("FactorLab")
st.sidebar.markdown("*Backtest your portfolio strategies*")

# Multiselect with predefined options - searchable dropdown
# Users can type to filter, click to select multiple
# This is better UX than free text input

POPULAR_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC",
    "CRM", "ORCL", "ADBE", "NFLX", "PYPL", "SQ", "SHOP", "UBER", "SNAP",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "C",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "MRNA",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "DIS", "CMCSA",
    # Industrial
    "CAT", "BA", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "DE",
    # Energy (removed PXD - delisted/acquired)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "OXY", "HAL",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VUG", "VTV", "XLF",
]

selected_tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=POPULAR_TICKERS,
    default=["AAPL", "MSFT", "GOOGL"],
    help="Type to search, click to select. You can select multiple tickers."
)

# Custom ticker input - for tickers not in the predefined list
# This pattern: text input + combine with multiselect
custom_tickers_input = st.sidebar.text_input(
    "Add custom tickers (comma-separated)",
    placeholder="e.g., PLTR, RIVN, COIN",
    help="Enter any ticker not in the dropdown above"
)

# Combine selected + custom, remove duplicates, uppercase
custom_tickers = [t.strip().upper() for t in custom_tickers_input.split(",") if t.strip()]
tickers = list(dict.fromkeys(selected_tickers + custom_tickers))  # Preserves order, removes dupes

# Date inputs - Streamlit has built-in date picker
col1, col2 = st.sidebar.columns(2)  # Two columns side by side
with col1:
    start_date = st.date_input(
        "Start Date",
        value=date(2024, 1, 1)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=date(2024, 12, 31)
    )

# Selectbox for strategy - returns the selected string
strategy_type = st.sidebar.selectbox(
    "Strategy",
    options=["Static (Equal Weight)", "Momentum"],
    help="Static: fixed weights. Momentum: rotate into top performers."
)

# Conditional input - only show if momentum selected
# This pattern is common: show/hide based on other inputs
if strategy_type == "Momentum":
    # Handle edge case: slider needs min < max
    max_positions = max(2, min(10, len(tickers)))  # At least 2 to avoid slider error
    n_positions = st.sidebar.slider(
        "Number of Positions",
        min_value=1,
        max_value=max_positions,
        value=min(3, max_positions),
        help="How many top-momentum stocks to hold"
    )
else:
    n_positions = len(tickers) if tickers else 1

# Benchmark selection - None option included
benchmark = st.sidebar.selectbox(
    "Benchmark (optional)",
    options=[None, "SPY", "QQQ", "IWM", "DIA"],
    format_func=lambda x: "None" if x is None else x,
    help="Compare your portfolio against a market index"
)

# Number input for cash
initial_cash = st.sidebar.number_input(
    "Initial Cash ($)",
    min_value=1000,
    max_value=10_000_000,
    value=100_000,
    step=10_000
)

# Button - returns True when clicked
# This is how you trigger expensive operations
run_backtest = st.sidebar.button("Run Backtest", type="primary")

# ============================================================
# SIDEBAR FOOTER - Attribution
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("Made by **Gabe Meredith**")
st.sidebar.caption(
    "[Website](https://gabemeredith.github.io) · "
    "[GitHub](https://github.com/gabemeredith) · "
    "[LinkedIn](https://www.linkedin.com/in/gabriel-meredith)"
)


# ============================================================
# MAIN AREA - Results
# ============================================================

st.title("📈 FactorLab Dashboard")

# Show current configuration
st.markdown(f"**Tickers:** {', '.join(tickers)} | **Strategy:** {strategy_type}")

# The backtest only runs when button is clicked
# Without this check, it would run on every interaction
if run_backtest:
    # Validation
    if len(tickers) < 1:
        st.error("Please enter at least one ticker")
        st.stop()  # Halt execution

    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()

    # Progress indicator - good UX for slow operations
    with st.spinner("Downloading price data..."):
        # ============================================================
        # FETCH DATA - Now using cached functions!
        # ============================================================
        # tuple() is required because lists aren't hashable (can't be cache keys)
        try:
            prices = fetch_ticker_data(tuple(tickers), start_date, end_date)
        except Exception as e:
            st.error(f"Failed to download data: {e}")
            st.stop()

        # Download benchmark if selected (also cached)
        benchmark_prices = None
        if benchmark:
            benchmark_prices = fetch_benchmark_data(benchmark, start_date, end_date)

    # Validate that we got data for all tickers
    available_tickers = prices["ticker"].unique().to_list()
    missing_tickers = [t for t in tickers if t.lower() not in [at.lower() for at in available_tickers]]

    if missing_tickers:
        st.warning(f"Could not fetch data for: {', '.join(missing_tickers)}. They may be delisted or invalid.")
        # Filter to only tickers we have data for
        tickers = [t for t in tickers if t.lower() in [at.lower() for at in available_tickers]]
        if not tickers:
            st.error("No valid tickers remaining. Please select different tickers.")
            st.stop()

    st.success(f"Loaded {len(prices):,} rows for {len(available_tickers)} tickers")

    with st.spinner("Running backtest..."):
        # ============================================================
        # YOUR EXISTING CODE - Setup strategy
        # ============================================================
        factors = None

        if strategy_type == "Momentum":
            factors = calculate_returns(prices)
            factors = calculate_momentum(factors)
            bt_strategy = MomentumStrategy(n_positions=n_positions)
        else:
            # Equal weights with buffer for costs
            weight_per_ticker = 0.97 / len(tickers)
            weight_dict = {t.lower(): weight_per_ticker for t in tickers}
            bt_strategy = StaticWeightStrategy(weight_dict)

        # ============================================================
        # YOUR EXISTING CODE - Run backtest
        # ============================================================
        bt_config = BacktestConfig(
            start_date=prices["date"].min(),
            end_date=prices["date"].max(),
            initial_cash=initial_cash,
            rebalance_frequency="monthly",
            slippage_pct=0.001,
            commission_pct=0.001
        )

        backtester = Backtester()
        result = backtester.run(
            prices=prices,
            strategy=bt_strategy,
            config=bt_config,
            factors=factors
        )

        # Run benchmark backtest if selected
        benchmark_equity = None
        if benchmark and benchmark_prices is not None:
            benchmark_strategy = StaticWeightStrategy({benchmark.lower(): 0.97})
            benchmark_config = BacktestConfig(
                start_date=benchmark_prices["date"].min(),
                end_date=benchmark_prices["date"].max(),
                initial_cash=initial_cash,
                rebalance_frequency="never",
                slippage_pct=0.001,
                commission_pct=0.001
            )
            benchmark_result = backtester.run(
                prices=benchmark_prices,
                strategy=benchmark_strategy,
                config=benchmark_config
            )
            benchmark_equity = benchmark_result.equity_curve

    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    equity_curve = result.equity_curve
    daily_returns = equity_curve["portfolio_value"].pct_change().drop_nulls()

    # Calculate metrics
    total_ret = total_return(equity_curve)
    cagr_val = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)
    vol = annualized_volatility(daily_returns)
    sharpe_val = sharpe_ratio(daily_returns, risk_free_rate=0.05)
    sortino_val = sortino_ratio(daily_returns, risk_free_rate=0.05)

    # ============================================================
    # METRICS ROW - st.columns creates a horizontal layout
    # ============================================================
    st.subheader("Performance Metrics")

    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    # st.metric shows a value with optional delta (change indicator)
    col1.metric(
        "Total Return",
        f"{total_ret * 100:.2f}%",
        delta=None  # Could show vs benchmark here
    )
    col2.metric("CAGR", f"{cagr_val * 100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
    col4.metric("Max Drawdown", f"{max_dd * 100:.2f}%")

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility", f"{vol * 100:.2f}%")
    col2.metric("Sortino Ratio", f"{sortino_val:.2f}")
    col3.metric(
        "Initial Value",
        f"${equity_curve['portfolio_value'][0]:,.0f}"
    )
    col4.metric(
        "Final Value",
        f"${equity_curve['portfolio_value'][-1]:,.0f}"
    )

    # ============================================================
    # BENCHMARK COMPARISON (if selected)
    # ============================================================
    if benchmark and benchmark_equity is not None:
        st.subheader(f"vs {benchmark} Benchmark")

        benchmark_ret = total_return(benchmark_equity)
        benchmark_returns = benchmark_equity["portfolio_value"].pct_change().drop_nulls()
        comparison = compare_to_benchmark(daily_returns, benchmark_returns)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            f"{benchmark} Return",
            f"{benchmark_ret * 100:.2f}%"
        )
        col2.metric(
            "Excess Return",
            f"{(total_ret - benchmark_ret) * 100:.2f}%",
            delta=f"{(total_ret - benchmark_ret) * 100:.1f}%"
        )
        col3.metric("Alpha", f"{comparison['alpha'] * 100:.2f}%")
        col4.metric("Beta", f"{comparison['beta']:.2f}")

    # ============================================================
    # CHARTS - Using Streamlit's built-in chart (simple)
    # ============================================================
    st.subheader("Equity Curve")

    # Prepare data for chart
    # st.line_chart expects a DataFrame with date index
    chart_data = equity_curve.select([
        pl.col("date"),
        pl.col("portfolio_value").alias("Portfolio")
    ]).to_pandas().set_index("date")

    # Add benchmark to chart if available
    if benchmark and benchmark_equity is not None:
        # Scale benchmark to same starting value for visual comparison
        scale = initial_cash / benchmark_equity["portfolio_value"][0]
        benchmark_scaled = benchmark_equity.select([
            pl.col("date"),
            (pl.col("portfolio_value") * scale).alias(benchmark)
        ]).to_pandas().set_index("date")

        chart_data = chart_data.join(benchmark_scaled, how="left")

    # st.line_chart is the simplest way to show a line chart
    # For more control, use st.plotly_chart or st.altair_chart
    st.line_chart(chart_data)

    # ============================================================
    # TRADES TABLE - Expandable section
    # ============================================================
    with st.expander("View Trade History"):
        # st.dataframe shows an interactive table
        st.dataframe(
            result.trades.to_pandas(),
            width="stretch"  # Updated from deprecated use_container_width=True
        )

else:
    # Show instructions when backtest hasn't been run
    st.info("👈 Configure your backtest in the sidebar and click **Run Backtest**")

    # You can add example content or documentation here
    st.markdown("""
    ### How to use
    1. Enter stock tickers (comma-separated)
    2. Select date range
    3. Choose a strategy
    4. Optionally select a benchmark for comparison
    5. Click **Run Backtest**

    ### Strategies
    - **Static (Equal Weight)**: Divide your portfolio equally among all tickers, rebalance monthly
    - **Momentum**: Hold the top N stocks by recent performance, rebalance monthly
    """)

# ============================================================
# FOOTER - Always visible at bottom of main area
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; padding: 20px;">
        Made by <b>Gabe Meredith</b><br>
        <a href="https://gabemeredith.github.io" target="_blank">Website</a> ·
        <a href="https://github.com/gabemeredith" target="_blank">GitHub</a> ·
        <a href="https://www.linkedin.com/in/gabriel-meredith" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
