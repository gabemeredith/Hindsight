"""
Hindsight.py Streamlit Dashboard

Run with: streamlit run app.py
"""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import io

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Hindsight.py",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# IMPORTS - Backend modules
# ============================================================
from hindsightpy.data.ingest_yf import fetch_yf_data, YFIngestConfig, normalize_prices
from hindsightpy.backtest.backtester import BacktestConfig, Backtester
from hindsightpy.backtest.strategy import StaticWeightStrategy, MomentumStrategy
from hindsightpy.financialfeatures.factors import calculate_momentum, calculate_returns
from hindsightpy.analytics.metrics import (
    total_return, cagr, max_drawdown, sharpe_ratio,
    sortino_ratio, annualized_volatility
)
from hindsightpy.analytics.benchmark import compare_to_benchmark

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
# Session state persists data across Streamlit reruns
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "last_config" not in st.session_state:
    st.session_state.last_config = None

# ============================================================
# CUSTOM CSS FOR BETTER UX
# ============================================================
st.markdown("""
<style>
    /* Metric cards styling - equal size boxes */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #3d3d4d;
        border-radius: 8px;
        padding: 16px;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: hidden;
    }

    div[data-testid="stMetric"] > div {
        padding: 0;
    }

    div[data-testid="stMetric"] label {
        font-size: 14px;
        color: #a0a0a0;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }

    /* Make delta text smaller to fit in fixed height */
    div[data-testid="stMetricDelta"] {
        font-size: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px 4px 0 0;
    }

    /* Button improvements */
    .stButton > button[kind="primary"] {
        width: 100%;
        padding: 12px 24px;
        font-weight: 600;
    }

    /* Success/info boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHED FUNCTIONS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(tickers: tuple, start: date, end: date) -> pl.DataFrame:
    """Fetch and normalize price data. Cached for 1 hour."""
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
# HELPER FUNCTIONS
# ============================================================
def get_date_range_preset(preset: str) -> tuple[date, date]:
    """Calculate start and end dates based on preset."""
    today = date.today()
    end = today

    if preset == "YTD":
        start = date(today.year, 1, 1)
    elif preset == "1 Year":
        start = today - relativedelta(years=1)
    elif preset == "3 Years":
        start = today - relativedelta(years=3)
    elif preset == "5 Years":
        start = today - relativedelta(years=5)
    elif preset == "10 Years":
        start = today - relativedelta(years=10)
    elif preset == "Max":
        start = date(2000, 1, 1)
    else:  # Custom
        return None, None

    return start, end


def calculate_drawdown_series(equity_curve: pl.DataFrame) -> pl.DataFrame:
    """Calculate running drawdown from equity curve."""
    return equity_curve.with_columns([
        pl.col("portfolio_value").cum_max().alias("peak"),
    ]).with_columns([
        ((pl.col("portfolio_value") - pl.col("peak")) / pl.col("peak") * 100).alias("drawdown")
    ])


def create_equity_chart(equity_curve: pl.DataFrame, benchmark_equity: pl.DataFrame = None,
                        benchmark_name: str = None, initial_cash: float = 100000) -> go.Figure:
    """Create interactive Plotly equity curve chart."""
    fig = go.Figure()

    # Portfolio line
    fig.add_trace(go.Scatter(
        x=equity_curve["date"].to_list(),
        y=equity_curve["portfolio_value"].to_list(),
        mode='lines',
        name='Portfolio',
        line=dict(color='#ff4b4b', width=2),
        hovertemplate='%{x|%b %d, %Y}<br>Portfolio: $%{y:,.0f}<extra></extra>'
    ))

    # Benchmark line if available
    if benchmark_equity is not None and benchmark_name:
        scale = initial_cash / benchmark_equity["portfolio_value"][0]
        scaled_values = [v * scale for v in benchmark_equity["portfolio_value"].to_list()]

        fig.add_trace(go.Scatter(
            x=benchmark_equity["date"].to_list(),
            y=scaled_values,
            mode='lines',
            name=benchmark_name,
            line=dict(color='#636efa', width=2, dash='dot'),
            hovertemplate='%{x|%b %d, %Y}<br>' + benchmark_name + ': $%{y:,.0f}<extra></extra>'
        ))

    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#a0a0a0')
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickformat='$,.0f',
            tickfont=dict(color='#a0a0a0')
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='#a0a0a0')
        )
    )

    return fig


def create_drawdown_chart(equity_curve: pl.DataFrame) -> go.Figure:
    """Create interactive drawdown chart."""
    dd_data = calculate_drawdown_series(equity_curve)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd_data["date"].to_list(),
        y=dd_data["drawdown"].to_list(),
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#ef4444', width=1.5),
        fillcolor='rgba(239, 68, 68, 0.3)',
        hovertemplate='%{x|%b %d, %Y}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        hovermode='x unified',
        margin=dict(l=50, r=20, t=10, b=40),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            ticksuffix='%',
            tickfont=dict(color='#a0a0a0'),
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='#a0a0a0')
        ),
        showlegend=False
    )

    return fig


def create_returns_distribution_chart(daily_returns: pl.Series) -> go.Figure:
    """Create returns distribution histogram."""
    returns_list = (daily_returns * 100).to_list()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns_list,
        nbinsx=40,
        name='Daily Returns',
        marker_color='#ff4b4b',
        opacity=0.8,
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title='',
        xaxis_title='',
        yaxis_title='',
        margin=dict(l=50, r=20, t=10, b=40),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            tickfont=dict(color='#a0a0a0')
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            ticksuffix='%',
            tickfont=dict(color='#a0a0a0')
        ),
        showlegend=False,
        bargap=0.1
    )

    return fig


def format_metric_value(value: float, metric_type: str) -> str:
    """Format metric values consistently."""
    if metric_type == "percent":
        return f"{value * 100:.2f}%"
    elif metric_type == "ratio":
        return f"{value:.2f}"
    elif metric_type == "currency":
        return f"${value:,.0f}"
    return str(value)


def export_results_to_csv(result, metrics: dict) -> str:
    """Generate CSV string of backtest results."""
    output = io.StringIO()

    # Write metrics
    output.write("PERFORMANCE METRICS\n")
    for name, value in metrics.items():
        output.write(f"{name},{value}\n")

    output.write("\n\nEQUITY CURVE\n")
    equity_df = result.equity_curve.to_pandas()
    equity_df.to_csv(output, index=False)

    output.write("\n\nTRADE HISTORY\n")
    trades_df = result.trades.to_pandas()
    trades_df.to_csv(output, index=False)

    return output.getvalue()


# ============================================================
# SIDEBAR - User Inputs
# ============================================================
with st.sidebar:
    st.title("Hindsight.py")
    st.caption("Backtest your portfolio strategies")

    st.markdown("---")

    # ---- Ticker Selection ----
    st.subheader("Select Assets")

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
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "OXY", "HAL",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VUG", "VTV", "XLF",
    ]

    selected_tickers = st.multiselect(
        "Tickers",
        options=POPULAR_TICKERS,
        default=["AAPL", "MSFT", "GOOGL"],
        help="Type to search, click to select multiple tickers"
    )

    custom_tickers_input = st.text_input(
        "Add custom tickers",
        placeholder="PLTR, RIVN, COIN",
        help="Comma-separated list of additional tickers"
    )

    custom_tickers = [t.strip().upper() for t in custom_tickers_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(selected_tickers + custom_tickers))

    if tickers:
        st.caption(f"Selected: {len(tickers)} ticker(s)")

    st.markdown("---")

    # ---- Date Range Selection ----
    st.subheader("Time Period")

    # Quick preset buttons
    date_preset = st.radio(
        "Date Range",
        options=["Custom", "YTD", "1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        horizontal=True,
        label_visibility="collapsed"
    )

    today = date.today()

    if date_preset != "Custom":
        preset_start, preset_end = get_date_range_preset(date_preset)
        start_date = preset_start
        end_date = preset_end
        st.caption(f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start",
                value=date(2024, 1, 1),
                min_value=date(2000, 1, 1),
                max_value=today
            )
        with col2:
            end_date = st.date_input(
                "End",
                value=min(date(2024, 12, 31), today),
                min_value=date(2000, 1, 1),
                max_value=today
            )

    st.markdown("---")

    # ---- Strategy Selection ----
    st.subheader("Strategy")

    strategy_type = st.selectbox(
        "Type",
        options=["Static (Equal Weight)", "Momentum"],
        help="Static: Fixed equal weights, rebalanced monthly. Momentum: Rotate into top performers.",
        label_visibility="collapsed"
    )

    if strategy_type == "Momentum":
        max_positions = max(2, min(10, len(tickers)))
        n_positions = st.slider(
            "Number of Positions",
            min_value=1,
            max_value=max_positions,
            value=min(3, max_positions),
            help="How many top-momentum stocks to hold at any time"
        )
    else:
        n_positions = len(tickers) if tickers else 1

    st.markdown("---")

    # ---- Benchmark & Cash ----
    st.subheader("Settings")

    benchmark = st.selectbox(
        "Benchmark",
        options=[None, "SPY", "QQQ", "IWM", "DIA"],
        format_func=lambda x: "None" if x is None else x,
        help="Compare portfolio performance against a market index"
    )

    initial_cash = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Starting capital for the backtest"
    )

    st.markdown("---")

    # ---- Run Button ----
    run_backtest = st.button(
        "Run Backtest",
        type="primary",
        use_container_width=True
    )

    # Clear results button (only show if we have results)
    if st.session_state.backtest_results is not None:
        if st.button("Clear Results", use_container_width=True):
            st.session_state.backtest_results = None
            st.session_state.last_config = None
            st.rerun()

    st.markdown("---")

    # ---- Footer ----
    st.caption("Made by **Gabe Meredith**")
    st.caption(
        "[Website](https://gabemeredith.github.io) · "
        "[GitHub](https://github.com/gabemeredith) · "
        "[LinkedIn](https://www.linkedin.com/in/gabriel-meredith)"
    )


# ============================================================
# MAIN AREA
# ============================================================

# Header
st.title("Hindsight.py")
st.caption("A from-scratch portfolio backtesting engine")

# ============================================================
# RUN BACKTEST
# ============================================================
if run_backtest:
    # Validation
    if len(tickers) < 1:
        st.error("Please select at least one ticker to continue.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    # Create a progress container
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()

        try:
            # Step 1: Download data
            progress_bar.progress(10, text="Downloading price data...")
            status_text.caption(f"Fetching data for {len(tickers)} ticker(s)...")

            prices = fetch_ticker_data(tuple(tickers), start_date, end_date)

            # Validate data
            available_tickers = prices["ticker"].unique().to_list()
            missing_tickers = [t for t in tickers if t.lower() not in [at.lower() for at in available_tickers]]

            if missing_tickers:
                st.warning(f"Could not fetch data for: {', '.join(missing_tickers)}. Continuing with available tickers.")
                tickers = [t for t in tickers if t.lower() in [at.lower() for at in available_tickers]]
                if not tickers:
                    st.error("No valid tickers remaining.")
                    st.stop()

            progress_bar.progress(30, text="Downloading benchmark data...")

            # Step 2: Download benchmark
            benchmark_prices = None
            if benchmark:
                status_text.caption(f"Fetching {benchmark} benchmark data...")
                benchmark_prices = fetch_benchmark_data(benchmark, start_date, end_date)

            progress_bar.progress(50, text="Setting up strategy...")

            # Step 3: Setup strategy
            factors = None
            if strategy_type == "Momentum":
                factors = calculate_returns(prices)
                factors = calculate_momentum(factors)
                bt_strategy = MomentumStrategy(n_positions=n_positions)
            else:
                weight_per_ticker = 0.97 / len(tickers)
                weight_dict = {t.lower(): weight_per_ticker for t in tickers}
                bt_strategy = StaticWeightStrategy(weight_dict)

            progress_bar.progress(70, text="Running backtest...")
            status_text.caption("Simulating trades...")

            # Step 4: Run backtest
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

            # Step 5: Run benchmark backtest
            benchmark_equity = None
            if benchmark and benchmark_prices is not None:
                progress_bar.progress(85, text="Running benchmark comparison...")
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

            progress_bar.progress(100, text="Complete!")

            # Store results in session state
            st.session_state.backtest_results = {
                "result": result,
                "benchmark_equity": benchmark_equity,
                "benchmark_name": benchmark,
                "initial_cash": initial_cash,
                "tickers": tickers,
                "strategy": strategy_type,
                "start_date": start_date,
                "end_date": end_date
            }
            st.session_state.last_config = {
                "tickers": tickers,
                "strategy": strategy_type,
                "benchmark": benchmark,
                "initial_cash": initial_cash
            }

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            st.success(f"Backtest complete! Analyzed {len(result.trades):,} trades across {len(tickers)} tickers.")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Backtest failed: {str(e)}")
            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================
if st.session_state.backtest_results is not None:
    results = st.session_state.backtest_results
    result = results["result"]
    benchmark_equity = results["benchmark_equity"]
    benchmark_name = results["benchmark_name"]
    initial_cash = results["initial_cash"]

    equity_curve = result.equity_curve
    daily_returns = equity_curve["portfolio_value"].pct_change().drop_nulls()

    # Calculate all metrics
    total_ret = total_return(equity_curve)
    cagr_val = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)
    vol = annualized_volatility(daily_returns)
    sharpe_val = sharpe_ratio(daily_returns, risk_free_rate=0.05)
    sortino_val = sortino_ratio(daily_returns, risk_free_rate=0.05)

    # Benchmark metrics
    benchmark_metrics = None
    if benchmark_equity is not None and benchmark_name:
        benchmark_ret = total_return(benchmark_equity)
        benchmark_returns = benchmark_equity["portfolio_value"].pct_change().drop_nulls()
        comparison = compare_to_benchmark(daily_returns, benchmark_returns)
        benchmark_metrics = {
            "return": benchmark_ret,
            "excess_return": total_ret - benchmark_ret,
            "alpha": comparison["alpha"],
            "beta": comparison["beta"]
        }

    # Configuration summary and back button
    col_config, col_back = st.columns([4, 1])
    with col_config:
        st.markdown(f"""
        **Configuration:** {', '.join(results['tickers'])} | {results['strategy']} |
        {results['start_date'].strftime('%b %Y')} - {results['end_date'].strftime('%b %Y')}
        """)
    with col_back:
        if st.button("Back to Guide", use_container_width=True):
            st.session_state.backtest_results = None
            st.session_state.last_config = None
            st.rerun()

    # ============================================================
    # TABS FOR ORGANIZATION
    # ============================================================
    tab_overview, tab_charts, tab_trades = st.tabs(["Overview", "Charts", "Trade History"])

    # ---- OVERVIEW TAB ----
    with tab_overview:
        st.subheader("Performance Summary")

        # Primary metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_color = "normal" if total_ret >= 0 else "inverse"
            st.metric(
                "Total Return",
                f"{total_ret * 100:.2f}%",
                delta=f"{total_ret * 100:.1f}%" if total_ret != 0 else None,
                delta_color=delta_color
            )

        with col2:
            st.metric("CAGR", f"{cagr_val * 100:.2f}%")

        with col3:
            sharpe_color = "normal" if sharpe_val >= 1 else "off"
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_val:.2f}",
                help="Risk-adjusted return. Above 1.0 is good, above 2.0 is excellent."
            )

        with col4:
            st.metric(
                "Max Drawdown",
                f"{max_dd * 100:.2f}%",
                help="Largest peak-to-trough decline"
            )

        # Secondary metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Volatility", f"{vol * 100:.2f}%", help="Annualized standard deviation of returns")

        with col2:
            st.metric("Sortino Ratio", f"{sortino_val:.2f}", help="Like Sharpe but only penalizes downside volatility")

        with col3:
            st.metric("Starting Value", f"${equity_curve['portfolio_value'][0]:,.0f}")

        with col4:
            final_val = equity_curve['portfolio_value'][-1]
            profit = final_val - initial_cash
            st.metric(
                "Final Value",
                f"${final_val:,.0f}",
                delta=f"${profit:+,.0f}"
            )

        # Benchmark comparison section
        if benchmark_metrics:
            st.markdown("---")
            st.subheader(f"vs {benchmark_name} Benchmark")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(f"{benchmark_name} Return", f"{benchmark_metrics['return'] * 100:.2f}%")

            with col2:
                excess = benchmark_metrics['excess_return']
                st.metric(
                    "Excess Return",
                    f"{excess * 100:.2f}%",
                    delta=f"{excess * 100:.1f}%",
                    delta_color="normal" if excess >= 0 else "inverse"
                )

            with col3:
                st.metric(
                    "Alpha",
                    f"{benchmark_metrics['alpha'] * 100:.2f}%",
                    help="Excess return not explained by market movements"
                )

            with col4:
                st.metric(
                    "Beta",
                    f"{benchmark_metrics['beta']:.2f}",
                    help="Sensitivity to market movements. 1.0 = moves with market"
                )

        # Export section
        st.markdown("---")

        metrics_dict = {
            "Total Return": f"{total_ret * 100:.2f}%",
            "CAGR": f"{cagr_val * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe_val:.2f}",
            "Sortino Ratio": f"{sortino_val:.2f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Volatility": f"{vol * 100:.2f}%",
            "Initial Value": f"${initial_cash:,.0f}",
            "Final Value": f"${equity_curve['portfolio_value'][-1]:,.0f}"
        }

        csv_data = export_results_to_csv(result, metrics_dict)

        st.download_button(
            label="Download Results (CSV)",
            data=csv_data,
            file_name=f"backtest_results_{date.today().isoformat()}.csv",
            mime="text/csv",
            help="Download all metrics, equity curve, and trade history"
        )

    # ---- CHARTS TAB ----
    with tab_charts:
        # Equity Curve
        st.markdown("**Equity Curve**")
        equity_chart = create_equity_chart(
            equity_curve,
            benchmark_equity,
            benchmark_name,
            initial_cash
        )
        st.plotly_chart(equity_chart, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # Two-column layout for smaller charts
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**Drawdown**")
            drawdown_chart = create_drawdown_chart(equity_curve)
            st.plotly_chart(drawdown_chart, use_container_width=True, config={'displayModeBar': False})

        with col2:
            st.markdown("**Returns Distribution**")
            returns_chart = create_returns_distribution_chart(daily_returns)
            st.plotly_chart(returns_chart, use_container_width=True, config={'displayModeBar': False})

    # ---- TRADES TAB ----
    with tab_trades:
        st.subheader("Trade History")

        trades_df = result.trades.to_pandas()

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", f"{len(trades_df):,}")
        with col2:
            if "side" in trades_df.columns:
                buys = len(trades_df[trades_df["side"] == "buy"])
                st.metric("Buy Orders", f"{buys:,}")
        with col3:
            if "side" in trades_df.columns:
                sells = len(trades_df[trades_df["side"] == "sell"])
                st.metric("Sell Orders", f"{sells:,}")

        st.markdown("---")

        # Full trade table
        st.dataframe(
            trades_df,
            use_container_width=True,
            height=400
        )

        # Download trades
        trades_csv = trades_df.to_csv(index=False)
        st.download_button(
            label="Download Trade History (CSV)",
            data=trades_csv,
            file_name=f"trade_history_{date.today().isoformat()}.csv",
            mime="text/csv"
        )

else:
    # ============================================================
    # WELCOME STATE - No results yet
    # ============================================================

    # Info box
    st.info("Configure your backtest in the sidebar and click **Run Backtest** to get started.")

    # Quick start guide
    with st.expander("Quick Start Guide", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Getting Started

            1. **Select Tickers** - Choose stocks from the dropdown or add custom tickers
            2. **Set Time Period** - Use presets (YTD, 1Y, etc.) or custom dates
            3. **Choose Strategy** - Equal weight or momentum-based
            4. **Add Benchmark** - Compare against SPY, QQQ, etc.
            5. **Run Backtest** - Click the button and wait for results
            """)

        with col2:
            st.markdown("""
            ### Strategies Explained

            **Static (Equal Weight)**
            - Divides capital equally among all selected tickers
            - Rebalances monthly to maintain equal weights

            **Momentum**
            - Ranks stocks by recent performance
            - Holds only the top N performers
            - Rotates holdings monthly based on momentum
            """)

    # Educational content
    with st.expander("What is Backtesting?"):
        st.markdown("""
        **Backtesting** is the process of testing a trading strategy on historical data
        to see how it *would have* performed in the past.

        #### Why is it important?

        Before risking real money, investors want to answer questions like:
        - Would this strategy have made money over the last 5 years?
        - How much could I have lost during a market crash?
        - Does this strategy beat simply buying an index fund?

        #### Key Metrics Explained

        | Metric | What it measures |
        |--------|------------------|
        | **Total Return** | How much your portfolio grew (or shrank) over the entire period |
        | **CAGR** | Compound Annual Growth Rate - your average yearly return, accounting for compounding |
        | **Sharpe Ratio** | Risk-adjusted return. Above 1.0 is decent, above 2.0 is strong |
        | **Sortino Ratio** | Like Sharpe, but only penalizes *downside* volatility |
        | **Max Drawdown** | The worst peak-to-trough decline during the period |
        | **Volatility** | How much your returns bounce around (standard deviation) |

        #### Caveats

        - **Past performance doesn't guarantee future results**
        - **Overfitting** - strategies can be accidentally optimized for historical data
        - **Transaction costs** - this tool models slippage and commissions, but real costs vary
        - **Survivorship bias** - we test on stocks that still exist today
        """)

    with st.expander("About This Project"):
        st.markdown("""
        #### Built From Scratch

        **Hindsight.py** is a from-scratch backtesting engine. No black-box libraries.
        Every calculation is explicit and auditable.

        **Key design decisions:**
        - **Test-Driven Development**: 150+ tests with hand-calculated expected values
        - **Explicit time loop**: Processes one day at a time, just like real trading
        - **Sells before buys**: Frees up cash before deploying it (many tutorials get this wrong)

        #### Tech Stack

        - **Python 3.11** - Type hints throughout
        - **Polars** - Fast DataFrame operations
        - **Streamlit** - This interactive dashboard
        - **Plotly** - Interactive charts
        - **pytest** - Comprehensive test suite

        ---

        *Built by Gabe Meredith as a learning project. Not financial advice.*
        """)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #a0a0a0; padding: 20px; font-size: 14px;">
        Built with Streamlit by <b>Gabe Meredith</b><br>
        <a href="https://gabemeredith.github.io" target="_blank" style="color: #ff4b4b;">Website</a> ·
        <a href="https://github.com/gabemeredith" target="_blank" style="color: #ff4b4b;">GitHub</a> ·
        <a href="https://www.linkedin.com/in/gabriel-meredith" target="_blank" style="color: #ff4b4b;">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True
)
