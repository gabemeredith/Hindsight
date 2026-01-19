"""
FactorLab CLI - Main entry point.

Usage:
    # Simple - one command does everything:
    factorlab run AAPL MSFT GOOGL

    # With options:
    factorlab run AAPL MSFT --start 2024-01-01 --end 2024-06-01 --cash 50000

    # Advanced - individual commands:
    factorlab ingest AAPL MSFT --start 2024-01-01 --end 2024-12-31
    factorlab backtest data/prices.parquet --strategy static --weights "aapl:0.5,msft:0.5"
    factorlab metrics results/equity_curve.parquet
    factorlab plot equity results/equity_curve.parquet --output chart.png
"""
import matplotlib.pyplot as plt
import plotext as pltxt
import polars as pl
import typer
from typing import Optional
from typing_extensions import Annotated
from datetime import date
from pathlib import Path
from factorlabs.data.ingest_yf import fetch_yf_data, YFIngestConfig, normalize_prices
from factorlabs.backtest.backtester import BacktestConfig, Backtester
from factorlabs.backtest.strategy import StaticWeightStrategy, MomentumStrategy
from factorlabs.financialfeatures.factors import calculate_momentum, calculate_returns
from factorlabs.visualization.charts import plot_equity_curve,plot_drawdown,plot_returns_distribution,plot_weights_over_time
from factorlabs.analytics.metrics import total_return, sortino_ratio, annualized_volatility, cagr, max_drawdown, sharpe_ratio, calculate_drawdown_series
from factorlabs.analytics.benchmark import compare_to_benchmark
# Create the main app
app = typer.Typer(
    name="factorlab",
    help="FactorLab - A quantitative backtesting engine for equities.",
    add_completion=False,
)


@app.command()
def ingest(
    tickers: Annotated[list[str], typer.Argument(help="Stock tickers to download (e.g., AAPL MSFT GOOGL)")],
    start: Annotated[str, typer.Option("--start", "-s", help="Start date (YYYY-MM-DD)")] = "2024-01-01",
    end: Annotated[str, typer.Option("--end", "-e", help="End date (YYYY-MM-DD)")] = "2024-12-31",
    output: Annotated[str, typer.Option("--output", "-o", help="Output parquet file path")] = "data/prices.parquet",
):
    """
    Download price data from Yahoo Finance.

    Example:
        factorlab ingest AAPL MSFT GOOGL --start 2024-01-01 --end 2024-12-31
    """
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    
    config = YFIngestConfig(
        tickers=tickers,
        start=start_date,end=end_date,interval="1d"
    )
    
    typer.echo(f"Date range: {start} to {end}")
    typer.echo(f"Ingesting data for: {', '.join(tickers)}")
    raw_date = fetch_yf_data(config)
    prices = normalize_prices(raw_date)
    
    Path(output).parent.mkdir(parents=True, exist_ok=True) 
    prices.write_parquet(output)
                                                                    
    typer.echo(f" Saved {len(prices):,} rows to {output}")

@app.command()
def backtest(
    prices_file: Annotated[Path, typer.Argument(help="Path to prices parquet file")],
    strategy: Annotated[str, typer.Option("--strategy", "-st", help="Strategy type: static, momentum")] = "static",
    cash: Annotated[float, typer.Option("--cash", "-c", help="Initial cash amount")] = 100000.0,
    rebalance: Annotated[str, typer.Option("--rebalance", "-r", help="Rebalance frequency: never, daily, weekly, monthly")] = "monthly",
    weights: Annotated[Optional[str], typer.Option("--weights", "-w", help="Static weights as 'AAPL:0.4,MSFT:0.3,GOOGL:0.3'")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output directory for results")] = "results/",
    slippage: Annotated[float, typer.Option("--slippage", help="Slippage percent (0.001 = 0.1%)")] = 0.001,                    
    commission: Annotated[float, typer.Option("--commission", help="Commission percent")] = 0.001 
):
    """
    Run a backtest simulation.

    Example:
        factorlab backtest data/prices.parquet --strategy static --weights "AAPL:0.4,MSFT:0.3,GOOGL:0.3"
        factorlab backtest data/prices.parquet --strategy momentum --rebalance monthly
    """

    typer.echo(f"Loading prices from: {prices_file}")
    prices = pl.read_parquet(prices_file)

    result = {}
    if weights:
        items = weights.split(',')
        for item in items:
            key_value = item.split(":")
            if len(key_value) == 2:
                result[key_value[0].lower()] = float(key_value[1])
        

    config = BacktestConfig(start_date=prices["date"].min(),end_date=prices["date"].max(),
                            initial_cash=cash,rebalance_frequency=rebalance, slippage_pct=slippage,
                            commission_pct=commission
                            )
    backtester = Backtester()

    if strategy == "momentum":
        bt_strat = MomentumStrategy()
    elif strategy == "static":
        bt_strat = StaticWeightStrategy(result)
    bt_result = backtester.run(prices=prices,strategy=bt_strat,config=config)
    # 6. Save results to output directory
    output_path = Path(output)                                                                                                 
    output_path.mkdir(parents=True, exist_ok=True)                                                                             
    bt_result.equity_curve.write_parquet(output_path / "equity_curve.parquet")                                                 
    bt_result.trades.write_parquet(output_path / "trades.parquet")     
    
    typer.echo(f"Strategy: {strategy}")
    typer.echo(f"Initial cash: ${cash:,.0f}")
    typer.echo(f"Rebalance: {rebalance}")




@app.command()
def plot(
    chart_type: Annotated[str, typer.Argument(help="Chart type: equity, drawdown, returns, weights")],
    data_file: Annotated[Path, typer.Argument(help="Path to data file (parquet)")],
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Output file path (PNG). If not specified, displays interactively.")] = None,
    title: Annotated[Optional[str], typer.Option("--title", "-t", help="Custom chart title")] = None,
):
    """
    Generate visualizations from backtest results.

    Example:
        factorlab plot equity results/equity_curve.parquet --output charts/equity.png
        factorlab plot drawdown results/equity_curve.parquet
    """
    typer.echo(f"Generating {chart_type} chart from: {data_file}")
    # 1. Load data from parquet
    data = pl.read_parquet(data_file)
    # 2. Call appropriate plot function based on chart_type
    if chart_type == "equity":
        if title:
            fig = plot_equity_curve(title=title,equity_curve=data)
        else:
            fig = plot_equity_curve(equity_curve=data)

    elif chart_type == "drawdown":
        if title:
            fig = plot_drawdown(equity_curve=data,title=title)
        else:
            fig = plot_drawdown(equity_curve=data)
    elif chart_type == "weights":
        if title:
            fig = plot_weights_over_time(weights_df=data,title=title)
        else:
            fig = plot_weights_over_time(weights_df=data)
    elif chart_type == "returns":
        if title:
            fig = plot_returns_distribution(returns_or_equity=data,title=title)        
        else:
            fig = plot_returns_distribution(returns_or_equity=data)        
    # 3. If output specified, save with fig.savefig()
    if output:
        fig.savefig(fname=output)
    if output is None:
        plt.show()



@app.command()
def metrics(
    equity_file: Annotated[Path, typer.Argument(help="Path to equity curve parquet file")],
    risk_free: Annotated[float, typer.Option("--rf", help="Risk-free rate (annual, e.g., 0.05 for 5%)")] = 0.05,
):
    """
    Calculate and display performance metrics.

    Example:
        factorlab metrics results/equity_curve.parquet --rf 0.05
    """
    equity_curve = pl.read_parquet(equity_file)                                                                            
                                                                                                                            
    # Calculate daily returns for risk metrics                                                                             
    daily_returns = equity_curve["portfolio_value"].pct_change().drop_nulls()                                              
                                                                                                                            
    # Calculate metrics (each returns a float)                                                                             
    total_ret = total_return(equity_curve)                                                                                 
    cagr_val = cagr(equity_curve)                                                                                          
    max_dd = max_drawdown(equity_curve)                                                                                    
    vol = annualized_volatility(daily_returns)                                                                             
    sharpe = sharpe_ratio(daily_returns, risk_free_rate=risk_free)                                                         
    sortino = sortino_ratio(daily_returns, risk_free_rate=risk_free)                                                       
                                                                                                                            
    # Print formatted results
    typer.echo("\n Performance Metrics")
    typer.echo("=" * 30)
    typer.echo(f"Total Return:     {total_ret * 100:>8.2f}%")
    typer.echo(f"CAGR:             {cagr_val * 100:>8.2f}%")
    typer.echo(f"Max Drawdown:     {max_dd * 100:>8.2f}%")
    typer.echo(f"Volatility:       {vol * 100:>8.2f}%")
    typer.echo(f"Sharpe Ratio:     {sharpe:>8.2f}")
    typer.echo(f"Sortino Ratio:    {sortino:>8.2f}")


@app.command()
def run(
    tickers: Annotated[list[str], typer.Argument(help="Stock tickers (e.g., AAPL MSFT GOOGL)")],
    start: Annotated[str, typer.Option("--start", "-s", help="Start date (YYYY-MM-DD)")] = "2024-01-01",
    end: Annotated[str, typer.Option("--end", "-e", help="End date (YYYY-MM-DD)")] = "2024-12-31",
    cash: Annotated[float, typer.Option("--cash", "-c", help="Initial cash amount")] = 100000.0,
    rebalance: Annotated[str, typer.Option("--rebalance", "-r", help="Rebalance frequency: never, daily, weekly, monthly")] = "monthly",
    strategy: Annotated[str, typer.Option("--strategy", "-st", help="Strategy type: static, momentum")] = "static",
    weights: Annotated[Optional[str], typer.Option("--weights", "-w", help="Custom weights as 'AAPL:0.4,MSFT:0.3,GOOGL:0.3' (static strategy only)")] = None,
    n_positions: Annotated[int, typer.Option("--n-positions", "-n", help="Number of top stocks to hold (momentum strategy only)")] = 3,
    benchmark: Annotated[Optional[str], typer.Option("--benchmark", "-b", help="Benchmark ticker for comparison (e.g., SPY, QQQ)")] = None,
    save_charts: Annotated[bool, typer.Option("--save-charts", help="Save charts as PNG files to charts/")] = False,
    no_charts: Annotated[bool, typer.Option("--no-charts", help="Don't display charts")] = False,
):
    """
    Run a complete backtest pipeline in one command.

    Downloads data, runs backtest, shows metrics and charts.

    Example:
        factorlab run AAPL MSFT GOOGL
        factorlab run AAPL MSFT --strategy momentum --n-positions 2
        factorlab run AAPL MSFT --weights "AAPL:0.6,MSFT:0.4"
        factorlab run AAPL MSFT GOOGL --benchmark SPY
        factorlab run AAPL MSFT --start 2024-01-01 --end 2024-06-01 --cash 50000
    """
    typer.echo("\n" + "=" * 50)
    typer.echo("  FactorLab - Running Complete Backtest Pipeline")
    typer.echo("=" * 50)

    # Step 1: Ingest data
    typer.echo(f"\n Downloading price data...")
    typer.echo(f"   Tickers: {', '.join(tickers)}")
    if benchmark:
        typer.echo(f"   Benchmark: {benchmark.upper()}")
    typer.echo(f"   Period: {start} to {end}")

    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)

    config = YFIngestConfig(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d"
    )

    raw_data = fetch_yf_data(config)
    prices = normalize_prices(raw_data)
    typer.echo(f"    Downloaded {len(prices):,} rows")

    # Download benchmark data if specified
    benchmark_prices = None
    benchmark_equity = None
    if benchmark:
        benchmark_config = YFIngestConfig(
            tickers=[benchmark],
            start=start_date,
            end=end_date,
            interval="1d"
        )
        benchmark_raw = fetch_yf_data(benchmark_config)
        benchmark_prices = normalize_prices(benchmark_raw)
        typer.echo(f"    Downloaded {benchmark.upper()} benchmark data")
    # Step 2: Set up strategy
    typer.echo(f"\n  Step 2: Setting up {strategy} strategy...")
    factors = None

    if strategy == "momentum":
        # Compute momentum factors for ranking
        factors = calculate_returns(prices)
        factors = calculate_momentum(factors)
        bt_strategy = MomentumStrategy(n_positions=n_positions)
        typer.echo(f"   Type: Momentum (top {n_positions} stocks by 10-day momentum)")
        typer.echo(f"   Stocks will be equal-weighted each rebalance")
    else:
        # Static strategy: use custom weights or equal weights
        if weights:
            # Parse custom weights string: "AAPL:0.4,MSFT:0.3,GOOGL:0.3"
            parsed_weights = {}
            for item in weights.split(','):
                parts = item.strip().split(":")
                if len(parts) == 2:
                    parsed_weights[parts[0].strip().lower()] = float(parts[1].strip())
            weight_dict = parsed_weights
            typer.echo(f"   Type: Static (custom weights)")
        else:
            # Equal weights with 3% buffer for costs
            n_tickers = len(tickers)
            weight_per_ticker = 0.97 / n_tickers
            weight_dict = {t.lower(): weight_per_ticker for t in tickers}
            typer.echo(f"   Type: Static (equal weights)")

        for t, w in weight_dict.items():
            typer.echo(f"   {t.upper()}: {w * 100:.1f}%")

        bt_strategy = StaticWeightStrategy(weight_dict)

    # Step 3: Run backtest
    typer.echo(f"\n🚀 Step 3: Running backtest...")
    typer.echo(f"   Initial cash: ${cash:,.0f}")
    typer.echo(f"   Rebalance: {rebalance}")

    bt_config = BacktestConfig(
        start_date=prices["date"].min(),
        end_date=prices["date"].max(),
        initial_cash=cash,
        rebalance_frequency=rebalance,
        slippage_pct=0.001,
        commission_pct=0.001
    )

    backtester = Backtester()
    result = backtester.run(prices=prices, strategy=bt_strategy, config=bt_config, factors=factors)

    # Save results
    output_path = Path("results")
    output_path.mkdir(parents=True, exist_ok=True)
    result.equity_curve.write_parquet(output_path / "equity_curve.parquet")
    result.trades.write_parquet(output_path / "trades.parquet")
    typer.echo(f"    Results saved to results/")

    # Step 4: Calculate and display metrics
    typer.echo(f"\n Step 4: Performance Summary")
    typer.echo("=" * 40)

    equity_curve = result.equity_curve
    daily_returns = equity_curve["portfolio_value"].pct_change().drop_nulls()

    total_ret = total_return(equity_curve)
    cagr_val = cagr(equity_curve)
    max_dd = max_drawdown(equity_curve)
    vol = annualized_volatility(daily_returns)
    sharpe_val = sharpe_ratio(daily_returns, risk_free_rate=0.05)
    sortino_val = sortino_ratio(daily_returns, risk_free_rate=0.05)

    initial_value = equity_curve["portfolio_value"][0]
    final_value = equity_curve["portfolio_value"][-1]

    typer.echo(f"   Initial Value:  ${initial_value:>12,.2f}")
    typer.echo(f"   Final Value:    ${final_value:>12,.2f}")
    typer.echo(f"   Total Return:   {total_ret * 100:>12.2f}%")
    typer.echo(f"   CAGR:           {cagr_val * 100:>12.2f}%")
    typer.echo(f"   Max Drawdown:   {max_dd * 100:>12.2f}%")
    typer.echo(f"   Volatility:     {vol * 100:>12.2f}%")
    typer.echo(f"   Sharpe Ratio:   {sharpe_val:>12.2f}")
    typer.echo(f"   Sortino Ratio:  {sortino_val:>12.2f}")

    # Benchmark comparison (if specified)
    benchmark_values = None
    if benchmark and benchmark_prices is not None:
        typer.echo(f"\n   vs {benchmark.upper()} Benchmark")
        typer.echo("   " + "-" * 36)

        # Run buy-and-hold backtest on benchmark (100% in benchmark)
        benchmark_strategy = StaticWeightStrategy({benchmark.lower(): 0.97})
        benchmark_config = BacktestConfig(
            start_date=benchmark_prices["date"].min(),
            end_date=benchmark_prices["date"].max(),
            initial_cash=cash,
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

        # Calculate benchmark metrics
        benchmark_total_ret = total_return(benchmark_equity)
        benchmark_returns = benchmark_equity["portfolio_value"].pct_change().drop_nulls()

        typer.echo(f"   {benchmark.upper()} Return: {benchmark_total_ret * 100:>12.2f}%")
        typer.echo(f"   Portfolio:      {total_ret * 100:>12.2f}%")
        excess = (total_ret - benchmark_total_ret) * 100
        excess_sign = "+" if excess >= 0 else ""
        typer.echo(f"   Excess Return:  {excess_sign}{excess:>11.2f}%")

        # Calculate alpha, beta, correlation
        comparison = compare_to_benchmark(daily_returns, benchmark_returns)
        typer.echo(f"   Alpha:          {comparison['alpha'] * 100:>12.2f}%")
        typer.echo(f"   Beta:           {comparison['beta']:>12.2f}")
        typer.echo(f"   Correlation:    {comparison['correlation']:>12.2f}")

        # Scale benchmark equity to same starting value for chart comparison
        benchmark_values = benchmark_equity["portfolio_value"].to_list()
        scale_factor = initial_value / benchmark_values[0]
        benchmark_values = [v * scale_factor for v in benchmark_values]

    # Step 5: Display charts
    if not no_charts:
        typer.echo(f"\n Step 5: Charts")

        # Get data for plotting
        dates = equity_curve["date"].to_list()
        values = equity_curve["portfolio_value"].to_list()

        # Convert dates to numeric for plotext (days since start)
        date_labels = [d.strftime("%b %d") for d in dates]

        # Equity curve chart
        pltxt.clear_figure()
        pltxt.plot(values, marker="braille", label="Portfolio")
        if benchmark_values:
            pltxt.plot(benchmark_values, marker="braille", color="gray", label=benchmark.upper())
        pltxt.title("Portfolio vs Benchmark" if benchmark_values else "Portfolio Equity Curve")
        pltxt.xlabel("Time")
        pltxt.ylabel("Value ($)")
        # Show fewer x-axis labels
        n_labels = min(6, len(date_labels))
        step = max(1, len(date_labels) // n_labels)
        xticks = list(range(0, len(date_labels), step))
        xlabels = [date_labels[i] for i in xticks]
        pltxt.xticks(xticks, xlabels)
        pltxt.theme("clear")
        pltxt.plot_size(60, 15)
        pltxt.show()

        # Drawdown chart
        drawdown_df = calculate_drawdown_series(equity_curve)
        drawdowns = drawdown_df["drawdown"].to_list()

        pltxt.clear_figure()
        pltxt.plot(drawdowns, marker="braille", color="red")
        pltxt.title("Portfolio Drawdown")
        pltxt.xlabel("Time")
        pltxt.ylabel("Drawdown (%)")
        pltxt.xticks(xticks, xlabels)
        pltxt.theme("clear")
        pltxt.plot_size(60, 12)
        pltxt.show()

    # Save PNG charts if requested
    if save_charts:
        typer.echo(f"\n Saving PNG charts...")
        charts_path = Path("charts")
        charts_path.mkdir(parents=True, exist_ok=True)

        fig1 = plot_equity_curve(equity_curve=equity_curve)
        fig1.savefig(charts_path / "equity_curve.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)

        fig2 = plot_drawdown(equity_curve=equity_curve)
        fig2.savefig(charts_path / "drawdown.png", dpi=150, bbox_inches='tight')
        plt.close(fig2)

        typer.echo(f"    Saved charts/equity_curve.png")
        typer.echo(f"    Saved charts/drawdown.png")

    typer.echo(f"\n{'=' * 50}")
    typer.echo("   Pipeline complete")
    typer.echo("=" * 50 + "\n")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()