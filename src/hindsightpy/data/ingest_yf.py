from dataclasses import dataclass
from datetime import date
import datetime
from typing import Sequence
import yfinance as yf
import polars as pl
import time
import ast
import duckdb
from .io_utils import write_prices
# From the project root:
# EXAMPLE RUN 
# python -m hindsightpy.data.ingest_yf NVDA 2020-01-01 2020-02-01


@dataclass
class YFIngestConfig:
    tickers: Sequence[str]
    start: date
    end: date
    interval: str
    adjust: bool = True
    out_path: str = "data/yf_prices.parquet"
    
# ========================== PUBLIC PIPELINE ==========================

def run_ingest(cfg: YFIngestConfig) -> None:
    """
    Runs the entire ingestion pipeline, which fetchs yf data, normalizes it,
    and stores it into an external file. Currently supports .duckdb and parquet files
    """
    raw = fetch_yf_data(cfg)
    normalized = normalize_prices(raw)
    print(normalized)
    write_prices(normalized, cfg)

# ========================== STAGE 1: FETCH ========================== 
def fetch_yf_data(cfg: YFIngestConfig) -> pl.DataFrame:
    """
    Function to fetch yahoo finance data and put it into a polars dataframe
    
    Parameters:cfg(YFIngestConfig)
    Preconditions:cfg is a YFIngestConfig
    """
    assert isinstance(cfg,YFIngestConfig), "cfg isnt a YFIngestConfig"
    data = yf.download(tickers=cfg.tickers,start=cfg.start,
                       end=cfg.end,interval=cfg.interval,
                       auto_adjust=cfg.adjust,
                       progress=False,)
    if data.empty:
        raise ValueError("No data returned from yfinance for given config")
    
    data = data.reset_index()

    df = pl.from_pandas(data)
    return df

# ========================== STAGE 2: NORMALIZE ==========================

def normalize_prices(df: pl.DataFrame) -> pl.DataFrame:
    # standardize columns, dtypes, timezone, multiindex -> tidy, etc.
    """
    Normalize a raw price DataFrame into the standard Hindsight.py price schema.

    This function takes the raw output returned by the data source (typically 
    yfinance) and performs the following transformations:

    1. Column normalization:
       - Ensures all column names are lower_snake_case.
       - Renames vendor-specific fields (e.g., 'Adj Close' -> 'adj_close').
       - Ensures a consistent set of price fields when available:
         ['open', 'high', 'low', 'close', 'adj_close', 'volume'].

    2. Index & date normalization:
       - Ensures there is a 'date' column of type Date.
       - Removes timezone information and normalizes to UTC if present.
       - Sorts the DataFrame by ['ticker', 'date'] when a ticker column exists.

    3. Multi-ticker handling:
       - If the raw DataFrame is in yfinance's wide/multi-index format, 
         it is converted to a tidy long-form schema with columns:
         ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'].

    4. Data-quality fixes:
       - Drops rows with entirely null OHLCV values.
       - Enforces non-negative prices and integer volume where applicable.
       - Ensures no duplicate (ticker, date) combinations.

    Returns
    -------
    pl.DataFrame
        A clean, canonical, long-form price dataset suitable for all downstream 
        Hindsight.py pipeline stages (feature engineering, sleeves, backtesting).

    Notes
    -----
    This function should contain **no vendor-specific API calls**. Its only job 
    is to take raw price data and standardize its structure and schema.
    """
    #step 1 -> normalize the column names
    df = normalize_column_names(df)
    df = normalize_date_column(df)
    df = data_quality_fixing(df)
    df = wide_to_long(df)
    #Final Step -> Turn from date       ┆ close_aapl ┆ close_msft ┆ high_aapl  ┆ … ┆ open_aapl ┆ open_msft 
    # to ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume'] 
    #because our current df is really inefficient apprently bc it adds columns not rows
   
     
    return df
    
# ---------------------------------------------HELPER FUNCTIONS FOR NORMALIZE_PRICES----------------------------------------------
import polars as pl

def normalize_date_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure the 'date' column is of dtype pl.Date.

    Handles:
    - pl.Utf8 (string dates)      -> parsed to pl.Date
    - pl.Datetime (with/without tz) -> converted to pl.Date
    - pl.Date or pl.Object        -> left as-is (assumed already date-like)
    """
    dtype = df["date"].dtype

    # Case 1: string dates -> parse to datetime -> date
    if dtype == pl.Utf8:
        return df.with_columns(
            pl.col("date")
            .str.to_datetime()
            .dt.date()
            .alias("date")
        )

    # Case 2: datetime (with or without timezone) -> date
    # In Polars, timezone-aware dtypes are pl.Datetime(time_zone=...)
    if isinstance(dtype, pl.Datetime):
        return df.with_columns(
            pl.col("date")
            .dt.date()
            .alias("date")
        )

    # Case 3: already date-like -> do nothing
    # Depending on how the df is built, this might come through as pl.Date or pl.Object.
    if dtype in (pl.Date, pl.Object):
        return df

    # Anything else is weird
    raise TypeError(f"Unsupported date dtype: {dtype}")

def data_quality_fixing(df: pl.DataFrame) -> pl.DataFrame:
    """
    - Drops rows with entirely null OHLCV values.
    - Drops rows with null close prices (required for backtesting).
    - Enforces non-negative prices and integer volume where applicable.
    - Ensures no duplicate (ticker, date) combinations.
    """
    #filter out Null columns
    df = df.filter(~pl.all_horizontal(pl.all().exclude('ticker',"date").is_null()))

    price_cols = [col for col in df.columns if col not in ['date', 'ticker']]

    # Replace negative values with None
    for col in price_cols:
        df = df.with_columns(
            pl.when(pl.col(col) < 0)
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )

    # Drop rows with null close prices - required for backtesting
    close_cols = [col for col in df.columns if col == 'close' or col.startswith('close_')]
    if close_cols:
        df = df.filter(~pl.any_horizontal([pl.col(c).is_null() for c in close_cols]))

    return df

        
def normalize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """
    Flatten yfinance-style MultiIndex columns into lower_snake_case strings.

    Examples
    --------
    ('Date', '')        -> 'date'
    ('Close', 'AAPL')   -> 'close_aapl'
    ('Volume', 'MSFT')  -> 'volume_msft'
    """
    rename_map = {}

    for col in df.columns:
        name = str(col)

        # Case 1: looks like "('close', 'aapl')" -> parse as a tuple
        if name.startswith("(") and name.endswith(")"):
            try:
                level1, level2 = ast.literal_eval(name)  # ('close', 'aapl')
            except (SyntaxError, ValueError):
                # fallback: just lowercase the raw string
                new_name = name.lower()
            else:
                level1 = (level1 or "").strip().lower()
                level2 = (level2 or "").strip().lower()

                if level2 == "":
                    # e.g. ('date', '') -> 'date'
                    new_name = level1
                else:
                    # e.g. ('close', 'aapl') -> 'close_aapl'
                    new_name = f"{level1}_{level2}"
        else:
            # Case 2: already a simple string column
            new_name = name.lower()

        rename_map[col] = new_name

    return df.rename(rename_map)

def wide_to_long(df: pl.DataFrame):
    """
    turns columns from date       ┆ close_aapl ┆ close_msft ┆ high_aapl  ┆ … ┆ open_aapl ┆ open_msft
    to: Data Ticker Open Close High Low Volume 
    
    How? wide df → unpivot → split into field+ticker → pivot by field
    """
    has_ticker_suffix = any('_' in col for col in df.columns if col != 'date')
    
    if not has_ticker_suffix:
        # Data is already in long format (single ticker case)
        # Just return as-is
        # NOTE: Single-ticker data won't have a 'ticker' column
        # May want to add one in the future
        return df
    df = df.unpivot(
        index="date",
        on= df.columns[1:],
    )
    df = df.sort("date")
    df = df.with_columns([
        pl.col("variable").str.split("_").list.get(0).alias("feature"),
        pl.col("variable").str.split("_").list.get(1).str.to_lowercase().alias("ticker"),
    ]
    ).drop("variable")
    df = df.select(
        "date",
        "ticker",
        "feature",
        "value"
    )
    df = df.pivot(
        index=["date","ticker"],
        on="feature",
        values="value",
        aggregate_function="first",
    )

    return df
# ========================== SCRIPT ENTRYPOINT ==========================

def main():
    """
    This function parses the users arguments, then assigns variables to YFIngestConfig,
    then runs the pipeline
    """
    # parse argparse / typer, build YFIngestConfig, call run_ingest
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers",nargs="+",
                        help="Tickers, e.g. AAPL MSFT")
    parser.add_argument("start_date",type=str,
                        help="first date")
    parser.add_argument("end_date",type=str,
                        help="ending date")
    #lets use a 1d interval as default? or actually 
    #TODO: have an interval liek 1d - 1w - 1m and the closer amt the user puts round
    #could be cool idk
    
    
    
    args = parser.parse_args()
    
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    cfg = YFIngestConfig(
        tickers=args.tickers,
        start=start,
        end = end,
        interval="1d",
        out_path="data/yf_prices.parquet",
    )
    t0 = time.time()
    run_ingest(cfg)
    t1 = time.time()

    print(f"Completed ingest in {t1 - t0:.3f}s")
    print(f"Data written to: {cfg.out_path}")


if __name__ == "__main__":
    main()
