from dataclasses import dataclass
from datetime import date
import datetime
from typing import Sequence
import yfinance as yf
import polars as pl
import time
import ast
import duckdb
@dataclass
class YFIngestConfig:
    tickers: Sequence[str]
    start: date
    end: date
    interval: str
    adjust: bool = True
    out_path: str = "src/data/yf_prices.parquet"
    
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

def normalize_prices(df: pl.DataFrame) -> pl.DataFrame:
    # standardize columns, dtypes, timezone, multiindex -> tidy, etc.
    """
    Normalize a raw price DataFrame into the standard FactorLabs price schema.

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
        FactorLabs pipeline stages (feature engineering, sleeves, backtesting).

    Notes
    -----
    This function should contain **no vendor-specific API calls**. Its only job 
    is to take raw price data and standardize its structure and schema.
    """
    #step 1 -> normalize the column names
    df = normalize_column_names(df)
    df = normalize_date_column(df)
    df = data_quality_fixing(df)
    #step 2 -> ensure there is a date column w/ datatypes of date
    
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
    - Enforces non-negative prices and integer volume where applicable.
    - Ensures no duplicate (ticker, date) combinations.
    """
    print(f"DF BEFORE CHANGES: {df}")
    df = df.filter(~pl.all_horizontal(pl.all().exclude('ticker',"date").is_null()))
    print(f"DF AFTER CHANGES: {df}")
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

def write_prices(df: pl.DataFrame, cfg: YFIngestConfig) -> None:
    """
    Persist a normalized OHLCV price DataFrame to disk.

    This function writes the processed Polars DataFrame produced by the
    ingestion pipeline to the output destination specified in
    `cfg.out_path`. The write format is inferred from the file extension
    (e.g., `.parquet`, `.duckdb`).

    Responsibilities:
    - Ensure required price columns exist (e.g., date, open, high, low, close, volume).
    - Create parent directories if needed.
    - Write the DataFrame atomically, overwriting any existing output file.

    Parameters
    ----------
    df : pl.DataFrame
        Normalized price data to b saved.

    cfg : YFIngestConfig
        Ingest configuration containing the output path.

    Returns
    -------
    None
        Performs I/O only.
    """
    from pathlib import Path
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == 'parquet':
        df.write_parquet(cfg.out_path)
    elif suffix == ".duckdb":
        # Create or open a DuckDB database file
        con = duckdb.connect(out_path.as_posix())

        # Register Polars DataFrame as a DuckDB view via Arrow
        con.register("prices_df", df.to_arrow())

        # Create or replace a table in the database
        con.execute("""
            CREATE OR REPLACE TABLE prices AS
            SELECT * FROM prices_df
        """)

        con.close()
    
    else:
        raise ValueError(f"Unsupported output format for: {out_path}")
        

        #lets write the prices to parquet 
        
def run_ingest(cfg: YFIngestConfig) -> None:
    raw = fetch_yf_data(cfg)
    normalized = normalize_prices(raw)
    write_prices(normalized, cfg)

def main():
    # parse argparse / typer, build YFIngestConfig, call run_ingest
    ...

if __name__ == "__main__":
    cfg = YFIngestConfig(
        tickers=["AAPL", "MSFT"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        interval="1d",
    )
    time1 = time.time()
    df = fetch_yf_data(cfg)
    time2 = time.time()
    print(f"TIME TAKEN TO COMPUTE: {time2 - time1}")
    df = normalize_prices(df)
    # df = normalize_column_names(df)
    print(df.head())
    print(df.columns)
