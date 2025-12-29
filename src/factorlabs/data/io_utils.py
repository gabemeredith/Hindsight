

"""
I/O Utilities for FactorLabs
----------------------------

Centralized utilities for reading/writing data, managing filesystem paths,
interacting with DuckDB, and providing standardized access points for
datasets used across the FactorLabs pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Any

import polars as pl
import duckdb




# ============================================
# 1. PROJECT & DIRECTORY HELPERS
# ============================================

def get_project_root() -> Path:
   """
    Return the root directory of the FactorLabs project.

    This is used to resolve all dataset paths relative to a stable directory.
    Typically this returns the parent folder containing the 'data/' directory.
   """
   return Path(__file__).resolve().parents[3]

def get_data_root() -> Path:
    """
   Return the base directory where all data lives.

   Expected structure:
      data/
         raw/
         interim/
         processed/
         external/

   This function ensures the directory exists before returning it.
    """
    return get_project_root() / "data"




#NOT APPLICABLE 
def get_partition_dir(partition: Literal["raw", "interim", "processed", "external"]) -> Path:
   """
   Return a directory path for a given data partition (e.g., 'processed').

   Creates the directory if it does not already exist.
   """


def get_dataset_path(
    name: str,
    partition: Literal["raw", "interim", "processed", "external"] = "processed",
    suffix: str = ".parquet",
) -> Path:
   """
   Build a standardized dataset path such as:

      data/processed/{name}.parquet

   Parameters
   ----------
   name : str
      The logical name of the dataset (e.g., "yf_prices").
   partition : str
      Which data partition to use (raw/interim/processed/external).
   suffix : str
      File extension, defaulting to '.parquet'.

   Returns
   -------
   Path
      Full path to the dataset file.
   """


# ============================================
# 2. PARQUET I/O HELPERS
# ============================================

def load_prices_df(name: str = "yf_prices") -> pl.DataFrame:
    """
    Load yf_prices.parquet from the data/ folder next to this file.
    """
    path = get_data_root() / f"{name}.parquet"
    return pl.read_parquet(path)


def write_prices(df: pl.DataFrame, cfg) -> None:
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
    if suffix == '.parquet':
        df.write_parquet(cfg.out_path)
    elif suffix == ".duckdb":
        #TODO work on this later down the line
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
        

# ============================================
# 3. DUCKDB CONNECTION
# ============================================

def get_duckdb_con(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection.

    If no path is given, uses the default FactorLabs DuckDB file under
    data/external/factorlabs.duckdb (or similar project directory).

    Returned connections should be closed by the caller unless used
    with a context-managed wrapper.
    """


# ============================================
# 4. DUCKDB QUERY/TABLE HELPERS
# ============================================

def duckdb_query(
    sql: str,
    *,
    params: dict[str, Any] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pl.DataFrame:
    """
    Execute a SQL query through DuckDB and return the results as a Polars DataFrame.

    If no connection is supplied, a new one is opened and then closed
    after executing the query.

    Parameters
    ----------
    sql : str
        SQL query string.
    params : dict[str, Any], optional
        Named parameters to substitute into the SQL.
    con : duckdb.DuckDBPyConnection, optional
        Existing connection. If None, a temporary one is created.

    Returns
    -------
    pl.DataFrame
        Results of the query.
    """

#TODO later down the line integrate duckdb
def write_df_to_duckdb(
    df: pl.DataFrame,
    table: str,
    *,
    mode: Literal["replace", "append"] = "replace",
    con: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """
    Write a Polars DataFrame into a DuckDB table.

    Parameters
    ----------
    df : pl.DataFrame
        The data to write.
    table : str
        Name of the DuckDB table.
    mode : {"replace", "append"}
        - replace: Drops the existing table and recreates it.
        - append: Inserts rows into an existing table.
    con : duckdb.DuckDBPyConnection, optional
        Existing connection. If None, a temporary one is created.

    Raises
    ------
    ValueError
        If given an unsupported mode.
    """


# ============================================
# 5. HIGH-LEVEL DATASET HELPERS
# ============================================

def save_prices_df(df: pl.DataFrame, *, name: str = "yf_prices") -> Path:
    """
    Save the canonical price dataset into the processed partition.

    Parameters
    ----------
    df : pl.DataFrame
        Price data after ingestion or after cleaning.
    name : str
        Logical dataset name (filename without extension).

    Returns
    -------
    Path
        Path where the dataset was written.
    """




# ============================================
# 6. OPTIONAL UTILITIES (CACHING / SAFE LOAD)
# ============================================

def try_load_dataset(
    name: str,
    partition: Literal["raw", "interim", "processed", "external"] = "processed",
    suffix: str = ".parquet",
) -> pl.DataFrame | None:
    """
    Try to load a dataset if it exists; return None if it does not.

    Parameters
    ----------
    name : str
        Logical dataset name.
    partition : str
        Directory partition to look in.
    suffix : str
        File extension.

    Returns
    -------
    pl.DataFrame or None
        The loaded DataFrame, or None if the file is missing.
    """
    
if __name__ == "__main__":

    df = load_prices_df()
    print(df)