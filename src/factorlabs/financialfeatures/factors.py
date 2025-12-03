import polars as pl
from factorlabs.data.src import io_utils
## NOTE: YOU NEED TO BE IN SRC THEN DO python -m factorlabs.financialfeatures.factors
#WILL FIND A FIX LATER 
df = io_utils.load_prices_df()

def calculate_log_return(df: pl.DataFrame,delay=1) -> pl.DataFrame:
    """
    Calculates the returns using the formula ln(close / close(day - delay))

    Args:
        df (pl.DataFrame): Polars dataframe with OHLCV data 
        delay (int): How many days in between to calculate

  
    """
    df = df.with_columns(
    ((pl.col("close") / pl.col("close").shift(delay)).log())
        .over("ticker")
        .alias("log_return")
    )
    return df

def calculate_momentum(df:pl.DataFrame,delay=10) -> pl.DataFrame:
    """
    Calculates the momentum using the formula close today - close delay

    Args:
        df (pl.DataFrame): Polars dataframe with OHLCV data 
        delay (int): How many days in between to calculate 
    """ 
    df = df.with_columns(
    ((pl.col("close") - pl.col("close").shift(delay)))
        .over("ticker")
        .alias("mom_10d")
    )
    return df
if __name__ == "__main__":
    df = calculate_log_return(df)
    df = calculate_momentum(df)
    with pl.Config(tbl_cols=-1):
        print(df.head(100))
