import polars as pl
from factorlabs.data.src import io_utils
## NOTE: YOU NEED TO BE IN SRC THEN DO python -m factorlabs.financialfeatures.factors
#WILL FIND A FIX LATER 
cfg = pl.Config()
cfg.set_tbl_rows(2000)



def calculate_returns(df: pl.DataFrame,delay=1) -> pl.DataFrame:
    """
    Calculates the returns using the formula close / close(day - delay)

    Args:
        df (pl.DataFrame): Polars dataframe with OHLCV data 
        delay (int): How many days in between to calculate

  
    """
    df = df.with_columns(
    ((pl.col("close") / pl.col("close").shift(delay) - 1))
        .over("ticker")
        .alias("ret_1d")
    )
    return df

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

def calculate_momentum(df:pl.DataFrame,delay=10,title='mom_10d') -> pl.DataFrame:
    """
    Calculates the momentum using the formula close today / close delay - 1

    Args:
        df (pl.DataFrame): Polars dataframe with OHLCV data 
        delay (int): How many days in between to calculate 
    """ 
    df = df.with_columns(
    ((pl.col("close") / pl.col("close").shift(delay)) - 1)
        .over("ticker")
        .alias(title)
    )
    return df

def calculate_sma(df:pl.DataFrame,delay=10,title='sma_10d') -> pl.DataFrame:
    """
    Calculates the simple moving average by summing the close over day periods, then dividing 
    by num of periods

    Args:
        df (pl.DataFrame): Polars dataframe with OHLCV data 
        delay (int): How many days in between to calculate 
    """ 
    df = df.with_columns(
    (pl.col("close").rolling_sum(delay) / delay)
        .over("ticker")
        .alias(title)
    )
    return df

def calculate_volitility(df:pl.DataFrame,delay=10,title='vol_10d'):
    """
    calculates volitility using daily returns over delay amount of days
    """
    df = df.with_columns(
        pl.col("ret_1d").rolling_std(window_size=delay)
        .over("ticker")
        .alias(title)
    )
    return df 

if __name__ == "__main__":
    df = io_utils.load_prices_df()

    df = calculate_returns(df)
    df = calculate_log_return(df)
    df = calculate_momentum(df)
    df = calculate_momentum(df,delay=21,title='mom_21d')
    df = calculate_sma(df)
    df = calculate_sma(df,delay=21,title='sma_21d') 
    df = calculate_volitility(df)
    print(df)
