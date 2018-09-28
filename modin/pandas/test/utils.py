import pandas
import modin.pandas as pd
from modin.pandas.utils import to_pandas


def df_equals(df1, df2):
    if isinstance(df1, pd.DataFrame):
        df1 = to_pandas(df1)
    if isinstance(df2, pd.DataFrame):
        df2 = to_pandas(df2)
    
    if isinstance(df1, (pd.DataFrame, pd.Series)) and isinstance(df2, (pd.DataFrame, pd.Series)):
        return df1.equals(df2)
    else:
        return df1 == df2


def df_is_empty(df):
    assert df.size == 0 and df.empty
    assert df.shape[0] == 0 or df.shape[1] == 0

