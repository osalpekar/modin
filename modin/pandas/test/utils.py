import pandas
import modin.pandas as pd
from modin.pandas.utils import to_pandas


def df_equals(df1, df2):
    """Tests if df1 and df2 are equal.

    Args:
        df1: (pandas or modin DataFrame or series) dataframe to test if equal.
        df2: (pandas or modin DataFrame or series) dataframe to test if equal.

    Returns:
        True if df1 is equal to df2.
    """
    types_for_equals = (pandas.DataFrame, pandas.Series, pandas.core.indexes.range.RangeIndex, pandas.core.indexes.base.Index)
    if isinstance(df1, pd.DataFrame):
        df1 = to_pandas(df1)
    if isinstance(df2, pd.DataFrame):
        df2 = to_pandas(df2)
    
    if isinstance(df1, types_for_equals) and isinstance(df2, types_for_equals):
        return df1.equals(df2)
    else:
        return df1 == df2


def df_is_empty(df):
    """Tests if df is empty.

    Args:
        df: (pandas or modin DataFrame) dataframe to test if empty.

    Returns:
        True if df is empty.
    """
    assert df.size == 0 and df.empty
    assert df.shape[0] == 0 or df.shape[1] == 0


def arg_keys(arg_name, keys):
    """Appends arg_name to the front of all values in keys.

    Args:
        arg_name: (string) String containing argument name.
        keys: (list of strings) Possible inputs of argument.

    Returns:
        List of strings with arg_name append to front of keys.
    """
    return ["{0} {1}".format(arg_name, key) for key in keys]


def name_contains(test_name, vals):
    """Determines if any string in vals is a substring of test_name.
    
    Args:
        test_name: (string) String to determine if contains substrings.
        vals: (list of strings) List of substrings to test for.

    Returns:
        True if a substring in vals is in test_name, else False.
    """
    return any([val in test_name for val in vals])

