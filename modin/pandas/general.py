from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from .dataframe import DataFrame




def merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
):
    """Database style join, where common columns in "on" are merged.

    Args:
        left: DataFrame.
        right: DataFrame.
        how: What type of join to use.
        on: The common column name(s) to join on. If None, and left_on and
            right_on  are also None, will default to all commonly named
            columns.
        left_on: The column(s) on the left to use for the join.
        right_on: The column(s) on the right to use for the join.
        left_index: Use the index from the left as the join keys.
        right_index: Use the index from the right as the join keys.
        sort: Sort the join keys lexicographically in the result.
        suffixes: Add this suffix to the common names not in the "on".
        copy: Does nothing in our implementation
        indicator: Adds a column named _merge to the DataFrame with
            metadata from the merge about each row.
        validate: Checks if merge is a specific type.

    Returns:
         A merged Dataframe
        """
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )

    return left.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        copy=copy,
        indicator=indicator,
        validate=validate,
    )


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name="All",
):
    if not isinstance(data, DataFrame):
        raise ValueError(
            "can not create pivot table with instance of type {}".format(type(data))
        )

    return data.pivot_table(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
    )

def melt(
    frame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name="value",
    col_level=None,
):
    return frame.melt(
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
    )

def wide_to_long(df, stubnames, i, j, sep='', suffix='\\d+'):
    # move _default_to_pandas_func to some kinda utils.py for dataframes
    result = pandas.wide_to_long(df._data_manager.to_pandas(),
            i, j, sep=sep, suffix=suffix)
    return DataFrame(result)

def isna(obj):
    """
    Detect missing values for an array-like object.
    Args:
        obj: Object to check for null or missing values.

    Returns:
        bool or array-like of bool
    """
    if isinstance(obj, DataFrame):
        return obj.isna()
    else:
        return pandas.isna(obj)

def isnull(obj):
    """
    Detect missing values for an array-like object.
    Args:
        obj: Object to check for null or missing values.

    Returns:
        bool or array-like of bool
    """
    if isinstance(obj, DataFrame):
        return obj.isnull()
    else:
        return pandas.isnull(obj)

def notna(obj):
    """
    Detect missing values for an array-like object.
    Args:
        obj: Object to check for null or missing values.

    Returns:
        bool or array-like of bool
    """
    if isinstance(obj, DataFrame):
        return obj.notna()
    else:
        return pandas.notna(obj)

def notnull(obj):
    """
    Detect missing values for an array-like object.
    Args:
        obj: Object to check for null or missing values.

    Returns:
        bool or array-like of bool
    """
    if isinstance(obj, DataFrame):
        return obj.notnull()
    else:
        return pandas.notnull(obj)

#not sure about this one
# def pivot
# def merge_ordered
# def merge_asof


