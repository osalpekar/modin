from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import sys
import pandas
import numpy as np
import modin.pandas as pd
from modin.pandas.utils import from_pandas, to_pandas

from .utils import (
    df_equals,
    name_contains,
    arg_keys,
    test_dfs_keys,
    numeric_dfs,
    test_dfs_values,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    bool_none_arg_keys,
    bool_none_arg_values,
    int_arg_keys,
    int_arg_values,
    quantiles_keys,
    quantiles_values,
    groupby_apply_func_keys,
    groupby_apply_func_values,
    groupby_agg_func_keys,
    groupby_agg_func_values,
    groupby_transform_func_keys,
    groupby_transform_func_values,
    groupby_pipe_func_keys,
    groupby_pipe_func_values,
)

PY2 = False
if sys.version_info.major < 3:
    PY2 = True

# Create test_groupby objects
test_groupby = dict()
for axis_name, axis in zip(axis_keys, axis_values):
    for df_name, dfs in zip(test_dfs_keys, test_dfs_values):
        if "empty_data" not in df_name and not (
            axis_name == "over rows" and "columns_only" in df_name
        ):
            modin_df, pandas_df = dfs
            index = modin_df.columns if axis_name == "over columns" else modin_df.index
            vals = (
                modin_df.groupby([str(i) for i in index], axis=axis),
                pandas_df.groupby([str(i) for i in index], axis=axis),
            )
            test_groupby["{}-{}".format(df_name, axis_name)] = vals

<<<<<<< 729ac7d2ac751f1d0e03a22d0b144489323b87d8
@pytest.fixture
def ray_df_equals_pandas(ray_df, pandas_df):
    assert isinstance(ray_df, pd.DataFrame)
    assert to_pandas(ray_df).equals(pandas_df) or (
        all(ray_df.isna().all()) and all(pandas_df.isna().all())
    )


@pytest.fixture
def ray_df_almost_equals_pandas(ray_df, pandas_df):
    assert isinstance(ray_df, pd.DataFrame)
    difference = to_pandas(ray_df) - pandas_df
    diff_max = difference.max().max()
    assert (
        to_pandas(ray_df).equals(pandas_df)
        or diff_max < 0.0001
        or (all(ray_df.isna().all()) and all(pandas_df.isna().all()))
    )


@pytest.fixture
def ray_series_equals_pandas(ray_df, pandas_df):
    assert ray_df.equals(pandas_df)


@pytest.fixture
def ray_df_equals(ray_df1, ray_df2):
    assert to_pandas(ray_df1).equals(to_pandas(ray_df2))


@pytest.fixture
def ray_groupby_equals_pandas(ray_groupby, pandas_groupby):
    for g1, g2 in zip(ray_groupby, pandas_groupby):
        assert g1[0] == g2[0]
        ray_df_equals_pandas(g1[1], g2[1])


def test_simple_row_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [3, 8, 12, 10],
            "col4": [17, 13, 16, 15],
            "col5": [-4, -5, -6, -7],
        }
    )

    ray_df = from_pandas(pandas_df)

    by = [1, 2, 1, 2]
    n = 1

    ray_groupby = ray_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    ray_groupby_equals_pandas(ray_groupby, pandas_groupby)
    test_ngroups(ray_groupby, pandas_groupby)
    test_skew(ray_groupby, pandas_groupby)
    test_ffill(ray_groupby, pandas_groupby)
    test_sem(ray_groupby, pandas_groupby)
    test_mean(ray_groupby, pandas_groupby)
    test_any(ray_groupby, pandas_groupby)
    test_min(ray_groupby, pandas_groupby)
    test_idxmax(ray_groupby, pandas_groupby)
    test_ndim(ray_groupby, pandas_groupby)
    test_cumsum(ray_groupby, pandas_groupby)
    test_pct_change(ray_groupby, pandas_groupby)
    test_cummax(ray_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        test_apply(ray_groupby, pandas_groupby, func)

    test_dtypes(ray_groupby, pandas_groupby)
    test_first(ray_groupby, pandas_groupby)
    test_backfill(ray_groupby, pandas_groupby)
    test_cummin(ray_groupby, pandas_groupby)
    test_bfill(ray_groupby, pandas_groupby)
    test_idxmin(ray_groupby, pandas_groupby)
    test_prod(ray_groupby, pandas_groupby)
    test_std(ray_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        test_agg(ray_groupby, pandas_groupby, func)
        test_aggregate(ray_groupby, pandas_groupby, func)

    test_last(ray_groupby, pandas_groupby)
    test_mad(ray_groupby, pandas_groupby)
    test_rank(ray_groupby, pandas_groupby)
    test_max(ray_groupby, pandas_groupby)
    test_var(ray_groupby, pandas_groupby)
    test_len(ray_groupby, pandas_groupby)
    test_sum(ray_groupby, pandas_groupby)
    test_ngroup(ray_groupby, pandas_groupby)
    test_nunique(ray_groupby, pandas_groupby)
    test_median(ray_groupby, pandas_groupby)
    test_head(ray_groupby, pandas_groupby, n)
    test_cumprod(ray_groupby, pandas_groupby)
    test_cov(ray_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        test_transform(ray_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        test_pipe(ray_groupby, pandas_groupby, func)

    test_corr(ray_groupby, pandas_groupby)
    test_fillna(ray_groupby, pandas_groupby)
    test_count(ray_groupby, pandas_groupby)
    test_tail(ray_groupby, pandas_groupby, n)
    test_quantile(ray_groupby, pandas_groupby)
    test_take(ray_groupby, pandas_groupby)


def test_single_group_row_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 36, 7],
            "col3": [3, 8, 12, 10],
            "col4": [17, 3, 16, 15],
            "col5": [-4, 5, -6, -7],
        }
    )

    ray_df = from_pandas(pandas_df)

    by = ["1", "1", "1", "1"]
    n = 6

    ray_groupby = ray_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ngroups(modin_groupby, pandas_groupby):
    assert modin_groupby.ngroups == pandas_groupby.ngroups


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_skew(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.skew(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.skew(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ffill(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.ffill()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_sem(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.sem()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_mean(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.mean(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.mean(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_none_arg_values, ids=arg_keys("bool_only", bool_none_arg_keys)
)
def test_any(modin_groupby, pandas_groupby, axis, skipna, bool_only):
    modin_result = modin_groupby.any(axis=axis, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_groupby.any(axis=axis, skipna=skipna, bool_only=bool_only)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_min(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.min(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.min(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_idxmax(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.idxmax(), pandas_groupby.idxmax())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ndim(modin_groupby, pandas_groupby):
    assert modin_groupby.ndim == pandas_groupby.ndim


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumsum(request, modin_groupby, pandas_groupby, axis, skipna):
    if name_contains(request.node.name, numeric_groupby):
        modin_result = modin_groupby.cumsum(axis=axis, skipna=skipna)
        pandas_result = pandas_groupby.cumsum(axis=axis, skipna=skipna)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.cumsum(axis=axis, skipna=skipna)


@pytest.fixture
def test_pct_change(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.pct_change()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummax(request, modin_groupby, pandas_groupby, axis, skipna):
    if name_contains(request.node.name, numeric_groupby):
        modin_result = modin_groupby.cummax(axis=axis, skipna=skipna)
        pandas_result = pandas_groupby.cummax(axis=axis, skipna=skipna)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.cummax(axis=axis, skipna=skipna)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", groupby_apply_func_values, ids=groupby_apply_func_keys)
def test_apply(request, modin_groupby, pandas_groupby, func, axis):
    if name_contains(request.node.name, ["over rows"]) or not name_contains(
        request.node.name, numeric_dfs
    ):
        modin_result = modin_groupby.apply(func, axis)
        pandas_result = pandas_groupby.apply(func, axis)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.apply(func, axis)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_dtypes(modin_groupby, pandas_groupby):
    assert df_equals(modin_groupby.dtypes, pandas_groupby.dtypes)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_first(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.first()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_backfill(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.backfill()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummin(modin_groupby, pandas_groupby, axis, skipna):
    modin_result = modin_groupby.cummin(axis=axis, skipna=skipna)
    pandas_result = pandas_groupby.cummin(axis=axis, skipna=skipna)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_bfill(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.bfill()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_idxmin(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.idxmin(), pandas_groupby.idxmin())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_prod(modin_groupby, pandas_groupby, axis, skipna, numeric_only, min_count):
    if numeric_only or name_contains(request.node.name, numeric_dfs):
        modin_result = modin_groupby.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        pandas_result = pandas_groupby.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_groupby.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(modin_groupby, pandas_groupby, axis, skipna, numeric_only, ddof):
    modin_result = modin_groupby.std(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    pandas_result = pandas_groupby.std(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", groupby_agg_func_values, ids=groupby_agg_func_keys)
def test_aggregate(request, modin_groupby, pandas_groupby, axis, func):
    # if (name_contains(request.node.name, ["over rows"]) or
    #         not name_contains(request.node.name, numeric_dfs)):
    #     modin_result = modin_groupby.aggregate(func, axis)
    #     pandas_result = pandas_groupby.aggregate(func, axis)
    #     assert df_equals(modin_result, pandas_result)
    # else:
    #     with pytest.raises(TypeError):
    #         modin_result = modin_groupby.aggregate(func, axis)
    with pytest.raises(NotImplementedError):
        modin_groupby.aggregate()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", groupby_agg_func_values, ids=groupby_agg_func_keys)
def test_agg(request, modin_groupby, pandas_groupby, axis, func):
    # if (name_contains(request.node.name, ["over rows"]) or
    #         not name_contains(request.node.name, numeric_dfs)):
    #     modin_result = modin_groupby.agg(func, axis)
    #     pandas_result = pandas_groupby.agg(func, axis)
    #     assert df_equals(modin_result, pandas_result)
    # else:
    #     with pytest.raises(TypeError):
    #         modin_result = modin_groupby.agg(func, axis)
    with pytest.raises(NotImplementedError):
        modin_groupby.agg()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_last(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.last()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_mad(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.mad()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "method",
    ["average", "min", "max", "first", "dense"],
    ids=["average", "min", "max", "first", "dense"],
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
@pytest.mark.parametrize(
    "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("pct", bool_arg_values, ids=arg_keys("pct", bool_arg_keys))
def test_rank(
    modin_groupby, pandas_groupby, axis, method, numeric_only, na_option, ascending, pct
):
    modin_result = modin_groupby.rank(
        axis=axis,
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    pandas_result = pandas_groupby.rank(
        axis=axis,
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_max(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.max(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.max(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(modin_groupby, pandas_groupby, axis, skipna, numeric_only, ddof):
    modin_result = modin_groupby.var(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    pandas_result = pandas_groupby.var(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_len(modin_groupby, pandas_groupby):
    assert len(modin_groupby) == len(pandas_groupby)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_sum(
    request, modin_groupby, pandas_groupby, axis, skipna, numeric_only, min_count
):
    if numeric_only or name_contains(request.node.name, numeric_groupbys):
        modin_result = modin_groupby.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        pandas_result = pandas_groupby.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_groupby.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ngroup(modin_groupby, pandas_groupby):
    assert df_equals(modin_groupby.ngroup(), pandas_groupby.ngroup())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_nunique(modin_groupby, pandas_groupby):
    assert df_equals(modin_groupby.nunique(), pandas_groupby.nunique())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_median(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.median(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.median(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(modin_groupby, pandas_groupby, n):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.head(n=n), pandas_groupby.head(n=n))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumprod(request, modin_groupby, pandas_groupby, axis, skipna):
    if name_contains(request.node.name, numeric_groupbys):
        modin_result = modin_groupby.cumprod(axis=axis, skipna=skipna)
        pandas_result = pandas_groupby.cumprod(axis=axis, skipna=skipna)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.cumprod(axis=axis, skipna=skipna)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_cov(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.cov()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize(
    "func", groupby_transform_func_values, ids=groupby_transform_func_keys
)
def test_transform(request, modin_groupby, pandas_groupby, func):
    if "empty_data" not in request.node.name:
        modin_result = modin_groupby.agg(func)
        pandas_result = pandas_groupby.agg(func)
        assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_corr(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.corr()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize(
    "method",
    ["backfill", "bfill", "pad", "ffill", None],
    ids=["backfill", "bfill", "pad", "ffill", "None"],
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_fillna(modin_groupby, pandas_groupby, method, axis):
    modin_result = modin_groupby.fillna(method=method, axis=axis, inplace=False)
    pandas_result = pandas_groupby.fillna(method=method, axis=axis, inplace=False)
    assert df_equals(modin_result, pandas_result)

    modin_groupby.fillna(method=method, axis=axis, inplace=True)
    pandas_groupby.fillna(method=method, axis=axis, inplace=True)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_count(modin_groupby, pandas_groupby, axis, numeric_only):
    modin_result = modin_groupby.count(axis=axis, numeric_only=numeric_only)
    pandas_result = pandas_groupby.count(axis=axis, numeric_only=numeric_only)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("func", groupby_pipe_func_values, ids=groupby_pipe_func_keys)
def test_pipe(modin_groupby, pandas_groupby, func):
    assert df_equals(modin_groupby.pipe(func), pandas_groupby.pipe(func))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(modin_groupby, pandas_groupby, n):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.tail(n=n), pandas_groupby.tail(n=n))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(modin_groupby, pandas_groupby, q):
    assert df_equals(modin_groupby.quantile(q), pandas_groupby.quantile(q))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_take(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.take(indices=[1])
