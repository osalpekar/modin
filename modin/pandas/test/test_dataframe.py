from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import io
import numpy as np
import pandas
import pandas.util.testing as tm
from pandas.tests.frame.common import TestData
import matplotlib
import modin.pandas as pd
from modin.pandas.utils import to_pandas
from numpy.testing import assert_array_equal
import sys

from .utils import (df_equals, df_is_empty, arg_keys, name_contains, test_dfs_keys, 
        test_dfs_values, numeric_dfs, test_func_keys, test_func_values, 
        query_func_keys, query_func_values, agg_func_keys, agg_func_values, 
        numeric_agg_funcs, quantiles_keys, quantiles_values, indices_keys, 
        indices_values, axis_keys, axis_values, bool_arg_keys, bool_arg_values,
        bool_none_arg_keys, bool_none_arg_values, int_arg_keys, int_arg_values)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Test inter df math functions
def inter_df_math_helper(op):
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    ray_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    assert df_equals(getattr(ray_df, op)(ray_df), getattr(pandas_df, op)(pandas_df))
    assert df_equals(getattr(ray_df, op)(4), getattr(pandas_df, op)(4))
    assert df_equals(getattr(ray_df, op)(4.0), getattr(pandas_df, op)(4.0))

    frame_data = {"A": [0, 2], "col1": [0, 19], "col2": [1, 1]}
    ray_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    assert df_equals(getattr(ray_df, op)(ray_df2), getattr(pandas_df, op)(pandas_df2))

    list_test = [0, 1, 2, 4]

    assert df_equals(getattr(ray_df, op)(list_test, axis=1), getattr(pandas_df, op)(list_test, axis=1))

    assert df_equals(
        getattr(ray_df, op)(list_test, axis=0),
        getattr(pandas_df, op)(list_test, axis=0),
    )


def test_add():
    inter_df_math_helper("add")


def test_div():
    inter_df_math_helper("div")


def test_divide():
    inter_df_math_helper("divide")


def test_floordiv():
    inter_df_math_helper("floordiv")


def test_mod():
    inter_df_math_helper("mod")


def test_mul():
    inter_df_math_helper("mul")


def test_multiply():
    inter_df_math_helper("multiply")


def test_pow():
    inter_df_math_helper("pow")


def test_sub():
    inter_df_math_helper("sub")


def test_subtract():
    inter_df_math_helper("subtract")


def test_truediv():
    inter_df_math_helper("truediv")


def test___div__():
    inter_df_math_helper("__div__")

# END test inter df math functions


# Test comparison of inter operation functions
def comparison_inter_ops_helper(op):
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    ray_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    assert df_equals(
        getattr(ray_df, op)(ray_df), getattr(pandas_df, op)(pandas_df)
    )
    assert df_equals(getattr(ray_df, op)(4), getattr(pandas_df, op)(4))
    assert df_equals(getattr(ray_df, op)(4.0), getattr(pandas_df, op)(4.0))

    frame_data = {"A": [0, 2], "col1": [0, 19], "col2": [1, 1]}

    ray_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    assert df_equals(
        getattr(ray_df2, op)(ray_df2), getattr(pandas_df2, op)(pandas_df2)
    )


def test_eq():
    comparison_inter_ops_helper("eq")


def test_ge():
    comparison_inter_ops_helper("ge")


def test_gt():
    comparison_inter_ops_helper("gt")


def test_le():
    comparison_inter_ops_helper("le")


def test_lt():
    comparison_inter_ops_helper("lt")


def test_ne():
    comparison_inter_ops_helper("ne")
# END test comparison of inter operation functions


# Test dataframe right operations
def inter_df_math_right_ops_helper(op):
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    ray_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    assert df_equals(getattr(ray_df, op)(4), getattr(pandas_df, op)(4))
    assert df_equals(getattr(ray_df, op)(4.0), getattr(pandas_df, op)(4.0))


def test_radd():
    inter_df_math_right_ops_helper("radd")


def test_rdiv():
    inter_df_math_right_ops_helper("rdiv")


def test_rfloordiv():
    inter_df_math_right_ops_helper("rfloordiv")


def test_rmod():
    inter_df_math_right_ops_helper("rmod")


def test_rmul():
    inter_df_math_right_ops_helper("rmul")


def test_rpow():
    inter_df_math_right_ops_helper("rpow")


def test_rsub():
    inter_df_math_right_ops_helper("rsub")


def test_rtruediv():
    inter_df_math_right_ops_helper("rtruediv")


def test___rsub__():
    inter_df_math_right_ops_helper("__rsub__")
# END test dataframe right operations


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_abs(request, ray_df, pandas_df):
    if name_contains(request.node.name, numeric_dfs):
        assert df_equals(ray_df.abs(), pandas_df.abs())
    else:
        with pytest.raises(TypeError):
            ray_df.abs()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_add_prefix(ray_df, pandas_df):
    test_prefix = "TEST"
    new_ray_df = ray_df.add_prefix(test_prefix)
    new_pandas_df = pandas_df.add_prefix(test_prefix)
    assert df_equals(new_ray_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap(request, ray_df, pandas_df, testfunc):
    if (not name_contains(request.node.name, numeric_test_funcs) or 
            name_contains(request.node.name, numeric_dfs)):
        new_ray_df = ray_df.applymap(testfunc)
        new_pandas_df = pandas_df.applymap(testfunc)

        assert df_equals(new_ray_df, new_pandas_df)
    else:
        with pytest.raises(TypeError):
            ray_df.applymap(testfunc)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_add_suffix(ray_df, pandas_df):
    test_suffix = "TEST"
    new_ray_df = ray_df.add_suffix(test_suffix)
    new_pandas_df = pandas_df.add_suffix(test_suffix)

    assert df_equals(new_ray_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_at(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.at()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_axes(ray_df, pandas_df):
    for ray_axis, pd_axis in zip(ray_df.axes, pandas_df.axes):
        assert np.array_equal(ray_axis, pd_axis)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_copy(ray_df, pandas_df):
    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    new_ray_df = ray_df.copy()

    assert new_ray_df is not ray_df
    assert df_equals(ray_df, new_ray_df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dtypes(ray_df, pandas_df):
    assert df_equals(ray_df.dtypes, pandas_df.dtypes)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ftypes(ray_df, pandas_df):
    assert df_equals(ray_df.ftypes, pandas_df.ftypes)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("key", indices_values, ids=indices_keys)
def test_get(ray_df, pandas_df, key):
    assert df_equals(ray_df.get(key), pandas_df.get(key))
    assert df_equals(ray_df.get(key, default="default"), pandas_df.get(key, default="default"))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_dtype_counts(ray_df, pandas_df):
    assert df_equals(ray_df.get_dtype_counts(), pandas_df.get_dtype_counts())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("dummy_na", bool_arg_values, ids=arg_keys("dummy_na", bool_arg_keys))
@pytest.mark.parametrize("drop_first", bool_arg_values, ids=arg_keys("drop_first", bool_arg_keys))
def test_get_dummies(ray_df, pandas_df, dummy_na, drop_first):
    result = pd.get_dummies(ray_df, dummy_na=dummy_na, drop_first=drop_first)
    expected = pandas.get_dummies(pandas_df, dummy_na=dummy_na, drop_first=drop_first)
    assert df_equals(result, expected)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_ftype_counts(ray_df, pandas_df):
    assert df_equals(ray_df.get_ftype_counts(), pandas_df.get_ftype_counts())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg(request, ray_df, pandas_df, axis, func):
    if (name_contains(request.node.name, ["over rows"]) or
            not name_contains(request.node.name, numeric_agg_funcs)):
        ray_result = ray_df.agg(func, axis)
        pandas_result = pandas_df.agg(func, axis)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.agg(func, axis)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(request, ray_df, pandas_df, func, axis):
    if (name_contains(request.node.name, ["over rows"]) or
            not name_contains(request.node.name, numeric_agg_funcs)):
        ray_result = ray_df.aggregate(func, axis)
        pandas_result = pandas_df.aggregate(func, axis)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.aggregate(func, axis)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_align(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.align(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("bool_only", bool_none_arg_values, ids=arg_keys("bool_only", bool_none_arg_keys))
def test_all(ray_df, pandas_df, axis, skipna, bool_only):
    ray_result = ray_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("bool_only", bool_none_arg_values, ids=arg_keys("bool_only", bool_none_arg_keys))
def test_any(ray_df, pandas_df, axis, skipna, bool_only):
    ray_result = ray_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    assert df_equals(ray_result, pandas_result)


def test_append():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    ray_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    frame_data2 = {"col5": [0], "col6": [1]}

    ray_df2 = pd.DataFrame(frame_data2)
    pandas_df2 = pandas.DataFrame(frame_data2)

    assert df_equals(ray_df.append(ray_df2), pandas_df.append(pandas_df2))

    with pytest.raises(ValueError):
        ray_df.append(ray_df2, verify_integrity=True)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply(request, ray_df, pandas_df, func, axis):
    if (name_contains(request.node.name, ["over rows"]) or
            not name_contains(request.node.name, numeric_agg_funcs)):
        ray_result = ray_df.apply(func, axis)
        pandas_result = pandas_df.apply(func, axis)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.apply(func, axis)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_apply_special(request, ray_df, pandas_df):
    if (name_contains(request.node.name, numeric_dfs)):
        ray_result = ray_df.apply(lambda df: -df, axis=0)
        pandas_result = pandas_df.apply(lambda df: -df, axis=0)
        assert df_equals(ray_result, pandas_result)
        ray_result = ray_df.apply(lambda df: -df, axis=1)
        pandas_result = pandas_df.apply(lambda df: -df, axis=1)
        assert df_equals(ray_result, pandas_result)
    elif "empty_data" not in request.node.name:
        key = ray_df.columns[0]
        ray_result = ray_df.apply(lambda df: df.drop(key), axis=1)
        pandas_result = pandas_df.apply(lambda df: df.drop(key), axis=1)
        assert df_equals(ray_result, pandas_result)



@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_as_blocks(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.as_blocks()


def test_as_matrix():
    test_data = TestData()
    frame = pd.DataFrame(test_data.frame)
    mat = frame.as_matrix()

    frame_columns = frame.columns
    for i, row in enumerate(mat):
        for j, value in enumerate(row):
            col = frame_columns[j]
            if np.isnan(value):
                assert np.isnan(frame[col][i])
            else:
                assert value == frame[col][i]

    # mixed type
    mat = pd.DataFrame(test_data.mixed_frame).as_matrix(["foo", "A"])
    assert mat[0, 0] == "bar"

    df = pd.DataFrame({"real": [1, 2, 3], "complex": [1j, 2j, 3j]})
    mat = df.as_matrix()
    if PY2:
        assert mat[0, 0] == 1j
    else:
        assert mat[0, 1] == 1j

    # single block corner case
    mat = pd.DataFrame(test_data.frame).as_matrix(["A", "B"])
    expected = test_data.frame.reindex(columns=["A", "B"]).values
    tm.assert_almost_equal(mat, expected)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_asfreq(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.asfreq(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_asof(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.asof(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_assign(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.assign()


def test_astype():
    td = TestData()
    ray_df = pd.DataFrame(
        td.frame.values, index=td.frame.index, columns=td.frame.columns
    )
    expected_df = pandas.DataFrame(
        td.frame.values, index=td.frame.index, columns=td.frame.columns
    )

    ray_df_casted = ray_df.astype(np.int32)
    expected_df_casted = expected_df.astype(np.int32)

    assert df_equals(ray_df_casted, expected_df_casted)

    ray_df_casted = ray_df.astype(np.float64)
    expected_df_casted = expected_df.astype(np.float64)

    assert df_equals(ray_df_casted, expected_df_casted)

    ray_df_casted = ray_df.astype(str)
    expected_df_casted = expected_df.astype(str)

    assert df_equals(ray_df_casted, expected_df_casted)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_at_time(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.at_time(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_between_time(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.between_time(None, None)


@pytest.fixture
def test_bfill():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    ray_df = pd.DataFrame(test_data.tsframe)
    assert df_equals(ray_df.bfill(), test_data.tsframe.bfill())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_blocks(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.blocks


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_bool(ray_df, pandas_df):
    with pytest.raises(ValueError):
        ray_df.bool()
        ray_df.__bool__()

    single_bool_pandas_df = pandas.DataFrame([True])
    single_bool_ray_df = pd.DataFrame([True])

    assert single_bool_pandas_df.bool() == single_bool_ray_df.bool()

    with pytest.raises(ValueError):
        # __bool__ always raises this error for DataFrames
        single_bool_ray_df.__bool__()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_boxplot(ray_df, pandas_df):
    assert ray_df.boxplot() == to_pandas(ray_df).boxplot()


@pytest.fixture
def test_clip(ray_df, pandas_df):
    # set bounds
    lower, upper = 2, 9
    lower_0 = [0, 14, 6, 1]
    upper_0 = [12, 1, 10, 7]

    # test no input
    assert ray_df_equals_pandas(ray_df.clip(), pandas_df.clip())
    # test only upper scalar bound
    assert ray_df_equals_pandas(ray_df.clip(None, lower), pandas_df.clip(None, lower))
    # test lower and upper scalar bound
    assert ray_df_equals_pandas(ray_df.clip(lower, upper), pandas_df.clip(lower, upper))
    # test lower and upper list bound on each column
    assert ray_df_equals_pandas(
        ray_df.clip(lower_0, upper_0, axis=0), pandas_df.clip(lower_0, upper_0, axis=0)
    )
    # test only upper list bound on each column
    assert ray_df_equals_pandas(
        ray_df.clip(np.nan, upper_0, axis=0), pandas_df.clip(np.nan, upper_0, axis=0)
    )


@pytest.fixture
def test_clip_lower(ray_df, pandas_df):
    # set bounds
    lower = 2
    lower_0 = [0, 14, 6, 1]

    # test lower scalar bound
    assert ray_df_equals_pandas(ray_df.clip_lower(lower), pandas_df.clip_lower(lower))
    # test lower list bound on each column
    assert ray_df_equals_pandas(
        ray_df.clip_lower(lower_0, axis=0), pandas_df.clip_lower(lower_0, axis=0)
    )


@pytest.fixture
def test_clip_upper(ray_df, pandas_df):
    # set bounds
    upper = 9
    upper_0 = [12, 1, 10, 7]

    # test upper scalar bound
    assert ray_df_equals_pandas(ray_df.clip_upper(upper), pandas_df.clip_upper(upper))
    # test upper list bound on each column
    assert ray_df_equals_pandas(
        ray_df.clip_upper(upper_0, axis=0), pandas_df.clip_upper(upper_0, axis=0)
    )


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_combine(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.combine(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_combine_first(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.combine_first(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_compound(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.compound()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_consolidate(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.consolidate()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_convert_objects(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.convert_objects()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_corr(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.corr()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_corrwith(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.corrwith(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys))
def test_count(ray_df, pandas_df, axis, numeric_only):
    ray_result = ray_df.count(axis=axis, numeric_only=numeric_only)
    pandas_result = pandas_df.count(axis=axis, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_cov(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.cov()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
def test_cummax(request, ray_df, pandas_df, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.cummax(axis=axis, skipna=skipna)
        pandas_result = pandas_df.cummax(axis=axis, skipna=skipna)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.cummax(axis=axis, skipna=skipna)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
def test_cummin(request, ray_df, pandas_df, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.cummin(axis=axis, skipna=skipna)
        pandas_result = pandas_df.cummin(axis=axis, skipna=skipna)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.cummin(axis=axis, skipna=skipna)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
def test_cumprod(request, ray_df, pandas_df, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.cumprod(axis=axis, skipna=skipna)
        pandas_result = pandas_df.cumprod(axis=axis, skipna=skipna)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.cumprod(axis=axis, skipna=skipna)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
def test_cumsum(request, ray_df, pandas_df, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.cumsum(axis=axis, skipna=skipna)
        pandas_result = pandas_df.cumsum(axis=axis, skipna=skipna)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_result = ray_df.cumsum(axis=axis, skipna=skipna)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_describe(ray_df, pandas_df):
    assert df_equals(ray_df.describe(), pandas_df.describe())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("periods", int_arg_values, ids=arg_keys("periods", int_arg_keys))
def test_diff(request,ray_df, pandas_df, axis, periods):
    if name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.diff(axis=axis, periods=periods)
        pandas_result = pandas_df.diff(axis=axis, periods=periods)
        assert df_equals(ray_result, pandas_result)


def test_drop():
    frame_data = {"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]}
    simple = pandas.DataFrame(frame_data)
    ray_simple = pd.DataFrame(frame_data)
    assert df_equals(ray_simple.drop("A", axis=1), simple[["B"]])
    assert df_equals(ray_simple.drop(["A", "B"], axis="columns"), simple[[]])
    assert df_equals(ray_simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
    assert df_equals(
        ray_simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :]
    )

    pytest.raises(ValueError, ray_simple.drop, 5)
    pytest.raises(ValueError, ray_simple.drop, "C", 1)
    pytest.raises(ValueError, ray_simple.drop, [1, 5])
    pytest.raises(ValueError, ray_simple.drop, ["A", "C"], 1)

    # errors = 'ignore'
    assert df_equals(ray_simple.drop(5, errors="ignore"), simple)
    assert df_equals(
        ray_simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :]
    )
    assert df_equals(ray_simple.drop("C", axis=1, errors="ignore"), simple)
    assert df_equals(
        ray_simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]]
    )

    # non-unique
    nu_df = pandas.DataFrame(
        pandas.compat.lzip(range(3), range(-3, 1), list("abc")), columns=["a", "a", "b"]
    )
    ray_nu_df = pd.DataFrame(nu_df)
    assert df_equals(ray_nu_df.drop("a", axis=1), nu_df[["b"]])
    assert df_equals(ray_nu_df.drop("b", axis="columns"), nu_df["a"])
    assert df_equals(ray_nu_df.drop([]), nu_df)

    nu_df = nu_df.set_index(pandas.Index(["X", "Y", "X"]))
    nu_df.columns = list("abc")
    ray_nu_df = pd.DataFrame(nu_df)
    assert df_equals(ray_nu_df.drop("X", axis="rows"), nu_df.loc[["Y"], :])
    assert df_equals(ray_nu_df.drop(["X", "Y"], axis=0), nu_df.loc[[], :])

    # inplace cache issue
    frame_data = np.random.randn(10, 3)
    df = pandas.DataFrame(frame_data, columns=list("abc"))
    ray_df = pd.DataFrame(frame_data, columns=list("abc"))
    expected = df[~(df.b > 0)]
    ray_df.drop(labels=df[df.b > 0].index, inplace=True)
    assert df_equals(ray_df, expected)


def test_drop_api_equivalence():
    # equivalence of the labels/axis and index/columns API's
    frame_data = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

    ray_df = pd.DataFrame(frame_data, index=["a", "b", "c"], columns=["d", "e", "f"])

    ray_df1 = ray_df.drop("a")
    ray_df2 = ray_df.drop(index="a")
    assert df_equals(ray_df1, ray_df2)

    ray_df1 = ray_df.drop("d", 1)
    ray_df2 = ray_df.drop(columns="d")
    assert df_equals(ray_df1, ray_df2)

    ray_df1 = ray_df.drop(labels="e", axis=1)
    ray_df2 = ray_df.drop(columns="e")
    assert df_equals(ray_df1, ray_df2)

    ray_df1 = ray_df.drop(["a"], axis=0)
    ray_df2 = ray_df.drop(index=["a"])
    assert df_equals(ray_df1, ray_df2)

    ray_df1 = ray_df.drop(["a"], axis=0).drop(["d"], axis=1)
    ray_df2 = ray_df.drop(index=["a"], columns=["d"])
    assert df_equals(ray_df1, ray_df2)

    with pytest.raises(ValueError):
        ray_df.drop(labels="a", index="b")

    with pytest.raises(ValueError):
        ray_df.drop(labels="a", columns="b")

    with pytest.raises(ValueError):
        ray_df.drop(axis=1)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_drop_duplicates(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.drop_duplicates()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
@pytest.mark.parametrize("inplace", bool_arg_values, ids=arg_keys("inplace", bool_arg_keys))
def test_dropna(ray_df, pandas_df, axis, how, inplace):
    pandas_result = pandas_df.dropna(axis=axis, how=how, inplace=inplace)
    ray_result = ray_df.dropna(axis=axis, how=how, inplace=inplace)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_inplace(ray_df, pandas_df):
    pandas_result = pandas_df.dropna()
    ray_df.dropna(inplace=True)
    assert df_equals(ray_df, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_multiple_axes(ray_df, pandas_df):
    assert df_equals(ray_df.dropna(how="all", axis=[0, 1]), pandas_df.dropna(how="all", axis=[0, 1]))
    assert df_equals(ray_df.dropna(how="all", axis=(0, 1)), pandas_df.dropna(how="all", axis=(0, 1)))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_multiple_axes_inplace(ray_df, pandas_df):
    ray_df_copy = ray_df.copy()
    pd_df_copy = pandas_df.copy()

    ray_df_copy.dropna(how="all", axis=[0, 1], inplace=True)
    pandas_df_copy.dropna(how="all", axis=[0, 1], inplace=True)

    assert df_equals(ray_df_copy, pandas_df_copy)

    ray_df_copy = ray_df.copy()
    pandas_df_copy = pandas_df.copy()

    ray_df_copy.dropna(how="all", axis=(0, 1), inplace=True)
    pandas_df_copy.dropna(how="all", axis=(0, 1), inplace=True)

    assert df_equals(ray_df_copy, pandas_df_copy)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_subset(request, ray_df, pandas_df):
    if not name_contains(request.node.name, ['empty_data']):
        column_subset = ray_df.columns[0:2]
        assert df_equals(ray_df.dropna(how="all", subset=column_subset), pandas_df.dropna(how="all", subset=column_subset))
        assert df_equals(ray_df.dropna(how="any", subset=column_subset), pandas_df.dropna(how="any", subset=column_subset))

        row_subset = ray_df.index[0:2]
        assert df_equals(ray_df.dropna(how="all", axis=1, subset=row_subset), pandas_df.dropna(how="all", axis=1, subset=row_subset))
        assert df_equals(ray_df.dropna(how="any", axis=1, subset=row_subset), pandas_df.dropna(how="any", axis=1, subset=row_subset))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_subset_error(ray_df, pandas_df):
    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    with pytest.raises(KeyError):
        ray_df.dropna(subset=list("EF"))

    with pytest.raises(KeyError):
        ray_df.dropna(axis=1, subset=[4, 5])


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dot(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.dot(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_duplicated(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.duplicated()


def test_empty_df():
    df = pd.DataFrame(index=["a", "b"])
    dt_is_empty(df)
    tm.assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    dt_is_empty(df)
    assert len(df.index) == 0
    tm.assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    dt_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0

    df = pd.DataFrame(index=["a", "b"])
    dt_is_empty(df)
    tm.assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    dt_is_empty(df)
    assert len(df.index) == 0
    tm.assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    dt_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0


def test_equals():
    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 4, 1]}
    ray_df1 = pd.DataFrame(frame_data)
    ray_df2 = pd.DataFrame(frame_data)

    assert df_equals(ray_df1, ray_df2)

    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 5, 1]}
    ray_df3 = pd.DataFrame(frame_data)

    assert not df_equals(ray_df3, ray_df1)
    assert not df_equals(ray_df3, ray_df2)


def test_eval_df_use_case():
    frame_data = {"a": np.random.randn(10), "b": np.random.randn(10)}
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)

    # test eval for series results
    tmp_pandas = df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_ray = ray_df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")

    assert isinstance(tmp_ray, pandas.Series)
    assert df_equals(tmp_ray, tmp_pandas)

    # Test not inplace assignments
    tmp_pandas = df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_ray = ray_df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas")
    assert df_equals(tmp_ray, tmp_pandas)

    # Test inplace assignments
    df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True)
    ray_df.eval(
        "e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True
    )
    # TODO: Use a series equality validator.
    assert df_equals(ray_df, df)


def test_eval_df_arithmetic_subexpression():
    frame_data = {"a": np.random.randn(10), "b": np.random.randn(10)}
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
    ray_df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
    # TODO: Use a series equality validator.
    assert df_equals(ray_df, df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ewm(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.ewm()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_expanding(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.expanding()


def test_ffill():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    ray_df = pd.DataFrame(test_data.tsframe)

    assert df_equals(ray_df.ffill(), test_data.tsframe.ffill())


def test_fillna_sanity():
    test_data = TestData()
    tf = test_data.tsframe
    tf.loc[tf.index[:5], "A"] = np.nan
    tf.loc[tf.index[-5:], "A"] = np.nan

    zero_filled = test_data.tsframe.fillna(0)
    ray_df = pd.DataFrame(test_data.tsframe).fillna(0)
    assert df_equals(ray_df, zero_filled)

    padded = test_data.tsframe.fillna(method="pad")
    ray_df = pd.DataFrame(test_data.tsframe).fillna(method="pad")
    assert df_equals(ray_df, padded)

    # mixed type
    mf = test_data.mixed_frame
    mf.loc[mf.index[5:20], "foo"] = np.nan
    mf.loc[mf.index[-10:], "A"] = np.nan

    result = test_data.mixed_frame.fillna(value=0)
    ray_df = pd.DataFrame(test_data.mixed_frame).fillna(value=0)
    assert df_equals(ray_df, result)

    result = test_data.mixed_frame.fillna(method="pad")
    ray_df = pd.DataFrame(test_data.mixed_frame).fillna(method="pad")
    assert df_equals(ray_df, result)

    pytest.raises(ValueError, test_data.tsframe.fillna)
    pytest.raises(ValueError, pd.DataFrame(test_data.tsframe).fillna)
    with pytest.raises(ValueError):
        pd.DataFrame(test_data.tsframe).fillna(5, method="ffill")

    # mixed numeric (but no float16)
    mf = test_data.mixed_float.reindex(columns=["A", "B", "D"])
    mf.loc[mf.index[-10:], "A"] = np.nan
    result = mf.fillna(value=0)
    ray_df = pd.DataFrame(mf).fillna(value=0)
    assert df_equals(ray_df, result)

    result = mf.fillna(method="pad")
    ray_df = pd.DataFrame(mf).fillna(method="pad")
    assert df_equals(ray_df, result)

    # TODO: Use this when Arrow issue resolves:
    # (https://issues.apache.org/jira/browse/ARROW-2122)
    # empty frame
    # df = DataFrame(columns=['x'])
    # for m in ['pad', 'backfill']:
    #     df.x.fillna(method=m, inplace=True)
    #     df.x.fillna(method=m)

    # with different dtype
    frame_data = [
        ["a", "a", np.nan, "a"],
        ["b", "b", np.nan, "b"],
        ["c", "c", np.nan, "c"],
    ]
    df = pandas.DataFrame(frame_data)

    result = df.fillna({2: "foo"})
    ray_df = pd.DataFrame(frame_data).fillna({2: "foo"})

    assert df_equals(ray_df, result)

    ray_df = pd.DataFrame(df)
    df.fillna({2: "foo"}, inplace=True)
    ray_df.fillna({2: "foo"}, inplace=True)
    assert df_equals(ray_df, result)

    frame_data = {
        "Date": [pandas.NaT, pandas.Timestamp("2014-1-1")],
        "Date2": [pandas.Timestamp("2013-1-1"), pandas.NaT],
    }
    df = pandas.DataFrame(frame_data)
    result = df.fillna(value={"Date": df["Date2"]})
    ray_df = pd.DataFrame(frame_data).fillna(value={"Date": df["Date2"]})
    assert df_equals(ray_df, result)

    # TODO: Use this when Arrow issue resolves:
    # (https://issues.apache.org/jira/browse/ARROW-2122)
    # with timezone
    """
    frame_data = {'A': [pandas.Timestamp('2012-11-11 00:00:00+01:00'),
                        pandas.NaT]}
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    assert df_equals(ray_df.fillna(method='pad'), df.fillna(method='pad'))

    frame_data = {'A': [pandas.NaT,
                        pandas.Timestamp('2012-11-11 00:00:00+01:00')]}
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data).fillna(method='bfill')
    assert df_equals(ray_df, df.fillna(method='bfill'))
    """


def test_fillna_downcast():
    # infer int64 from float64
    frame_data = {"a": [1.0, np.nan]}
    df = pandas.DataFrame(frame_data)
    result = df.fillna(0, downcast="infer")
    ray_df = pd.DataFrame(frame_data).fillna(0, downcast="infer")
    assert df_equals(ray_df, result)

    # infer int64 from float64 when fillna value is a dict
    df = pandas.DataFrame(frame_data)
    result = df.fillna({"a": 0}, downcast="infer")
    ray_df = pd.DataFrame(frame_data).fillna({"a": 0}, downcast="infer")
    assert df_equals(ray_df, result)


def test_ffill2():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    ray_df = pd.DataFrame(test_data.tsframe)
    assert df_equals(ray_df.fillna(method="ffill"), test_data.tsframe.fillna(method="ffill"))


def test_bfill2():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    ray_df = pd.DataFrame(test_data.tsframe)
    assert df_equals(ray_df.fillna(method="bfill"), test_data.tsframe.fillna(method="bfill"))


def test_fillna_inplace():
    frame_data = np.random.randn(10, 4)
    df = pandas.DataFrame(frame_data)
    df[1][:4] = np.nan
    df[3][-4:] = np.nan

    ray_df = pd.DataFrame(df)
    df.fillna(value=0, inplace=True)
    assert not df_equals(ray_df, df)

    ray_df.fillna(value=0, inplace=True)
    assert df_equals(ray_df, df)

    ray_df = pd.DataFrame(df).fillna(value={0: 0}, inplace=True)
    assert ray_df is None

    df[1][:4] = np.nan
    df[3][-4:] = np.nan
    ray_df = pd.DataFrame(df)
    df.fillna(method="ffill", inplace=True)

    assert not df_equals(ray_df, df)

    ray_df.fillna(method="ffill", inplace=True)
    assert df_equals(ray_df, df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_frame_fillna_limit(ray_df, pandas_df):
    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    ray_df = pd.DataFrame(result)
    assert df_equals(ray_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2))

    result = pandas_df[-2:].reindex(index)
    ray_df = pd.DataFrame(result)
    assert df_equals(ray_df.fillna(method="backfill", limit=2), result.fillna(method="backfill", limit=2))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_frame_pad_backfill_limit(ray_df, pandas_df):
    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    ray_df = pd.DataFrame(result)
    assert df_equals(ray_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2))

    result = pandas_df[-2:].reindex(index)
    ray_df = pd.DataFrame(result)
    assert df_equals(ray_df.fillna(method="backfill", limit=2), result.fillna(method="backfill", limit=2))


def test_fillna_dtype_conversion():
    # make sure that fillna on an empty frame works
    df = pandas.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
    ray_df = pd.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
    assert df_equals(ray_df.fillna("nan"), df.fillna("nan"))

    frame_data = {"A": [1, np.nan], "B": [1.0, 2.0]}
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    for v in ["", 1, np.nan, 1.0]:
        assert df_equals(ray_df.fillna(v), df.fillna(v))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_skip_certain_blocks(ray_df, pandas_df):
    # don't try to fill boolean, int blocks
    assert df_equals(ray_df.fillna(np.nan), pandas_df.fillna(np.nan))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_dict_series(request, ray_df, pandas_df):
    # Only test if nonempty data because no column names for empty dataframe
    if "empty_data" not in request.node.name:
        #if request.node.name != 
        col1 = ray_df.columns[0]
        col2 = ray_df.columns[-1]
        col3 = ray_df.columns[int(ray_df.shape[1]/2)]

        assert df_equals(ray_df.fillna({col1: 0, col2: 5}), pandas_df.fillna({col1: 0, col2: 5}))

        assert df_equals(ray_df.fillna({col1: 0, col2: 5, col3: 7}), pandas_df.fillna({col1: 0, col2: 5, col3: 7}))

        with pytest.raises(NotImplementedError):
            # Series treated same as dict
            assert df_equals(ray_df.fillna(pandas_df.max()), pandas_df.fillna(pandas_df.max()))


def test_fillna_dataframe():
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data, index=list("VWXYZ"))
    ray_df = pd.DataFrame(frame_data, index=list("VWXYZ"))

    # df2 may have different index and columns
    df2 = pandas.DataFrame(
        {"a": [np.nan, 10, 20, 30, 40], "b": [50, 60, 70, 80, 90], "foo": ["bar"] * 5},
        index=list("VWXuZ"),
    )

    # only those columns and indices which are shared get filled
    with pytest.raises(NotImplementedError):
        assert df_equals(ray_df.fillna(df2), df.fillna(df2))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_columns(ray_df, pandas_df):
    assert df_equals(ray_df.fillna(method="ffill", axis=1), pandas_df.fillna(method="ffill", axis=1))

    assert df_equals(ray_df.fillna(method="ffill", axis=1), pandas_df.fillna(method="ffill", axis=1))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_invalid_method(ray_df, pandas_df):
    with tm.assert_raises_regex(ValueError, "ffil"):
        ray_df.fillna(method="ffil")


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_invalid_value(request, ray_df, pandas_df):
    print(request.node.name)
    # list
    pytest.raises(TypeError, ray_df.fillna, [1, 2])
    # tuple
    pytest.raises(TypeError, ray_df.fillna, (1, 2))

    # frame with series
    # empty dataframe should have index error as there are no columns
    if "empty_data" in request.node.name:
        pytest.raises(ValueError, ray_df.iloc[:, 0].fillna, ray_df)
    else:
        with pytest.raises(IndexError):
            ray_df.iloc[:, 0].fillna(ray_df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_col_reordering(ray_df, pandas_df):
    assert df_equals(ray_df.fillna(method="ffill"), pandas_df.fillna(method="ffill"))


"""
TODO: Use this when Arrow issue resolves:
(https://issues.apache.org/jira/browse/ARROW-2122)
@pytest.fixture
def test_fillna_datetime_columns():
    frame_data = {'A': [-1, -2, np.nan],
                  'B': date_range('20130101', periods=3),
                  'C': ['foo', 'bar', None],
                  'D': ['foo2', 'bar2', None]}
    df = pandas.DataFrame(frame_data, index=date_range('20130110', periods=3))
    ray_df = pd.DataFrame(frame_data, index=date_range('20130110', periods=3))
    assert df_equals(ray_df.fillna('?'), df.fillna('?'))

    frame_data = {'A': [-1, -2, np.nan],
                  'B': [pandas.Timestamp('2013-01-01'),
                        pandas.Timestamp('2013-01-02'), pandas.NaT],
                  'C': ['foo', 'bar', None],
                  'D': ['foo2', 'bar2', None]}
    df = pandas.DataFrame(frame_data, index=date_range('20130110', periods=3))
    ray_df = pd.DataFrame(frame_data, index=date_range('20130110', periods=3))
    assert df_equals(ray_df.fillna('?'), df.fillna('?'))
"""


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_filter(ray_df, pandas_df):
    by = {"items": ["col1", "col5"], "regex": "4$|3$", "like": "col"}
    assert df_equals(ray_df.filter(items=by["items"]), pandas_df.filter(items=by["items"]))

    assert df_equals(ray_df.filter(regex=by["regex"], axis=0), pandas_df.filter(regex=by["regex"], axis=0))
    assert df_equals(ray_df.filter(regex=by["regex"], axis=1), pandas_df.filter(regex=by["regex"], axis=1))

    assert df_equals(ray_df.filter(like=by["like"]), pandas_df.filter(like=by["like"]))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_first(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.first(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_first_valid_index(ray_df, pandas_df):
    assert ray_df.first_valid_index() == (pandas_df.first_valid_index())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_csv(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_csv(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_dict(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_dict(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_items(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_items(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_records(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_records(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_value(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.get_value(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_values(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.get_values()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(ray_df, pandas_df, n):
    assert df_equals(ray_df.head(n), pandas_df.head(n))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_hist(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.hist(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iat(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.iat()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
def test_idxmax(ray_df, pandas_df, axis, skipna):
    ray_result = ray_df.all(axis=axis, skipna=skipna)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
def test_idxmin(ray_df, pandas_df, axis, skipna):
    ray_result = ray_df.all(axis=axis, skipna=skipna)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_infer_objects(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.infer_objects()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iloc(request, ray_df, pandas_df):
    if not name_contains(request.node.name, ["empty_data"]):
        # Scaler
        assert ray_df.iloc[0, 1] == pandas_df.iloc[0, 1]

        # Series
        assert df_equals(ray_df.iloc[0], pandas_df.iloc[0])
        assert df_equals(ray_df.iloc[1:, 0], pandas_df.iloc[1:, 0])
        assert df_equals(ray_df.iloc[1:2, 0], pandas_df.iloc[1:2, 0])

        # DataFrame
        assert df_equals(ray_df.iloc[[1, 2]], pandas_df.iloc[[1, 2]])
        # See issue #80
        # assert df_equals(ray_df.iloc[[1, 2], [1, 0]], pandas_df.iloc[[1, 2], [1, 0]])
        assert df_equals(ray_df.iloc[1:2, 0:2], pandas_df.iloc[1:2, 0:2])

        # Issue #43
        ray_df.iloc[0:3, :]

        # Write Item
        ray_df.iloc[[1, 2]] = 42
        pandas_df.iloc[[1, 2]] = 42
        assert df_equals(ray_df, pandas_df)
    else:
        with pytest.raises(IndexError):
            ray_df.iloc[0, 1]


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_index(ray_df, pandas_df):
    assert df_equals(ray_df.index, pandas_df.index)
    ray_df_cp = ray_df.copy()
    pandas_df_cp = pandas_df.copy()

    ray_df_cp.index = [str(i) for i in ray_df_cp.index]
    pandas_df_cp.index = [str(i) for i in pandas_df_cp.index]
    assert df_equals(ray_df_cp.index, pandas_df_cp.index)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_info(request, ray_df, pandas_df):
    # Test to make sure that it does not crash
    ray_df.info(memory_usage="deep")

    if not name_contains(request.node.name, ["empty_data"]):
        with io.StringIO() as buf:
            ray_df.info(buf=buf)
            info_string = buf.getvalue()
            assert "<class 'modin.pandas.dataframe.DataFrame'>\n" in info_string
            assert "memory usage: " in info_string
            assert "Data columns (total {} columns):".format(ray_df.shape[1]) in info_string

        with io.StringIO() as buf:
            ray_df.info(buf=buf, verbose=False, memory_usage=False)
            info_string = buf.getvalue()
            assert "memory usage: " not in info_string
            assert "Columns: {0} entries, {1} to {2}".format(ray_df.shape[1], ray_df.columns[0], ray_df.columns[-1]) in info_string


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("loc", int_arg_values, ids=arg_keys("loc", int_arg_keys))
def test_insert(ray_df, pandas_df, loc):
    ray_df = ray_df.copy()
    pandas_df = pandas_df.copy()
    loc %= ray_df.shape[1] + 1
    column = "New Column"
    key = loc if loc < ray_df.shape[1] else loc-1
    value = ray_df.iloc[:, key]
    ray_df.insert(loc, column, value)
    pandas_df.insert(loc, column, value)
    assert df_equals(ray_df, pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_interpolate(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.interpolate()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_is_copy(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.is_copy


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_items(ray_df, pandas_df):
    ray_items = ray_df.items()
    pandas_items = pandas_df.items()
    for ray_item, pandas_item in zip(ray_items, pandas_items):
        ray_index, ray_series = ray_item
        pandas_index, pandas_series = pandas_item
        assert df_equals(pandas_series, ray_series)
        assert pandas_index == ray_index


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iteritems(ray_df, pandas_df):
    ray_items = ray_df.iteritems()
    pandas_items = pandas_df.iteritems()
    for ray_item, pandas_item in zip(ray_items, pandas_items):
        ray_index, ray_series = ray_item
        pandas_index, pandas_series = pandas_item
        assert df_equals(pandas_series, ray_series)
        assert pandas_index == ray_index


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iterrows(ray_df, pandas_df):
    ray_iterrows = ray_df.iterrows()
    pandas_iterrows = pandas_df.iterrows()
    for ray_row, pandas_row in zip(ray_iterrows, pandas_iterrows):
        ray_index, ray_series = ray_row
        pandas_index, pandas_series = pandas_row
        assert df_equals(pandas_series, ray_series)
        assert pandas_index == ray_index


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_itertuples(ray_df, pandas_df):
    # test default
    ray_it_default = ray_df.itertuples()
    pandas_it_default = pandas_df.itertuples()
    for ray_row, pandas_row in zip(ray_it_default, pandas_it_default):
        np.testing.assert_equal(ray_row, pandas_row)

    # test all combinations of custom params
    indices = [True, False]
    names = [None, "NotPandas", "Pandas"]

    for index in indices:
        for name in names:
            ray_it_custom = ray_df.itertuples(index=index, name=name)
            pandas_it_custom = pandas_df.itertuples(index=index, name=name)
            for ray_row, pandas_row in zip(ray_it_custom, pandas_it_custom):
                np.testing.assert_equal(ray_row, pandas_row)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ix(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.ix()


def test_join():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    ray_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    frame_data2 = {"col5": [0], "col6": [1]}
    ray_df2 = pd.DataFrame(frame_data2)
    pandas_df2 = pandas.DataFrame(frame_data2)

    join_types = ["left", "right", "outer", "inner"]
    for how in join_types:
        ray_join = ray_df.join(ray_df2, how=how)
        pandas_join = pandas_df.join(pandas_df2, how=how)
        assert df_equals(ray_join, pandas_join)

    frame_data3 = {"col7": [1, 2, 3, 5, 6, 7, 8]}

    ray_df3 = pd.DataFrame(frame_data3)
    pandas_df3 = pandas.DataFrame(frame_data3)

    join_types = ["left", "outer", "inner"]
    for how in join_types:
        ray_join = ray_df.join([ray_df2, ray_df3], how=how)
        pandas_join = pandas_df.join([pandas_df2, pandas_df3], how=how)
        assert df_equals(ray_join, pandas_join)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_keys(ray_df, pandas_df):
    assert df_equals(ray_df.keys(), pandas_df.keys())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_kurt(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.kurt()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_kurtosis(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.kurtosis()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_last(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.last(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_last_valid_index(ray_df, pandas_df):
    assert ray_df.last_valid_index() == (pandas_df.last_valid_index())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_loc(request, ray_df, pandas_df):
    if "empty_data" not in request.node.name:
        key1 = ray_df.columns[0]
        key2 = ray_df.columns[1]
        # Scaler
        assert ray_df.loc[0, key1] == pandas_df.loc[0, key1]

        # Series
        assert df_equals(ray_df.loc[0], pandas_df.loc[0])
        assert df_equals(ray_df.loc[1:, key1], pandas_df.loc[1:, key1])
        assert df_equals(ray_df.loc[1:2, key1], pandas_df.loc[1:2, key1])

        # DataFrame
        assert df_equals(ray_df.loc[[1, 2]], pandas_df.loc[[1, 2]])

        # See issue #80
        # assert df_equals(ray_df.loc[[1, 2], ['col1']], pandas_df.loc[[1, 2], ['col1']])
        assert df_equals(ray_df.loc[1:2, key1:key2], pandas_df.loc[1:2, key1:key2])

        # Write Item
        ray_df_copy = ray_df.copy()
        pandas_df_copy = pandas_df.copy()
        ray_df_copy.loc[[1, 2]] = 42
        pandas_df_copy.loc[[1, 2]] = 42
        assert df_equals(ray_df_copy, pandas_df_copy)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_lookup(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.lookup(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_mad(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.mad()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_mask(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.mask(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys))
def test_max(ray_df, pandas_df, axis, skipna, numeric_only):
    ray_result = ray_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
    pandas_result = pandas_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
def test_mean(ray_df, pandas_df, axis, skipna, numeric_only):
    ray_result = ray_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
    pandas_result = pandas_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
def test_median(ray_df, pandas_df, axis, skipna, numeric_only):
    ray_result = ray_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
    pandas_result = pandas_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)

@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_melt(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.melt()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_memory_usage(ray_df, pandas_df):
    assert ray_df.memory_usage(index=True).at["Index"] is not None
    assert ray_df.memory_usage(deep=True).sum() >= ray_df.memory_usage(deep=False).sum()


def test_merge():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    ray_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    frame_data2 = {"col1": [0, 1, 2], "col2": [1, 5, 6]}
    ray_df2 = pd.DataFrame(frame_data2)
    pandas_df2 = pandas.DataFrame(frame_data2)

    join_types = ["outer", "inner"]
    for how in join_types:
        with pytest.raises(NotImplementedError):
            # Defaults
            ray_result = ray_df.merge(ray_df2, how=how)
            pandas_result = pandas_df.merge(pandas_df2, how=how)
            df_equals(ray_result, pandas_result)

            # left_on and right_index
            ray_result = ray_df.merge(
                ray_df2, how=how, left_on="col1", right_index=True
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_on="col1", right_index=True
            )
            df_equals(ray_result, pandas_result)

            # left_index and right_on
            ray_result = ray_df.merge(
                ray_df2, how=how, left_index=True, right_on="col1"
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_index=True, right_on="col1"
            )
            df_equals(ray_result, pandas_result)

            # left_on and right_on col1
            ray_result = ray_df.merge(ray_df2, how=how, left_on="col1", right_on="col1")
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_on="col1", right_on="col1"
            )
            df_equals(ray_result, pandas_result)

            # left_on and right_on col2
            ray_result = ray_df.merge(ray_df2, how=how, left_on="col2", right_on="col2")
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_on="col2", right_on="col2"
            )
            df_equals(ray_result, pandas_result)

        # left_index and right_index
        ray_result = ray_df.merge(ray_df2, how=how, left_index=True, right_index=True)
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_index=True, right_index=True
        )
        df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
def test_min(ray_df, pandas_df, axis, skipna, numeric_only):
    ray_result = ray_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
    pandas_result = pandas_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys))
def test_mode(ray_df, pandas_df, axis, skipna, numeric_only):
    ray_result = ray_df.mode(axis=axis, skipna=skipna, numeric_only=numeric_only)
    pandas_result = pandas_df.mode(axis=axis, skipna=skipna, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ndim(ray_df, pandas_df):
    assert ray_df.ndim == pandas_df.ndim


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_nlargest(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.nlargest(None, None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_notna(ray_df, pandas_df):
    assert df_equals(ray_df.notna(), pandas_df.notna())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_notnull(ray_df, pandas_df):
    assert df_equals(ray_df.notnull(), pandas_df.notnull())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_nsmallest(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.nsmallest(None, None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("dropna", bool_arg_values, ids=arg_keys("dropna", bool_arg_keys))
def test_nunique(ray_df, pandas_df, axis, dropna):
    ray_result = ray_df.nunique(axis=axis, dropna=dropna)
    pandas_result = pandas_df.nunique(axis=axis, dropna=dropna)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pct_change(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.pct_change()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pipe(ray_df, pandas_df):
    n = len(ray_df.index)
    a, b, c = 2 % n, 0, 3 % n
    col = ray_df.columns[3 % len(ray_df.columns)]

    def h(x):
        return x.drop(columns=[col])

    def g(x, arg1=0):
        for _ in range(arg1):
            x = x.append(x)
        return x

    def f(x, arg2=0, arg3=0):
        return x.drop([arg2, arg3])

    assert df_equals(f(g(h(ray_df), arg1=a), arg2=b, arg3=c), (ray_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)))

    assert df_equals((ray_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)), (pandas_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pivot(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.pivot()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pivot_table(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.pivot_table()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_plot(ray_df, pandas_df):
    # We have to test this way because equality in plots means same object.
    zipped_plot_lines = zip(ray_df.plot().lines, to_pandas(ray_df).plot().lines)

    for l, r in zipped_plot_lines:
        assert np.array_equal(l.get_xdata(), r.get_xdata())
        assert np.array_equal(l.get_ydata(), r.get_ydata())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pop(request, ray_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = ray_df.columns[0]
        temp_ray_df = ray_df.copy()
        temp_pandas_df = pandas_df.copy()
        ray_popped = temp_ray_df.pop(key)
        pandas_popped = temp_pandas_df.pop(key)
        assert df_equals(ray_popped, pandas_popped)
        assert df_equals(temp_ray_df, temp_pandas_df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys))
@pytest.mark.parametrize("min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys))
def test_prod(request, ray_df, pandas_df, axis, skipna, numeric_only, min_count):
    if numeric_only or name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
        pandas_result = pandas_df.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_df.prod(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys))
@pytest.mark.parametrize("min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys))
def test_product(request, ray_df, pandas_df, axis, skipna, numeric_only, min_count):
    if numeric_only or name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.product(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
        pandas_result = pandas_df.product(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_df.product(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(ray_df, pandas_df, q):
    assert df_equals(ray_df.quantile(q), pandas_df.quantile(q))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("funcs", query_func_values, ids=query_func_keys)
def test_query(request, ray_df, pandas_df, funcs):
    if name_contains(request.node.name, numeric_dfs) and "empty_data" not in request.node.name:
        ray_result = ray_df.query(funcs)
        pandas_result = pandas_df.query(funcs)
        assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("method", ['average', 'min', 'max', 'first', 'dense'], ids=['average', 'min', 'max', 'first', 'dense'])
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
@pytest.mark.parametrize("na_option", ['keep', 'top', 'bottom'], ids=['keep', 'top', 'bottom'])
@pytest.mark.parametrize("ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys))
@pytest.mark.parametrize("pct", bool_arg_values, ids=arg_keys("pct", bool_arg_keys))
def test_rank(ray_df, pandas_df, axis, method, numeric_only, na_option, ascending, pct):
    ray_result = ray_df.rank(axis=axis, method=method, numeric_only=numeric_only, na_option=na_option, ascending=ascending, pct=pct)
    pandas_result = pandas_df.rank(axis=axis, method=method, numeric_only=numeric_only, na_option=na_option, ascending=ascending, pct=pct)
    assert df_equals(ray_result, pandas_result)


def test_reindex():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 10, 11],
        "col4": [12, 13, 14, 15],
        "col5": [0, 0, 0, 0],
    }
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)

    assert df_equals(ray_df.reindex([0, 3, 2, 1]), pandas_df.reindex([0, 3, 2, 1]))

    assert df_equals(ray_df.reindex([0, 6, 2]), pandas_df.reindex([0, 6, 2]))

    assert df_equals(ray_df.reindex(["col1", "col3", "col4", "col2"], axis=1), pandas_df.reindex(["col1", "col3", "col4", "col2"], axis=1))

    assert df_equals(ray_df.reindex(["col1", "col7", "col4", "col8"], axis=1), pandas_df.reindex(["col1", "col7", "col4", "col8"], axis=1))

    assert df_equals(ray_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]), pandas_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reindex_axis(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.reindex_axis(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reindex_like(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.reindex_like(None)


def test_rename_sanity():
    test_data = TestData()
    mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}

    ray_df = pd.DataFrame(test_data.frame)
    assert df_equals(ray_df.rename(columns=mapping), test_data.frame.rename(columns=mapping))

    renamed2 = test_data.frame.rename(columns=str.lower)
    assert df_equals(ray_df.rename(columns=str.lower), renamed2)

    ray_df = pd.DataFrame(renamed2)
    assert df_equals(ray_df.rename(columns=str.upper), renamed2.rename(columns=str.upper))

    # index
    data = {"A": {"foo": 0, "bar": 1}}

    # gets sorted alphabetical
    df = pandas.DataFrame(data)
    ray_df = pd.DataFrame(data)
    tm.assert_index_equal(
        ray_df.rename(index={"foo": "bar", "bar": "foo"}).index,
        df.rename(index={"foo": "bar", "bar": "foo"}).index,
    )

    tm.assert_index_equal(
        ray_df.rename(index=str.upper).index, df.rename(index=str.upper).index
    )

    # have to pass something
    pytest.raises(TypeError, ray_df.rename)

    # partial columns
    renamed = test_data.frame.rename(columns={"C": "foo", "D": "bar"})
    ray_df = pd.DataFrame(test_data.frame)
    tm.assert_index_equal(
        ray_df.rename(columns={"C": "foo", "D": "bar"}).index,
        test_data.frame.rename(columns={"C": "foo", "D": "bar"}).index,
    )

    # TODO: Uncomment when transpose works
    # other axis
    # renamed = test_data.frame.T.rename(index={'C': 'foo', 'D': 'bar'})
    # tm.assert_index_equal(
    #     test_data.frame.T.rename(index={'C': 'foo', 'D': 'bar'}).index,
    #     ray_df.T.rename(index={'C': 'foo', 'D': 'bar'}).index)

    # index with name
    index = pandas.Index(["foo", "bar"], name="name")
    renamer = pandas.DataFrame(data, index=index)
    ray_df = pd.DataFrame(data, index=index)

    renamed = renamer.rename(index={"foo": "bar", "bar": "foo"})
    ray_renamed = ray_df.rename(index={"foo": "bar", "bar": "foo"})
    tm.assert_index_equal(renamed.index, ray_renamed.index)

    assert renamed.index.name == ray_renamed.index.name


def test_rename_multiindex():
    tuples_index = [("foo1", "bar1"), ("foo2", "bar2")]
    tuples_columns = [("fizz1", "buzz1"), ("fizz2", "buzz2")]
    index = pandas.MultiIndex.from_tuples(tuples_index, names=["foo", "bar"])
    columns = pandas.MultiIndex.from_tuples(tuples_columns, names=["fizz", "buzz"])

    frame_data = [(0, 0), (1, 1)]
    df = pandas.DataFrame(frame_data, index=index, columns=columns)
    ray_df = pd.DataFrame(frame_data, index=index, columns=columns)

    #
    # without specifying level -> accross all levels
    renamed = df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    ray_renamed = ray_df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    tm.assert_index_equal(renamed.index, ray_renamed.index)

    renamed = df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)
    assert renamed.index.names == ray_renamed.index.names
    assert renamed.columns.names == ray_renamed.columns.names

    #
    # with specifying a level

    # dict
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0)
    ray_renamed = ray_df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0)
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz")
    ray_renamed = ray_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz"
    )
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)

    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1)
    ray_renamed = ray_df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1)
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz")
    ray_renamed = ray_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz"
    )
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)

    # function
    func = str.upper
    renamed = df.rename(columns=func, level=0)
    ray_renamed = ray_df.rename(columns=func, level=0)
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)
    renamed = df.rename(columns=func, level="fizz")
    ray_renamed = ray_df.rename(columns=func, level="fizz")
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)

    renamed = df.rename(columns=func, level=1)
    ray_renamed = ray_df.rename(columns=func, level=1)
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)
    renamed = df.rename(columns=func, level="buzz")
    ray_renamed = ray_df.rename(columns=func, level="buzz")
    tm.assert_index_equal(renamed.columns, ray_renamed.columns)

    # index
    renamed = df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
    ray_renamed = ray_df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
    tm.assert_index_equal(ray_renamed.index, renamed.index)


def test_rename_nocopy():
    test_data = TestData().frame
    ray_df = pd.DataFrame(test_data)
    ray_renamed = ray_df.rename(columns={"C": "foo"}, copy=False)
    ray_renamed["foo"] = 1
    assert (ray_df["C"] == 1).all()


def test_rename_inplace():
    test_data = TestData().frame
    ray_df = pd.DataFrame(test_data)

    assert df_equals(ray_df.rename(columns={"C": "foo"}), test_data.rename(columns={"C": "foo"}))

    frame = test_data.copy()
    ray_frame = ray_df.copy()
    frame.rename(columns={"C": "foo"}, inplace=True)
    ray_frame.rename(columns={"C": "foo"}, inplace=True)

    assert df_equals(ray_frame, frame)


def test_rename_bug():
    # rename set ref_locs, and set_index was not resetting
    frame_data = {0: ["foo", "bar"], 1: ["bah", "bas"], 2: [1, 2]}
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    df = df.rename(columns={0: "a"})
    df = df.rename(columns={1: "b"})
    # TODO: Uncomment when set_index is implemented
    # df = df.set_index(['a', 'b'])
    # df.columns = ['2001-01-01']

    ray_df = ray_df.rename(columns={0: "a"})
    ray_df = ray_df.rename(columns={1: "b"})
    # TODO: Uncomment when set_index is implemented
    # ray_df = ray_df.set_index(['a', 'b'])
    # ray_df.columns = ['2001-01-01']

    assert df_equals(ray_df, df)


def test_rename_axis_inplace():
    test_frame = TestData().frame
    ray_df = pd.DataFrame(test_frame)

    result = test_frame.copy()
    ray_result = ray_df.copy()
    no_return = result.rename_axis("foo", inplace=True)
    ray_no_return = ray_result.rename_axis("foo", inplace=True)

    assert no_return is ray_no_return
    assert df_equals(ray_result, result)

    result = test_frame.copy()
    ray_result = ray_df.copy()
    no_return = result.rename_axis("bar", axis=1, inplace=True)
    ray_no_return = ray_result.rename_axis("bar", axis=1, inplace=True)

    assert no_return is ray_no_return
    assert df_equals(ray_result, result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reorder_levels(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.reorder_levels(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_replace(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.replace()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_resample(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.resample(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reset_index(ray_df, pandas_df, inplace=False):
    if not inplace:
        assert df_equals(ray_df.reset_index(inplace=inplace), pandas_df.reset_index(inplace=inplace))
    else:
        ray_df_cp = ray_df.copy()
        pd_df_cp = pandas_df.copy()
        ray_df_cp.reset_index(inplace=inplace)
        pd_df_cp.reset_index(inplace=inplace)
        assert df_equals(ray_df_cp, pd_df_cp)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_rolling(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.rolling(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_round(ray_df, pandas_df):
    assert df_equals(ray_df.round(), pandas_df.round())
    assert df_equals(ray_df.round(1), pandas_df.round(1))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_sample(ray_df, pandas_df, axis):
    with pytest.raises(ValueError):
        ray_df.sample(n=3, frac=0.4, axis=axis)

    assert df_equals(ray_df.sample(frac=0.5, random_state=42, axis=axis), pandas_df.sample(frac=0.5, random_state=42, axis=axis))
    assert df_equals(ray_df.sample(n=2, random_state=42, axis=axis), pandas_df.sample(n=2, random_state=42, axis=axis))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_select(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.select(None)


def test_select_dtypes():
    frame_data = {
        "test1": list("abc"),
        "test2": np.arange(3, 6).astype("u1"),
        "test3": np.arange(8.0, 11.0, dtype="float64"),
        "test4": [True, False, True],
        "test5": pandas.date_range("now", periods=3).values,
        "test6": list(range(5, 8)),
    }
    df = pandas.DataFrame(frame_data)
    rd = pd.DataFrame(frame_data)

    include = np.float, "integer"
    exclude = (np.bool_,)
    r = rd.select_dtypes(include=include, exclude=exclude)

    e = df[["test2", "test3", "test6"]]
    assert df_equals(r, e)

    try:
        pd.DataFrame().select_dtypes()
        assert False
    except ValueError:
        assert True


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_sem(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.sem()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_set_axis(ray_df, pandas_df, axis):
    if type(axis) == str:
        x = 0 if axis == "rows" else 1
    else:
        x = axis
    labels = ["{}".format(i) for i in range(ray_df.shape[x])]
    ray_result = ray_df.set_axis(labels, axis=axis, inplace=False)
    pandas_result = pandas_df.set_axis(labels, axis=axis, inplace=False)
    assert df_equals(ray_result, pandas_result)

    ray_df_copy = ray_df.copy()
    ray_df.set_axis(labels, axis=axis, inplace=True)
    assert not df_equals(ray_df, ray_df_copy)
    pandas_df.set_axis(labels, axis=axis, inplace=True)
    assert df_equals(ray_df, pandas_df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("drop", bool_arg_values, ids=arg_keys("drop", bool_arg_keys))
@pytest.mark.parametrize("append", bool_arg_values, ids=arg_keys("append", bool_arg_keys))
def test_set_index(request, ray_df, pandas_df, drop, append):
    if "empty_data" not in request.node.name:
        key = ray_df.columns[0]
        ray_result = ray_df.set_index(key, drop=drop, append=append, inplace=False)
        pandas_result = pandas_df.set_index(key, drop=drop, append=append, inplace=False)
        assert df_equals(ray_result, pandas_result)

        ray_df_copy = ray_df.copy()
        ray_df.set_index(key, drop=drop, append=append, inplace=True)
        assert not df_equals(ray_df, ray_df_copy)
        pandas_df.set_index(key, drop=drop, append=append, inplace=True)
        assert df_equals(ray_df, pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_set_value(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.set_value(None, None, None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_shape(ray_df, pandas_df):
    assert ray_df.shape == pandas_df.shape


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_shift(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.shift()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_size(ray_df, pandas_df):
    assert ray_df.size == pandas_df.size


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
def test_skew(ray_df, pandas_df, axis, skipna, numeric_only):
    ray_result = ray_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    pandas_result = pandas_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    assert df_equals(ray_result, pandas_result)

@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_slice_shift(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.slice_shift()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys))
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
@pytest.mark.parametrize("sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys))
def test_sort_index(ray_df, pandas_df, axis, ascending, na_position, sort_remaining):
    ray_result = ray_df.sort_index(axis=axis, ascending=ascending, na_position=na_position, inplace=False)
    pandas_result = pandas_df.sort_index(axis=axis, ascending=ascending, na_position=na_position, inplace=False)
    assert df_equals(ray_result, pandas_result)

    ray_df_copy = ray_df.copy()
    ray_df.sort_index(axis=axis, ascending=ascending, na_position=na_position, inplace=True)
    pandas_df.sort_index(axis=axis, ascending=ascending, na_position=na_position, inplace=True)
    assert df_equals(ray_df, pandas_df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys))
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_values(request, ray_df, pandas_df, axis, ascending, na_position):
    if "empty_data" not in request.node.name and ((axis == 0 or axis == 'over rows') or name_contains(request.node.name, numeric_dfs)):
        index = ray_df.index if axis or axis == "columns" else ray_df.columns
        key = index[0]
        ray_result = ray_df.sort_values(key, axis=axis, ascending=ascending, na_position=na_position, inplace=False)
        pandas_result = pandas_df.sort_values(key, axis=axis, ascending=ascending, na_position=na_position, inplace=False)
        assert df_equals(ray_result, pandas_result)

        ray_df_cp = ray_df.copy()
        pandas_df_cp = pandas_df.copy()
        ray_df_cp.sort_values(key, axis=axis, ascending=ascending, na_position=na_position, inplace=True)
        pandas_df_cp.sort_values(key, axis=axis, ascending=ascending, na_position=na_position, inplace=True)
        assert df_equals(ray_df_cp, pandas_df_cp)

        keys = [key, index[-1]]
        ray_result = ray_df.sort_values(keys, axis=axis, ascending=ascending, na_position=na_position, inplace=False)
        pandas_result = pandas_df.sort_values(keys, axis=axis, ascending=ascending, na_position=na_position, inplace=False)
        assert df_equals(ray_result, pandas_result)

        ray_df_cp = ray_df.copy()
        pandas_df_cp = pandas_df.copy()
        ray_df_cp.sort_values(keys, axis=axis, ascending=ascending, na_position=na_position, inplace=True)
        pandas_df_cp.sort_values(keys, axis=axis, ascending=ascending, na_position=na_position, inplace=True)
        assert df_equals(ray_df_cp, pandas_df_cp)



@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_sortlevel(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.sortlevel()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_squeeze(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.squeeze()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_stack(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.stack()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(ray_df, pandas_df, axis, skipna, numeric_only, ddof):
    ray_result = ray_df.std(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    pandas_result = pandas_df.std(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    assert df_equals(ray_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_style(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.style


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys))
@pytest.mark.parametrize("min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys))
def test_sum(request, ray_df, pandas_df, axis, skipna, numeric_only, min_count):
    if numeric_only or name_contains(request.node.name, numeric_dfs):
        ray_result = ray_df.sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
        pandas_result = pandas_df.sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)
        assert df_equals(ray_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            ray_df.sum(axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_swapaxes(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.swapaxes(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_swaplevel(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.swaplevel()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(ray_df, pandas_df, n):
    assert df_equals(ray_df.tail(n), pandas_df.tail(n))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_take(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.take(None)


def test_to_datetime():
    frame_data = {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
    ray_df = pd.DataFrame(frame_data)
    pd_df = pandas.DataFrame(frame_data)

    assert df_equals(pd.to_datetime(ray_df), pandas.to_datetime(pd_df))


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_records(ray_df, pandas_df):
    assert np.array_equal(ray_df.to_records(), to_pandas(ray_df).to_records())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_sparse(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.to_sparse()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_string(ray_df, pandas_df):
    assert ray_df.to_string() == to_pandas(ray_df).to_string()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_timestamp(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.to_timestamp()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_xarray(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.to_xarray()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform(request, ray_df, pandas_df, func):
    if "empty_data" not in request.node.name:
        ray_result = ray_df.agg(func)
        pandas_result = pandas_df.agg(func)
        assert df_equals(ray_result, pandas_result)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_transpose(ray_df, pandas_df):
    assert df_equals(ray_df.T, pandas_df.T)
    assert df_equals(ray_df.transpose(), pandas_df.transpose())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_truncate(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.truncate()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_tshift(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.tshift()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_tz_convert(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.tz_convert(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_tz_localize(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.tz_localize(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_unstack(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.unstack()


def test_update():
    df = pd.DataFrame(
        [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
    )
    other = pd.DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

    df.update(other)
    expected = pd.DataFrame(
        [[1.5, np.nan, 3], [3.6, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
    )
    assert df_equals(df, expected)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_values(ray_df, pandas_df):
    np.testing.assert_equal(ray_df.values, pandas_df.values)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys))
@pytest.mark.parametrize("numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys))
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(ray_df, pandas_df, axis, skipna, numeric_only, ddof):
    ray_result = ray_df.var(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    pandas_result = pandas_df.var(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    assert df_equals(ray_result, pandas_result)


def test_where():
    frame_data = np.random.randn(100, 10)
    pandas_df = pandas.DataFrame(frame_data, columns=list("abcdefghij"))
    ray_df = pd.DataFrame(frame_data, columns=list("abcdefghij"))
    pandas_cond_df = pandas_df % 5 < 2
    ray_cond_df = ray_df % 5 < 2

    pandas_result = pandas_df.where(pandas_cond_df, -pandas_df)
    ray_result = ray_df.where(ray_cond_df, -ray_df)
    assert all((to_pandas(ray_result) == pandas_result).all())

    other = pandas_df.loc[3]
    pandas_result = pandas_df.where(pandas_cond_df, other, axis=1)
    ray_result = ray_df.where(ray_cond_df, other, axis=1)
    assert all((to_pandas(ray_result) == pandas_result).all())

    other = pandas_df["e"]
    pandas_result = pandas_df.where(pandas_cond_df, other, axis=0)
    ray_result = ray_df.where(ray_cond_df, other, axis=0)
    assert all((to_pandas(ray_result) == pandas_result).all())

    pandas_result = pandas_df.where(pandas_df < 2, True)
    ray_result = ray_df.where(ray_df < 2, True)
    assert all((to_pandas(ray_result) == pandas_result).all())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_xs(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.xs(None)


def test__doc__():
    assert pd.DataFrame.__doc__ != pandas.DataFrame.__doc__
    assert pd.DataFrame.__init__ != pandas.DataFrame.__init__
    for attr, obj in pd.DataFrame.__dict__.items():
        if (callable(obj) or isinstance(obj, property)) and attr != "__init__":
            pd_obj = getattr(pandas.DataFrame, attr, None)
            if callable(pd_obj) or isinstance(pd_obj, property):
                assert obj.__doc__ == pd_obj.__doc__


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___getitem__(request, ray_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = ray_df.columns[0]
        ray_col = ray_df.__getitem__(key)
        assert isinstance(ray_col, pandas.Series)

        pd_col = pandas_df[key]
        assert df_equals(pd_col, ray_col)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___getattr__(request, ray_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = ray_df.columns[0]
        col = ray_df.__getattr__(key)
        assert isinstance(col, pandas.Series)

        col = getattr(ray_df, key)
        assert isinstance(col, pandas.Series)

        col = ray_df.col1
        assert isinstance(col, pandas.Series)

        # Check that lookup in column doesn't override other attributes
        df2 = ray_df.rename(index=str, columns={key: "columns"})
        assert isinstance(df2.columns, pandas.Index)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___setitem__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__setitem__(None, None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___len__(ray_df, pandas_df):
    assert len(ray_df) == len(pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___unicode__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__unicode__()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___neg__(ray_df, pandas_df):
    ray_df_neg = ray_df.__neg__()
    assert df_equals(pandas_df.__neg__(), ray_df_neg)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___invert__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__invert__()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___hash__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__hash__()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___iter__(ray_df, pandas_df):
    ray_iterator = ray_df.__iter__()

    # Check that ray_iterator implements the iterator interface
    assert hasattr(ray_iterator, "__iter__")
    assert hasattr(ray_iterator, "next") or hasattr(ray_iterator, "__next__")

    pd_iterator = pandas_df.__iter__()
    assert list(ray_iterator) == list(pd_iterator)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___contains__(request, ray_df, pandas_df):
    result = False
    key = "Not Exist"
    assert result == ray_df.__contains__(key)
    assert result == (key in ray_df)

    if "empty_data" not in request.node.name:
        result = True
        key = pandas_df.columns[0]
        assert result == ray_df.__contains__(key)
        assert result == (key in ray_df)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___nonzero__(ray_df, pandas_df):
    with pytest.raises(ValueError):
        # Always raises ValueError
        ray_df.__nonzero__()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___abs__(ray_df, pandas_df):
    assert df_equals(abs(ray_df), abs(pandas_df))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___round__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__round__()


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___array__(ray_df, pandas_df):
    assert_array_equal(ray_df.__array__(), pandas_df.__array__())


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___bool__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__bool__()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___getstate__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__getstate__()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___setstate__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__setstate__(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___delitem__(request, ray_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = pandas_df.columns[0]

        ray_df = ray_df.copy()
        pandas_df = pandas_df.copy()
        ray_df.__delitem__(key)
        pandas_df.__delitem__(key)
        assert df_equals(ray_df, pandas_df)

        # Issue 2027
        last_label = pandas_df.iloc[:, -1].name
        ray_df.__delitem__(last_label)
        pandas_df.__delitem__(last_label)
        df_equals(ray_df, pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___finalize__(ray_df, pandas_df):
    with pytest.raises(NotImplementedError):
        ray_df.__finalize__(None)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___copy__(ray_df, pandas_df):
    ray_df_copy, pandas_df_copy = ray_df.__copy__(), pandas_df.__copy__()
    assert df_equals(ray_df_copy, pandas_df_copy)


@pytest.mark.parametrize("ray_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___deepcopy__(ray_df, pandas_df):
    ray_df_copy, pandas_df_copy = ray_df.__deepcopy__(), pandas_df.__deepcopy__()
    assert df_equals(ray_df_copy, pandas_df_copy)


def test___repr__():
    frame_data = np.random.randint(0, 100, size=(1000, 100))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(ray_df)

    frame_data = np.random.randint(0, 100, size=(1000, 99))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(ray_df)

    frame_data = np.random.randint(0, 100, size=(1000, 101))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(ray_df)

    frame_data = np.random.randint(0, 100, size=(1000, 102))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(ray_df)

    # ___repr___ method has a different code path depending on
    # whether the number of rows is >60; and a different code path
    # depending on the number of columns is >20.
    # Previous test cases already check the case when cols>20
    # and rows>60. The cases that follow exercise the other three
    # combinations.
    # rows <= 60, cols > 20
    frame_data = np.random.randint(0, 100, size=(10, 100))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(ray_df)

    # rows <= 60, cols <= 20
    frame_data = np.random.randint(0, 100, size=(10, 10))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(ray_df)

    # rows > 60, cols <= 20
    frame_data = np.random.randint(0, 100, size=(100, 10))
    pandas_df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(ray_df)

