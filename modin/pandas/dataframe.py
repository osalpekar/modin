from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.api.types import is_scalar
from pandas.compat import to_str, string_types, cPickle as pkl
import pandas.core.common as com
from pandas.core.dtypes.common import (_get_dtype_from_object, is_bool_dtype,
                                       is_list_like, is_numeric_dtype,
                                       is_timedelta64_dtype)
from pandas.core.index import _ensure_index_from_sequences
from pandas.core.indexing import (check_bool_indexer, convert_to_index_sliceable)
from pandas.errors import MergeError
from pandas.util._validators import validate_bool_kwarg

import itertools
import io
import functools
import numpy as np
import ray
import re
import sys
import warnings

from .utils import (from_pandas, to_pandas, _blocks_to_col, _blocks_to_row,
                    _compile_remote_dtypes, _concat_index, _co_op_helper,
                    _create_block_partitions, _deploy_func,
                    _fix_blocks_dimensions, _inherit_docstrings,
                    _map_partitions, _match_partitioning,_reindex_helper)
from ..data_management.data_manager import RayPandasDataManager
from .index_metadata import _IndexMetadata
from .iterator import PartitionIterator


@_inherit_docstrings(
    pandas.DataFrame, excluded=[pandas.DataFrame, pandas.DataFrame.__init__])
class DataFrame(object):
    def __init__(self,
                 data=None,
                 index=None,
                 columns=None,
                 dtype=None,
                 copy=False,
                 col_partitions=None,
                 row_partitions=None,
                 block_partitions=None,
                 row_metadata=None,
                 col_metadata=None,
                 dtypes_cache=None,
                 data_manager=None):
        """Distributed DataFrame object backed by Pandas dataframes.

        Args:
            data (numpy ndarray (structured or homogeneous) or dict):
                Dict can contain Series, arrays, constants, or list-like
                objects.
            index (pandas.Index, list, ObjectID): The row index for this
                DataFrame.
            columns (pandas.Index): The column names for this DataFrame, in
                pandas Index object.
            dtype: Data type to force. Only a single dtype is allowed.
                If None, infer
            copy (boolean): Copy data from inputs.
                Only affects DataFrame / 2d ndarray input
            col_partitions ([ObjectID]): The list of ObjectIDs that contain
                the column DataFrame partitions.
            row_partitions ([ObjectID]): The list of ObjectIDs that contain the
                row DataFrame partitions.
            block_partitions: A 2D numpy array of block partitions.
            row_metadata (_IndexMetadata):
                Metadata for the new DataFrame's rows
            col_metadata (_IndexMetadata):
                Metadata for the new DataFrame's columns
        """
        if isinstance(data, DataFrame):
            self._data_manager = data._data_manager
            return

        self._dtypes_cache = dtypes_cache

        # Check type of data and use appropriate constructor
        if data is not None or (col_partitions is None
                                and row_partitions is None
                                and block_partitions is None
                                and data_manager is None):

            pandas_df = pandas.DataFrame(
                data=data,
                index=index,
                columns=columns,
                dtype=dtype,
                copy=copy)

            # Cache dtypes
            self._dtypes_cache = pandas_df.dtypes

            self._data_manager = from_pandas(pandas_df)._data_manager
        else:
            if data_manager is not None:
                self._data_manager = data_manager
            else:
                # created this invariant to make sure we never have to go into the
                # partitions to get the columns
                assert columns is not None or col_metadata is not None, \
                    "Columns not defined, must define columns or col_metadata " \
                    "for internal DataFrame creations"

                if block_partitions is not None:
                    axis = 0
                    # put in numpy array here to make accesses easier since it's 2D
                    self._block_partitions = np.array(block_partitions)
                    self._block_partitions = \
                        _fix_blocks_dimensions(self._block_partitions, axis)

                else:
                    if row_partitions is not None:
                        axis = 0
                        partitions = row_partitions
                        axis_length = len(columns) if columns is not None else \
                            len(col_metadata)
                    elif col_partitions is not None:
                        axis = 1
                        partitions = col_partitions
                        axis_length = len(index) if index is not None else \
                            len(row_metadata)
                        # All partitions will already have correct dtypes
                        self._dtypes_cache = [
                            _deploy_func.remote(lambda df: df.dtypes, pandas_df)
                            for pandas_df in col_partitions
                        ]

                    # TODO: write explicit tests for "short and wide"
                    # column partitions
                    self._block_partitions = \
                        _create_block_partitions(partitions, axis=axis,
                                                 length=axis_length)

            if self._dtypes_cache is None and data_manager is None:
                self._get_remote_dtypes()

            if data_manager is None:
                self._data_manager = RayPandasDataManager._from_old_block_partitions(self._block_partitions, index, columns)

    def _get_row_partitions(self):
        empty_rows_mask = self._row_metadata._lengths > 0
        if any(empty_rows_mask):
            self._row_metadata._lengths = \
                self._row_metadata._lengths[empty_rows_mask]
            self._block_partitions = self._block_partitions[empty_rows_mask, :]
        return [
            _blocks_to_row.remote(*part)
            for i, part in enumerate(self._block_partitions)
        ]

    def _set_row_partitions(self, new_row_partitions):
        self._block_partitions = \
            _create_block_partitions(new_row_partitions, axis=0,
                                     length=len(self.columns))

    _row_partitions = property(_get_row_partitions, _set_row_partitions)

    def _get_col_partitions(self):
        empty_cols_mask = self._col_metadata._lengths > 0
        if any(empty_cols_mask):
            self._col_metadata._lengths = \
                self._col_metadata._lengths[empty_cols_mask]
            self._block_partitions = self._block_partitions[:, empty_cols_mask]
        return [
            _blocks_to_col.remote(*self._block_partitions[:, i])
            for i in range(self._block_partitions.shape[1])
        ]

    def _set_col_partitions(self, new_col_partitions):
        self._block_partitions = \
            _create_block_partitions(new_col_partitions, axis=1,
                                     length=len(self.index))

    _col_partitions = property(_get_col_partitions, _set_col_partitions)

    def __str__(self):
        return repr(self)

    def _build_repr_df(self, num_rows, num_cols):
        # Add one here so that pandas automatically adds the dots
        # It turns out to be faster to extract 2 extra rows and columns than to
        # build the dots ourselves.
        num_rows_for_head = num_rows // 2 + 1
        num_cols_for_front = num_cols // 2 + 1

        if len(self.index) <= num_rows:
            head = self._data_manager
            tail = None
        else:
            head = self._data_manager.head(num_rows_for_head)
            tail = self._data_manager.tail(num_rows_for_head)

        if len(self.columns) <= num_cols:
            head_front = head.to_pandas()
            # Creating these empty to make the concat logic simpler
            head_back = pandas.DataFrame()
            tail_back = pandas.DataFrame()

            if tail is not None:
                tail_front = tail.to_pandas()
            else:
                tail_front = pandas.DataFrame()
        else:
            head_front = head.front(num_cols_for_front).to_pandas()
            head_back = head.back(num_cols_for_front).to_pandas()

            if tail is not None:
                tail_front = tail.front(num_cols_for_front).to_pandas()
                tail_back = tail.back(num_cols_for_front).to_pandas()
            else:
                tail_front = tail_back = pandas.DataFrame()

        head_for_repr = pandas.concat([head_front, head_back], axis=1)
        tail_for_repr = pandas.concat([tail_front, tail_back], axis=1)

        return pandas.concat([head_for_repr, tail_for_repr])

    def __repr__(self):
        # In the future, we can have this be configurable, just like Pandas.
        num_rows = 60
        num_cols = 20

        result = repr(self._build_repr_df(num_rows, num_cols))
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # The split here is so that we don't repr pandas row lengths.
            return result.rsplit("\n\n", 1)[0] + "\n\n[{0} rows x {1} columns]".format(len(self.index), len(self.columns))
        else:
            return result

    def _repr_html_(self):
        """repr function for rendering in Jupyter Notebooks like Pandas
        Dataframes.

        Returns:
            The HTML representation of a Dataframe.
        """
        # In the future, we can have this be configurable, just like Pandas.
        num_rows = 60
        num_cols = 20

        # We use pandas _repr_html_ to get a string of the HTML representation
        # of the dataframe.
        result = self._build_repr_df(num_rows, num_cols)._repr_html_()
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # We split so that we insert our correct dataframe dimensions.
            return result.split("<p>")[0] + "<p>{0} rows x {1} columns</p>\n</div>".format(len(self.index), len(self.columns))
        else:
            return result

    def _get_index(self):
        """Get the index for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._data_manager.index

    def _get_columns(self):
        """Get the columns for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._data_manager.columns

    def _set_index(self, new_index):
        """Set the index for this DataFrame.

        Args:
            new_index: The new index to set this
        """
        self._data_manager.index = new_index

    def _set_columns(self, new_columns):
        """Set the columns for this DataFrame.

        Args:
            new_index: The new index to set this
        """
        self._data_manager.columns = new_columns

    index = property(_get_index, _set_index)
    columns = property(_get_columns, _set_columns)

    def _map_reduce(self, *args, **kwargs):
        raise ValueError("Fix this implementation")

    def _validate_eval_query(self, expr, **kwargs):
        """Helper function to check the arguments to eval() and query()

        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.
        """
        if isinstance(expr, str) and expr is '':
            raise ValueError("expr cannot be an empty string")

        if isinstance(expr, str) and '@' in expr:
            raise NotImplementedError("Local variables not yet supported in "
                                      "eval.")

        if isinstance(expr, str) and 'not' in expr:
            if 'parser' in kwargs and kwargs['parser'] == 'python':
                raise NotImplementedError("'Not' nodes are not implemented.")

    @property
    def size(self):
        """Get the number of elements in the DataFrame.

        Returns:
            The number of elements in the DataFrame.
        """
        return len(self.index) * len(self.columns)

    @property
    def ndim(self):
        """Get the number of dimensions for this DataFrame.

        Returns:
            The number of dimensions for this DataFrame.
        """
        # DataFrames have an invariant that requires they be 2 dimensions.
        return 2

    @property
    def ftypes(self):
        """Get the ftypes for this DataFrame.

        Returns:
            The ftypes for this DataFrame.
        """
        # The ftypes are common across all partitions.
        # The first partition will be enough.
        result = ray.get(
            _deploy_func.remote(lambda df: df.ftypes, self._row_partitions[0]))
        result.index = self.columns
        return result

    def _get_remote_dtypes(self):
        """Finds and caches ObjectIDs for the dtypes of each column partition.
        """
        self._dtypes_cache = [
            _compile_remote_dtypes.remote(*column)
            for column in self._block_partitions.T
        ]

    @property
    def dtypes(self):
        """Get the dtypes for this DataFrame.

        Returns:
            The dtypes for this DataFrame.
        """
        assert self._dtypes_cache is not None

        if isinstance(self._dtypes_cache, list) and \
                isinstance(self._dtypes_cache[0],
                           ray.ObjectID):
            self._dtypes_cache = pandas.concat(
                ray.get(self._dtypes_cache), copy=False)
            self._dtypes_cache.index = self.columns

        return self._dtypes_cache

    @property
    def empty(self):
        """Determines if the DataFrame is empty.

        Returns:
            True if the DataFrame is empty.
            False otherwise.
        """
        return len(self.columns) == 0 or len(self.index) == 0

    @property
    def values(self):
        """Create a numpy array with the values from this DataFrame.

        Returns:
            The numpy representation of this DataFrame.
        """
        return self.as_matrix()

    @property
    def axes(self):
        """Get the axes for the DataFrame.

        Returns:
            The axes for the DataFrame.
        """
        return [self.index, self.columns]

    @property
    def shape(self):
        """Get the size of each of the dimensions in the DataFrame.

        Returns:
            A tuple with the size of each dimension as they appear in axes().
        """
        return len(self.index), len(self.columns)

    def _update_inplace(self, new_manager):
        """Updates the current DataFrame inplace.

        Args:
            new_manager: The new DataManager to use to manage the data
        """
        old_manager = self._data_manager
        self._data_manager = new_manager
        old_manager.free()
        # self._get_remote_dtypes()

    def add_prefix(self, prefix):
        """Add a prefix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(data_manager=self._data_manager.add_prefix(prefix))

    def add_suffix(self, suffix):
        """Add a suffix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(data_manager=self._data_manager.add_suffix(suffix))

    def applymap(self, func):
        """Apply a function to a DataFrame elementwise.

        Args:
            func (callable): The function to apply.
        """
        if not callable(func):
            raise ValueError("\'{0}\' object is not callable".format(
                type(func)))

        return DataFrame(data_manager=self._data_manager.applymap(func))

    def copy(self, deep=True):
        """Creates a shallow copy of the DataFrame.

        Returns:
            A new DataFrame pointing to the same partitions as this one.
        """
        return DataFrame(data_manager=self._data_manager.copy())

    def groupby(self,
                by=None,
                axis=0,
                level=None,
                as_index=True,
                sort=True,
                group_keys=True,
                squeeze=False,
                **kwargs):
        """Apply a groupby to this DataFrame. See _groupby() remote task.
        Args:
            by: The value to groupby.
            axis: The axis to groupby.
            level: The level of the groupby.
            as_index: Whether or not to store result as index.
            sort: Whether or not to sort the result by the index.
            group_keys: Whether or not to group the keys.
            squeeze: Whether or not to squeeze.
        Returns:
            A new DataFrame resulting from the groupby.
        """
        axis = pandas.DataFrame()._get_axis_number(axis)
        if callable(by):
            by = by(self.index)
        elif isinstance(by, string_types):
            by = self.__getitem__(by).values.tolist()
        elif is_list_like(by):
            if isinstance(by, pandas.Series):
                by = by.values.tolist()

            mismatch = len(by) != len(self) if axis == 0 \
                else len(by) != len(self.columns)

            if all(obj in self for obj in by) and mismatch:
                raise NotImplementedError(
                    "Groupby with lists of columns not yet supported.")
            elif mismatch:
                raise KeyError(next(x for x in by if x not in self))

        from .groupby import DataFrameGroupBy
        return DataFrameGroupBy(self, by, axis, level, as_index, sort,
                                group_keys, squeeze, **kwargs)

    def sum(self,
            axis=None,
            skipna=True,
            level=None,
            numeric_only=None,
            min_count=1,
            **kwargs):
        """Perform a sum across the DataFrame.

        Args:
            axis (int): The axis to sum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The sum of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs)

    def abs(self):
        """Apply an absolute value function to all numeric columns.

        Returns:
            A new DataFrame with the applied absolute value.
        """
        for t in self.dtypes:
            if np.dtype('O') == t:
                # TODO Give a more accurate error to Pandas
                raise TypeError("bad operand type for abs():", "str")

        return DataFrame(data_manager=self._data_manager.abs())

    def isin(self, values):
        """Fill a DataFrame with booleans for cells contained in values.

        Args:
            values (iterable, DataFrame, Series, or dict): The values to find.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is in values.
            True: cell is contained in values.
            False: otherwise
        """
        return DataFrame(data_manager=self._data_manager.isin(values=values))

    def isna(self):
        """Fill a DataFrame with booleans for cells containing NA.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is NA.
            True: cell contains NA.
            False: otherwise.
        """
        return DataFrame(data_manager=self._data_manager.isna())

    def isnull(self):
        """Fill a DataFrame with booleans for cells containing a null value.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
                is null.
            True: cell contains null.
            False: otherwise.
        """
        return DataFrame(data_manager=self._data_manager.isnull())

    def keys(self):
        """Get the info axis for the DataFrame.

        Returns:
            A pandas Index for this DataFrame.
        """
        return self.columns

    def transpose(self, *args, **kwargs):
        """Transpose columns and rows for the DataFrame.

        Returns:
            A new DataFrame transposed from this DataFrame.
        """
        return DataFrame(data_manager=self._data_manager.transpose(*args, **kwargs))

    T = property(transpose)

    def dropna(self,
               axis=0,
               how='any',
               thresh=None,
               subset=None,
               inplace=False):
        """Create a new DataFrame from the removed NA values from this one.

        Args:
            axis (int, tuple, or list): The axis to apply the drop.
            how (str): How to drop the NA values.
                'all': drop the label if all values are NA.
                'any': drop the label if any values are NA.
            thresh (int): The minimum number of NAs to require.
            subset ([label]): Labels to consider from other axis.
            inplace (bool): Change this DataFrame or return a new DataFrame.
                True: Modify the data for this DataFrame, return None.
                False: Create a new DataFrame and return it.

        Returns:
            If inplace is set to True, returns None, otherwise returns a new
            DataFrame with the dropna applied.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if is_list_like(axis):
            axis = [pandas.DataFrame()._get_axis_number(ax) for ax in axis]

            result = self

            for ax in axis:
                result = result.dropna(
                    axis=ax, how=how, thresh=thresh, subset=subset)
            if not inplace:
                return result

            self._update_inplace(new_manager=result._data_manager)
            return

        axis = pandas.DataFrame()._get_axis_number(axis)

        if how is not None and how not in ['any', 'all']:
            raise ValueError('invalid how option: %s' % how)
        if how is None and thresh is None:
            raise TypeError('must specify how or thresh')

        if subset is not None:
            if axis == 1:
                indices = self.index.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
            else:
                indices = self.columns.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))

        new_manager = self._data_manager.dropna(axis=axis, how=how, thresh=thresh, subset=subset)

        if not inplace:
            return DataFrame(data_manager=new_manager)
        else:
            self._update_inplace(new_manager=new_manager)

    def add(self, other, axis='columns', level=None, fill_value=None):
        """Add this DataFrame to another or a scalar/list.

        Args:
            other: What to add this this DataFrame.
            axis: The axis to apply addition over. Only applicaable to Series
                or list 'other'.
            level: A level in the multilevel axis to add over.
            fill_value: The value to fill NaN.

        Returns:
            A new DataFrame with the applied addition.
        """
        other = self._validate_single_op(other, axis)
        new_manager = self._data_manager.add(other=other,
                                             axis=axis,
                                             level=level,
                                             fill_value=fill_value)
        return self._create_dataframe_from_manager(new_manager)

    def agg(self, func, axis=0, *args, **kwargs):
        return self.aggregate(func, axis, *args, **kwargs)

    def aggregate(self, func, axis=0, *args, **kwargs):
        axis = pandas.DataFrame()._get_axis_number(axis)

        result = None

        if axis == 0:
            try:
                result = self._aggregate(func, axis=axis, *args, **kwargs)
            except TypeError:
                pass

        if result is None:
            kwargs.pop('is_transform', None)
            return self.apply(func, axis=axis, args=args, **kwargs)

        return result

    def _aggregate(self, arg, *args, **kwargs):
        _axis = kwargs.pop('_axis', None)
        if _axis is None:
            _axis = getattr(self, 'axis', 0)
        kwargs.pop('_level', None)

        if isinstance(arg, string_types):
            return self._string_function(arg, *args, **kwargs)

        # Dictionaries have complex behavior because they can be renamed here.
        elif isinstance(arg, dict):
            raise NotImplementedError(
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")
        elif is_list_like(arg) or callable(arg):
            return self.apply(arg, axis=_axis, args=args, **kwargs)
        else:
            # TODO Make pandas error
            raise ValueError("type {} is not callable".format(type(arg)))

    def _string_function(self, func, *args, **kwargs):
        assert isinstance(func, string_types)

        f = getattr(self, func, None)

        if f is not None:
            if callable(f):
                return f(*args, **kwargs)

            assert len(args) == 0
            assert len([
                kwarg for kwarg in kwargs if kwarg not in ['axis', '_level']
            ]) == 0
            return f

        f = getattr(np, func, None)
        if f is not None:
            raise NotImplementedError("Numpy aggregates not yet supported.")

        raise ValueError("{} is an unknown string function".format(func))

    def align(self,
              other,
              join='outer',
              axis=None,
              level=None,
              copy=True,
              fill_value=None,
              method=None,
              limit=None,
              fill_axis=0,
              broadcast_axis=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def all(self, axis=None, bool_only=None, skipna=None, level=None,
            **kwargs):
        """Return whether all elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies df.all(axis=1)
                to the transpose of df.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.all(
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
            level=level,
            **kwargs)

    def any(self, axis=None, bool_only=None, skipna=None, level=None,
            **kwargs):
        """Return whether any elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies on the column partitions,
                otherwise operates on row partitions
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.any(
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
            level=level,
            **kwargs)

    def append(self, other, ignore_index=False, verify_integrity=False):
        """Append another DataFrame/list/Series to this one.

        Args:
            other: The object to append to this.
            ignore_index: Ignore the index on appending.
            verify_integrity: Verify the integrity of the index on completion.

        Returns:
            A new DataFrame containing the concatenated values.
        """
        if isinstance(other, (pandas.Series, dict)):
            if isinstance(other, dict):
                other = pandas.Series(other)
            if other.name is None and not ignore_index:
                raise TypeError('Can only append a Series if ignore_index=True'
                                ' or if the Series has a name')

            if other.name is None:
                index = None
            else:
                # other must have the same index name as self, otherwise
                # index name will be reset
                index = pandas.Index([other.name], name=self.index.name)

            # Create a Modin DataFrame from this Series for ease of development
            other = DataFrame(pandas.DataFrame(other).T, index=index)._data_manager
        elif isinstance(other, list):
            if not isinstance(other[0], DataFrame):
                other = pandas.DataFrame(other)
                if (self.columns.get_indexer(other.columns) >= 0).all():
                    other = DataFrame(other.loc[:, self.columns])._data_manager
                else:
                    other = DataFrame(other)._data_manager
            else:
                other = [obj._data_manager for obj in other]
        else:
            other = other._data_manager

        # If ignore_index is False, by definition the Index will be correct.
        # We also do this first to ensure that we don't waste compute/memory.
        if verify_integrity and not ignore_index:
            raise NotImplementedError("Implement this!")

        data_manager = self._data_manager.concat(0, other, ignore_index=ignore_index)
        return DataFrame(data_manager=data_manager)

    def apply(self,
              func,
              axis=0,
              broadcast=False,
              raw=False,
              reduce=None,
              args=(),
              **kwds):
        """Apply a function along input axis of DataFrame.

        Args:
            func: The function to apply
            axis: The axis over which to apply the func.
            broadcast: Whether or not to broadcast.
            raw: Whether or not to convert to a Series.
            reduce: Whether or not to try to apply reduction procedures.

        Returns:
            Series or DataFrame, depending on func.
        """
        axis = pandas.DataFrame()._get_axis_number(axis)

        if isinstance(func, string_types):
            if axis == 1:
                kwds['axis'] = axis
            return getattr(self, func)(*args, **kwds)
        elif isinstance(func, dict):
            if axis == 1:
                raise TypeError("(\"'dict' object is not callable\", "
                                "'occurred at index {0}'".format(
                                    self.index[0]))
            if len(self.columns) != len(set(self.columns)):
                warnings.warn(
                    'duplicate column names not supported with apply().',
                    FutureWarning,
                    stacklevel=2)
        elif is_list_like(func):
            if axis == 1:
                raise TypeError("(\"'list' object is not callable\", "
                                "'occurred at index {0}'".format(
                                    self.index[0]))
        elif not callable(func):
            return

        data_manager = self._data_manager.apply(func, axis, *args, **kwds)
        if isinstance(data_manager, pandas.Series):
            return data_manager
        return DataFrame(data_manager=data_manager)

    def as_blocks(self, copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def as_matrix(self, columns=None):
        """Convert the frame to its Numpy-array representation.

        Args:
            columns: If None, return all columns, otherwise,
                returns specified columns.

        Returns:
            values: ndarray
        """
        # TODO this is very inefficient, also see __array__
        return to_pandas(self).as_matrix(columns)

    def asfreq(self,
               freq,
               method=None,
               how=None,
               normalize=False,
               fill_value=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def asof(self, where, subset=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def assign(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def astype(self, dtype, copy=True, errors='raise', **kwargs):
        if isinstance(dtype, dict):
            if (not set(dtype.keys()).issubset(set(self.columns))
                    and errors == 'raise'):
                raise KeyError("Only a column name can be used for the key in"
                               "a dtype mappings argument.")
            columns = list(dtype.keys())
            col_idx = [(self.columns.get_loc(columns[i]), columns[i]) if
                       columns[i] in self.columns else (columns[i], columns[i])
                       for i in range(len(columns))]
            new_dict = {}
            for idx, key in col_idx:
                new_dict[idx] = dtype[key]
            new_rows = _map_partitions(lambda df, dt: df.astype(dtype=dt,
                                                                copy=True,
                                                                errors=errors,
                                                                **kwargs),
                                       self._row_partitions, new_dict)
            if copy:
                return DataFrame(
                    row_partitions=new_rows,
                    columns=self.columns,
                    index=self.index)
            self._row_partitions = new_rows
        else:
            new_blocks = [_map_partitions(lambda d: d.astype(dtype=dtype,
                                                             copy=True,
                                                             errors=errors,
                                                             **kwargs),
                                          block)
                          for block in self._block_partitions]
            if copy:
                return DataFrame(
                    block_partitions=new_blocks,
                    columns=self.columns,
                    index=self.index)
            self._block_partitions = new_blocks

    def at_time(self, time, asof=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def between_time(self,
                     start_time,
                     end_time,
                     include_start=True,
                     include_end=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for DataFrame.fillna(method='bfill')
        """
        new_df = self.fillna(
            method='bfill',
            axis=axis,
            limit=limit,
            downcast=downcast,
            inplace=inplace)
        if not inplace:
            return new_df

    def bool(self):
        """Return the bool of a single element PandasObject.

        This must be a boolean scalar value, either True or False.  Raise a
        ValueError if the PandasObject does not have exactly 1 element, or that
        element is not boolean
        """
        shape = self.shape
        if shape != (1, ) and shape != (1, 1):
            raise ValueError("""The PandasObject does not have exactly
                                1 element. Return the bool of a single
                                element PandasObject. The truth value is
                                ambiguous. Use a.empty, a.item(), a.any()
                                or a.all().""")
        else:
            return to_pandas(self).bool()

    def boxplot(self,
                column=None,
                by=None,
                ax=None,
                fontsize=None,
                rot=0,
                grid=True,
                figsize=None,
                layout=None,
                return_type=None,
                **kwds):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def clip(self,
             lower=None,
             upper=None,
             axis=None,
             inplace=False,
             *args,
             **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def clip_lower(self, threshold, axis=None, inplace=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def clip_upper(self, threshold, axis=None, inplace=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def combine(self, other, func, fill_value=None, overwrite=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def combine_first(self, other):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def compound(self, axis=None, skipna=None, level=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def consolidate(self, inplace=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def convert_objects(self,
                        convert_dates=True,
                        convert_numeric=False,
                        convert_timedeltas=True,
                        copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def corr(self, method='pearson', min_periods=1):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def corrwith(self, other, axis=0, drop=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def count(self, axis=0, level=None, numeric_only=False):
        """Get the count of non-null objects in the DataFrame.

        Arguments:
            axis: 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            level: If the axis is a MultiIndex (hierarchical), count along a
                particular level, collapsing into a DataFrame.
            numeric_only: Include only float, int, boolean data

        Returns:
            The count, in a Series (or DataFrame if level is specified).
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return self._data_manager.count(axis=axis, level=level, numeric_only=numeric_only)

    def cov(self, min_periods=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative maximum across the DataFrame.

        Args:
            axis (int): The axis to take maximum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative maximum of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return DataFrame(data_manager=self._data_manager.cummax(axis=axis, skipna=skipna, **kwargs))

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative minimum across the DataFrame.

        Args:
            axis (int): The axis to cummin on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative minimum of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return DataFrame(data_manager=self._data_manager.cummin(axis=axis, skipna=skipna, **kwargs))

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative product across the DataFrame.

        Args:
            axis (int): The axis to take product on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative product of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return DataFrame(data_manager=self._data_manager.cumprod(axis=axis, skipna=skipna, **kwargs))

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative sum across the DataFrame.

        Args:
            axis (int): The axis to take sum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative sum of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return DataFrame(data_manager=self._data_manager.cumsum(axis=axis, skipna=skipna, **kwargs))

    def describe(self, percentiles=None, include=None, exclude=None):
        """
        Generates descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding NaN values.

        Args:
            percentiles (list-like of numbers, optional):
                The percentiles to include in the output.
            include: White-list of data types to include in results
            exclude: Black-list of data types to exclude in results

        Returns: Series/DataFrame of summary statistics
        """
        # This is important because we don't have communication between
        # partitions. We need to communicate to the partitions if they should
        # be operating on object data or not.
        # TODO uncomment after dtypes is fixed
        # if not all(t == np.dtype("O") for t in self.dtypes):
        if exclude is None:
            exclude = "object"
        elif "object" not in include:
            exclude = ([exclude] + "object") if isinstance(exclude, str) else list(exclude) + "object"

        if percentiles is not None:
            pandas.DataFrame()._check_percentile(percentiles)

        return DataFrame(data_manager=self._data_manager.describe(percentiles=percentiles, include=include, exclude=exclude))

    def diff(self, periods=1, axis=0):
        """Finds the difference between elements on the axis requested

        Args:
            periods: Periods to shift for forming difference
            axis: Take difference over rows or columns

        Returns:
            DataFrame with the diff applied
        """
        return DataFrame(data_manager=self._data_manager.diff(periods=periods, axis=axis))

    def div(self, other, axis='columns', level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._operator_helper(pandas.DataFrame.div, other, axis, level,
                                     fill_value)

    def divide(self, other, axis='columns', level=None, fill_value=None):
        """Synonym for div.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self.div(other, axis, level, fill_value)

    def dot(self, other):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def drop(self,
             labels=None,
             axis=0,
             index=None,
             columns=None,
             level=None,
             inplace=False,
             errors='raise'):
        """Return new object with labels in requested axis removed.
        Args:
            labels: Index or column labels to drop.
            axis: Whether to drop labels from the index (0 / 'index') or
                columns (1 / 'columns').
            index, columns: Alternative to specifying axis (labels, axis=1 is
                equivalent to columns=labels).
            level: For MultiIndex
            inplace: If True, do operation inplace and return None.
            errors: If 'ignore', suppress error and existing labels are
                dropped.
        Returns:
            dropped : type of caller
        """
        # TODO implement level
        if level is not None:
            raise NotImplementedError("Level not yet supported for drop")

        inplace = validate_bool_kwarg(inplace, "inplace")
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and "
                                 "'index'/'columns'")
            axis = pandas.DataFrame()._get_axis_name(axis)
            axes = {axis: labels}
        elif index is not None or columns is not None:
            axes, _ = pandas.DataFrame() \
                ._construct_axes_from_arguments((index, columns), {})
        else:
            raise ValueError("Need to specify at least one of 'labels', "
                             "'index' or 'columns'")

        # TODO Clean up this error checking
        if "index" not in axes:
            axes["index"] = None
        elif axes["index"] is not None:
            if not is_list_like(axes["index"]):
                axes["index"] = [axes["index"]]
            if errors == 'raise':
                non_existant = [obj for obj in axes["index"] if obj not in self.index]
                if len(non_existant):
                    raise ValueError("labels {} not contained in axis".format(non_existant))
            else:
                axes["index"] = [obj for obj in axes["index"] if obj in self.index]
                # If the length is zero, we will just do nothing
                if not len(axes["index"]):
                    axes["index"] = None

        if "columns" not in axes:
            axes["columns"] = None
        elif axes["columns"] is not None:
            if not is_list_like(axes["columns"]):
                axes["columns"] = [axes["columns"]]
            if errors == 'raise':
                non_existant = [obj for obj in axes["columns"] if obj not in self.columns]
                if len(non_existant):
                    raise ValueError("labels {} not contained in axis".format(non_existant))
            else:
                axes["columns"] = [obj for obj in axes["columns"] if obj in self.columns]
                # If the length is zero, we will just do nothing
                if not len(axes["columns"]):
                    axes["columns"] = None

        new_manager = self._data_manager.drop(index=axes["index"], columns=axes["columns"])

        if inplace:
            self._update_inplace(new_manager=new_manager)

        return DataFrame(data_manager=new_manager)

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def duplicated(self, subset=None, keep='first'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def eq(self, other, axis='columns', level=None):
        """Checks element-wise that this is equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the eq over.
            level: The Multilevel index level to apply eq over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._operator_helper(pandas.DataFrame.eq, other, axis, level)

    def equals(self, other):
        """
        Checks if other DataFrame is elementwise equal to the current one

        Returns:
            Boolean: True if equal, otherwise False
        """
        if isinstance(other, pandas.DataFrame):
            # Copy into a Ray DataFrame to simplify logic below
            other = DataFrame(other)

        if not self.index.equals(other.index) or not \
                self.columns.equals(other.columns):
            return False

        return all(self.eq(other).all())

    def eval(self, expr, inplace=False, **kwargs):
        """Evaluate a Python expression as a string using various backends.
        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.

            parser: The parser to use to construct the syntax tree from the
                expression. The default of 'pandas' parses code slightly
                different than standard Python. Alternatively, you can parse
                an expression using the 'python' parser to retain strict
                Python semantics. See the enhancing performance documentation
                for more details.

            engine: The engine used to evaluate the expression.

            truediv: Whether to use true division, like in Python >= 3

            local_dict: A dictionary of local variables, taken from locals()
                by default.

            global_dict: A dictionary of global variables, taken from
                globals() by default.

            resolvers: A list of objects implementing the __getitem__ special
                method that you can use to inject an additional collection
                of namespaces to use for variable lookup. For example, this is
                used in the query() method to inject the index and columns
                variables that refer to their respective DataFrame instance
                attributes.

            level: The number of prior stack frames to traverse and add to
                the current scope. Most users will not need to change this
                parameter.

            target: This is the target object for assignment. It is used when
                there is variable assignment in the expression. If so, then
                target must support item assignment with string keys, and if a
                copy is being returned, it must also support .copy().

            inplace: If target is provided, and the expression mutates target,
                whether to modify target inplace. Otherwise, return a copy of
                target with the mutation.
        Returns:
            ndarray, numeric scalar, DataFrame, Series
        """
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")

        data_manager = self._data_manager.eval(expr, **kwargs)

        if inplace:
            self._update_inplace(new_manager=data_manager)
        else:
            return DataFrame(data_manager=data_manager)

    def ewm(self,
            com=None,
            span=None,
            halflife=None,
            alpha=None,
            min_periods=0,
            freq=None,
            adjust=True,
            ignore_na=False,
            axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def expanding(self, min_periods=1, freq=None, center=False, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for DataFrame.fillna(method='ffill')
        """
        new_df = self.fillna(
            method='ffill',
            axis=axis,
            limit=limit,
            downcast=downcast,
            inplace=inplace)
        if not inplace:
            return new_df

    def fillna(self,
               value=None,
               method=None,
               axis=None,
               inplace=False,
               limit=None,
               downcast=None,
               **kwargs):
        """Fill NA/NaN values using the specified method.

        Args:
            value: Value to use to fill holes. This value cannot be a list.

            method: Method to use for filling holes in reindexed Series pad.
                ffill: propagate last valid observation forward to next valid
                backfill.
                bfill: use NEXT valid observation to fill gap.

            axis: 0 or 'index', 1 or 'columns'.

            inplace: If True, fill in place. Note: this will modify any other
                views on this object.

            limit: If method is specified, this is the maximum number of
                consecutive NaN values to forward/backward fill. In other
                words, if there is a gap with more than this number of
                consecutive NaNs, it will only be partially filled. If method
                is not specified, this is the maximum number of entries along
                the entire axis where NaNs will be filled. Must be greater
                than 0 if not None.

            downcast: A dict of item->dtype of what to downcast if possible,
                or the string 'infer' which will try to downcast to an
                appropriate equal type.

        Returns:
            filled: DataFrame
        """
        # TODO implement value passed as DataFrame
        if isinstance(value, pandas.DataFrame):
            raise NotImplementedError("Passing a DataFrame as the value for "
                                      "fillna is not yet supported.")

        inplace = validate_bool_kwarg(inplace, 'inplace')

        axis = pandas.DataFrame()._get_axis_number(axis) \
            if axis is not None \
            else 0

        if isinstance(value, (list, tuple)):
            raise TypeError('"value" parameter must be a scalar or dict, but '
                            'you passed a "{0}"'.format(type(value).__name__))
        if value is None and method is None:
            raise ValueError('must specify a fill method or value')
        if value is not None and method is not None:
            raise ValueError('cannot specify both a fill method and value')
        if method is not None and method not in [
                'backfill', 'bfill', 'pad', 'ffill'
        ]:
            expecting = 'pad (ffill) or backfill (bfill)'
            msg = 'Invalid fill method. Expecting {expecting}. Got {method}'\
                  .format(expecting=expecting, method=method)
            raise ValueError(msg)

        if isinstance(value, pandas.Series):
            raise NotImplementedError("value as a Series not yet supported.")

        new_manager = self._data_manager.fillna(value=value, method=method, axis=axis, inplace=False, limit=limit, downcast=downcast, **kwargs)

        if inplace:
            self._update_inplace(new_manager=new_manager)
        else:
            return DataFrame(data_manager=new_manager)

    def filter(self, items=None, like=None, regex=None, axis=None):
        """Subset rows or columns based on their labels

        Args:
            items (list): list of labels to subset
            like (string): retain labels where `arg in label == True`
            regex (string): retain labels matching regex input
            axis: axis to filter on

        Returns:
            A new DataFrame with the filter applied.
        """
        nkw = com._count_not_none(items, like, regex)
        if nkw > 1:
            raise TypeError('Keyword arguments `items`, `like`, or `regex` '
                            'are mutually exclusive')
        if nkw == 0:
            raise TypeError('Must pass either `items`, `like`, or `regex`')

        if axis is None:
            axis = 'columns'  # This is the default info axis for dataframes

        axis = pandas.DataFrame()._get_axis_number(axis)
        labels = self.columns if axis else self.index

        if items is not None:
            bool_arr = labels.isin(items)
        elif like is not None:

            def f(x):
                return like in to_str(x)

            bool_arr = labels.map(f).tolist()
        else:

            def f(x):
                return matcher.search(to_str(x)) is not None

            matcher = re.compile(regex)
            bool_arr = labels.map(f).tolist()

        if not axis:
            return self[bool_arr]
        return self[self.columns[bool_arr]]

    def first(self, offset):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def first_valid_index(self):
        """Return index for first non-NA/null value.

        Returns:
            scalar: type of index
        """
        return self._data_manager.first_valid_index()

    def floordiv(self, other, axis='columns', level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._operator_helper(pandas.DataFrame.floordiv, other, axis,
                                     level, fill_value)

    @classmethod
    def from_csv(self,
                 path,
                 header=0,
                 sep=', ',
                 index_col=0,
                 parse_dates=True,
                 encoding=None,
                 tupleize_cols=None,
                 infer_datetime_format=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @classmethod
    def from_dict(self, data, orient='columns', dtype=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @classmethod
    def from_items(self, items, columns=None, orient='columns'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @classmethod
    def from_records(self,
                     data,
                     index=None,
                     exclude=None,
                     columns=None,
                     coerce_float=False,
                     nrows=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def ge(self, other, axis='columns', level=None):
        """Checks element-wise that this is greater than or equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._operator_helper(pandas.DataFrame.ge, other, axis, level)

    def get(self, key, default=None):
        """Get item from object for given key (DataFrame column, Panel
        slice, etc.). Returns default value if not found.

        Args:
            key (DataFrame column, Panel slice) : the key for which value
            to get

        Returns:
            value (type of items contained in object) : A value that is
            stored at the key
        """
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def get_dtype_counts(self):
        """Get the counts of dtypes in this object.

        Returns:
            The counts of dtypes in this object.
        """
        return ray.get(
            _deploy_func.remote(lambda df: df.get_dtype_counts(),
                                self._row_partitions[0]))

    def get_ftype_counts(self):
        """Get the counts of ftypes in this object.

        Returns:
            The counts of ftypes in this object.
        """
        return ray.get(
            _deploy_func.remote(lambda df: df.get_ftype_counts(),
                                self._row_partitions[0]))

    def get_value(self, index, col, takeable=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def get_values(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def gt(self, other, axis='columns', level=None):
        """Checks element-wise that this is greater than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._operator_helper(pandas.DataFrame.gt, other, axis, level)

    def head(self, n=5):
        """Get the first n rows of the DataFrame.

        Args:
            n (int): The number of rows to return.

        Returns:
            A new DataFrame with the first n rows of the DataFrame.
        """
        if n >= len(self.index):
            return self.copy()

        return DataFrame(data_manager=self._data_manager.head(n))

    def hist(self,
             data,
             column=None,
             by=None,
             grid=True,
             xlabelsize=None,
             xrot=None,
             ylabelsize=None,
             yrot=None,
             ax=None,
             sharex=False,
             sharey=False,
             figsize=None,
             layout=None,
             bins=10,
             **kwds):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def idxmax(self, axis=0, skipna=True):
        """Get the index of the first occurrence of the max value of the axis.

        Args:
            axis (int): Identify the max over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each maximum value for the axis
                specified.
        """
        if not all(d != np.dtype('O') for d in self.dtypes):
            raise TypeError(
                "reduction operation 'argmax' not allowed for this dtype")

        return self._data_manager.idxmax(axis=axis, skipna=skipna)

    def idxmin(self, axis=0, skipna=True):
        """Get the index of the first occurrence of the min value of the axis.

        Args:
            axis (int): Identify the min over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each minimum value for the axis
                specified.
        """
        if not all(d != np.dtype('O') for d in self.dtypes):
            raise TypeError(
                "reduction operation 'argmax' not allowed for this dtype")

        return self._data_manager.idxmin(axis=axis, skipna=skipna)

    def infer_objects(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def info(self,
             verbose=None,
             buf=None,
             max_cols=None,
             memory_usage=None,
             null_counts=None):
        def info_helper(df):
            output_buffer = io.StringIO()
            df.info(
                verbose=verbose,
                buf=output_buffer,
                max_cols=max_cols,
                memory_usage=memory_usage,
                null_counts=null_counts)
            return output_buffer.getvalue()

        # Combine the per-partition info and split into lines
        result = ''.join(
            ray.get(_map_partitions(info_helper, self._col_partitions)))
        lines = result.split('\n')

        # Class denoted in info() output
        class_string = '<class \'modin.pandas.dataframe.DataFrame\'>\n'

        # Create the Index info() string by parsing self.index
        index_string = self.index.summary() + '\n'

        # A column header is needed in the inf() output
        col_header = 'Data columns (total {0} columns):\n' \
            .format(len(self.columns))

        # Parse the per-partition values to get the per-column details
        # Find all the lines in the output that start with integers
        prog = re.compile('^[0-9]+.+')
        col_lines = [prog.match(line) for line in lines]
        cols = [c.group(0) for c in col_lines if c is not None]
        # replace the partition columns names with real column names
        columns = [
            "{0}\t{1}\n".format(self.columns[i], cols[i].split(" ", 1)[1])
            for i in range(len(cols))
        ]
        col_string = ''.join(columns) + '\n'

        # A summary of the dtypes in the dataframe
        dtypes_string = "dtypes: "
        for dtype, count in self.dtypes.value_counts().iteritems():
            dtypes_string += "{0}({1}),".format(dtype, count)
        dtypes_string = dtypes_string[:-1] + '\n'

        # Compute the memory usage by summing per-partitions return values
        # Parse lines for memory usage number
        prog = re.compile('^memory+.+')
        mems = [prog.match(line) for line in lines]
        mem_vals = [
            float(re.search(r'\d+', m.group(0)).group()) for m in mems
            if m is not None
        ]

        memory_string = ""

        if len(mem_vals) != 0:
            # Sum memory usage from each partition
            if memory_usage != 'deep':
                memory_string = 'memory usage: {0}+ bytes' \
                    .format(sum(mem_vals))
            else:
                memory_string = 'memory usage: {0} bytes'.format(sum(mem_vals))

        # Combine all the components of the info() output
        result = ''.join([
            class_string, index_string, col_header, col_string, dtypes_string,
            memory_string
        ])

        # Write to specified output buffer
        if buf:
            buf.write(result)
        else:
            sys.stdout.write(result)

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert column into DataFrame at specified location.

        Args:
            loc (int): Insertion index. Must verify 0 <= loc <= len(columns).
            column (hashable object): Label of the inserted column.
            value (int, Series, or array-like): The values to insert.
            allow_duplicates (bool): Whether to allow duplicate column names.
        """
        if not is_list_like(value):
            value = np.full(len(self.index), value)

        if len(value) != len(self.index):
            raise ValueError("Length of values does not match length of index")
        if not allow_duplicates and column in self.columns:
            raise ValueError(
                "cannot insert {0}, already exists".format(column))
        if loc > len(self.columns):
            raise IndexError(
                "index {0} is out of bounds for axis 0 with size {1}".format(
                    loc, len(self.columns)))
        if loc < 0:
            raise ValueError("unbounded slice")

        new_manager = self._data_manager.insert(loc, column, value)
        self._update_inplace(new_manager=new_manager)

    def interpolate(self,
                    method='linear',
                    axis=0,
                    limit=None,
                    inplace=False,
                    limit_direction='forward',
                    downcast=None,
                    **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def iterrows(self):
        """Iterate over DataFrame rows as (index, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the rows of the frame.
        """
        index_iter = (self._row_metadata.partition_series(i).index
                      for i in range(len(self._row_partitions)))

        def iterrow_helper(part):
            df = ray.get(part)
            df.columns = self.columns
            df.index = next(index_iter)
            return df.iterrows()

        partition_iterator = PartitionIterator(self._row_partitions,
                                               iterrow_helper)

        for v in partition_iterator:
            yield v

    def items(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the columns of the frame.
        """
        col_iter = (self._col_metadata.partition_series(i).index
                    for i in range(len(self._col_partitions)))

        def items_helper(part):
            df = ray.get(part)
            df.columns = next(col_iter)
            df.index = self.index
            return df.items()

        partition_iterator = PartitionIterator(self._col_partitions,
                                               items_helper)

        for v in partition_iterator:
            yield v

    def iteritems(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Returns the same thing as .items()

        Returns:
            A generator that iterates over the columns of the frame.
        """
        return self.items()

    def itertuples(self, index=True, name='Pandas'):
        """Iterate over DataFrame rows as namedtuples.

        Args:
            index (boolean, default True): If True, return the index as the
                first element of the tuple.
            name (string, default "Pandas"): The name of the returned
            namedtuples or None to return regular tuples.
        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A tuple representing row data. See args for varying tuples.
        """
        index_iter = (self._row_metadata.partition_series(i).index
                      for i in range(len(self._row_partitions)))

        def itertuples_helper(part):
            df = ray.get(part)
            df.columns = self.columns
            df.index = next(index_iter)
            return df.itertuples(index=index, name=name)

        partition_iterator = PartitionIterator(self._row_partitions,
                                               itertuples_helper)

        for v in partition_iterator:
            yield v

    def join(self,
             other,
             on=None,
             how='left',
             lsuffix='',
             rsuffix='',
             sort=False):
        """Join two or more DataFrames, or a DataFrame with a collection.

        Args:
            other: What to join this DataFrame with.
            on: A column name to use from the left for the join.
            how: What type of join to conduct.
            lsuffix: The suffix to add to column names that match on left.
            rsuffix: The suffix to add to column names that match on right.
            sort: Whether or not to sort.

        Returns:
            The joined DataFrame.
        """

        if on is not None:
            raise NotImplementedError("Not yet.")

        if isinstance(other, pandas.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = DataFrame({other.name: other})

        if isinstance(other, DataFrame):
            # Joining the empty DataFrames with either index or columns is
            # fast. It gives us proper error checking for the edge cases that
            # would otherwise require a lot more logic.
            pandas.DataFrame(columns=self.columns).join(pandas.DataFrame(columns=other.columns), lsuffix=lsuffix, rsuffix=rsuffix).columns

            return DataFrame(data_manager=self._data_manager.concat(1, other._data_manager, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort))
        else:
            # This constraint carried over from Pandas.
            if on is not None:
                raise ValueError("Joining multiple DataFrames only supported"
                                 " for joining on index")

            # See note above about error checking with an empty join.
            pandas.DataFrame(columns=self.columns).join(
                [pandas.DataFrame(columns=obj.columns) for obj in other],
                lsuffix=lsuffix,
                rsuffix=rsuffix).columns

            return DataFrame(data_manager=self._data_manager.concat(1, [obj._data_manager for obj in other], how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort))

    def kurt(self,
             axis=None,
             skipna=None,
             level=None,
             numeric_only=None,
             **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def kurtosis(self,
                 axis=None,
                 skipna=None,
                 level=None,
                 numeric_only=None,
                 **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def last(self, offset):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def last_valid_index(self):
        """Return index for last non-NA/null value.

        Returns:
            scalar: type of index
        """
        return self._data_manager.last_valid_index()

    def le(self, other, axis='columns', level=None):
        """Checks element-wise that this is less than or equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the le over.
            level: The Multilevel index level to apply le over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._operator_helper(pandas.DataFrame.le, other, axis, level)

    def lookup(self, row_labels, col_labels):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def lt(self, other, axis='columns', level=None):
        """Checks element-wise that this is less than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the lt over.
            level: The Multilevel index level to apply lt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._operator_helper(pandas.DataFrame.lt, other, axis, level)

    def mad(self, axis=None, skipna=None, level=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def mask(self,
             cond,
             other=np.nan,
             inplace=False,
             axis=None,
             level=None,
             errors='raise',
             try_cast=False,
             raise_on_error=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def max(self,
            axis=None,
            skipna=None,
            level=None,
            numeric_only=None,
            **kwargs):
        """Perform max across the DataFrame.

        Args:
            axis (int): The axis to take the max on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The max of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.max(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs)

    def mean(self,
             axis=None,
             skipna=None,
             level=None,
             numeric_only=None,
             **kwargs):
        """Computes mean across the DataFrame.

        Args:
            axis (int): The axis to take the mean on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The mean of the DataFrame. (Pandas series)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.mean(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs)

    def median(self,
               axis=None,
               skipna=None,
               level=None,
               numeric_only=None,
               **kwargs):
        """Computes median across the DataFrame.

        Args:
            axis (int): The axis to take the median on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The median of the DataFrame. (Pandas series)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        return self._data_manager.median(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs)

    def melt(self,
             id_vars=None,
             value_vars=None,
             var_name=None,
             value_name='value',
             col_level=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def memory_usage(self, index=True, deep=False):
        def remote_func(df):
            return df.memory_usage(index=False, deep=deep)

        result = self._map_reduce(axis=0, map_func=remote_func)

        result.index = self.columns
        if index:
            index_value = self._row_metadata.index.memory_usage(deep=deep)
            return pandas.Series(index_value, index=['Index']).append(result)

        return result

    def merge(self,
              right,
              how='inner',
              on=None,
              left_on=None,
              right_on=None,
              left_index=False,
              right_index=False,
              sort=False,
              suffixes=('_x', '_y'),
              copy=True,
              indicator=False,
              validate=None):
        """Database style join, where common columns in "on" are merged.

        Args:
            right: The DataFrame to merge against.
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

        if not isinstance(right, DataFrame):
            raise ValueError("can not merge DataFrame with instance of type "
                             "{}".format(type(right)))

        if left_index is False or right_index is False:
            raise NotImplementedError(
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")

        args = (how, on, left_on, right_on, left_index, right_index, sort,
                suffixes, False, indicator, validate)

        left_cols = ray.put(self.columns)
        right_cols = ray.put(right.columns)

        # This can be put in a remote function because we don't need it until
        # the end, and the columns can be built asynchronously. This takes the
        # columns defining off the critical path and speeds up the overall
        # merge.
        new_columns = _merge_columns.remote(left_cols, right_cols, *args)

        if on is not None:
            if left_on is not None or right_on is not None:
                raise MergeError("Can only pass argument \"on\" OR \"left_on\""
                                 " and \"right_on\", not a combination of "
                                 "both.")
            if not is_list_like(on):
                on = [on]

            if next((True for key in on if key not in self), False) or \
                    next((True for key in on if key not in right), False):

                missing_key = \
                    next((str(key) for key in on if key not in self), "") + \
                    next((str(key) for key in on if key not in right), "")
                raise KeyError(missing_key)

        elif right_on is not None or right_index is True:
            if left_on is None and left_index is False:
                # Note: This is not the same error as pandas, but pandas throws
                # a ValueError NoneType has no len(), and I don't think that
                # helps enough.
                raise TypeError("left_on must be specified or left_index must "
                                "be true if right_on is specified.")

        elif left_on is not None or left_index is True:
            if right_on is None and right_index is False:
                # Note: See note above about TypeError.
                raise TypeError("right_on must be specified or right_index "
                                "must be true if right_on is specified.")

        if left_on is not None:
            if not is_list_like(left_on):
                left_on = [left_on]

            if next((True for key in left_on if key not in self), False):
                raise KeyError(next(key for key in left_on if key not in self))

        if right_on is not None:
            if not is_list_like(right_on):
                right_on = [right_on]

            if next((True for key in right_on if key not in right), False):
                raise KeyError(
                    next(key for key in right_on if key not in right))

        # There's a small chance that our partitions are already perfect, but
        # if it's not, we need to adjust them. We adjust the right against the
        # left because the defaults of merge rely on the order of the left. We
        # have to push the index down here, so if we're joining on the right's
        # index we go ahead and push it down here too.
        if not np.array_equal(self._row_metadata._lengths,
                              right._row_metadata._lengths) or right_index:

            repartitioned_right = np.array([
                _match_partitioning._submit(
                    args=(df, self._row_metadata._lengths, right.index),
                    num_return_vals=len(self._row_metadata._lengths))
                for df in right._col_partitions
            ]).T
        else:
            repartitioned_right = right._block_partitions

        if not left_index and not right_index:
            # Passing None to each call specifies that we don't care about the
            # left's index for the join.
            left_idx = itertools.repeat(None)

            # We only return the index if we need to update it, and that only
            # happens when either left_index or right_index is True. We will
            # use this value to add the return vals if we are getting an index
            # back.
            return_index = False
        else:
            # We build this to push the index down so that we can use it for
            # the join.
            left_idx = \
                (v.index for k, v in
                 self._row_metadata._coord_df.copy().groupby('partition'))
            return_index = True

        new_blocks = \
            np.array([_co_op_helper._submit(
                args=tuple([lambda x, y: x.merge(y, *args),
                            left_cols, right_cols,
                            len(self._block_partitions.T), next(left_idx)] +
                           np.concatenate(obj).tolist()),
                num_return_vals=len(self._block_partitions.T) + return_index)
                for obj in zip(self._block_partitions,
                               repartitioned_right)])

        if not return_index:
            # Default to RangeIndex if left_index and right_index both false.
            new_index = None
        else:
            new_index_parts = new_blocks[:, -1]
            new_index = _concat_index.remote(*new_index_parts)
            new_blocks = new_blocks[:, :-1]

        return DataFrame(
            block_partitions=new_blocks, columns=new_columns, index=new_index)

    def min(self,
            axis=None,
            skipna=None,
            level=None,
            numeric_only=None,
            **kwargs):
        """Perform min across the DataFrame.

        Args:
            axis (int): The axis to take the min on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The min of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.min(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs)

    def mod(self, other, axis='columns', level=None, fill_value=None):
        """Mods this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the mod against this.
            axis: The axis to mod over.
            level: The Multilevel index level to apply mod over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Mod applied.
        """
        return self._operator_helper(pandas.DataFrame.mod, other, axis, level,
                                     fill_value)

    def mode(self, axis=0, numeric_only=False):
        """Perform mode across the DataFrame.

        Args:
            axis (int): The axis to take the mode on.
            numeric_only (bool): if True, only apply to numeric columns.

        Returns:
            DataFrame: The mode of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis)

        return DataFrame(data_manager=self._data_manager.mode(axis=axis, numeric_only=numeric_only))

    def mul(self, other, axis='columns', level=None, fill_value=None):
        """Multiplies this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the multiply against this.
            axis: The axis to multiply over.
            level: The Multilevel index level to apply multiply over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Multiply applied.
        """
        return self._operator_helper(pandas.DataFrame.mul, other, axis, level,
                                     fill_value)

    def multiply(self, other, axis='columns', level=None, fill_value=None):
        """Synonym for mul.

        Args:
            other: The object to use to apply the multiply against this.
            axis: The axis to multiply over.
            level: The Multilevel index level to apply multiply over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Multiply applied.
        """
        return self.mul(other, axis, level, fill_value)

    def ne(self, other, axis='columns', level=None):
        """Checks element-wise that this is not equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the ne over.
            level: The Multilevel index level to apply ne over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._operator_helper(pandas.DataFrame.ne, other, axis, level)

    def nlargest(self, n, columns, keep='first'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def notna(self):
        """Perform notna across the DataFrame.

        Returns:
            Boolean DataFrame where value is False if corresponding
            value is NaN, True otherwise
        """
        return DataFrame(data_manager=self._data_manager.notna())

    def notnull(self):
        """Perform notnull across the DataFrame.

        Returns:
            Boolean DataFrame where value is False if corresponding
            value is NaN, True otherwise
        """
        return DataFrame(data_manager=self._data_manager.notnull())

    def nsmallest(self, n, columns, keep='first'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def nunique(self, axis=0, dropna=True):
        """Return Series with number of distinct
           observations over requested axis.

        Args:
            axis : {0 or 'index', 1 or 'columns'}, default 0
            dropna : boolean, default True

        Returns:
            nunique : Series
        """
        return self._data_manager.nunique(axis=axis, dropna=dropna)

    def pct_change(self,
                   periods=1,
                   fill_method='pad',
                   limit=None,
                   freq=None,
                   **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, *args, **kwargs)

        Args:
            func: function to apply to the df.
            args: positional arguments passed into ``func``.
            kwargs: a dictionary of keyword arguments passed into ``func``.

        Returns:
            object: the return type of ``func``.
        """
        return com._pipe(self, func, *args, **kwargs)

    def pivot(self, index=None, columns=None, values=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def pivot_table(self,
                    values=None,
                    index=None,
                    columns=None,
                    aggfunc='mean',
                    fill_value=None,
                    margins=False,
                    dropna=True,
                    margins_name='All'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def plot(self,
             x=None,
             y=None,
             kind='line',
             ax=None,
             subplots=False,
             sharex=None,
             sharey=False,
             layout=None,
             figsize=None,
             use_index=True,
             title=None,
             grid=None,
             legend=True,
             style=None,
             logx=False,
             logy=False,
             loglog=False,
             xticks=None,
             yticks=None,
             xlim=None,
             ylim=None,
             rot=None,
             fontsize=None,
             colormap=None,
             table=False,
             yerr=None,
             xerr=None,
             secondary_y=False,
             sort_columns=False,
             **kwds):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def pop(self, item):
        """Pops an item from this DataFrame and returns it.

        Args:
            item (str): Column label to be popped

        Returns:
            A Series containing the popped values. Also modifies this
            DataFrame.
        """
        result = self[item]
        del self[item]
        return result

    def pow(self, other, axis='columns', level=None, fill_value=None):
        """Pow this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the pow against this.
            axis: The axis to pow over.
            level: The Multilevel index level to apply pow over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Pow applied.
        """
        return self._operator_helper(pandas.DataFrame.pow, other, axis, level,
                                     fill_value)

    def prod(self,
             axis=None,
             skipna=None,
             level=None,
             numeric_only=None,
             min_count=1,
             **kwargs):
        """Return the product of the values for the requested axis

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            level : int or level name, default None
            numeric_only : boolean, default None
            min_count : int, default 1

        Returns:
            prod : Series or DataFrame (if level specified)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs)

    def product(self,
                axis=None,
                skipna=None,
                level=None,
                numeric_only=None,
                min_count=1,
                **kwargs):
        """Return the product of the values for the requested axis

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            level : int or level name, default None
            numeric_only : boolean, default None
            min_count : int, default 1

        Returns:
            product : Series or DataFrame (if level specified)
        """
        return self.prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs)

    def quantile(self,
                 q=0.5,
                 axis=0,
                 numeric_only=True,
                 interpolation='linear'):
        """Return values at the given quantile over requested axis,
            a la numpy.percentile.

        Args:
            q (float): 0 <= q <= 1, the quantile(s) to compute
            axis (int): 0 or 'index' for row-wise,
                        1 or 'columns' for column-wise
            interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
                Specifies which interpolation method to use

        Returns:
            quantiles : Series or DataFrame
                    If q is an array, a DataFrame will be returned where the
                    index is q, the columns are the columns of self, and the
                    values are the quantiles.

                    If q is a float, a Series will be returned where the
                    index is the columns of self and the values
                    are the quantiles.
        """

        def check_bad_dtype(t):
            return t == np.dtype('O') or is_timedelta64_dtype(t)

        if not numeric_only:
            # check if there are any object columns
            if all(check_bad_dtype(t) for t in self.dtypes):
                raise TypeError("can't multiply sequence by non-int of type "
                                "'float'")
            else:
                if next((True for t in self.dtypes if check_bad_dtype(t)),
                        False):
                    dtype = next(t for t in self.dtypes if check_bad_dtype(t))
                    raise ValueError("Cannot compare type '{}' with type '{}'"
                                     .format(type(dtype), float))
        else:
            # Normally pandas returns this near the end of the quantile, but we
            # can't afford the overhead of running the entire operation before
            # we error.
            if all(check_bad_dtype(t) for t in self.dtypes):
                raise ValueError("need at least one array to concatenate")

        # check that all qs are between 0 and 1
        pandas.DataFrame()._check_percentile(q)

        axis = pandas.DataFrame()._get_axis_number(axis)

        if isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list)):
            return DataFrame(data_manager=self._data_manager.quantile_for_list_of_values(q=q, axis=axis, numeric_only=numeric_only, interpolation=interpolation))

        else:
            return self._data_manager.quantile_for_single_value(q=q, axis=axis, numeric_only=numeric_only, interpolation=interpolation)

    def query(self, expr, inplace=False, **kwargs):
        """Queries the Dataframe with a boolean expression

        Returns:
            A new DataFrame if inplace=False
        """
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")

        new_manager = self._data_manager.query(expr, **kwargs)

        if inplace:
            self._update_inplace(new_manager=new_manager)
        else:
            return DataFrame(data_manager=new_manager)

    def radd(self, other, axis='columns', level=None, fill_value=None):
        return self.add(other, axis, level, fill_value)

    def rank(self,
             axis=0,
             method='average',
             numeric_only=None,
             na_option='keep',
             ascending=True,
             pct=False):
        """
        Compute numerical data ranks (1 through n) along axis.
        Equal values are assigned a rank that is the [method] of
        the ranks of those values.

        Args:
            axis (int): 0 or 'index' for row-wise,
                        1 or 'columns' for column-wise
            method: {'average', 'min', 'max', 'first', 'dense'}
                Specifies which method to use for equal vals
            numeric_only (boolean)
                Include only float, int, boolean data.
            na_option: {'keep', 'top', 'bottom'}
                Specifies how to handle NA options
            ascending (boolean):
                Decedes ranking order
            pct (boolean):
                Computes percentage ranking of data
        Returns:
            A new DataFrame
        """
        axis = pandas.DataFrame()._get_axis_number(axis)

        return DataFrame(data_manager=self._data_manager.rank(
            axis=axis,
            method=method,
            numeric_only=numeric_only,
            na_option=na_option,
            ascending=ascending,
            pct=pct))

    def rdiv(self, other, axis='columns', level=None, fill_value=None):
        return self.div(other, axis, level, fill_value)

    def reindex(self,
                labels=None,
                index=None,
                columns=None,
                axis=None,
                method=None,
                copy=True,
                level=None,
                fill_value=np.nan,
                limit=None,
                tolerance=None):
        if level is not None:
            raise NotImplementedError(
                "Multilevel Index not Implemented. "
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")

        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None \
            else 0
        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels

        if index is not None:
            new_manager = self._data_manager.reindex(0, index, method=method, fill_value=fill_value, limit=limit, tolerance=tolerance)
        else:
            new_manager = self._data_manager

        if columns is not None:
            final_manager = new_manager.reindex(1, columns, method=method, fill_value=fill_value, limit=limit, tolerance=tolerance)
        else:
            final_manager = new_manager

        if copy:
            return DataFrame(data_manager=final_manager)

        self._update_inplace(new_manager=final_manager)

    def reindex_axis(self,
                     labels,
                     axis=0,
                     method=None,
                     level=None,
                     copy=True,
                     limit=None,
                     fill_value=np.nan):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def reindex_like(self,
                     other,
                     method=None,
                     copy=True,
                     limit=None,
                     tolerance=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def rename(self,
               mapper=None,
               index=None,
               columns=None,
               axis=None,
               copy=True,
               inplace=False,
               level=None):
        """Alters axes labels.

        Args:
            mapper, index, columns: Transformations to apply to the axis's
                values.
            axis: Axis to target with mapper.
            copy: Also copy underlying data.
            inplace: Whether to return a new DataFrame.
            level: Only rename a specific level of a MultiIndex.

        Returns:
            If inplace is False, a new DataFrame with the updated axes.
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')

        # We have to do this with the args because of how rename handles
        # kwargs. It doesn't ignore None values passed in, so we have to filter
        # them ourselves.
        args = locals()
        kwargs = {
            k: v
            for k, v in args.items() if v is not None and k != "self"
        }
        # inplace should always be true because this is just a copy, and we
        # will use the results after.
        kwargs['inplace'] = True

        df_to_rename = pandas.DataFrame(index=self.index, columns=self.columns)
        df_to_rename.rename(**kwargs)

        if inplace:
            obj = self
        else:
            obj = self.copy()

        obj.index = df_to_rename.index
        obj.columns = df_to_rename.columns

        if not inplace:
            return obj

    def rename_axis(self, mapper, axis=0, copy=True, inplace=False):
        axes_is_columns = axis == 1 or axis == "columns"
        renamed = self if inplace else self.copy()
        if axes_is_columns:
            renamed.columns.name = mapper
        else:
            renamed.index.name = mapper
        if not inplace:
            return renamed

    def _set_axis_name(self, name, axis=0, inplace=False):
        """Alter the name or names of the axis.

        Args:
            name: Name for the Index, or list of names for the MultiIndex
            axis: 0 or 'index' for the index; 1 or 'columns' for the columns
            inplace: Whether to modify `self` directly or return a copy

        Returns:
            Type of caller or None if inplace=True.
        """
        axes_is_columns = axis == 1 or axis == "columns"
        renamed = self if inplace else self.copy()
        if axes_is_columns:
            renamed.columns.set_names(name)
        else:
            renamed.index.set_names(name)

        if not inplace:
            return renamed

    def reorder_levels(self, order, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def replace(self,
                to_replace=None,
                value=None,
                inplace=False,
                limit=None,
                regex=False,
                method='pad',
                axis=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def resample(self,
                 rule,
                 how=None,
                 axis=0,
                 fill_method=None,
                 closed=None,
                 label=None,
                 convention='start',
                 kind=None,
                 loffset=None,
                 limit=None,
                 base=0,
                 on=None,
                 level=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def reset_index(self,
                    level=None,
                    drop=False,
                    inplace=False,
                    col_level=0,
                    col_fill=''):
        """Reset this index to default and create column from current index.

        Args:
            level: Only remove the given levels from the index. Removes all
                levels by default
            drop: Do not try to insert index into DataFrame columns. This
                resets the index to the default integer index.
            inplace: Modify the DataFrame in place (do not create a new object)
            col_level : If the columns have multiple levels, determines which
                level the labels are inserted into. By default it is inserted
                into the first level.
            col_fill: If the columns have multiple levels, determines how the
                other levels are named. If None then the index name is
                repeated.

        Returns:
            A new DataFrame if inplace is False, None otherwise.
        """
        # TODO Implement level
        if level is not None:
            raise NotImplementedError("Level not yet supported!")
        inplace = validate_bool_kwarg(inplace, 'inplace')

        # Error checking for matching Pandas. Pandas does not allow you to
        # insert a dropped index into a DataFrame if these columns already
        # exist.
        if not drop and all(n in self.columns for n in ["level_0", "index"]):
            raise ValueError("cannot insert level_0, already exists")

        new_manager = self._data_manager.reset_index(drop=drop, level=level)
        if inplace:
            self._update_inplace(new_manager=new_manager)
        else:
            return DataFrame(data_manager=new_manager)

    def rfloordiv(self, other, axis='columns', level=None, fill_value=None):
        return self.floordiv(other, axis, level, fill_value)

    def rmod(self, other, axis='columns', level=None, fill_value=None):
        return self.mod(other, axis, level, fill_value)

    def rmul(self, other, axis='columns', level=None, fill_value=None):
        return self.mul(other, axis, level, fill_value)

    def rolling(self,
                window,
                min_periods=None,
                freq=None,
                center=False,
                win_type=None,
                on=None,
                axis=0,
                closed=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def round(self, decimals=0, *args, **kwargs):
        """Round each element in the DataFrame.

        Args:
            decimals: The number of decimals to round to.

        Returns:
             A new DataFrame.
        """
        return DataFrame(data_manager=self._data_manager.round(decimals=decimals, **kwargs))

    def rpow(self, other, axis='columns', level=None, fill_value=None):
        return self.pow(other, axis, level, fill_value)

    def rsub(self, other, axis='columns', level=None, fill_value=None):
        return self.sub(other, axis, level, fill_value)

    def rtruediv(self, other, axis='columns', level=None, fill_value=None):
        return self.truediv(other, axis, level, fill_value)

    def sample(self,
               n=None,
               frac=None,
               replace=False,
               weights=None,
               random_state=None,
               axis=None):
        """Returns a random sample of items from an axis of object.

        Args:
            n: Number of items from axis to return. Cannot be used with frac.
                Default = 1 if frac = None.
            frac: Fraction of axis items to return. Cannot be used with n.
            replace: Sample with or without replacement. Default = False.
            weights: Default 'None' results in equal probability weighting.
                If passed a Series, will align with target object on index.
                Index values in weights not found in sampled object will be
                ignored and index values in sampled object not in weights will
                be assigned weights of zero. If called on a DataFrame, will
                accept the name of a column when axis = 0. Unless weights are
                a Series, weights must be same length as axis being sampled.
                If weights do not sum to 1, they will be normalized to sum
                to 1. Missing values in the weights column will be treated as
                zero. inf and -inf values not allowed.
            random_state: Seed for the random number generator (if int), or
                numpy RandomState object.
            axis: Axis to sample. Accepts axis number or name.

        Returns:
            A new Dataframe
        """

        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None \
            else 0

        if axis == 0:
            axis_labels = self._data_manager.index
            axis_length = len(axis_labels)
        else:
            axis_labels = self._data_manager.column
            axis_length = len(axis_labels)

        if weights is not None:

            # Index of the weights Series should correspond to the index of the
            # Dataframe in order to sample
            if isinstance(weights, pandas.Series):
                weights = weights.reindex(self.axes[axis])

            # If weights arg is a string, the weights used for sampling will
            # the be values in the column corresponding to that string
            if isinstance(weights, string_types):
                if axis == 0:
                    try:
                        weights = self[weights]
                    except KeyError:
                        raise KeyError("String passed to weights not a "
                                       "valid column")
                else:
                    raise ValueError("Strings can only be passed to "
                                     "weights when sampling from rows on "
                                     "a DataFrame")

            weights = pandas.Series(weights, dtype='float64')

            if len(weights) != axis_length:
                raise ValueError("Weights and axis to be sampled must be of "
                                 "same length")

            if (weights == np.inf).any() or (weights == -np.inf).any():
                raise ValueError("weight vector may not include `inf` values")

            if (weights < 0).any():
                raise ValueError("weight vector many not include negative "
                                 "values")

            # weights cannot be NaN when sampling, so we must set all nan
            # values to 0
            weights = weights.fillna(0)

            # If passed in weights are not equal to 1, renormalize them
            # otherwise numpy sampling function will error
            weights_sum = weights.sum()
            if weights_sum != 1:
                if weights_sum != 0:
                    weights = weights / weights_sum
                else:
                    raise ValueError("Invalid weights: weights sum to zero")

            weights = weights.values

        if n is None and frac is None:
            # default to n = 1 if n and frac are both None (in accordance with
            # Pandas specification)
            n = 1
        elif n is not None and frac is None and n % 1 != 0:
            # n must be an integer
            raise ValueError("Only integers accepted as `n` values")
        elif n is None and frac is not None:
            # compute the number of samples based on frac
            n = int(round(frac * axis_length))
        elif n is not None and frac is not None:
            # Pandas specification does not allow both n and frac to be passed
            # in
            raise ValueError('Please enter a value for `frac` OR `n`, not '
                             'both')
        if n < 0:
            raise ValueError("A negative number of rows requested. Please "
                             "provide positive value.")

        if n == 0:
            # An Empty DataFrame is returned if the number of samples is 0.
            # The Empty Dataframe should have either columns or index specified
            # depending on which axis is passed in.
            return DataFrame(
                columns=[] if axis == 1 else self.columns,
                index=self.index if axis == 1 else [])

        if random_state is not None:
            # Get a random number generator depending on the type of
            # random_state that is passed in
            if isinstance(random_state, int):
                random_num_gen = np.random.RandomState(random_state)
            elif isinstance(random_state, np.random.randomState):
                random_num_gen = random_state
            else:
                # random_state must be an int or a numpy RandomState object
                raise ValueError("Please enter an `int` OR a "
                                 "np.random.RandomState for random_state")

            # choose random numbers and then get corresponding labels from
            # chosen axis
            sample_indices = random_num_gen.choice(
                    np.arange(0, axis_length), size=n, replace=replace)
            samples = axis_labels[sample_indices]
        else:
            # randomly select labels from chosen axis
            samples = np.random.choice(
                a=axis_labels, size=n, replace=replace, p=weights)

        if axis == 1:
            data_manager = self._data_manager.getitem_col_array(samples)
            return DataFrame(data_manager=data_manager)
        else:
            data_manager = self._data_manager.getitem_row_array(samples)
            return DataFrame(data_manager=data_manager)

    def select(self, crit, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def select_dtypes(self, include=None, exclude=None):
        # Validates arguments for whether both include and exclude are None or
        # if they are disjoint. Also invalidates string dtypes.
        pandas.DataFrame().select_dtypes(include, exclude)

        if include and not is_list_like(include):
            include = [include]
        elif not include:
            include = []

        if exclude and not is_list_like(exclude):
            exclude = [exclude]
        elif not exclude:
            exclude = []

        sel = tuple(map(set, (include, exclude)))

        include, exclude = map(lambda x: set(map(_get_dtype_from_object, x)),
                               sel)

        include_these = pandas.Series(not bool(include), index=self.columns)
        exclude_these = pandas.Series(not bool(exclude), index=self.columns)

        def is_dtype_instance_mapper(column, dtype):
            return column, functools.partial(issubclass, dtype.type)

        for column, f in itertools.starmap(is_dtype_instance_mapper,
                                           self.dtypes.iteritems()):
            if include:  # checks for the case of empty include or exclude
                include_these[column] = any(map(f, include))
            if exclude:
                exclude_these[column] = not any(map(f, exclude))

        dtype_indexer = include_these & exclude_these
        indicate = [
            i for i in range(len(dtype_indexer.values))
            if not dtype_indexer.values[i]
        ]
        return self.drop(columns=self.columns[indicate], inplace=False)

    def sem(self,
            axis=None,
            skipna=None,
            level=None,
            ddof=1,
            numeric_only=None,
            **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def set_axis(self, labels, axis=0, inplace=None):
        """Assign desired index to given axis.

        Args:
            labels (pandas.Index or list-like): The Index to assign.
            axis (string or int): The axis to reassign.
            inplace (bool): Whether to make these modifications inplace.

        Returns:
            If inplace is False, returns a new DataFrame, otherwise None.
        """
        if is_scalar(labels):
            warnings.warn(
                'set_axis now takes "labels" as first argument, and '
                '"axis" as named parameter. The old form, with "axis" as '
                'first parameter and \"labels\" as second, is still supported '
                'but will be deprecated in a future version of pandas.',
                FutureWarning,
                stacklevel=2)
            labels, axis = axis, labels

        if inplace is None:
            warnings.warn(
                'set_axis currently defaults to operating inplace.\nThis '
                'will change in a future version of pandas, use '
                'inplace=True to avoid this warning.',
                FutureWarning,
                stacklevel=2)
            inplace = True
        if inplace:
            setattr(self, pandas.DataFrame()._get_axis_name(axis), labels)
        else:
            obj = self.copy()
            obj.set_axis(labels, axis=axis, inplace=True)
            return obj

    def set_index(self,
                  keys,
                  drop=True,
                  append=False,
                  inplace=False,
                  verify_integrity=False):
        """Set the DataFrame index using one or more existing columns.

        Args:
            keys: column label or list of column labels / arrays.
            drop (boolean): Delete columns to be used as the new index.
            append (boolean): Whether to append columns to existing index.
            inplace (boolean): Modify the DataFrame in place.
            verify_integrity (boolean): Check the new index for duplicates.
                Otherwise defer the check until necessary. Setting to False
                will improve the performance of this method

        Returns:
            If inplace is set to false returns a new DataFrame, otherwise None.
        """
        inplace = validate_bool_kwarg(inplace, 'inplace')
        if not isinstance(keys, list):
            keys = [keys]

        if inplace:
            frame = self
        else:
            frame = self.copy()

        arrays = []
        names = []
        if append:
            names = [x for x in self.index.names]
            if isinstance(self.index, pandas.MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)

        to_remove = []
        for col in keys:
            if isinstance(col, pandas.MultiIndex):
                # append all but the last column so we don't have to modify
                # the end of this loop
                for n in range(col.nlevels - 1):
                    arrays.append(col._get_level_values(n))

                level = col._get_level_values(col.nlevels - 1)
                names.extend(col.names)
            elif isinstance(col, pandas.Series):
                level = col._values
                names.append(col.name)
            elif isinstance(col, pandas.Index):
                level = col
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray, pandas.Index)):
                level = col
                names.append(None)
            else:
                level = frame[col]._values
                names.append(col)
                if drop:
                    to_remove.append(col)
            arrays.append(level)

        index = _ensure_index_from_sequences(arrays, names)

        if verify_integrity and not index.is_unique:
            duplicates = index.get_duplicates()
            raise ValueError('Index has duplicate keys: %s' % duplicates)

        for c in to_remove:
            del frame[c]

        # clear up memory usage
        index._cleanup()

        frame.index = index

        if not inplace:
            return frame

    def set_value(self, index, col, value, takeable=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def shift(self, periods=1, freq=None, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def skew(self,
             axis=None,
             skipna=None,
             level=None,
             numeric_only=None,
             **kwargs):
        """Return unbiased skew over requested axis Normalized by N-1

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
            Exclude NA/null values when computing the result.
            level : int or level name, default None
            numeric_only : boolean, default None

        Returns:
            skew : Series or DataFrame (if level specified)
        """
        return self._data_manager.skew(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs)

    def slice_shift(self, periods=1, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def sort_index(self,
                   axis=0,
                   level=None,
                   ascending=True,
                   inplace=False,
                   kind='quicksort',
                   na_position='last',
                   sort_remaining=True,
                   by=None):
        """Sort a DataFrame by one of the indices (columns or index).

        Args:
            axis: The axis to sort over.
            level: The MultiIndex level to sort over.
            ascending: Ascending or descending
            inplace: Whether or not to update this DataFrame inplace.
            kind: How to perform the sort.
            na_position: Where to position NA on the sort.
            sort_remaining: On Multilevel Index sort based on all levels.
            by: (Deprecated) argument to pass to sort_values.

        Returns:
            A sorted DataFrame
        """
        if level is not None:
            raise NotImplementedError("Multilevel index not yet implemented.")

        if by is not None:
            warnings.warn(
                "by argument to sort_index is deprecated, "
                "please use .sort_values(by=...)",
                FutureWarning,
                stacklevel=2)
            if level is not None:
                raise ValueError("unable to simultaneously sort by and level")
            return self.sort_values(
                by, axis=axis, ascending=ascending, inplace=inplace)

        axis = pandas.DataFrame()._get_axis_number(axis)

        if not axis:
            new_index = self.index.sort_values(ascending=ascending)
            new_columns = None
        else:
            new_index = None
            new_columns = self.columns.sort_values(ascending=ascending)

        return self.reindex(index=new_index, columns=new_columns)

    def sort_values(self,
                    by,
                    axis=0,
                    ascending=True,
                    inplace=False,
                    kind='quicksort',
                    na_position='last'):
        """Sorts by a column/row or list of columns/rows.

        Args:
            by: A list of labels for the axis to sort over.
            axis: The axis to sort.
            ascending: Sort in ascending or descending order.
            inplace: If true, do the operation inplace.
            kind: How to sort.
            na_position: Where to put np.nan values.

        Returns:
             A sorted DataFrame.
        """

        axis = pandas.DataFrame()._get_axis_number(axis)

        if not is_list_like(by):
            by = [by]

        if axis == 0:
            broadcast_value_dict = {str(col): self[col] for col in by}
            broadcast_values = pandas.DataFrame(broadcast_value_dict)
        else:
            broadcast_value_list = [
                to_pandas(self[row::len(self.index)]) for row in by
            ]

            index_builder = list(zip(broadcast_value_list, by))

            for row, idx in index_builder:
                row.index = [str(idx)]

            broadcast_values = \
                pandas.concat([row for row, idx in index_builder], copy=False)

        # We are converting the by to string here so that we don't have a
        # collision with the RangeIndex on the inner frame. It is cheap and
        # gaurantees that we sort by the correct column.
        by = [str(col) for col in by]

        args = (by, axis, ascending, False, kind, na_position)

        def _sort_helper(df, broadcast_values, axis, *args):
            """Sorts the data on a partition.

            Args:
                df: The DataFrame to sort.
                broadcast_values: The by DataFrame to use for the sort.
                axis: The axis to sort over.
                args: The args for the sort.

            Returns:
                 A new sorted DataFrame.
            """
            if axis == 0:
                broadcast_values.index = df.index
                names = broadcast_values.columns
            else:
                broadcast_values.columns = df.columns
                names = broadcast_values.index

            return pandas.concat([df, broadcast_values], axis=axis ^ 1,
                                 copy=False).sort_values(*args) \
                .drop(names, axis=axis ^ 1)

        if axis == 0:
            new_column_partitions = _map_partitions(
                lambda df: _sort_helper(df, broadcast_values, axis, *args),
                self._col_partitions)

            new_row_partitions = None
            new_columns = self.columns

            # This is important because it allows us to get the axis that we
            # aren't sorting over. We need the order of the columns/rows and
            # this will provide that in the return value.
            new_index = broadcast_values.sort_values(*args).index
        else:
            new_row_partitions = _map_partitions(
                lambda df: _sort_helper(df, broadcast_values, axis, *args),
                self._row_partitions)

            new_column_partitions = None
            new_columns = broadcast_values.sort_values(*args).columns
            new_index = self.index

        if inplace:
            self._update_inplace(
                row_partitions=new_row_partitions,
                col_partitions=new_column_partitions,
                columns=new_columns,
                index=new_index)
        else:
            return DataFrame(
                row_partitions=new_row_partitions,
                col_partitions=new_column_partitions,
                columns=new_columns,
                index=new_index,
                dtypes_cache=self._dtypes_cache)

    def sortlevel(self,
                  level=0,
                  axis=0,
                  ascending=True,
                  inplace=False,
                  sort_remaining=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def squeeze(self, axis=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def stack(self, level=-1, dropna=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def std(self,
            axis=None,
            skipna=None,
            level=None,
            ddof=1,
            numeric_only=None,
            **kwargs):
        """Computes standard deviation across the DataFrame.

        Args:
            axis (int): The axis to take the std on.
            skipna (bool): True to skip NA values, false otherwise.
            ddof (int): degrees of freedom

        Returns:
            The std of the DataFrame (Pandas Series)
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.std(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs)

    def sub(self, other, axis='columns', level=None, fill_value=None):
        """Subtract a DataFrame/Series/scalar from this DataFrame.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: THe axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self._operator_helper(pandas.DataFrame.sub, other, axis, level,
                                     fill_value)

    def subtract(self, other, axis='columns', level=None, fill_value=None):
        """Alias for sub.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: THe axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self.sub(other, axis, level, fill_value)

    def swapaxes(self, axis1, axis2, copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def swaplevel(self, i=-2, j=-1, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def tail(self, n=5):
        """Get the last n rows of the DataFrame.

        Args:
            n (int): The number of rows to return.

        Returns:
            A new DataFrame with the last n rows of this DataFrame.
        """
        if n >= len(self.index):
            return self.copy()

        return DataFrame(data_manager=self._data_manager.tail(n))

    def take(self, indices, axis=0, convert=None, is_copy=True, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_clipboard(self, excel=None, sep=None, **kwargs):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_clipboard(excel, sep, **kwargs)

    def to_csv(self,
               path_or_buf=None,
               sep=",",
               na_rep="",
               float_format=None,
               columns=None,
               header=True,
               index=True,
               index_label=None,
               mode="w",
               encoding=None,
               compression=None,
               quoting=None,
               quotechar='"',
               line_terminator="\n",
               chunksize=None,
               tupleize_cols=None,
               date_format=None,
               doublequote=True,
               escapechar=None,
               decimal="."):

        kwargs = {
            'path_or_buf': path_or_buf,
            'sep': sep,
            'na_rep': na_rep,
            'float_format': float_format,
            'columns': columns,
            'header': header,
            'index': index,
            'index_label': index_label,
            'mode': mode,
            'encoding': encoding,
            'compression': compression,
            'quoting': quoting,
            'quotechar': quotechar,
            'line_terminator': line_terminator,
            'chunksize': chunksize,
            'tupleize_cols': tupleize_cols,
            'date_format': date_format,
            'doublequote': doublequote,
            'escapechar': escapechar,
            'decimal': decimal
        }

        if compression is not None:
            warnings.warn("Defaulting to Pandas implementation",
                          PendingDeprecationWarning)
            return to_pandas(self).to_csv(**kwargs)

        if tupleize_cols is not None:
            warnings.warn(
                "The 'tupleize_cols' parameter is deprecated and "
                "will be removed in a future version",
                FutureWarning,
                stacklevel=2)
        else:
            tupleize_cols = False

        remote_kwargs_id = ray.put(dict(kwargs, path_or_buf=None))
        columns_id = ray.put(self.columns)

        def get_csv_str(df, index, columns, header, kwargs):
            df.index = index
            df.columns = columns
            kwargs["header"] = header
            return df.to_csv(**kwargs)

        idxs = [0] + np.cumsum(self._row_metadata._lengths).tolist()
        idx_args = [
            self.index[idxs[i]:idxs[i + 1]]
            for i in range(len(self._row_partitions))
        ]
        csv_str_ids = _map_partitions(
            get_csv_str, self._row_partitions, idx_args,
            [columns_id] * len(self._row_partitions),
            [header] + [False] * (len(self._row_partitions) - 1),
            [remote_kwargs_id] * len(self._row_partitions))

        if path_or_buf is None:
            buf = io.StringIO()
        elif isinstance(path_or_buf, str):
            buf = open(path_or_buf, mode)
        else:
            buf = path_or_buf

        for csv_str_id in csv_str_ids:
            buf.write(ray.get(csv_str_id))
            buf.flush()

        result = None
        if path_or_buf is None:
            result = buf.getvalue()
            buf.close()
        elif isinstance(path_or_buf, str):
            buf.close()
        return result

    def to_dense(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_dict(self, orient='dict', into=dict):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_excel(self,
                 excel_writer,
                 sheet_name='Sheet1',
                 na_rep='',
                 float_format=None,
                 columns=None,
                 header=True,
                 index=True,
                 index_label=None,
                 startrow=0,
                 startcol=0,
                 engine=None,
                 merge_cells=True,
                 encoding=None,
                 inf_rep='inf',
                 verbose=True,
                 freeze_panes=None):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_excel(excel_writer, sheet_name, na_rep, float_format,
                            columns, header, index, index_label, startrow,
                            startcol, engine, merge_cells, encoding, inf_rep,
                            verbose, freeze_panes)

    def to_feather(self, fname):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_feather(fname)

    def to_gbq(self,
               destination_table,
               project_id,
               chunksize=10000,
               verbose=True,
               reauth=False,
               if_exists='fail',
               private_key=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_hdf(self, path_or_buf, key, **kwargs):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_hdf(path_or_buf, key, **kwargs)

    def to_html(self,
                buf=None,
                columns=None,
                col_space=None,
                header=True,
                index=True,
                na_rep='np.NaN',
                formatters=None,
                float_format=None,
                sparsify=None,
                index_names=True,
                justify=None,
                bold_rows=True,
                classes=None,
                escape=True,
                max_rows=None,
                max_cols=None,
                show_dimensions=False,
                notebook=False,
                decimal='.',
                border=None):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_html(buf, columns, col_space, header, index, na_rep,
                           formatters, float_format, sparsify, index_names,
                           justify, bold_rows, classes, escape, max_rows,
                           max_cols, show_dimensions, notebook, decimal,
                           border)

    def to_json(self,
                path_or_buf=None,
                orient=None,
                date_format=None,
                double_precision=10,
                force_ascii=True,
                date_unit='ms',
                default_handler=None,
                lines=False,
                compression=None):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_json(path_or_buf, orient, date_format, double_precision,
                           force_ascii, date_unit, default_handler, lines,
                           compression)

    def to_latex(self,
                 buf=None,
                 columns=None,
                 col_space=None,
                 header=True,
                 index=True,
                 na_rep='np.NaN',
                 formatters=None,
                 float_format=None,
                 sparsify=None,
                 index_names=True,
                 bold_rows=False,
                 column_format=None,
                 longtable=None,
                 escape=None,
                 encoding=None,
                 decimal='.',
                 multicolumn=None,
                 multicolumn_format=None,
                 multirow=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_msgpack(self, path_or_buf=None, encoding='utf-8', **kwargs):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_msgpack(path_or_buf, encoding, **kwargs)

    def to_panel(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_parquet(self, fname, engine='auto', compression='snappy', **kwargs):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_parquet(fname, engine, compression, **kwargs)

    def to_period(self, freq=None, axis=0, copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_pickle(self,
                  path,
                  compression='infer',
                  protocol=pkl.HIGHEST_PROTOCOL):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_pickle(path, compression, protocol)

    def to_records(self, index=True, convert_datetime64=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_sparse(self, fill_value=None, kind='block'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_sql(self,
               name,
               con,
               flavor=None,
               schema=None,
               if_exists='fail',
               index=True,
               index_label=None,
               chunksize=None,
               dtype=None):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_sql(name, con, flavor, schema, if_exists, index,
                          index_label, chunksize, dtype)

    def to_stata(self,
                 fname,
                 convert_dates=None,
                 write_index=True,
                 encoding='latin-1',
                 byteorder=None,
                 time_stamp=None,
                 data_label=None,
                 variable_labels=None):

        warnings.warn("Defaulting to Pandas implementation",
                      PendingDeprecationWarning)

        port_frame = to_pandas(self)
        port_frame.to_stata(fname, convert_dates, write_index, encoding,
                            byteorder, time_stamp, data_label, variable_labels)

    def to_string(self,
                  buf=None,
                  columns=None,
                  col_space=None,
                  header=True,
                  index=True,
                  na_rep='np.NaN',
                  formatters=None,
                  float_format=None,
                  sparsify=None,
                  index_names=True,
                  justify=None,
                  line_width=None,
                  max_rows=None,
                  max_cols=None,
                  show_dimensions=False):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_timestamp(self, freq=None, how='start', axis=0, copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def to_xarray(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def transform(self, func, *args, **kwargs):
        kwargs["is_transform"] = True
        result = self.agg(func, *args, **kwargs)
        try:
            result.columns = self.columns
            result.index = self.index
        except ValueError:
            raise ValueError("transforms cannot produce aggregated results")
        return result

    def truediv(self, other, axis='columns', level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._operator_helper(pandas.DataFrame.truediv, other, axis,
                                     level, fill_value)

    def truncate(self, before=None, after=None, axis=None, copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def tshift(self, periods=1, freq=None, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def tz_localize(self, tz, axis=0, level=None, copy=True,
                    ambiguous='raise'):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def unstack(self, level=-1, fill_value=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def update(self,
               other,
               join='left',
               overwrite=True,
               filter_func=None,
               raise_conflict=False):
        """Modify DataFrame in place using non-NA values from other.

        Args:
            other: DataFrame, or object coercible into a DataFrame
            join: {'left'}, default 'left'
            overwrite: If True then overwrite values for common keys in frame
            filter_func: Can choose to replace values other than NA.
            raise_conflict: If True, will raise an error if the DataFrame and
                other both contain data in the same place.

        Returns:
            None
        """
        if raise_conflict:
            raise NotImplementedError(
                "raise_conflict parameter not yet supported. "
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")

        if not isinstance(other, DataFrame):
            other = DataFrame(other)

        def update_helper(x, y):
            x.update(y, join, overwrite, filter_func, False)
            return x

        self._inter_df_op_helper(update_helper, other, join, None, inplace=True)

    def var(self,
            axis=None,
            skipna=None,
            level=None,
            ddof=1,
            numeric_only=None,
            **kwargs):
        """Computes variance across the DataFrame.

        Args:
            axis (int): The axis to take the variance on.
            skipna (bool): True to skip NA values, false otherwise.
            ddof (int): degrees of freedom

        Returns:
            The variance of the DataFrame.
        """
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        return self._data_manager.var(
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs)

    def where(self,
              cond,
              other=np.nan,
              inplace=False,
              axis=None,
              level=None,
              errors='raise',
              try_cast=False,
              raise_on_error=None):
        """Replaces values not meeting condition with values in other.

        Args:
            cond: A condition to be met, can be callable, array-like or a
                DataFrame.
            other: A value or DataFrame of values to use for setting this.
            inplace: Whether or not to operate inplace.
            axis: The axis to apply over. Only valid when a Series is passed
                as other.
            level: The MultiLevel index level to apply over.
            errors: Whether or not to raise errors. Does nothing in Pandas.
            try_cast: Try to cast the result back to the input type.
            raise_on_error: Whether to raise invalid datatypes (deprecated).

        Returns:
            A new DataFrame with the replaced values.
        """

        inplace = validate_bool_kwarg(inplace, 'inplace')

        if isinstance(other, pandas.Series) and axis is None:
            raise ValueError("Must specify axis=0 or 1")

        if level is not None:
            raise NotImplementedError("Multilevel Index not yet supported on "
                                      "Pandas on Ray.")

        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None \
            else 0

        cond = cond(self) if callable(cond) else cond

        if not isinstance(cond, DataFrame):
            if not hasattr(cond, 'shape'):
                cond = np.asanyarray(cond)
            if cond.shape != self.shape:
                raise ValueError("Array conditional must be same shape as "
                                 "self")
            cond = DataFrame(cond, index=self.index, columns=self.columns)

        zipped_partitions = self._copartition(cond, self.index)
        args = (False, axis, level, errors, try_cast, raise_on_error)

        if isinstance(other, DataFrame):
            other_zipped = (v for k, v in self._copartition(other, self.index))

            new_partitions = [
                _where_helper.remote(k, v, next(other_zipped), self.columns,
                                     cond.columns, other.columns, *args)
                for k, v in zipped_partitions
            ]

        # Series has to be treated specially because we're operating on row
        # partitions from here on.
        elif isinstance(other, pandas.Series):
            if axis == 0:
                # Pandas determines which index to use based on axis.
                other = other.reindex(self.index)
                other.index = pandas.RangeIndex(len(other))

                # Since we're working on row partitions, we have to partition
                # the Series based on the partitioning of self (since both
                # self and cond are co-partitioned by self.
                other_builder = []
                for length in self._row_metadata._lengths:
                    other_builder.append(other[:length])
                    other = other[length:]
                    # Resetting the index here ensures that we apply each part
                    # to the correct row within the partitions.
                    other.index = pandas.RangeIndex(len(other))

                other = (obj for obj in other_builder)

                new_partitions = [
                    _where_helper.remote(k, v, next(other, pandas.Series()),
                                         self.columns, cond.columns, None,
                                         *args) for k, v in zipped_partitions
                ]
            else:
                other = other.reindex(self.columns)
                new_partitions = [
                    _where_helper.remote(k, v, other, self.columns,
                                         cond.columns, None, *args)
                    for k, v in zipped_partitions
                ]

        else:
            new_partitions = [
                _where_helper.remote(k, v, other, self.columns, cond.columns,
                                     None, *args) for k, v in zipped_partitions
            ]

        if inplace:
            self._update_inplace(
                row_partitions=new_partitions,
                row_metadata=self._row_metadata,
                col_metadata=self._col_metadata)
        else:
            return DataFrame(
                row_partitions=new_partitions,
                row_metadata=self._row_metadata,
                col_metadata=self._col_metadata)

    def xs(self, key, axis=0, level=None, drop_level=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __getitem__(self, key):
        """Get the column specified by key for this DataFrame.

        Args:
            key : The column name.

        Returns:
            A Pandas Series representing the value for the column.
        """
        key = com._apply_if_callable(key, self)

        # Shortcut if key is an actual column
        is_mi_columns = isinstance(self.columns, pandas.MultiIndex)
        try:
            if key in self.columns and not is_mi_columns:
                return self._getitem_column(key)
        except (KeyError, ValueError, TypeError):
            pass

        # see if we can slice the rows
        # This lets us reuse code in Pandas to error check
        indexer = convert_to_index_sliceable(pandas.DataFrame(index=self.index), key)
        if indexer is not None:
            return self._getitem_slice(indexer)

        if isinstance(key, (pandas.Series, np.ndarray, pandas.Index, list)):
            return self._getitem_array(key)
        elif isinstance(key, DataFrame):
            raise NotImplementedError("To contribute to Pandas on Ray, please"
                                      "visit github.com/modin-project/modin.")
            # return self._getitem_frame(key)
        elif is_mi_columns:
            raise NotImplementedError("To contribute to Pandas on Ray, please"
                                      "visit github.com/modin-project/modin.")
            # return self._getitem_multilevel(key)
        else:
            return self._getitem_column(key)

    def _getitem_column(self, key):
        return self._data_manager.getitem_single_key(key)

    def _getitem_array(self, key):
        if com.is_bool_indexer(key):
            if isinstance(key, pandas.Series) and \
                    not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match "
                    "DataFrame index.",
                    UserWarning,
                    stacklevel=3)
            elif len(key) != len(self.index):
                raise ValueError('Item wrong length {} instead of {}.'.format(
                    len(key), len(self.index)))
            key = check_bool_indexer(self.index, key)

            # We convert here because the data_manager assumes it is a list of
            # indices. This greatly decreases the complexity of the code.
            key = self.index[key]
            return DataFrame(data_manager=self._data_manager.getitem_row_array(key))
        else:
            return DataFrame(data_manager=self._data_manager.getitem_column_array(key))

    def _getitem_slice(self, key):
        # We convert here because the data_manager assumes it is a list of
        # indices. This greatly decreases the complexity of the code.
        key = self.index[key]
        return DataFrame(data_manager=self._data_manager.getitem_row_array(key))

    def __getattr__(self, key):
        """After regular attribute access, looks up the name in the columns

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self.columns:
                return self[key]
            raise e

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise NotImplementedError(
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")
        if key not in self.columns:
            self.insert(loc=len(self.columns), column=key, value=value)
        else:
            loc = self.columns.get_loc(key)
            self.__delitem__(key)
            self.insert(loc=loc, column=key, value=value)

    def __len__(self):
        """Gets the length of the DataFrame.

        Returns:
            Returns an integer length of the DataFrame object.
        """
        return len(self.index)

    def __unicode__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __invert__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __hash__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __iter__(self):
        """Iterate over the columns

        Returns:
            An Iterator over the columns of the DataFrame.
        """
        return iter(self.columns)

    def __contains__(self, key):
        """Searches columns for specific key

        Args:
            key : The column name

        Returns:
            Returns a boolean if the specified key exists as a column name
        """
        return self.columns.__contains__(key)

    def __nonzero__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __bool__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __abs__(self):
        """Creates a modified DataFrame by taking the absolute value.

        Returns:
            A modified DataFrame
        """
        return self.abs()

    def __round__(self, decimals=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __array__(self, dtype=None):
        # TODO: This is very inefficient and needs fix, also see as_matrix
        return to_pandas(self).__array__(dtype=dtype)

    def __array_wrap__(self, result, context=None):
        # TODO: This is very inefficient, see also __array__ and as_matrix
        return to_pandas(self).__array_wrap__(result, context=context)

    def __getstate__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __setstate__(self, state):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __delitem__(self, key):
        """Delete a column by key. `del a[key]` for example.
           Operation happens in place.

           Notes: This operation happen on row and column partition
                  simultaneously. No rebuild.
        Args:
            key: key to delete
        """
        if key not in self:
            raise KeyError(key)

        self._update_inplace(new_manager=self._data_manager.delitem(key))

    def __finalize__(self, other, method=None, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __copy__(self, deep=True):
        """Make a copy using modin.DataFrame.copy method

        Args:
            deep: Boolean, deep copy or not.
                  Currently we do not support deep copy.

        Returns:
            A Ray DataFrame object.
        """
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        """Make a -deep- copy using modin.DataFrame.copy method
           This is equivalent to copy(deep=True).

        Args:
            memo: No effect. Just to comply with Pandas API.

        Returns:
            A Ray DataFrame object.
        """
        return self.copy(deep=True)

    def __and__(self, other):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __or__(self, other):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __xor__(self, other):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)

    def __radd__(self, other, axis="columns", level=None, fill_value=None):
        return self.radd(other, axis, level, fill_value)

    def __mul__(self, other):
        return self.mul(other)

    def __imul__(self, other):
        return self.mul(other)

    def __rmul__(self, other, axis="columns", level=None, fill_value=None):
        return self.rmul(other, axis, level, fill_value)

    def __pow__(self, other):
        return self.pow(other)

    def __ipow__(self, other):
        return self.pow(other)

    def __rpow__(self, other, axis="columns", level=None, fill_value=None):
        return self.rpow(other, axis, level, fill_value)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub(other)

    def __rsub__(self, other, axis="columns", level=None, fill_value=None):
        return self.rsub(other, axis, level, fill_value)

    def __floordiv__(self, other):
        return self.floordiv(other)

    def __ifloordiv__(self, other):
        return self.floordiv(other)

    def __rfloordiv__(self, other, axis="columns", level=None,
                      fill_value=None):
        return self.rfloordiv(other, axis, level, fill_value)

    def __truediv__(self, other):
        return self.truediv(other)

    def __itruediv__(self, other):
        return self.truediv(other)

    def __rtruediv__(self, other, axis="columns", level=None, fill_value=None):
        return self.rtruediv(other, axis, level, fill_value)

    def __mod__(self, other):
        return self.mod(other)

    def __imod__(self, other):
        return self.mod(other)

    def __rmod__(self, other, axis="columns", level=None, fill_value=None):
        return self.rmod(other, axis, level, fill_value)

    def __div__(self, other, axis="columns", level=None, fill_value=None):
        return self.div(other, axis, level, fill_value)

    def __rdiv__(self, other, axis="columns", level=None, fill_value=None):
        return self.rdiv(other, axis, level, fill_value)

    def __neg__(self):
        """Computes an element wise negative DataFrame

        Returns:
            A modified DataFrame where every element is the negation of before
        """
        for t in self.dtypes:
            if not (is_bool_dtype(t) or is_numeric_dtype(t)
                    or is_timedelta64_dtype(t)):
                raise TypeError("Unary negative expects numeric dtype, not {}"
                                .format(t))

        return DataFrame(data_manager=self._data_manager.negative())

    def __sizeof__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def __doc__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def blocks(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def style(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def iat(self, axis=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def loc(self):
        """Purely label-location based indexer for selection by label.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _Loc_Indexer
        return _Loc_Indexer(self)

    @property
    def is_copy(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def at(self, axis=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def ix(self, axis=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _iLoc_Indexer
        return _iLoc_Indexer(self)

    def _copartition(self, other, new_index):
        """Colocates the values of other with this for certain operations.

        NOTE: This method uses the indexes of each DataFrame to order them the
            same. This operation does an implicit shuffling of data and zips
            the two DataFrames together to be operated on.

        Args:
            other: The other DataFrame to copartition with.

        Returns:
            Two new sets of partitions, copartitioned and zipped.
        """
        # Put in the object store so they aren't serialized each iteration.
        old_self_index = ray.put(self.index)
        new_index = ray.put(new_index)
        old_other_index = ray.put(other.index)

        new_num_partitions = max(
            len(self._block_partitions.T), len(other._block_partitions.T))

        new_partitions_self = \
            np.array([_reindex_helper._submit(
                args=tuple([old_self_index, new_index, 1,
                            new_num_partitions] + block.tolist()),
                num_return_vals=new_num_partitions)
                for block in self._block_partitions.T]).T

        new_partitions_other = \
            np.array([_reindex_helper._submit(
                args=tuple([old_other_index, new_index, 1,
                            new_num_partitions] + block.tolist()),
                num_return_vals=new_num_partitions)
                for block in other._block_partitions.T]).T

        return zip(new_partitions_self, new_partitions_other)

    def _operator_helper(self, func, other, axis, level, *args):
        """Helper method for inter-DataFrame and scalar operations"""
        if isinstance(other, DataFrame):
            return self._inter_df_op_helper(
                lambda x, y: func(x, y, axis, level, *args), other, "outer",
                level)
        else:
            return self._single_df_op_helper(
                lambda df: func(df, other, axis, level, *args), other, axis,
                level)

    def _create_dataframe_from_manager(self, new_manager, inplace=False):
        """Returns or updates a DataFrame given new data_manager"""
        if not inplace:
            return DataFrame(data_manager=new_manager)
        else:
            self._update_inplace(new_manager=new_manager)

    def _inter_df_op_helper(self, func, other, how, level, inplace=False):
        if level is not None:
            raise NotImplementedError("Mutlilevel index not yet supported "
                                      "in Pandas on Ray")
        new_manager = self._data_manager.inter_manager_operations(other._data_manager, how, func)

        if not inplace:
            return DataFrame(data_manager=new_manager)
        else:
            self._update_inplace(new_manager=new_manager)

    def _single_df_op_helper(self, func, other, axis, level):
        if level is not None:
            raise NotImplementedError("Multilevel index not yet supported "
                                      "in Pandas on Ray")
        axis = pandas.DataFrame()._get_axis_number(axis)

        if is_list_like(other):
            if axis == 0:
                if len(other) != len(self.index):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self.index), len(other)))
            else:
                if len(other) != len(self.columns):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self.columns), len(other)))

        return DataFrame(data_manager=self._data_manager.scalar_operations(axis, other, func))

    def _validate_other_df(self, other, axis):
        """Helper method to check validity of other in inter-df operations"""
        axis = pandas.DataFrame()._get_axis_number(axis)

        if isinstance(other, DataFrame):
            return other._data_manager
        elif is_list_like(other):
            if axis == 0:
                if len(other) != len(self.index):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self.index), len(other)))
            else:
                if len(other) != len(self.columns):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self.columns), len(other)))
        return other



@ray.remote
def _merge_columns(left_columns, right_columns, *args):
    """Merge two columns to get the correct column names and order.

    Args:
        left_columns: The columns on the left side of the merge.
        right_columns: The columns on the right side of the merge.
        args: The arguments for the merge.

    Returns:
         The columns for the merge operation.
    """
    return pandas.DataFrame(columns=left_columns, index=[0], dtype='uint8') \
        .merge(pandas.DataFrame(columns=right_columns, index=[0],
                                dtype='uint8'), *args).columns


@ray.remote
def _where_helper(left, cond, other, left_columns, cond_columns, other_columns,
                  *args):

    left = pandas.concat(ray.get(left.tolist()), axis=1, copy=False)
    # We have to reset the index and columns here because we are coming
    # from blocks and the axes are set according to the blocks. We have
    # already correctly copartitioned everything, so there's no
    # correctness problems with doing this.
    left.reset_index(inplace=True, drop=True)
    left.columns = left_columns

    cond = pandas.concat(ray.get(cond.tolist()), axis=1, copy=False)
    cond.reset_index(inplace=True, drop=True)
    cond.columns = cond_columns

    if isinstance(other, np.ndarray):
        other = pandas.concat(ray.get(other.tolist()), axis=1, copy=False)
        other.reset_index(inplace=True, drop=True)
        other.columns = other_columns

    return left.where(cond, other, *args)
