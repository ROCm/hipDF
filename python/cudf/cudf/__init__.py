# Copyright (c) 2018-2023, NVIDIA CORPORATION.

# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# _setup_numba _must be called before numba.cuda is imported, because
# it sets the numba config variable responsible for enabling
# Minor Version Compatibility. Setting it after importing numba.cuda has no effect.
from cudf.utils._numba import _setup_numba
from cudf.utils.gpu_utils import validate_setup

_setup_numba()
validate_setup()

try:
    from numba import hip
    hip.pose_as_cuda()
except ImportError:
    pass

import cupy
from numba import config as numba_config, cuda

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager

from cudf import api, core, datasets, testing
from cudf.api.extensions import (
    register_dataframe_accessor,
    register_index_accessor,
    register_series_accessor,
)
from cudf.api.types import dtype
from cudf.core.algorithms import factorize
from cudf.core.cut import cut
from cudf.core.dataframe import DataFrame, from_dataframe, from_pandas, merge
from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)
from cudf.core.groupby import Grouper
from cudf.core.index import (
    BaseIndex,
    CategoricalIndex,
    DatetimeIndex,
    Float32Index,
    Float64Index,
    GenericIndex,
    Index,
    Int8Index,
    Int16Index,
    Int32Index,
    Int64Index,
    IntervalIndex,
    RangeIndex,
    StringIndex,
    TimedeltaIndex,
    UInt8Index,
    UInt16Index,
    UInt32Index,
    UInt64Index,
    interval_range,
)
from cudf.core.missing import NA, NaT
from cudf.core.multiindex import MultiIndex
from cudf.core.reshape import (
    concat,
    crosstab,
    get_dummies,
    melt,
    pivot,
    pivot_table,
    unstack,
)
from cudf.core.scalar import Scalar
from cudf.core.series import Series, isclose
from cudf.core.tools.datetimes import DateOffset, date_range, to_datetime
from cudf.core.tools.numeric import to_numeric
from cudf.io import (
    from_dlpack,
    read_avro,
    read_csv,
    read_feather,
    read_hdf,
    read_json,
    read_orc,
    read_parquet,
    read_text,
)
from cudf.options import (
    describe_option,
    get_option,
    option_context,
    set_option,
)
from cudf.utils.utils import clear_cache

cuda.set_memory_manager(RMMNumbaManager)
cupy.cuda.set_allocator(rmm_cupy_allocator)


rmm.register_reinitialize_hook(clear_cache)

from cuda import cuda as _cuda_python_cuda
__is_hip_amd_port__ = hasattr(_cuda_python_cuda, "HIP_PYTHON")
del _cuda_python_cuda

__version__ = "23.10.00"

__all__ = [
    "BaseIndex",
    "CategoricalDtype",
    "CategoricalIndex",
    "DataFrame",
    "DateOffset",
    "DatetimeIndex",
    "Decimal32Dtype",
    "Decimal64Dtype",
    "Float32Index",
    "Float64Index",
    "GenericIndex",
    "Grouper",
    "Index",
    "Int16Index",
    "Int32Index",
    "Int64Index",
    "Int8Index",
    "IntervalDtype",
    "IntervalIndex",
    "ListDtype",
    "MultiIndex",
    "NA",
    "NaT",
    "RangeIndex",
    "Scalar",
    "Series",
    "StringIndex",
    "StructDtype",
    "TimedeltaIndex",
    "UInt16Index",
    "UInt32Index",
    "UInt64Index",
    "UInt8Index",
    "api",
    "concat",
    "crosstab",
    "cut",
    "date_range",
    "describe_option",
    "factorize",
    "from_dataframe",
    "from_dlpack",
    "from_pandas",
    "get_dummies",
    "get_option",
    "interval_range",
    "isclose",
    "melt",
    "merge",
    "pivot",
    "pivot_table",
    "read_avro",
    "read_csv",
    "read_feather",
    "read_hdf",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_text",
    "set_option",
    "testing",
    "to_datetime",
    "to_numeric",
    "unstack",
]