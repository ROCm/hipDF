# Copyright (c) 2020-2023, NVIDIA CORPORATION.

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

import numba.cuda
import pytest

import dask
from dask import dataframe as dd
from dask.distributed import Client
from distributed.utils_test import cleanup, loop, loop_in_thread  # noqa: F401

import cudf
from cudf.testing._utils import assert_eq

import dask_cudf

dask_cuda = pytest.importorskip("dask_cuda")


def more_than_two_gpus():
    ngpus = len(numba.cuda.gpus)
    return ngpus >= 2


@pytest.mark.parametrize("delayed", [True, False])
def test_basic(loop, delayed):  # noqa: F811
    with dask_cuda.LocalCUDACluster(loop=loop) as cluster:
        with Client(cluster):
            pdf = dask.datasets.timeseries(dtypes={"x": int}).reset_index()
            gdf = pdf.map_partitions(cudf.DataFrame.from_pandas)
            if delayed:
                gdf = dd.from_delayed(gdf.to_delayed())
            assert_eq(pdf.head(), gdf.head())


def test_merge():
    # Repro Issue#3366
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            r1 = cudf.DataFrame()
            r1["a1"] = range(4)
            r1["a2"] = range(4, 8)
            r1["a3"] = range(4)

            r2 = cudf.DataFrame()
            r2["b0"] = range(4)
            r2["b1"] = range(4)
            r2["b1"] = r2.b1.astype("str")

            d1 = dask_cudf.from_cudf(r1, 2)
            d2 = dask_cudf.from_cudf(r2, 2)

            res = d1.merge(d2, left_on=["a3"], right_on=["b0"])
            assert len(res) == 4


@pytest.mark.skipif(
    not more_than_two_gpus(), reason="Machine does not have more than two GPUs"
)
def test_ucx_seriesgroupby():
    pytest.importorskip("ucp")

    # Repro Issue#3913
    with dask_cuda.LocalCUDACluster(n_workers=2, protocol="ucx") as cluster:
        with Client(cluster):
            df = cudf.DataFrame({"a": [1, 2, 3, 4], "b": [5, 1, 2, 5]})
            dask_df = dask_cudf.from_cudf(df, npartitions=2)
            dask_df_g = dask_df.groupby(["a"]).b.sum().compute()

            assert dask_df_g.name == "b"


def test_str_series_roundtrip():
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            expected = cudf.Series(["hi", "hello", None])
            dask_series = dask_cudf.from_cudf(expected, npartitions=2)

            actual = dask_series.compute()
            assert_eq(actual, expected)


def test_p2p_shuffle():
    # Check that we can use `shuffle="p2p"`
    with dask_cuda.LocalCUDACluster(n_workers=1) as cluster:
        with Client(cluster):
            ddf = (
                dask.datasets.timeseries(
                    start="2000-01-01",
                    end="2000-01-08",
                    dtypes={"x": int},
                )
                .reset_index(drop=True)
                .to_backend("cudf")
            )
            dd.assert_eq(
                ddf.sort_values("x", shuffle="p2p").compute(),
                ddf.compute().sort_values("x"),
                check_index=False,
            )
