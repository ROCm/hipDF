# Copyright (c) 2023, NVIDIA CORPORATION.
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2, 3], "int8"),
        ([1, 2, 3], "int64"),
        ([1.1, 2.2, 3.3], "float32"),
        ([1.0, 2.0, 3.0], "float32"),
        ([1.0, 2.0, 3.0], "float64"),
        (["a", "b", "c"], "str"),
        (["a", "b", "c"], "category"),
        (["2001-01-01", "2001-01-02", "2001-01-03"], "datetime64[ns]"),
    ],
)
def test_convert_dtypes(data, dtype):
    s = pd.Series(data, dtype=dtype)
    gs = cudf.Series(data, dtype=dtype)
    expect = s.convert_dtypes()

    # because we don't have distinct nullable types, we check that we
    # get the same result if we convert to nullable pandas types:
    got = gs.convert_dtypes().to_pandas(nullable=True)
    assert_eq(expect, got)


# Now write the same test, but construct a DataFrame
# as input instead of parametrizing:
