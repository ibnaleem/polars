from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
import polars.interchange.from_dataframe
from polars.testing import assert_frame_equal


def test_from_dataframe_polars() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    result = pl.from_dataframe(df, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_polars_interchange_fast_path() -> None:
    df = pl.DataFrame(
        {"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]},
        schema_overrides={"c": pl.Categorical},
    )
    dfi = df.__dataframe__()
    result = pl.from_dataframe(dfi, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_categorical_zero_copy() -> None:
    df = pl.DataFrame({"a": ["foo", "bar"]}, schema={"a": pl.Categorical})
    df_pa = df.to_arrow()

    with pytest.raises(TypeError):
        pl.from_dataframe(df_pa, allow_copy=False)


def test_from_dataframe_pandas() -> None:
    data = {"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]}

    # Pandas dataframe
    df = pd.DataFrame(data)
    result = pl.from_dataframe(df)
    expected = pl.DataFrame(data)
    assert_frame_equal(result, expected)


def test_from_dataframe_pyarrow_table_zero_copy() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    df_pa = df.to_arrow()

    result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_pyarrow_recordbatch_zero_copy() -> None:
    a = pa.array([1, 2])
    b = pa.array([3.0, 4.0])
    c = pa.array(["foo", "bar"])

    batch = pa.record_batch([a, b, c], names=["a", "b", "c"])
    result = pl.from_dataframe(batch, allow_copy=False)
    expected = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})

    assert_frame_equal(result, expected)


def test_from_dataframe_allow_copy() -> None:
    # Zero copy only allowed when input is already a Polars dataframe
    df = pl.DataFrame({"a": [1, 2]})
    result = pl.from_dataframe(df, allow_copy=True)
    assert_frame_equal(result, df)

    df1_pandas = pd.DataFrame({"a": [1, 2]})
    result_from_pandas = pl.from_dataframe(df1_pandas, allow_copy=False)
    assert_frame_equal(result_from_pandas, df)

    # Zero copy cannot be guaranteed for other inputs at this time
    df2_pandas = pd.DataFrame({"a": ["A", "B"]})
    with pytest.raises(RuntimeError):
        pl.from_dataframe(df2_pandas, allow_copy=False)


def test_from_dataframe_invalid_type() -> None:
    df = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        pl.from_dataframe(df)  # type: ignore[arg-type]


def test_from_dataframe_empty_arrow_interchange_object() -> None:
    df = pl.Series("a", dtype=pl.Int8).to_frame()
    df_pa = df.to_arrow()
    dfi = df_pa.__dataframe__()

    result = pl.from_dataframe(dfi)

    assert_frame_equal(result, df)
