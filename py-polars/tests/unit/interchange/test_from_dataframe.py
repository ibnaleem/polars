from __future__ import annotations

from datetime import date

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
import polars.interchange.from_dataframe
from polars.interchange.buffer import PolarsBuffer
from polars.interchange.from_dataframe import (
    _construct_data_buffer,
    _construct_offsets_buffer,
    _construct_validity_buffer,
    _construct_validity_buffer_from_bitmask,
    _construct_validity_buffer_from_bytemask,
)
from polars.interchange.protocol import CopyNotAllowedError, DtypeKind, Endianness
from polars.testing import assert_frame_equal, assert_series_equal

NE = Endianness.NATIVE

# def test_from_dataframe_polars() -> None:
#     df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
#     result = pl.from_dataframe(df, allow_copy=False)
#     assert_frame_equal(result, df)


# def test_from_dataframe_polars_interchange_fast_path() -> None:
#     df = pl.DataFrame(
#         {"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]},
#         schema_overrides={"c": pl.Categorical},
#     )
#     dfi = df.__dataframe__()
#     result = pl.from_dataframe(dfi, allow_copy=False)
#     assert_frame_equal(result, df)


# def test_from_dataframe_categorical_zero_copy() -> None:
#     df = pl.DataFrame({"a": ["foo", "bar"]}, schema={"a": pl.Categorical})
#     df_pa = df.to_arrow()

#     with pytest.raises(TypeError):
#         pl.from_dataframe(df_pa, allow_copy=False)


# def test_from_dataframe_pandas() -> None:
#     data = {"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]}

#     # Pandas dataframe
#     df = pd.DataFrame(data)
#     result = pl.from_dataframe(df)
#     expected = pl.DataFrame(data)
#     assert_frame_equal(result, expected)


# def test_from_dataframe_pyarrow_table_zero_copy() -> None:
#     df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
#     df_pa = df.to_arrow()

#     result = pl.from_dataframe(df_pa, allow_copy=False)
#     assert_frame_equal(result, df)


# def test_from_dataframe_pyarrow_recordbatch_zero_copy() -> None:
#     a = pa.array([1, 2])
#     b = pa.array([3.0, 4.0])
#     c = pa.array(["foo", "bar"])

#     batch = pa.record_batch([a, b, c], names=["a", "b", "c"])
#     result = pl.from_dataframe(batch, allow_copy=False)
#     expected = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})

#     assert_frame_equal(result, expected)


# def test_from_dataframe_allow_copy() -> None:
#     # Zero copy only allowed when input is already a Polars dataframe
#     df = pl.DataFrame({"a": [1, 2]})
#     result = pl.from_dataframe(df, allow_copy=True)
#     assert_frame_equal(result, df)

#     df1_pandas = pd.DataFrame({"a": [1, 2]})
#     result_from_pandas = pl.from_dataframe(df1_pandas, allow_copy=False)
#     assert_frame_equal(result_from_pandas, df)

#     # Zero copy cannot be guaranteed for other inputs at this time
#     df2_pandas = pd.DataFrame({"a": ["A", "B"]})
#     with pytest.raises(RuntimeError):
#         pl.from_dataframe(df2_pandas, allow_copy=False)


# def test_from_dataframe_invalid_type() -> None:
#     df = [[1, 2], [3, 4]]
#     with pytest.raises(TypeError):
#         pl.from_dataframe(df)  # type: ignore[arg-type]


# def test_from_dataframe_empty_arrow_interchange_object() -> None:
#     df = pl.Series("a", dtype=pl.Int8).to_frame()
#     df_pa = df.to_arrow()
#     dfi = df_pa.__dataframe__()

#     result = pl.from_dataframe(dfi)

#     assert_frame_equal(result, df)


def test_construct_data_buffer() -> None:
    data = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.INT, 64, "l", NE)

    result = _construct_data_buffer(buffer, dtype, length=5)
    assert_series_equal(result, data)


def test_construct_data_buffer_boolean_sliced() -> None:
    data = pl.Series([False, True, True, False])
    data_sliced = data[2:]
    buffer = PolarsBuffer(data_sliced)
    dtype = (DtypeKind.BOOL, 1, "b", NE)

    result = _construct_data_buffer(buffer, dtype, length=2, offset=2)
    assert_series_equal(result, data_sliced)


def test_construct_data_buffer_logical_dtype() -> None:
    data = pl.Series([0, 1, 3, 3, 9], dtype=pl.UInt32)
    # data = pl.Series([date(2023, 12, 31), date(2024, 1, 1)], dtype=pl.Date)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.DATETIME, 32, "tdD", NE)

    result = _construct_data_buffer(buffer, dtype, length=5)
    assert_series_equal(result, data)


def test_construct_offsets_buffer() -> None:
    data = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.INT, 64, "l", NE)
    offsets_buffer_info = (buffer, dtype)

    result = _construct_offsets_buffer(offsets_buffer_info)
    assert_series_equal(result, data)


def test_construct_offsets_buffer_copy() -> None:
    data = pl.Series([0, 1, 3, 3, 9], dtype=pl.UInt32)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.UINT, 32, "I", NE)
    offsets_buffer_info = (buffer, dtype)

    with pytest.raises(CopyNotAllowedError):
        _construct_offsets_buffer(offsets_buffer_info, allow_copy=False)

    result = _construct_offsets_buffer(offsets_buffer_info)
    expected = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    assert_series_equal(result, expected)


def test_construct_offsets_buffer_none() -> None:
    result = _construct_offsets_buffer(None)
    assert result is None


def test_construct_validity_buffer() -> None:
    pass


@pytest.fixture()
def bitmask() -> PolarsBuffer:
    data = pl.Series([False, True, True, False])
    return PolarsBuffer(data)


@pytest.mark.parametrize("allow_copy", [True, False])
def test_construct_validity_buffer_from_bitmask(
    allow_copy: bool, bitmask: PolarsBuffer
) -> None:
    result = _construct_validity_buffer_from_bitmask(
        bitmask, null_value=0, offset=0, length=4, allow_copy=allow_copy
    )
    expected = pl.Series([False, True, True, False])
    assert_series_equal(result, expected)


def test_construct_validity_buffer_from_bitmask_inverted(bitmask: PolarsBuffer) -> None:
    result = _construct_validity_buffer_from_bitmask(
        bitmask, null_value=1, offset=0, length=4
    )
    expected = pl.Series([True, False, False, True])
    assert_series_equal(result, expected)


def test_construct_validity_buffer_from_bitmask_zero_copy_fails(
    bitmask: PolarsBuffer,
) -> None:
    with pytest.raises(CopyNotAllowedError):
        _construct_validity_buffer_from_bitmask(
            bitmask, null_value=1, offset=0, length=4, allow_copy=False
        )


def test_construct_validity_buffer_from_bitmask_sliced() -> None:
    data = pl.Series([False, True, True, False])
    data_sliced = data[2:]
    bitmask = PolarsBuffer(data_sliced)

    result = _construct_validity_buffer_from_bitmask(
        bitmask, null_value=0, offset=2, length=2
    )
    assert_series_equal(result, data_sliced)


@pytest.fixture()
def bytemask() -> PolarsBuffer:
    data = pl.Series([0, 1, 1, 0], dtype=pl.UInt8)
    return PolarsBuffer(data)


def test_construct_validity_buffer_from_bytemask(bytemask: PolarsBuffer) -> None:
    result = _construct_validity_buffer_from_bytemask(bytemask, null_value=0)
    expected = pl.Series([False, True, True, False])
    assert_series_equal(result, expected)


def test_construct_validity_buffer_from_bytemask_inverted(
    bytemask: PolarsBuffer,
) -> None:
    result = _construct_validity_buffer_from_bytemask(bytemask, null_value=1)
    expected = pl.Series([True, False, False, True])
    assert_series_equal(result, expected)


def test_construct_validity_buffer_from_bytemask_zero_copy_fails(
    bytemask: PolarsBuffer,
) -> None:
    with pytest.raises(CopyNotAllowedError):
        _construct_validity_buffer_from_bytemask(
            bytemask, null_value=0, allow_copy=False
        )
