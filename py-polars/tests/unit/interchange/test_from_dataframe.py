from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
import polars.interchange.from_dataframe
from polars.interchange.buffer import PolarsBuffer
from polars.interchange.column import PolarsColumn
from polars.interchange.from_dataframe import (
    _categorical_column_to_series,
    _construct_data_buffer,
    _construct_offsets_buffer,
    _construct_validity_buffer,
    _construct_validity_buffer_from_bitmask,
    _construct_validity_buffer_from_bytemask,
    _string_column_to_series,
)
from polars.interchange.protocol import (
    ColumnNullType,
    CopyNotAllowedError,
    DtypeKind,
    Endianness,
)
from polars.testing import assert_frame_equal, assert_series_equal

NE = Endianness.NATIVE


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


def test_from_dataframe_categorical() -> None:
    df = pl.DataFrame({"a": ["foo", "bar"]}, schema={"a": pl.Categorical})
    df_pa = df.to_arrow()

    result = pl.from_dataframe(df_pa)
    expected = pl.DataFrame(
        {"a": ["foo", "bar"]}, schema={"a": pl.Enum(["foo", "bar"])}
    )
    assert_frame_equal(result, expected)

    with pytest.raises(
        CopyNotAllowedError, match="categorical mapping must be constructed"
    ):
        pl.from_dataframe(df_pa, allow_copy=False)


def test_from_dataframe_pandas_zero_copy() -> None:
    data = {"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]}

    df = pd.DataFrame(data)
    result = pl.from_dataframe(df, allow_copy=False)
    expected = pl.DataFrame(data)
    assert_frame_equal(result, expected)


def test_from_dataframe_pyarrow_table_zero_copy() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3.0, 4.0],
            "c": ["foo", None],
        }
    )
    df_pa = df.to_arrow()

    result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_pyarrow_empty_table() -> None:
    df = pl.Series("a", dtype=pl.Int8).to_frame()
    df_pa = df.to_arrow()

    result = pl.from_dataframe(df_pa, allow_copy=False)
    assert_frame_equal(result, df)


def test_from_dataframe_pyarrow_recordbatch_zero_copy() -> None:
    a = pa.array([1, 2])
    b = pa.array([3.0, 4.0])
    c = pa.array(["foo", "bar"], type=pa.large_string())

    batch = pa.record_batch([a, b, c], names=["a", "b", "c"])
    result = pl.from_dataframe(batch, allow_copy=False)

    expected = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    assert_frame_equal(result, expected)


def test_from_dataframe_invalid_type() -> None:
    df = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        pl.from_dataframe(df)  # type: ignore[arg-type]


def test_from_dataframe_categorical_offsets_copy() -> None:
    values = ["a", "b", None, "a"]

    dtype = pa.dictionary(pa.int32(), pa.utf8())
    arr = pa.array(values, dtype)
    df_pa = pa.Table.from_arrays([arr], names=["a"])

    result = pl.from_dataframe(df_pa)
    expected = pl.Series("a", values, dtype=pl.Enum(["a", "b"])).to_frame()
    assert_frame_equal(result, expected)

    with pytest.raises(
        CopyNotAllowedError, match="categorical mapping must be constructed"
    ):
        result = pl.from_dataframe(df_pa, allow_copy=False)


def test_from_dataframe_categorical_non_string_keys() -> None:
    values = [1, 2, None, 1]

    dtype = pa.dictionary(pa.uint32(), pa.int32())
    arr = pa.array(values, dtype)
    df_pa = pa.Table.from_arrays([arr], names=["a"])

    with pytest.raises(
        NotImplementedError, match="non-string categories are not supported"
    ):
        pl.from_dataframe(df_pa)


class PatchableColumn(PolarsColumn):
    """Helper class that allows patching certain PolarsColumn properties."""

    describe_null: tuple[ColumnNullType, Any] = (ColumnNullType.USE_BITMASK, 0)
    describe_categorical: dict[str, Any] = {}  # type: ignore[assignment]  # noqa: RUF012
    null_count = 0


def test_string_column_to_series_no_offsets() -> None:
    s = pl.Series([97, 98, 99])
    col = PolarsColumn(s)
    with pytest.raises(
        RuntimeError,
        match="cannot create String column without an offsets buffer",
    ):
        _string_column_to_series(col)


def test_categorical_column_to_series_non_dictionary() -> None:
    s = pl.Series(["a", "b", None, "a"], dtype=pl.Categorical)

    col = PatchableColumn(s)
    col.describe_categorical = {"is_dictionary": False}

    with pytest.raises(
        NotImplementedError, match="non-dictionary categoricals are not yet supported"
    ):
        _categorical_column_to_series(col)


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
    data = pl.Series([100, 200, 300], dtype=pl.Int32)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.DATETIME, 32, "tdD", NE)

    result = _construct_data_buffer(buffer, dtype, length=3)
    assert_series_equal(result, data)


def test_construct_offsets_buffer() -> None:
    data = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.INT, 64, "l", NE)

    result = _construct_offsets_buffer(buffer, dtype)
    assert_series_equal(result, data)


def test_construct_offsets_buffer_copy() -> None:
    data = pl.Series([0, 1, 3, 3, 9], dtype=pl.UInt32)
    buffer = PolarsBuffer(data)
    dtype = (DtypeKind.UINT, 32, "I", NE)

    with pytest.raises(CopyNotAllowedError):
        _construct_offsets_buffer(buffer, dtype, allow_copy=False)

    result = _construct_offsets_buffer(buffer, dtype)
    expected = pl.Series([0, 1, 3, 3, 9], dtype=pl.Int64)
    assert_series_equal(result, expected)


@pytest.fixture()
def bitmask() -> PolarsBuffer:
    data = pl.Series([False, True, True, False])
    return PolarsBuffer(data)


@pytest.fixture()
def bytemask() -> PolarsBuffer:
    data = pl.Series([0, 1, 1, 0], dtype=pl.UInt8)
    return PolarsBuffer(data)


def test_construct_validity_buffer_non_nullable() -> None:
    s = pl.Series([1, 2, 3])

    col = PatchableColumn(s)
    col.describe_null = (ColumnNullType.NON_NULLABLE, None)
    col.null_count = 1

    result = _construct_validity_buffer(None, col, s)
    assert result is None


def test_construct_validity_buffer_null_count() -> None:
    s = pl.Series([1, 2, 3])

    col = PatchableColumn(s)
    col.describe_null = (ColumnNullType.USE_SENTINEL, -1)
    col.null_count = 0

    result = _construct_validity_buffer(None, col, s)
    assert result is None


def test_construct_validity_buffer_use_bitmask(bitmask: PolarsBuffer) -> None:
    s = pl.Series([1, 2, 3, 4])

    col = PatchableColumn(s)
    col.describe_null = (ColumnNullType.USE_BITMASK, 0)
    col.null_count = 2

    dtype = (DtypeKind.BOOL, 1, "b", NE)
    validity_buffer_info = (bitmask, dtype)

    result = _construct_validity_buffer(validity_buffer_info, col, s)
    expected = pl.Series([False, True, True, False])
    assert_series_equal(result, expected)  # type: ignore[arg-type]

    result = _construct_validity_buffer(None, col, s)
    assert result is None


def test_construct_validity_buffer_use_bytemask(bytemask: PolarsBuffer) -> None:
    s = pl.Series([1, 2, 3, 4])

    col = PatchableColumn(s)
    col.describe_null = (ColumnNullType.USE_BYTEMASK, 0)
    col.null_count = 2

    dtype = (DtypeKind.UINT, 8, "C", NE)
    validity_buffer_info = (bytemask, dtype)

    result = _construct_validity_buffer(validity_buffer_info, col, s)
    expected = pl.Series([False, True, True, False])
    assert_series_equal(result, expected)  # type: ignore[arg-type]

    result = _construct_validity_buffer(None, col, s)
    assert result is None


def test_construct_validity_buffer_use_nan() -> None:
    s = pl.Series([1.0, 2.0, float("nan")])

    col = PatchableColumn(s)
    col.describe_null = (ColumnNullType.USE_NAN, None)
    col.null_count = 1

    result = _construct_validity_buffer(None, col, s)
    expected = pl.Series([True, True, False])
    assert_series_equal(result, expected)  # type: ignore[arg-type]

    with pytest.raises(CopyNotAllowedError, match="bitmask must be constructed"):
        _construct_validity_buffer(None, col, s, allow_copy=False)


def test_construct_validity_buffer_use_sentinel() -> None:
    s = pl.Series(["a", "bc", "NULL"])

    col = PatchableColumn(s)
    col.describe_null = (ColumnNullType.USE_SENTINEL, "NULL")
    col.null_count = 1

    result = _construct_validity_buffer(None, col, s)
    expected = pl.Series([True, True, False])
    assert_series_equal(result, expected)  # type: ignore[arg-type]

    with pytest.raises(CopyNotAllowedError, match="bitmask must be constructed"):
        _construct_validity_buffer(None, col, s, allow_copy=False)


def test_construct_validity_buffer_unsupported() -> None:
    s = pl.Series([1, 2, 3])

    col = PatchableColumn(s)
    col.describe_null = (100, None)  # type: ignore[assignment]
    col.null_count = 1

    with pytest.raises(NotImplementedError, match="unsupported null type: 100"):
        _construct_validity_buffer(None, col, s)


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
