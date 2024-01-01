from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import polars._reexport as pl
import polars.functions as F
from polars.datatypes import Boolean, Int64, UInt8
from polars.interchange.dataframe import PolarsDataFrame
from polars.interchange.protocol import ColumnNullType, CopyNotAllowedError
from polars.interchange.utils import dtype_to_polars_dtype, get_buffer_size_in_elements
from polars.utils._wrap import wrap_s

with contextlib.suppress(ImportError):  # Module not available when building docs
    from polars.polars import PySeries

if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.interchange.protocol import Buffer, Column, Dtype, SupportsInterchange
    from polars.interchange.protocol import DataFrame as InterchangeDataFrame


def from_dataframe(df: SupportsInterchange, *, allow_copy: bool = True) -> DataFrame:
    """
    Build a Polars DataFrame from any dataframe supporting the interchange protocol.

    Parameters
    ----------
    df
        Object supporting the dataframe interchange protocol, i.e. must have implemented
        the `__dataframe__` method.
    allow_copy
        Allow memory to be copied to perform the conversion. If set to False, causes
        conversions that are not zero-copy to fail.
    """
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, PolarsDataFrame):
        return df._df

    if not hasattr(df, "__dataframe__"):
        raise TypeError(
            f"`df` of type {type(df).__name__!r} does not support the dataframe interchange protocol"
        )

    return _from_dataframe(df.__dataframe__(allow_copy=allow_copy))


def _from_dataframe(df: InterchangeDataFrame, *, allow_copy: bool = True) -> DataFrame:
    chunks = [
        _protocol_df_chunk_to_polars(chunk, allow_copy=allow_copy)
        for chunk in df.get_chunks()
    ]
    # If copy is allowed, rechunk as it will speed up subsequent computation
    return F.concat(chunks, rechunk=allow_copy)


def _protocol_df_chunk_to_polars(
    df: InterchangeDataFrame, *, allow_copy: bool = True
) -> DataFrame:
    columns = [
        _column_to_series(column, allow_copy=allow_copy) for column in df.get_columns()
    ]
    return pl.DataFrame(columns)


def _column_to_series(column: Column, *, allow_copy: bool = True) -> Series:
    polars_dtype = dtype_to_polars_dtype(column.dtype)

    buffers = column.get_buffers()
    offset = column.offset

    data_buffer = _construct_data_buffer(*buffers["data"], column.size(), offset)
    offsets_buffer = _construct_offsets_buffer(
        buffers["offsets"], offset, allow_copy=allow_copy
    )

    data = wrap_s(
        PySeries.from_buffers(polars_dtype, data_buffer, None, offsets_buffer)
    )

    validity_buffer = _construct_validity_buffer(
        column,
        buffers["validity"],
        data,
        offset,
        allow_copy=allow_copy,
    )

    s = wrap_s(
        PySeries.from_buffers(
            polars_dtype, data_buffer, validity_buffer, offsets_buffer
        )
    )
    setattr(s, "__INTERCHANGE_BUFFERS", buffers)  # Keep buffers alive

    return s


def _construct_data_buffer(
    buffer: Buffer, dtype: Dtype, size: int, offset: int
) -> Series:
    # This requires the data buffer to have the correct dtype
    # See: https://github.com/pola-rs/polars/pull/10787
    polars_physical_dtype = dtype_to_polars_dtype(dtype)
    return wrap_s(PySeries.from_buffer(polars_physical_dtype, buffer.ptr, size, offset))


def _construct_offsets_buffer(
    buffer_info: tuple[Buffer, Dtype] | None, offset: int, *, allow_copy: bool = True
) -> Series | None:
    if buffer_info is None:
        return None

    buffer, dtype = buffer_info
    polars_physical_dtype = dtype_to_polars_dtype(dtype)

    size = get_buffer_size_in_elements(buffer, dtype)
    s = wrap_s(PySeries.from_buffer(polars_physical_dtype, buffer.ptr, size, offset))

    # Polars only supports Int64 offsets
    if polars_physical_dtype != Int64:
        if not allow_copy:
            raise CopyNotAllowedError(
                f"offset buffer must be cast from {polars_physical_dtype} to Int64"
            )
        s = s.cast(Int64)

    return s


def _construct_validity_buffer(
    column: Column,
    validity_buffer_info: tuple[Buffer, Dtype] | None,
    data_buffer: Series,
    offset: int,
    *,
    allow_copy: bool = True,
) -> Series | None:
    null_type, null_value = column.describe_null
    if null_type == ColumnNullType.NON_NULLABLE or column.null_count == 0:
        return None

    elif null_type == ColumnNullType.USE_BITMASK:
        if validity_buffer_info is None:
            return None
        buffer = validity_buffer_info[0]
        return _construct_validity_buffer_from_bitmask(
            buffer, column.size(), null_value, offset, allow_copy=allow_copy
        )

    elif null_type == ColumnNullType.USE_BYTEMASK:
        if validity_buffer_info is None:
            return None
        buffer = validity_buffer_info[0]
        return _construct_validity_buffer_from_bytemask(
            buffer, null_value, allow_copy=allow_copy
        )

    elif null_type == ColumnNullType.USE_NAN:
        if not allow_copy:
            raise CopyNotAllowedError("bitmask must be constructed")
        return data_buffer.is_not_nan()

    elif null_type == ColumnNullType.USE_SENTINEL:
        if not allow_copy:
            raise CopyNotAllowedError("bitmask must be constructed")
        return data_buffer != null_value

    else:
        raise NotImplementedError(f"unsupported null type: {null_type!r}")


def _construct_validity_buffer_from_bitmask(
    buffer: Buffer,
    null_value: int,
    offset: int,
    length: int,
    *,
    allow_copy: bool = True,
) -> Series:
    buffer_info = (buffer.ptr, offset, length)
    s = pl.Series._from_buffer(Boolean, buffer_info, buffer)

    if null_value != 0:
        if not allow_copy:
            raise CopyNotAllowedError("bitmask must be inverted")
        s = ~s

    return s


def _construct_validity_buffer_from_bytemask(
    buffer: Buffer,
    null_value: int,
    *,
    allow_copy: bool = True,
) -> Series:
    if not allow_copy:
        raise CopyNotAllowedError("bytemask must be converted into a bitmask")

    buffer_info = (buffer.ptr, 0, buffer.bufsize)
    s = pl.Series._from_buffer(UInt8, buffer_info, owner=buffer)
    s = s.cast(Boolean)

    if null_value != 0:
        s = ~s

    return s
