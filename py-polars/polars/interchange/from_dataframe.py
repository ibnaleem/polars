from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl
import polars.functions as F
from polars.datatypes import Boolean, Int64, String, UInt8
from polars.interchange.dataframe import PolarsDataFrame
from polars.interchange.protocol import ColumnNullType, CopyNotAllowedError
from polars.interchange.utils import (
    dtype_to_polars_dtype,
    get_buffer_length_in_elements,
    polars_dtype_to_data_buffer_dtype,
)

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
    chunks = []
    for chunk in df.get_chunks():
        c = _protocol_df_chunk_to_polars(chunk, allow_copy=allow_copy)
        chunks.append(c)

    # Handle implementations that yield no chunks for an empty dataframe
    if not chunks:
        c = _protocol_df_chunk_to_polars(df, allow_copy=allow_copy)
        chunks.append(c)

    return F.concat(chunks, rechunk=False)


def _protocol_df_chunk_to_polars(
    df: InterchangeDataFrame, *, allow_copy: bool = True
) -> DataFrame:
    columns = [
        _column_to_series(column, allow_copy=allow_copy).alias(name)
        for column, name in zip(df.get_columns(), df.column_names())
    ]
    return pl.DataFrame(columns)


def _column_to_series(column: Column, *, allow_copy: bool = True) -> Series:
    polars_dtype = dtype_to_polars_dtype(column.dtype)

    buffers = column.get_buffers()
    offset = column.offset

    # First construct the Series without a validity buffer
    buffer, dtype = buffers["data"]
    length = buffer.bufsize if polars_dtype == String else column.size()
    data_buffer = _construct_data_buffer(buffer, dtype, length, offset)

    offsets_buffer = _construct_offsets_buffer(
        buffers["offsets"], offset, allow_copy=allow_copy
    )

    if offsets_buffer is None:
        data_buffers = [data_buffer]
    else:
        data_buffers = [data_buffer, offsets_buffer]

    result = pl.Series._from_buffers(polars_dtype, data=data_buffers, validity=None)

    # Add the validity buffer if present
    validity_buffer = _construct_validity_buffer(
        buffers["validity"],
        column,
        result,
        offset,
        allow_copy=allow_copy,
    )
    if validity_buffer is not None:
        result = pl.Series._from_buffers(
            polars_dtype, data=data_buffers, validity=validity_buffer
        )

    return result


def _construct_data_buffer(
    buffer: Buffer, dtype: Dtype, length: int, offset: int = 0
) -> Series:
    polars_dtype = dtype_to_polars_dtype(dtype)

    # TODO: Remove the line below when backward compatibility is no longer required
    # https://github.com/pola-rs/polars/pull/10787
    polars_dtype = polars_dtype_to_data_buffer_dtype(polars_dtype)

    buffer_info = (buffer.ptr, offset, length)
    return pl.Series._from_buffer(polars_dtype, buffer_info, owner=buffer)


def _construct_offsets_buffer(
    offsets_buffer_info: tuple[Buffer, Dtype] | None,
    offset: int = 0,
    *,
    allow_copy: bool = True,
) -> Series | None:
    if offsets_buffer_info is None:
        return None

    buffer, dtype = offsets_buffer_info

    polars_dtype = dtype_to_polars_dtype(dtype)
    length = get_buffer_length_in_elements(buffer, dtype)
    buffer_info = (buffer.ptr, offset, length)

    s = pl.Series._from_buffer(polars_dtype, buffer_info, owner=buffer)

    # Polars only supports Int64 offsets
    if polars_dtype != Int64:
        if not allow_copy:
            raise CopyNotAllowedError(
                f"offset buffer must be cast from {polars_dtype} to Int64"
            )
        s = s.cast(Int64)

    return s


def _construct_validity_buffer(
    validity_buffer_info: tuple[Buffer, Dtype] | None,
    column: Column,
    data: Series,
    offset: int = 0,
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
            buffer, null_value, column.size(), offset, allow_copy=allow_copy
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
        return data.is_not_nan()

    elif null_type == ColumnNullType.USE_SENTINEL:
        if not allow_copy:
            raise CopyNotAllowedError("bitmask must be constructed")
        return data != null_value

    else:
        raise NotImplementedError(f"unsupported null type: {null_type!r}")


def _construct_validity_buffer_from_bitmask(
    buffer: Buffer,
    null_value: int,
    length: int,
    offset: int = 0,
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
