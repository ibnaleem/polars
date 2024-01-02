from __future__ import annotations

from typing import TYPE_CHECKING

import polars._reexport as pl
import polars.functions as F
from polars.datatypes import Boolean, Enum, Int64, String, UInt8, UInt32
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
    from polars.type_aliases import PolarsDataType


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
        df = _protocol_df_chunk_to_polars(chunk, allow_copy=allow_copy)
        chunks.append(df)

    # Handle implementations that yield no chunks for an empty dataframe
    if not chunks:
        df = _protocol_df_chunk_to_polars(df, allow_copy=allow_copy)
        chunks.append(df)

    return F.concat(chunks, rechunk=False)


def _protocol_df_chunk_to_polars(
    df: InterchangeDataFrame, *, allow_copy: bool = True
) -> DataFrame:
    columns = []
    for column, name in zip(df.get_columns(), df.column_names()):
        dtype = dtype_to_polars_dtype(column.dtype)
        if dtype == String:
            s = _string_to_series(column, allow_copy=allow_copy)
        elif dtype == Enum:
            s = _categorical_to_series(column, allow_copy=allow_copy)
        else:
            s = _column_to_series(column, dtype, allow_copy=allow_copy)
        columns.append(s.alias(name))

    return pl.DataFrame(columns)


def _column_to_series(
    column: Column, dtype: PolarsDataType, *, allow_copy: bool = True
) -> Series:
    buffers = column.get_buffers()
    offset = column.offset

    data_buffer = _construct_data_buffer(*buffers["data"], column.size(), offset)
    validity_buffer = _construct_validity_buffer(
        buffers["validity"], column, data_buffer, offset, allow_copy=allow_copy
    )
    return pl.Series._from_buffers(dtype, data=data_buffer, validity=validity_buffer)


def _string_to_series(column: Column, *, allow_copy: bool = True) -> Series:
    buffers = column.get_buffers()
    offset = column.offset

    buffer, dtype = buffers["data"]
    data_buffer = _construct_data_buffer(buffer, dtype, buffer.bufsize, offset)

    offsets_buffer = _construct_offsets_buffer(
        buffers["offsets"], offset, allow_copy=allow_copy
    )

    # Construct the
    data_buffers = [data_buffer, offsets_buffer]
    data = pl.Series._from_buffers(String, data=data_buffers, validity=None)

    # Add the validity buffer if present
    validity_buffer = _construct_validity_buffer(
        buffers["validity"], column, data, offset, allow_copy=allow_copy
    )
    if validity_buffer is not None:
        data = pl.Series._from_buffers(
            String, data=data_buffers, validity=validity_buffer
        )

    return data


def _categorical_to_series(column: Column, *, allow_copy: bool = True) -> Series:
    if not allow_copy:
        raise CopyNotAllowedError("categorical mapping must be constructed")

    categorical = column.describe_categorical
    if not categorical["is_dictionary"]:
        raise NotImplementedError("non-dictionary categoricals are not yet supported")

    buffers = column.get_buffers()
    offset = column.offset

    data_buffer = _construct_data_buffer(*buffers["data"], column.size(), offset)

    # Polars only supports UInt32 categoricals
    if data_buffer.dtype != UInt32:
        if not allow_copy:
            raise CopyNotAllowedError(
                f"data buffer must be cast from {data_buffer.dtype} to UInt32"
            )
        data_buffer = data_buffer.cast(UInt32)

    validity_buffer = _construct_validity_buffer(
        buffers["validity"], column, data_buffer, offset, allow_copy=allow_copy
    )

    dtype = Enum(categorical["categories"])
    return pl.Series._from_buffers(dtype, data=data_buffer, validity=validity_buffer)


def _construct_data_buffer(
    buffer: Buffer, dtype: Dtype, length: int, offset: int = 0
) -> Series:
    polars_dtype = dtype_to_polars_dtype(dtype)

    # TODO: Remove the line below and associated utility function when backward
    # compatibility is no longer required
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
