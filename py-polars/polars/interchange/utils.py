from __future__ import annotations

import re
from typing import TYPE_CHECKING

from polars.datatypes import (
    Boolean,
    Categorical,
    Date,
    Datetime,
    Duration,
    Enum,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    String,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
)
from polars.interchange.protocol import DtypeKind, Endianness

if TYPE_CHECKING:
    from polars.datatypes import DataTypeClass
    from polars.interchange.protocol import Buffer, Dtype
    from polars.type_aliases import PolarsDataType

NE = Endianness.NATIVE

polars_dtype_to_dtype_map: dict[DataTypeClass, Dtype] = {
    Int8: (DtypeKind.INT, 8, "c", NE),
    Int16: (DtypeKind.INT, 16, "s", NE),
    Int32: (DtypeKind.INT, 32, "i", NE),
    Int64: (DtypeKind.INT, 64, "l", NE),
    UInt8: (DtypeKind.UINT, 8, "C", NE),
    UInt16: (DtypeKind.UINT, 16, "S", NE),
    UInt32: (DtypeKind.UINT, 32, "I", NE),
    UInt64: (DtypeKind.UINT, 64, "L", NE),
    Float32: (DtypeKind.FLOAT, 32, "f", NE),
    Float64: (DtypeKind.FLOAT, 64, "g", NE),
    Boolean: (DtypeKind.BOOL, 1, "b", NE),
    String: (DtypeKind.STRING, 8, "U", NE),
    Date: (DtypeKind.DATETIME, 32, "tdD", NE),
    Time: (DtypeKind.DATETIME, 64, "ttu", NE),
    Datetime: (DtypeKind.DATETIME, 64, "tsu:", NE),
    Duration: (DtypeKind.DATETIME, 64, "tDu", NE),
    Categorical: (DtypeKind.CATEGORICAL, 32, "I", NE),
    Enum: (DtypeKind.CATEGORICAL, 32, "I", NE),
}


def polars_dtype_to_dtype(dtype: PolarsDataType) -> Dtype:
    """Convert Polars data type to interchange protocol data type."""
    try:
        result = polars_dtype_to_dtype_map[dtype.base_type()]
    except KeyError as exc:
        raise ValueError(
            f"data type {dtype!r} not supported by the interchange protocol"
        ) from exc

    # Handle instantiated data types
    if isinstance(dtype, Datetime):
        return _datetime_to_dtype(dtype)
    elif isinstance(dtype, Duration):
        return _duration_to_dtype(dtype)

    return result


def _datetime_to_dtype(dtype: Datetime) -> Dtype:
    tu = dtype.time_unit[0] if dtype.time_unit is not None else "u"
    tz = dtype.time_zone if dtype.time_zone is not None else ""
    arrow_c_type = f"ts{tu}:{tz}"
    return DtypeKind.DATETIME, 64, arrow_c_type, NE


def _duration_to_dtype(dtype: Duration) -> Dtype:
    tu = dtype.time_unit[0] if dtype.time_unit is not None else "u"
    arrow_c_type = f"tD{tu}"
    return DtypeKind.DATETIME, 64, arrow_c_type, NE


dtype_to_polars_dtype_map: dict[DtypeKind, dict[int, DataTypeClass]] = {
    DtypeKind.INT: {
        8: Int8,
        16: Int16,
        32: Int32,
        64: Int64,
    },
    DtypeKind.UINT: {
        8: UInt8,
        16: UInt16,
        32: UInt32,
        64: UInt64,
    },
    DtypeKind.FLOAT: {
        32: Float32,
        64: Float64,
    },
    DtypeKind.BOOL: {1: Boolean},
    DtypeKind.STRING: {8: Utf8},
}


def dtype_to_polars_dtype(dtype: Dtype) -> PolarsDataType:
    """Convert interchange protocol data type to Polars data type."""
    kind, bit_width, format_str, _ = dtype

    if kind == DtypeKind.DATETIME:
        return _temporal_dtype_to_polars_dtype(format_str)
    elif kind == DtypeKind.CATEGORICAL:
        return Categorical

    try:
        return dtype_to_polars_dtype_map[kind][bit_width]
    except KeyError as exc:
        raise NotImplementedError(f"unsupported data type: {format_str!r}") from exc


def _temporal_dtype_to_polars_dtype(format_str: str) -> PolarsDataType:
    if (match := re.fullmatch(r"ts([mun]):(.*)", format_str)) is not None:
        time_unit = match.group(1)
        time_zone = match.group(2) or None
        return Datetime(
            time_unit=time_unit,  # type: ignore[arg-type]
            time_zone=time_zone,
        )
    elif format_str == "tdD":
        return Date
    elif format_str == "ttu":
        return Time
    elif (match := re.fullmatch(r"tD([mun])", format_str)) is not None:
        time_unit = match.group(1)
        return Duration(time_unit=time_unit)  # type: ignore[arg-type]

    raise NotImplementedError(f"unsupported temporal data type: {format_str!r}")


def get_buffer_size_in_elements(buffer: Buffer, dtype: Dtype) -> int:
    """Get the length of a buffer in elements."""
    bits_per_element = dtype[1]
    bytes_per_element, rest = divmod(bits_per_element, 8)
    if rest > 0:
        raise ValueError(f"cannot get buffer length for buffer with dtype {dtype!r}")
    return buffer.bufsize // bytes_per_element
