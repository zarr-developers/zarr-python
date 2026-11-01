"""Strategies for each data type this package defines that holds no other field.

A fill value is drawn in every spelling the data type's spec gives it:
a float's as a number, a named non-finite value or a hex string of its
bits, a `bytes` value's as an array of byte values or base64.
"""

from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args

from hypothesis import strategies as st

from zarr_metadata._numpy import NUMPY_TIME_MAX_SCALE_FACTOR
from zarr_metadata.testing._strategy import EMPTY_CONFIGURATIONS, DataTypeStrategy
from zarr_metadata.v3.data_type._float import FloatSpecialFillValue
from zarr_metadata.v3.data_type._numpy_time import NUMPY_TIME_UNIT
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE
from zarr_metadata.v3.data_type.fixed_length_utf32 import (
    FIXED_LENGTH_UTF32_DATA_TYPE,
    FixedLengthUtf32Configuration,
)
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE, RawBytesConfiguration
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata._common import JSONValue
    from zarr_metadata.v3._definition import DataTypeDefinition

FloatWidth = Literal[16, 32, 64]

INTEGER_RANGES: Final[Mapping[DataTypeDefinition[Any], tuple[int, int]]] = {
    INT8_DATA_TYPE: (-(2**7), 2**7 - 1),
    INT16_DATA_TYPE: (-(2**15), 2**15 - 1),
    INT32_DATA_TYPE: (-(2**31), 2**31 - 1),
    INT64_DATA_TYPE: (-(2**63), 2**63 - 1),
    UINT8_DATA_TYPE: (0, 2**8 - 1),
    UINT16_DATA_TYPE: (0, 2**16 - 1),
    UINT32_DATA_TYPE: (0, 2**32 - 1),
    UINT64_DATA_TYPE: (0, 2**64 - 1),
}
"""The values each integer data type holds."""

FLOAT_WIDTHS: Final[Mapping[DataTypeDefinition[Any], FloatWidth]] = {
    FLOAT16_DATA_TYPE: 16,
    FLOAT32_DATA_TYPE: 32,
    FLOAT64_DATA_TYPE: 64,
}
"""The width in bits of each floating-point data type."""

COMPLEX_WIDTHS: Final[Mapping[DataTypeDefinition[Any], FloatWidth]] = {
    COMPLEX64_DATA_TYPE: 32,
    COMPLEX128_DATA_TYPE: 64,
}
"""The width in bits of each part of each complex data type."""

_MAX_RAW_BYTES: Final = 32
"""The widest raw bits drawn, in bytes: `r256`."""

_MAX_BYTES: Final = 16
"""The longest `bytes` fill value drawn."""


def finite_floats(width: FloatWidth) -> st.SearchStrategy[float]:
    """Finite numbers a float of `width` bits holds exactly."""
    return st.floats(width=width, allow_nan=False, allow_infinity=False)


_PACKED: Final[Mapping[FloatWidth, str]] = {16: ">e", 32: ">f", 64: ">d"}
"""How `struct` packs a float of each width, most significant byte first."""


def _hex_spellings(digits: str) -> st.SearchStrategy[str]:
    """`0x` and the digits, the letters in either case."""
    return st.sampled_from((f"0x{digits}", f"0x{digits.upper()}"))


def float_spellings(value: float, width: FloatWidth) -> st.SearchStrategy[JSONValue]:
    """`value`, a float of `width` bits, as a number or as the hex string of its bits."""
    return st.just(value) | _hex_spellings(struct.pack(_PACKED[width], value).hex())


def _float_fill_values(width: FloatWidth) -> st.SearchStrategy[JSONValue]:
    """A number, an integer the type holds exactly, a named non-finite value, or the bits in hex.

    A float's fill value is any of these
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79).
    Numbers are drawn from those the type holds, so none is rounded: a
    `float16` takes `1e300` by the letter of the spec, and a reader is
    free to make of it what it will.
    """
    # Every integer of magnitude up to 2**11 is a float16, the narrowest type.
    numbers = finite_floats(width) | st.integers(-(2**11), 2**11)
    named = st.sampled_from(get_args(FloatSpecialFillValue))
    bits = st.integers(0, 2**width - 1).flatmap(
        lambda value: _hex_spellings(f"{value:0{width // 4}x}")
    )
    return st.one_of(numbers, named, bits)


def _fill_values_of(values: st.SearchStrategy[JSONValue]) -> DataTypeStrategy:
    """A data type with nothing to configure, whose fill values are `values`."""
    return DataTypeStrategy(
        configurations=EMPTY_CONFIGURATIONS, fill_values=lambda _configuration: values
    )


def _byte_values(size: int) -> st.SearchStrategy[JSONValue]:
    """`size` byte values, each an integer in `[0, 255]`, as a JSON array."""
    return st.lists(st.integers(0, 255), min_size=size, max_size=size).map(
        lambda values: cast("JSONValue", tuple(values))
    )


def _raw_bytes_fill_values(configuration: Mapping[str, JSONValue]) -> st.SearchStrategy[JSONValue]:
    """One byte value for each 8 bits of the size."""
    return _byte_values(cast("RawBytesConfiguration", configuration)["bits"] // 8)


def _base64(value: bytes) -> JSONValue:
    return base64.b64encode(value).decode("ascii")


def _complex_fill_values(width: FloatWidth) -> st.SearchStrategy[JSONValue]:
    """A pair of float fill values of `width` bits, real then imaginary."""
    component = _float_fill_values(width)
    return st.tuples(component, component).map(lambda pair: cast("JSONValue", pair))


_BYTES_FILL_VALUES: Final = st.integers(0, _MAX_BYTES).flatmap(_byte_values) | st.binary(
    max_size=_MAX_BYTES
).map(_base64)
"""A `bytes` fill value: byte values, or the bytes in standard-alphabet base64."""

_TEXT: Final = st.text(st.characters(exclude_categories=("Cs",)), max_size=16)
"""Text a JSON document can hold: no lone surrogate, which is no Unicode scalar value."""

_MAX_CODE_POINTS: Final = 16
"""The most code points a drawn `fixed_length_utf32` value holds."""


def _fixed_length_utf32_fill_values(
    configuration: Mapping[str, JSONValue],
) -> st.SearchStrategy[JSONValue]:
    """Text of at most one code point for each 4 bytes of the size."""
    capacity = cast("FixedLengthUtf32Configuration", configuration)["length_bytes"] // 4
    return st.text(st.characters(exclude_categories=("Cs",)), max_size=capacity)


_NUMPY_TIME: Final = DataTypeStrategy(
    configurations=st.fixed_dictionaries(
        {
            "unit": st.sampled_from(NUMPY_TIME_UNIT),
            "scale_factor": st.integers(1, NUMPY_TIME_MAX_SCALE_FACTOR),
        }
    ),
    fill_values=lambda _configuration: st.integers(-(2**63), 2**63 - 1) | st.just("NaT"),
)
"""Both numpy time types: a unit, and how many of it one tick is; a signed 64-bit count of ticks, or `"NaT"`."""

DATA_TYPE_STRATEGIES: Final[Mapping[DataTypeDefinition[Any], DataTypeStrategy]] = {
    BOOL_DATA_TYPE: _fill_values_of(st.booleans()),
    **{
        definition: _fill_values_of(st.integers(low, high))
        for definition, (low, high) in INTEGER_RANGES.items()
    },
    **{
        definition: _fill_values_of(_float_fill_values(width))
        for definition, width in FLOAT_WIDTHS.items()
    },
    **{
        definition: _fill_values_of(_complex_fill_values(width))
        for definition, width in COMPLEX_WIDTHS.items()
    },
    RAW_BYTES_DATA_TYPE: DataTypeStrategy(
        configurations=st.integers(1, _MAX_RAW_BYTES).map(
            lambda size: RawBytesConfiguration(bits=8 * size)
        ),
        fill_values=_raw_bytes_fill_values,
    ),
    BYTES_DATA_TYPE: _fill_values_of(_BYTES_FILL_VALUES),
    STRING_DATA_TYPE: _fill_values_of(_TEXT),
    FIXED_LENGTH_UTF32_DATA_TYPE: DataTypeStrategy(
        configurations=st.integers(1, _MAX_CODE_POINTS).map(
            lambda count: FixedLengthUtf32Configuration(length_bytes=4 * count)
        ),
        fill_values=_fixed_length_utf32_fill_values,
    ),
    NUMPY_DATETIME64_DATA_TYPE: _NUMPY_TIME,
    NUMPY_TIMEDELTA64_DATA_TYPE: _NUMPY_TIME,
}
"""A strategy for each data type that holds no other field."""

__all__ = [
    "COMPLEX_WIDTHS",
    "DATA_TYPE_STRATEGIES",
    "FLOAT_WIDTHS",
    "INTEGER_RANGES",
    "FloatWidth",
    "finite_floats",
    "float_spellings",
]
