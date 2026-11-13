"""The types of the v3 data types.

For each data type: the value and `Literal` type of its name, its
configuration and object forms where it has them, the JSON shape of its
fill value, and the constants that spell its permitted values.

This module holds types only and imports no definition, so a definition
of any kind can import a data type's types at run time. Each data type's
submodule of `zarr_metadata.v3.data_type` re-exports its names, and holds
the data type's definition and rules.
"""

from collections.abc import Mapping
from typing import Annotated, Final, Literal, NewType, NotRequired, get_args

from annotated_types import Interval
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import JSONValue
from zarr_metadata._numpy import NUMPY_TIME_UNITS, Base64Bytes, NumpyTimeScaleFactor, NumpyTimeTicks
from zarr_metadata.v3._common import ZarrV3DataTypeDeclaration

# Shared by more than one data type

ByteValue = Annotated[int, Interval(ge=0, le=255)]
"""One byte of a fill value: a JSON integer in `[0, 255]`, as the fill values of raw bits and of `bytes` write each byte.

See https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L97-L99
for raw bits, and https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md?plain=1#L8
for `bytes`.
"""

FloatSpecialFillValue = Literal["NaN", "Infinity", "-Infinity"]
"""The named non-finite fill values every IEEE 754 floating-point type takes."""

FLOAT_SPECIAL_FILL_VALUES: Final[tuple[FloatSpecialFillValue, ...]] = get_args(
    FloatSpecialFillValue
)
"""The strings that the fill value rules accept for a float fill value in place of a number."""

NumpyTimeUnit = Literal[
    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic"
]
"""The time unit codes of the numpy time types, `numpy.datetime64` and `numpy.timedelta64`."""

NUMPY_TIME_UNIT: Final = NUMPY_TIME_UNITS
"""Runtime tuple of the permitted `numpy.timedelta64`/`numpy.datetime64` unit strings."""


# bool: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

BOOL_DATA_TYPE_NAME: Final = "bool"
"""The `data_type` value for the `bool` type."""

BoolDataTypeName = Literal["bool"]
"""Literal type of the `data_type` field for `bool`."""

BoolFillValue = bool
"""Permitted JSON shape of the `fill_value` field for `bool`: a JSON boolean."""


# int8: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

INT8_DATA_TYPE_NAME: Final = "int8"
"""The `data_type` value for the `int8` type."""

Int8DataTypeName = Literal["int8"]
"""Literal type of the `data_type` field for `int8`."""

Int8FillValue = Annotated[int, Interval(ge=-(2**7), le=2**7 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int8`: a JSON integer in [-128, 127]."""


# int16: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

INT16_DATA_TYPE_NAME: Final = "int16"
"""The `data_type` value for the `int16` type."""

Int16DataTypeName = Literal["int16"]
"""Literal type of the `data_type` field for `int16`."""

Int16FillValue = Annotated[int, Interval(ge=-(2**15), le=2**15 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int16`: a JSON integer in [-32768, 32767]."""


# int32: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

INT32_DATA_TYPE_NAME: Final = "int32"
"""The `data_type` value for the `int32` type."""

Int32DataTypeName = Literal["int32"]
"""Literal type of the `data_type` field for `int32`."""

Int32FillValue = Annotated[int, Interval(ge=-(2**31), le=2**31 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int32`: a JSON integer in [-2**31, 2**31 - 1]."""


# int64: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

INT64_DATA_TYPE_NAME: Final = "int64"
"""The `data_type` value for the `int64` type."""

Int64DataTypeName = Literal["int64"]
"""Literal type of the `data_type` field for `int64`."""

Int64FillValue = Annotated[int, Interval(ge=-(2**63), le=2**63 - 1)]
"""Permitted JSON shape of the `fill_value` field for `int64`: a JSON integer in [-2**63, 2**63 - 1]."""


# uint8: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

UINT8_DATA_TYPE_NAME: Final = "uint8"
"""The `data_type` value for the `uint8` type."""

Uint8DataTypeName = Literal["uint8"]
"""Literal type of the `data_type` field for `uint8`."""

Uint8FillValue = Annotated[int, Interval(ge=0, le=2**8 - 1)]
"""Permitted JSON shape of the `fill_value` field for `uint8`: a JSON integer in [0, 255]."""


# uint16: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

UINT16_DATA_TYPE_NAME: Final = "uint16"
"""The `data_type` value for the `uint16` type."""

Uint16DataTypeName = Literal["uint16"]
"""Literal type of the `data_type` field for `uint16`."""

Uint16FillValue = Annotated[int, Interval(ge=0, le=2**16 - 1)]
"""Permitted JSON shape of the `fill_value` field for `uint16`: a JSON integer in [0, 65535]."""


# uint32: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

UINT32_DATA_TYPE_NAME: Final = "uint32"
"""The `data_type` value for the `uint32` type."""

Uint32DataTypeName = Literal["uint32"]
"""Literal type of the `data_type` field for `uint32`."""

Uint32FillValue = Annotated[int, Interval(ge=0, le=2**32 - 1)]
"""Permitted JSON shape of the `fill_value` field for `uint32`: a JSON integer in [0, 2**32 - 1]."""


# uint64: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

UINT64_DATA_TYPE_NAME: Final = "uint64"
"""The `data_type` value for the `uint64` type."""

Uint64DataTypeName = Literal["uint64"]
"""Literal type of the `data_type` field for `uint64`."""

Uint64FillValue = Annotated[int, Interval(ge=0, le=2**64 - 1)]
"""Permitted JSON shape of the `fill_value` field for `uint64`: a JSON integer in [0, 2**64 - 1]."""


# float16: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

FLOAT16_DATA_TYPE_NAME: Final = "float16"
"""The `data_type` value for the `float16` type."""

Float16DataTypeName = Literal["float16"]
"""Literal type of the `data_type` field for `float16`."""

Float16SpecialFillValue = FloatSpecialFillValue
"""Named non-finite fill values permitted by the spec for IEEE 754 floats.

https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79
"""

HexFloat16 = NewType("HexFloat16", str)
"""A 6-character hex string (`0x` + 4 hex digits) encoding the
unsigned-integer representation of a float16."""

Float16FillValue = float | int | Float16SpecialFillValue | HexFloat16
"""Permitted JSON shape of the `fill_value` field for `float16`.

Either a JSON number, one of the named non-finite sentinels (`"NaN"`,
`"Infinity"`, `"-Infinity"`), or a `HexFloat16` (`0xYYYY` string encoding
the unsigned-integer representation of the IEEE 754 value).
"""

CANONICAL_NAN_HEX_FLOAT16: Final = "0x7e00"
"""Canonical hex form of the float16 NaN sentinel `"NaN"`.

Per spec (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L72-L74) the named `"NaN"` sentinel denotes the float with sign=0, the
most significant mantissa bit set, and all other mantissa bits zero
(the IEEE 754 default quiet NaN). Other NaN bit patterns must be
encoded with the explicit hex-string form.
"""

CANONICAL_POSITIVE_INFINITY_HEX_FLOAT16: Final = "0x7c00"
"""Canonical hex form of the float16 `"Infinity"` sentinel."""

CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT16: Final = "0xfc00"
"""Canonical hex form of the float16 `"-Infinity"` sentinel."""


# float32: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

FLOAT32_DATA_TYPE_NAME: Final = "float32"
"""The `data_type` value for the `float32` type."""

Float32DataTypeName = Literal["float32"]
"""Literal type of the `data_type` field for `float32`."""

Float32SpecialFillValue = FloatSpecialFillValue
"""Named non-finite fill values permitted by the spec for IEEE 754 floats.

https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79
"""

HexFloat32 = NewType("HexFloat32", str)
"""A 10-character hex string (`0x` + 8 hex digits) encoding the
unsigned-integer representation of a float32."""

Float32FillValue = float | int | Float32SpecialFillValue | HexFloat32
"""Permitted JSON shape of the `fill_value` field for `float32`.

Either a JSON number, one of the named non-finite sentinels (`"NaN"`,
`"Infinity"`, `"-Infinity"`), or a `HexFloat32` (`0xYYYYYYYY` string
encoding the unsigned-integer representation of the IEEE 754 value).
"""

CANONICAL_NAN_HEX_FLOAT32: Final = "0x7fc00000"
"""Canonical hex form of the float32 NaN sentinel `"NaN"`.

Per spec (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L72-L74) the named `"NaN"` sentinel denotes the float with sign=0, the
most significant mantissa bit set, and all other mantissa bits zero
(the IEEE 754 default quiet NaN). Other NaN bit patterns must be
encoded with the explicit hex-string form.
"""

CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32: Final = "0x7f800000"
"""Canonical hex form of the float32 `"Infinity"` sentinel."""

CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32: Final = "0xff800000"
"""Canonical hex form of the float32 `"-Infinity"` sentinel."""


# float64: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

FLOAT64_DATA_TYPE_NAME: Final = "float64"
"""The `data_type` value for the `float64` type."""

Float64DataTypeName = Literal["float64"]
"""Literal type of the `data_type` field for `float64`."""

Float64SpecialFillValue = FloatSpecialFillValue
"""Named non-finite fill values permitted by the spec for IEEE 754 floats.

https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L63-L79
"""

HexFloat64 = NewType("HexFloat64", str)
"""An 18-character hex string (`0x` + 16 hex digits) encoding the
unsigned-integer representation of a float64."""

Float64FillValue = float | int | Float64SpecialFillValue | HexFloat64
"""Permitted JSON shape of the `fill_value` field for `float64`.

Either a JSON number, one of the named non-finite sentinels (`"NaN"`,
`"Infinity"`, `"-Infinity"`), or a `HexFloat64` (`0xYYYYYYYYYYYYYYYY`
string encoding the unsigned-integer representation of the IEEE 754
value).
"""

CANONICAL_NAN_HEX_FLOAT64: Final = "0x7ff8000000000000"
"""Canonical hex form of the float64 NaN sentinel `"NaN"`.

Per spec (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L72-L74) the named `"NaN"` sentinel denotes the float with sign=0, the
most significant mantissa bit set, and all other mantissa bits zero
(the IEEE 754 default quiet NaN). Other NaN bit patterns must be
encoded with the explicit hex-string form.
"""

CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64: Final = "0x7ff0000000000000"
"""Canonical hex form of the float64 `"Infinity"` sentinel."""

CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64: Final = "0xfff0000000000000"
"""Canonical hex form of the float64 `"-Infinity"` sentinel."""


# complex64: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

COMPLEX64_DATA_TYPE_NAME: Final = "complex64"
"""The `data_type` value for the `complex64` type."""

Complex64DataTypeName = Literal["complex64"]
"""Literal type of the `data_type` field for `complex64`."""

Complex64Component = Float32FillValue
"""One real or imaginary component of a `complex64` fill value.

Same shape as a `float32` fill value: a JSON number, a named sentinel,
or a `HexFloat32` string.
"""

Complex64FillValue = tuple[Complex64Component, Complex64Component]
"""Permitted JSON shape of the `fill_value` field for `complex64`.

A two-element JSON array `[real, imag]` where each component is a
`Complex64Component`.
"""


# complex128: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

COMPLEX128_DATA_TYPE_NAME: Final = "complex128"
"""The `data_type` value for the `complex128` type."""

Complex128DataTypeName = Literal["complex128"]
"""Literal type of the `data_type` field for `complex128`."""

Complex128Component = Float64FillValue
"""One real or imaginary component of a `complex128` fill value.

Same shape as a `float64` fill value: a JSON number, a named sentinel,
or a `HexFloat64` string.
"""

Complex128FillValue = tuple[Complex128Component, Complex128Component]
"""Permitted JSON shape of the `fill_value` field for `complex128`.

A two-element JSON array `[real, imag]` where each component is a
`Complex128Component`.
"""


# raw: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html

RawBytesDataTypeName = NewType("RawBytesDataTypeName", str)
"""A spec-conformant `r<N>` raw-bytes name (e.g. `"r8"`, `"r16"`).

"raw bits, variable size given by *, limited to be a multiple of 8":
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/data-types/index.rst#L46-L47
"""

RawBytesFillValue = tuple[ByteValue, ...]
"""Permitted JSON shape of the `fill_value` field for `r<N>`.

A JSON array of N/8 integers in `[0, 255]` (one per byte).
"""


class RawBytesConfiguration(TypedDict, closed=True):
    """The configuration that a raw-bytes name carries: its size in bits, such as `{"bits": 16}` for `r16`.

    A document writes the name, never this object. The definition of `r*`
    reads a name into this object, and the simplest spelling writes it
    back as the name.
    """

    bits: int
    """The size in bits that the name writes, `16` for `r16`.

    A positive multiple of 8.
    """


# bytes: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/bytes/README.md

BYTES_DATA_TYPE_NAME: Final = "bytes"
"""The `data_type` value for the variable-length `bytes` type."""

BytesDataTypeName = Literal["bytes"]
"""Literal type of the `data_type` field for `bytes`."""

BytesFillValue = tuple[ByteValue, ...] | Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `bytes`.

Either a JSON array of integers in `[0, 255]` (one per byte), or a
`Base64Bytes` string encoding the byte sequence.
"""


# string: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/string/README.md

STRING_DATA_TYPE_NAME: Final = "string"
"""The `data_type` value for the `string` type."""

StringDataTypeName = Literal["string"]
"""Literal type of the `data_type` field for `string`."""

StringFillValue = str
"""Permitted JSON shape of the `fill_value` field for `string`: a JSON unicode string."""


# variable_length_bytes: https://github.com/zarr-developers/zarr-python/blob/a2f2415c86f7d35212ae8c0d5dc5e4a22a6c6191/src/zarr/core/dtype/npy/bytes.py#L1141-L1143

VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME: Final = "variable_length_bytes"
"""The `data_type` value for the `variable_length_bytes` type."""

VariableLengthBytesDataTypeName = Literal["variable_length_bytes"]
"""Literal type of the `data_type` field for `variable_length_bytes`."""

VariableLengthBytesFillValue = Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `variable_length_bytes`: a `Base64Bytes` string, which is what zarr-python writes and reads."""


# fixed_length_utf32: https://github.com/zarr-developers/zarr-extensions/blob/cadd1dab6c9aebafb66080b78b583f6507466b3d/data-types/fixed_length_utf32/README.md

FIXED_LENGTH_UTF32_DATA_TYPE_NAME: Final = "fixed_length_utf32"
"""The `name` field value of the `fixed_length_utf32` data type."""

FixedLengthUtf32DataTypeName = Literal["fixed_length_utf32"]
"""Literal type of the `name` field of the `fixed_length_utf32` data type."""


class FixedLengthUtf32Configuration(TypedDict, closed=True):
    """Configuration for the `fixed_length_utf32` data type."""

    length_bytes: ReadOnly[int]
    """The size in bytes of every value: a positive multiple of 4, since
    each code point takes 4 bytes."""


class FixedLengthUtf32(TypedDict, closed=True):
    """`fixed_length_utf32` data type metadata."""

    name: FixedLengthUtf32DataTypeName
    """The data type's name, `"fixed_length_utf32"`."""
    configuration: FixedLengthUtf32Configuration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


FixedLengthUtf32FillValue = str
"""Permitted JSON shape of the `fill_value` field for `fixed_length_utf32`: a JSON string of at most `length_bytes / 4` code points."""


# null_terminated_bytes: https://github.com/zarr-developers/zarr-python/blob/a2f2415c86f7d35212ae8c0d5dc5e4a22a6c6191/src/zarr/core/dtype/npy/bytes.py#L385-L390

NULL_TERMINATED_BYTES_DATA_TYPE_NAME: Final = "null_terminated_bytes"
"""The `name` field value of the `null_terminated_bytes` data type."""

NullTerminatedBytesDataTypeName = Literal["null_terminated_bytes"]
"""Literal type of the `name` field of the `null_terminated_bytes` data type."""


class NullTerminatedBytesConfiguration(TypedDict, closed=True):
    """Configuration for the `null_terminated_bytes` data type."""

    length_bytes: ReadOnly[int]
    """The size in bytes of every value, at least 1."""


class NullTerminatedBytes(TypedDict, closed=True):
    """`null_terminated_bytes` data type metadata."""

    name: NullTerminatedBytesDataTypeName
    """The data type's name, `"null_terminated_bytes"`."""
    configuration: NullTerminatedBytesConfiguration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


NullTerminatedBytesFillValue = Base64Bytes
"""Permitted JSON shape of the `fill_value` field for `null_terminated_bytes`: the bytes, without their trailing null bytes, as a `Base64Bytes` string."""


# zarr_python_raw_bytes: https://github.com/zarr-developers/zarr-python/blob/a2f2415c86f7d35212ae8c0d5dc5e4a22a6c6191/src/zarr/core/dtype/npy/bytes.py#L783-L785

ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME: Final = "raw_bytes"
"""The `name` field value of zarr-python's `raw_bytes` data type."""

ZarrPythonRawBytesDataTypeName = Literal["raw_bytes"]
"""Literal type of the `name` field of zarr-python's `raw_bytes` data type."""


class ZarrPythonRawBytesConfiguration(TypedDict, closed=True):
    """Configuration for zarr-python's `raw_bytes` data type."""

    length_bytes: ReadOnly[int]
    """The size in bytes of every value, at least 1."""


class ZarrPythonRawBytes(TypedDict, closed=True):
    """zarr-python's `raw_bytes` data type metadata."""

    name: ZarrPythonRawBytesDataTypeName
    """The data type's name, `"raw_bytes"`."""
    configuration: ZarrPythonRawBytesConfiguration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


ZarrPythonRawBytesFillValue = Base64Bytes
"""Permitted JSON shape of the `fill_value` field for zarr-python's `raw_bytes`: the bytes as a `Base64Bytes` string, which zarr-python writes `length_bytes` long."""


# numpy_datetime64: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md

NUMPY_DATETIME64_DATA_TYPE_NAME: Final = "numpy.datetime64"
"""The `name` field value of the `numpy.datetime64` data type."""

NumpyDatetime64DataTypeName = Literal["numpy.datetime64"]
"""Literal type of the `name` field of the `numpy.datetime64` data type."""


class NumpyDatetime64Configuration(TypedDict, closed=True):
    """Configuration for the `numpy.datetime64` data type."""

    unit: ReadOnly[NumpyTimeUnit]
    """A string encoding a unit of time.

    `"us"` and `"μs"` are equivalent spellings of microseconds; `"generic"`
    means the unit is unspecified.
    """
    scale_factor: ReadOnly[NumpyTimeScaleFactor]
    """The multiplier relative to the unit.

    An integer in `[1, 2147483647]`.
    """


class NumpyDatetime64(TypedDict, closed=True):
    """`numpy.datetime64` data type metadata."""

    name: NumpyDatetime64DataTypeName
    """The data type's name, `"numpy.datetime64"`."""
    configuration: NumpyDatetime64Configuration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


NumpyDatetime64FillValue = NumpyTimeTicks | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.datetime64`.

Either a JSON integer (count of `unit * scale_factor` since the epoch),
or the string `"NaT"` (equivalent to the integer `-2**63`).
"""


# numpy_timedelta64: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.timedelta64/README.md

NUMPY_TIMEDELTA64_DATA_TYPE_NAME: Final = "numpy.timedelta64"
"""The `name` field value of the `numpy.timedelta64` data type."""

NumpyTimedelta64DataTypeName = Literal["numpy.timedelta64"]
"""Literal type of the `name` field of the `numpy.timedelta64` data type."""


class NumpyTimedelta64Configuration(TypedDict, closed=True):
    """Configuration for the `numpy.timedelta64` data type."""

    unit: ReadOnly[NumpyTimeUnit]
    """A string encoding a unit of time.

    `"us"` and `"μs"` are equivalent spellings of microseconds; `"generic"`
    means the unit is unspecified.
    """
    scale_factor: ReadOnly[NumpyTimeScaleFactor]
    """The multiplier relative to the unit.

    An integer in `[1, 2147483647]`.
    """


class NumpyTimedelta64(TypedDict, closed=True):
    """`numpy.timedelta64` data type metadata."""

    name: NumpyTimedelta64DataTypeName
    """The data type's name, `"numpy.timedelta64"`."""
    configuration: NumpyTimedelta64Configuration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


NumpyTimedelta64FillValue = NumpyTimeTicks | Literal["NaT"]
"""Permitted JSON shape of the `fill_value` field for `numpy.timedelta64`.

Either a JSON integer (a count of `unit * scale_factor`), or the string
`"NaT"` (equivalent to the integer `-2**63`).
"""


# struct: https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/struct/README.md

STRUCT_DATA_TYPE_NAME: Final = "struct"
"""The `name` field value of the `struct` data type."""

StructDataTypeName = Literal["struct"]
"""Literal type of the `name` field of the `struct` data type."""


class StructField(TypedDict, closed=True):
    """A single field entry inside a structured dtype."""

    name: ReadOnly[str]
    """The field name (must be unique within a struct and non-empty)."""
    data_type: ReadOnly[ZarrV3DataTypeDeclaration]
    """The field's data type.

    Recursive: may be a bare-string primitive or a named-config envelope
    including another `struct`.
    """


class StructConfiguration(TypedDict, closed=True):
    """Configuration for the `struct` data type."""

    fields: ReadOnly[tuple[StructField, ...]]
    """The record's fields, in the order they are encoded.

    At least one. The order is significant: it is the order in which the
    `bytes` codec packs the fields' values.
    """


class Struct(TypedDict, closed=True):
    """`struct` data type metadata."""

    name: StructDataTypeName
    """The data type's name, `"struct"`."""
    configuration: StructConfiguration
    """The data type's parameters."""
    must_understand: NotRequired[bool]
    """Whether a reader that does not recognize this data type must refuse the document.

    Absent means `true`. The spec does not permit `false` for a data type
    (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1580-L1581).
    """


StructFillValue = Mapping[str, JSONValue]
"""Permitted JSON shape of the `fill_value` field for `struct`.

A JSON object mapping each field name to that field's fill value. Field
fill values are themselves shaped per the field's `data_type`, recursively.
"""


__all__ = [
    "BOOL_DATA_TYPE_NAME",
    "BYTES_DATA_TYPE_NAME",
    "CANONICAL_NAN_HEX_FLOAT16",
    "CANONICAL_NAN_HEX_FLOAT32",
    "CANONICAL_NAN_HEX_FLOAT64",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT16",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT32",
    "CANONICAL_NEGATIVE_INFINITY_HEX_FLOAT64",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT16",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT32",
    "CANONICAL_POSITIVE_INFINITY_HEX_FLOAT64",
    "COMPLEX64_DATA_TYPE_NAME",
    "COMPLEX128_DATA_TYPE_NAME",
    "FIXED_LENGTH_UTF32_DATA_TYPE_NAME",
    "FLOAT16_DATA_TYPE_NAME",
    "FLOAT32_DATA_TYPE_NAME",
    "FLOAT64_DATA_TYPE_NAME",
    "FLOAT_SPECIAL_FILL_VALUES",
    "INT8_DATA_TYPE_NAME",
    "INT16_DATA_TYPE_NAME",
    "INT32_DATA_TYPE_NAME",
    "INT64_DATA_TYPE_NAME",
    "NULL_TERMINATED_BYTES_DATA_TYPE_NAME",
    "NUMPY_DATETIME64_DATA_TYPE_NAME",
    "NUMPY_TIMEDELTA64_DATA_TYPE_NAME",
    "NUMPY_TIME_UNIT",
    "STRING_DATA_TYPE_NAME",
    "STRUCT_DATA_TYPE_NAME",
    "UINT8_DATA_TYPE_NAME",
    "UINT16_DATA_TYPE_NAME",
    "UINT32_DATA_TYPE_NAME",
    "UINT64_DATA_TYPE_NAME",
    "VARIABLE_LENGTH_BYTES_DATA_TYPE_NAME",
    "ZARR_PYTHON_RAW_BYTES_DATA_TYPE_NAME",
    "BoolDataTypeName",
    "BoolFillValue",
    "ByteValue",
    "BytesDataTypeName",
    "BytesFillValue",
    "Complex64Component",
    "Complex64DataTypeName",
    "Complex64FillValue",
    "Complex128Component",
    "Complex128DataTypeName",
    "Complex128FillValue",
    "FixedLengthUtf32",
    "FixedLengthUtf32Configuration",
    "FixedLengthUtf32DataTypeName",
    "FixedLengthUtf32FillValue",
    "Float16DataTypeName",
    "Float16FillValue",
    "Float16SpecialFillValue",
    "Float32DataTypeName",
    "Float32FillValue",
    "Float32SpecialFillValue",
    "Float64DataTypeName",
    "Float64FillValue",
    "Float64SpecialFillValue",
    "FloatSpecialFillValue",
    "HexFloat16",
    "HexFloat32",
    "HexFloat64",
    "Int8DataTypeName",
    "Int8FillValue",
    "Int16DataTypeName",
    "Int16FillValue",
    "Int32DataTypeName",
    "Int32FillValue",
    "Int64DataTypeName",
    "Int64FillValue",
    "NullTerminatedBytes",
    "NullTerminatedBytesConfiguration",
    "NullTerminatedBytesDataTypeName",
    "NullTerminatedBytesFillValue",
    "NumpyDatetime64",
    "NumpyDatetime64Configuration",
    "NumpyDatetime64DataTypeName",
    "NumpyDatetime64FillValue",
    "NumpyTimeUnit",
    "NumpyTimedelta64",
    "NumpyTimedelta64Configuration",
    "NumpyTimedelta64DataTypeName",
    "NumpyTimedelta64FillValue",
    "RawBytesConfiguration",
    "RawBytesDataTypeName",
    "RawBytesFillValue",
    "StringDataTypeName",
    "StringFillValue",
    "Struct",
    "StructConfiguration",
    "StructDataTypeName",
    "StructField",
    "StructFillValue",
    "Uint8DataTypeName",
    "Uint8FillValue",
    "Uint16DataTypeName",
    "Uint16FillValue",
    "Uint32DataTypeName",
    "Uint32FillValue",
    "Uint64DataTypeName",
    "Uint64FillValue",
    "VariableLengthBytesDataTypeName",
    "VariableLengthBytesFillValue",
    "ZarrPythonRawBytes",
    "ZarrPythonRawBytesConfiguration",
    "ZarrPythonRawBytesDataTypeName",
    "ZarrPythonRawBytesFillValue",
]
