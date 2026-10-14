"""The data types the arithmetic codecs, `cast_value` and `scale_offset`, take.

`cast_value` "is only defined for data types that model real numbers:
floating-point and integral data types"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9),
and `scale_offset` "for data types where multiplication, division,
addition, and subtraction are well-defined"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L5-L9);
each then lists "the following data types defined in this repository",
the same list for both. A data type defined elsewhere may qualify, so one
the list does not name is refused only where the spec's own words refuse
it.

The numpy time types are refused by neither: each "is compatible with any
codec that supports arrays of signed 64-bit integers"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120),
which both codecs do, while neither lists them.
"""

from typing import Any, Final

from zarr_metadata.v3._definition import RAW_BYTES_NAME, Resolved
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE_NAME

FLOATING_POINT: Final = frozenset(
    {
        "float4_e2m1fn",
        "float6_e2m3fn",
        "float6_e3m2fn",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "bfloat16",
        "float16",
        "float32",
        "float64",
    }
)
"""The floating-point data types the lists name: no integral type, which `cast_value` may wrap to."""

NOT_NUMBERS: Final = frozenset(
    {
        BOOL_DATA_TYPE_NAME,
        RAW_BYTES_NAME,
        STRING_DATA_TYPE_NAME,
        BYTES_DATA_TYPE_NAME,
        STRUCT_DATA_TYPE_NAME,
    }
)
"""Data types whose values are no numbers -- truth values, raw bits, text, byte strings, records -- which neither codec takes."""

COMPLEX: Final = frozenset({COMPLEX64_DATA_TYPE_NAME, COMPLEX128_DATA_TYPE_NAME})
"""Complex numbers, which model no real number, so `cast_value` does not take them.

Addition, subtraction, multiplication and division are well-defined on
them, so whether `scale_offset` does, its spec leaves open: the list
does not name them.
"""


def read_name(data_type: Resolved[Any] | None) -> str | None:
    """The name the definition that read `data_type` is filed under; None when no definition read it."""
    if data_type is None or data_type.resolution != "read" or data_type.definition is None:
        return None
    return data_type.definition.name


__all__ = ["COMPLEX", "FLOATING_POINT", "NOT_NUMBERS", "read_name"]
