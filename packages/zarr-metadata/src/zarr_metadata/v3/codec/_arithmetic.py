"""The data types that the arithmetic codecs, `cast_value` and `scale_offset`, accept.

`cast_value` "is only defined for data types that model real numbers:
floating-point and integral data types"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9),
and `scale_offset` "for data types where multiplication, division,
addition, and subtraction are well-defined"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L5-L9);
Each spec then lists "the following data types defined in this
repository", and the list is the same for both. A data type defined
elsewhere may also qualify, so a data type that the list does not name
is rejected only when the spec text itself excludes it.

Neither codec rejects the numpy time types. Each of them "is compatible
with any codec that supports arrays of signed 64-bit integers"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/data-types/numpy.datetime64/README.md?plain=1#L120),
and both codecs support those, even though neither list names the time
types.
"""

from typing import Final

from zarr_metadata.v3._definition import RAW_BYTES_NAME
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
"""The floating-point data types that the lists name. None of them is integral, so `cast_value` may not wrap to them."""

NOT_NUMBERS: Final = frozenset(
    {
        BOOL_DATA_TYPE_NAME,
        RAW_BYTES_NAME,
        STRING_DATA_TYPE_NAME,
        BYTES_DATA_TYPE_NAME,
        STRUCT_DATA_TYPE_NAME,
    }
)
"""Data types whose values are not numbers (truth values, raw bits, text, byte strings, records). Neither codec accepts them."""

COMPLEX: Final = frozenset({COMPLEX64_DATA_TYPE_NAME, COMPLEX128_DATA_TYPE_NAME})
"""Complex data types. They do not model real numbers, so `cast_value` does not accept them.

Addition, subtraction, multiplication and division are well-defined on
them, but the `scale_offset` spec does not name them in its list. So the
spec leaves open whether `scale_offset` accepts them.
"""


__all__ = ["COMPLEX", "FLOATING_POINT", "NOT_NUMBERS"]
