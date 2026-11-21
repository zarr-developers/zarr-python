"""The number classes that the arithmetic codecs, `cast_value` and `scale_offset`, refuse.

`cast_value` "is only defined for data types that model real numbers:
floating-point and integral data types"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9),
and `scale_offset` "for data types where multiplication, division,
addition, and subtraction are well-defined"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L5-L9).
Each spec then lists "the following data types defined in this
repository" that it supports, and the list is the same for both: the
integer, unsigned integer and floating-point types. That repository also
defines the complex types and the numpy time types, and neither list
names them. So both codecs accept the same data types.

Each data type says what kind of numbers its values are with its
`number_class`, and both codecs refuse the classes their lists leave out.
A data type that does not declare its number class is accepted, since
nothing is known of its values.
"""

from typing import Final

from zarr_metadata.v3._definition import NumberClass

NO_REAL_NUMBERS: Final[frozenset[NumberClass]] = frozenset({"non_numeric", "complex"})
"""The number classes that neither codec takes: values that are not plain numbers, and complex numbers.

Addition, subtraction, multiplication and division are well-defined on
complex numbers, but the `scale_offset` spec names none of the complex
types its repository defines among those it supports.
"""


__all__ = ["NO_REAL_NUMBERS"]
