"""The number classes that the arithmetic codecs, `cast_value` and `scale_offset`, refuse.

`cast_value` "is only defined for data types that model real numbers:
floating-point and integral data types"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/cast_value/README.md?plain=1#L5-L9),
and `scale_offset` "for data types where multiplication, division,
addition, and subtraction are well-defined"
(https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/scale_offset/README.md?plain=1#L5-L9).
Each data type says what kind of numbers its values are with its
`number_class`, and each codec refuses the classes its spec excludes. A
data type that does not declare its number class is accepted, since
nothing is known of its values.
"""

from typing import Final

from zarr_metadata.v3._definition import NumberClass

NOT_NUMBERS: Final[frozenset[NumberClass]] = frozenset({"non_numeric"})
"""The number class of values that are not numbers (truth values, raw bits, text, byte strings, records). Neither codec accepts it."""

NO_REAL_NUMBERS: Final[frozenset[NumberClass]] = frozenset({"non_numeric", "complex"})
"""The number classes that model no real number, which `cast_value` takes neither from nor to.

Addition, subtraction, multiplication and division are well-defined on
complex numbers, but the `scale_offset` spec does not name the complex
data types in its list. So the spec leaves open whether `scale_offset`
accepts them, and it does here.
"""


__all__ = ["NOT_NUMBERS", "NO_REAL_NUMBERS"]
