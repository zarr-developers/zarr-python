"""The types of the v2 data types: the configuration each family of typestrs is read into, and the JSON shapes of their fill values.

This module holds types only and imports no definition. Each data type's
submodule of `zarr_metadata.v2.data_type` re-exports its names, and holds
the data type's definition and rules.
"""

from typing import Annotated, Literal, NotRequired

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._numpy import NumpyTimeScaleFactor, NumpyTimeTicks
from zarr_metadata.v2._declaration_json import ZarrV2DataTypeDeclaration

# scalar

ZarrV2ByteOrder = Literal["<", ">", "|"]
"""The byte orders that a typestr writes: little-endian, big-endian, and not relevant."""


class ZarrV2ScalarConfiguration(TypedDict, closed=True):
    """The configuration that a scalar typestr carries: its byte order and its size in bytes, such as `{"byteorder": "<", "itemsize": 4}` for `<f4`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    """The byte order that the typestr writes.

    A family refuses `|` at a size where the order of the bytes matters, and
    its canonical form writes `|` at a size where it does not, such as `<i1`
    as `|i1`.
    """
    itemsize: ReadOnly[Annotated[int, Ge(0)]]
    """The size that the typestr writes: a count of bytes, or of code points for `str` (`U`)."""


ZarrV2FloatSpecial = Literal["NaN", "Infinity", "-Infinity"]
"""The non-finite values that the v2 spec spells by name (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L178-L190)."""

ZarrV2FloatFillValue = float | int | ZarrV2FloatSpecial | None
"""A v2 float fill value: a number, a named non-finite value, or null."""

ZarrV2ComplexComponent = float | int | ZarrV2FloatSpecial
"""One component of a complex fill value: a float fill value that is not null."""

ZarrV2ComplexFillValue = tuple[ZarrV2ComplexComponent, ZarrV2ComplexComponent] | None
"""A v2 complex fill value: `[real, imag]`, each a float fill value, or null."""


# fixed_width

ZarrV2Base64FillValue = str | None
"""A fill value that the v2 spec encodes as base64, for a fixed-length byte string, void, or a structured type (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L191-L193), or null."""


# object


class ZarrV2ObjectConfiguration(TypedDict, closed=True):
    """The configuration that `|O` carries: a byte order, which the type ignores."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    """The byte order that the typestr writes. The canonical form writes `|`, as NumPy does."""


# time


class ZarrV2TimeConfiguration(TypedDict, closed=True):
    """The configuration that a time typestr carries. For example, `<M8[10ns]` is `{"byteorder": "<", "itemsize": 8, "unit": "ns", "scale_factor": 10}`."""

    byteorder: ReadOnly[ZarrV2ByteOrder]
    """The byte order that the typestr writes, `<` or `>`."""
    itemsize: ReadOnly[Annotated[int, Ge(0)]]
    """The size in bytes that the typestr writes, which must be 8."""
    unit: NotRequired[ReadOnly[str]]
    """The NumPy time unit in brackets, such as `ns`. The canonical form writes `μs` as `us`.

    An array's dtype must have one. Only a codec's dtype parameter may leave
    it out, as in `<M8`.
    """
    scale_factor: NotRequired[ReadOnly[NumpyTimeScaleFactor]]
    """The number of units in one tick, such as the `10` of `[10ns]`, or 1 when the brackets write no number."""


ZarrV2TimeFillValue = NumpyTimeTicks | Literal["NaT"] | None
"""A v2 time fill value: a count of ticks, `"NaT"`, or null."""


# struct

ZarrV2StructRecord = (
    tuple[str, ZarrV2DataTypeDeclaration, tuple[Annotated[int, Ge(0)], ...]]
    | tuple[str, ZarrV2DataTypeDeclaration]
)
"""A field record: `[name, dtype]` or `[name, dtype, shape]`. The dtype is a nested declaration (https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v2/v2.0.rst#L152-L174)."""


class ZarrV2StructConfiguration(TypedDict, closed=True):
    """The records of a structured dtype. The document writes the records as the dtype itself."""

    fields: ReadOnly[tuple[ZarrV2StructRecord, ...]]
    """The field records, in order.

    Their names must be distinct, except `""`, which NumPy gives the padding
    of an aligned struct. There may be no records.
    """


__all__ = [
    "ZarrV2Base64FillValue",
    "ZarrV2ByteOrder",
    "ZarrV2ComplexComponent",
    "ZarrV2ComplexFillValue",
    "ZarrV2FloatFillValue",
    "ZarrV2FloatSpecial",
    "ZarrV2ObjectConfiguration",
    "ZarrV2ScalarConfiguration",
    "ZarrV2StructConfiguration",
    "ZarrV2StructRecord",
    "ZarrV2TimeConfiguration",
    "ZarrV2TimeFillValue",
]
