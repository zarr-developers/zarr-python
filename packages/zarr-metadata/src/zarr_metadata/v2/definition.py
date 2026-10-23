"""Read Zarr v2 fields against their definitions. This is the public module for v2 reading.

A v2 array document has two kinds of field: its `dtype`, and the codecs
in `compressor` and `filters`. Zarr v2 has no extension registry, but it
still has a scope: the data types that zarr-python 2.x writes (one
definition per NumPy family), and the codecs that numcodecs 0.16
configures (one per id that this package models). `CORE_V2` is that
scope. To read one field in it, call `resolve_dtype_v2` or
`resolve_codec_v2`. They return `AcceptedField`, `UnclaimedField` or
`RefusedField`, as `zarr_metadata.definition.resolve` does for a v3
field. The scope operations (`Context.of`, `extended_with`, `joined`,
`claimant`) use the same `Context` class, with format 2. A v2 reader
rejects a v3 scope with `TypeError`. `ZarrV2Context` is the type of a v2
scope.

A dtype reads as its family, and the byte order, size and unit of the
typestr are its configuration. For example, `<f4` is `float` with
`{"byteorder": "<", "itemsize": 4}`, and its simplest spelling is the
typestr again. A records array reads as `struct`, and the type of each
record is a nested field. A codec is read by its id, and its other
members are its parameters. A typestr with a type code that the spec
does not list, or a codec id that the package does not model, reads as
an `UnclaimedField`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from zarr_metadata.definition._context import Context, ZarrV2Context, scoped
from zarr_metadata.definition._definition import resolve
from zarr_metadata.v2._definition import (
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
    ZarrV2DataTypeField,
)
from zarr_metadata.v2.codec import V2_CODECS
from zarr_metadata.v2.data_type import V2_DATA_TYPES

if TYPE_CHECKING:
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.definition._definition import Problems, ResolvedField

CORE_V2: Final[ZarrV2Context] = Context.of(*V2_DATA_TYPES, *V2_CODECS)
"""The data types that zarr-python 2.x writes and the codecs that numcodecs 0.16 configures."""


def resolve_dtype_v2(
    value: object, context: ZarrV2Context | None = None, loc: Loc = ()
) -> tuple[ResolvedField[ZarrV2DataTypeDefinition[Any]], Problems]:
    """Read `value`, a v2 `dtype`, in `context` (`CORE_V2` when none is given). Return what the scope made of it, and every problem, each prefixed with `loc`."""
    return resolve(value, ZarrV2DataTypeDefinition, scoped(context, CORE_V2), loc)


def resolve_codec_v2(
    value: object, context: ZarrV2Context | None = None, loc: Loc = ()
) -> tuple[ResolvedField[ZarrV2CodecDefinition[Any]], Problems]:
    """Read `value`, a v2 `compressor` or one of its `filters`, in `context` (`CORE_V2` when none is given). Return what the scope made of it, and every problem, each prefixed with `loc`."""
    return resolve(value, ZarrV2CodecDefinition, scoped(context, CORE_V2), loc)


__all__ = [
    "CORE_V2",
    "ZarrV2CodecDefinition",
    "ZarrV2Context",
    "ZarrV2DataTypeDefinition",
    "ZarrV2DataTypeField",
    "resolve_codec_v2",
    "resolve_dtype_v2",
]
