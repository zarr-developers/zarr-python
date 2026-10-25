"""Read Zarr v2 fields against their definitions. This is the public module for v2 reading.

A v2 array document has two kinds of field: its `dtype`, and the codecs
in `compressor` and `filters`. Zarr v2 has no extension registry, but it
still has a scope: the data types that zarr-python 2.x writes (one
definition per NumPy family), and the codecs that numcodecs 0.16
configures (one per id that this package models). `CORE_V2` is that
scope. To read one field in it, call `zarr_metadata.definition.resolve`
with the field's kind, as for a v3 field:
`resolve("<f4", ZarrV2DataTypeDefinition, CORE_V2)` returns an
`AcceptedField`, `UnclaimedField` or `RefusedField`, and every problem.
`ZarrV2Context` is the class of a v2 scope. It has a slot for each kind
of v2 field, `data_types` and `codecs`, and shares `update`,
`union` and `claimant` with the v3 scope. A v2 reader rejects a v3
scope with `TypeError`.

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

from typing import Final

from zarr_metadata.v2._context import ZarrV2Context, ZarrV2ContextSlots, ZarrV2Kind
from zarr_metadata.v2._definition import (
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
    ZarrV2DataTypeField,
)
from zarr_metadata.v2.codec import CODECS_V2
from zarr_metadata.v2.data_type import DATA_TYPES_V2

CORE_V2: Final = ZarrV2Context(data_types=DATA_TYPES_V2, codecs=CODECS_V2)
"""The data types that zarr-python 2.x writes and the codecs that numcodecs 0.16 configures."""


__all__ = [
    "CORE_V2",
    "ZarrV2CodecDefinition",
    "ZarrV2Context",
    "ZarrV2ContextSlots",
    "ZarrV2DataTypeDefinition",
    "ZarrV2DataTypeField",
    "ZarrV2Kind",
]
