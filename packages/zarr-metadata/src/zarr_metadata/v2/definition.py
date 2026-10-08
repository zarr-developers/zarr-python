"""Zarr v2 fields read against their definitions: the public door.

A v2 array document has two kinds of field: its `dtype`, and the codecs
in `compressor` and `filters`. Zarr v2 has no extension registry, but it
has a scope all the same: the data types zarr-python 2.x writes, one
definition per NumPy family, and the codecs numcodecs 0.16 configures,
one per id this package models. `CORE_V2` is that scope. Read one field
in it with `resolve_dtype_v2` or `resolve_codec_v2`, which give `Read`,
`Unclaimed` or `Refused` as `zarr_metadata.v3.definition.resolve` does
for a v3 field; the scope algebra -- `Context.of`, `extended_with`,
`joined`, `claimant` -- is the same `Context`.

A dtype reads as its family, the typestr's byte order, size and unit its
configuration: `<f4` is `float` with `{"byteorder": "<", "itemsize": 4}`,
and its simplest spelling is the typestr again. A records array reads
as `struct`, each record's type a nested field. A codec reads by its id,
its other members the parameters. A typestr of a type code the spec does
not list, or a codec id the package does not model, is `Unclaimed`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from zarr_metadata.v2._definition import (
    ZarrV2CodecDefinition,
    ZarrV2DataTypeDefinition,
    ZarrV2DataTypeField,
)
from zarr_metadata.v2.codec import V2_CODECS
from zarr_metadata.v2.data_type import V2_DATA_TYPES
from zarr_metadata.v3._definition import (
    Read,
    Refused,
    Resolved,
    Unclaimed,
    canonical_fill_value,
    fill_value_problems,
    resolve,
)
from zarr_metadata.v3._registry import Context
from zarr_metadata.v3._scope import (
    ClaimKey,
    Claims,
    Conflict,
    Disagreements,
    ScopeConflictError,
)

if TYPE_CHECKING:
    from zarr_metadata._typed_json import Loc
    from zarr_metadata.v3._definition import Problems

CORE_V2: Final = Context.of(*V2_DATA_TYPES, *V2_CODECS)
"""The data types zarr-python 2.x writes and the codecs numcodecs 0.16 configures."""


def resolve_dtype_v2(
    value: object, context: Context | None = None, loc: Loc = ()
) -> tuple[Resolved[ZarrV2DataTypeDefinition[Any]], Problems]:
    """`value`, a v2 `dtype`, read in `context`, `CORE_V2` when none is given: what the scope made of it, and every problem, each prefixed with `loc`."""
    return resolve(value, ZarrV2DataTypeDefinition, CORE_V2 if context is None else context, loc)


def resolve_codec_v2(
    value: object, context: Context | None = None, loc: Loc = ()
) -> tuple[Resolved[ZarrV2CodecDefinition[Any]], Problems]:
    """`value`, a v2 `compressor` or one of its `filters`, read in `context`, `CORE_V2` when none is given: what the scope made of it, and every problem, each prefixed with `loc`."""
    return resolve(value, ZarrV2CodecDefinition, CORE_V2 if context is None else context, loc)


__all__ = [
    "CORE_V2",
    "V2_CODECS",
    "V2_DATA_TYPES",
    "ClaimKey",
    "Claims",
    "Conflict",
    "Context",
    "Disagreements",
    "Read",
    "Refused",
    "Resolved",
    "ScopeConflictError",
    "Unclaimed",
    "ZarrV2CodecDefinition",
    "ZarrV2DataTypeDefinition",
    "ZarrV2DataTypeField",
    "canonical_fill_value",
    "fill_value_problems",
    "resolve_codec_v2",
    "resolve_dtype_v2",
]
