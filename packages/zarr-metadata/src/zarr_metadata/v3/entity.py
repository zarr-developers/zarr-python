"""The extension layer: what an entity is, and what is in scope.

Every Zarr v3 extension point -- codecs, data types, chunk grids, chunk
key encodings, storage transformers -- is modelled as a class that
answers for itself. This module is the public door to that layer, for two
kinds of caller, and everything either needs is exported from it.

**Reading metadata.** `ArrayDocumentV3.from_json` is the fail-fast front
door: one call, and either every extension point is read or a single
`MetadataValidationError` carries every reason it is not, in
`.problems`. The entities it yields know things the document does not
spell out -- what a data type's scalars are, which position a codec
occupies, what a grid divides an array into. A name the scope does not
model is not a failure: it arrives as an `Opaque` marked `out_of_scope`,
for the reader to resolve elsewhere.

    from zarr_metadata.v3.entity import ArrayDocumentV3, CodecEntity

    array = ArrayDocumentV3.from_json(json.loads(raw))   # or raises
    array.parts.grid.rank
    for codec in array.codecs:
        if isinstance(codec, CodecEntity):
            codec.kind                  # 'array_bytes'
        else:
            codec.json, codec.reason    # 'out_of_scope': resolve it yourself

**What comes back.** Problems, not exceptions, wherever a document is
being judged rather than demanded. `zarr_metadata.rules.validate_array_metadata_v3(document, context=...)`
returns a tuple of `ValidationProblem(loc, message, kind)`, `kind` one of
`ProblemKind`, each `loc` indexing into the document:
`("codecs", 1, "configuration", "level")`. `SCOPE.coerce("codecs", entry)`
reads one metadata field and returns `(entity, problems)` where `entity`
is the entity or an `Opaque` -- never `None` -- with `loc` relative to the
entry: `("configuration", "level")`. An entity's own
`coerce(value, context)` returns `(entity or None, problems)`; that is
`Coerced`. Constructing an entity by hand raises `MetadataValidationError`
with `loc` relative to the configuration: `("level",)`.

**Writing an extension.** Subclass `CodecEntity`, `DataTypeEntity`,
`ChunkGridEntity` or `MetadataEntity`; declare the configuration as
dataclass fields; put every rule finer than a type in `__post_init__`;
add the class to a scope. Complete, and runnable as written:

    from dataclasses import dataclass
    from typing import ClassVar

    from zarr_metadata.rules import validate_array_metadata_v3
    from zarr_metadata.v3.entity import (
        CORE_AND_EXTENSIONS,
        UNSET,
        CodecEntity,
        CodecKind,
        MetadataValidationError,
        problem,
    )

    @dataclass(frozen=True)  # load-bearing: `coerce` builds the entity with cls(**members)
    class AcmeLz4Codec(CodecEntity):
        acceleration: int | UNSET = UNSET  # optional: defaults to UNSET, never to a value

        identifier: ClassVar[str] = "acme.lz4"
        kind: ClassVar[CodecKind] = "bytes_bytes"

        def __post_init__(self) -> None:
            if self.acceleration is not UNSET and not 1 <= self.acceleration <= 65537:
                raise MetadataValidationError(
                    problem(
                        ("acceleration",),
                        f"expected an integer in [1, 65537], got {self.acceleration}",
                        "invalid_value",
                    )
                )

    SCOPE = CORE_AND_EXTENSIONS.extended_with(codecs={AcmeLz4Codec.identifier: AcmeLz4Codec})
    validate_array_metadata_v3(document, context=SCOPE)

The fields are the only place the shape is written. Which members exist,
which may be left out (the type admits `UNSET`), and how each one is
type-checked are all read off the annotations, and the shapes are the
ones JSON takes: `int`, `float` (any JSON number), `bool`, `str`,
`JSONValue`, a `Literal` of names, `tuple[T, ...]` or `tuple[T1, T2]`, a
TypedDict or dataclass record, `Mapping[str, V]`, a `NewType`, and a
nested entity -- always with `Opaque`, `inner: CodecEntity | Opaque`,
because that is what the field holds when the inner name is out of
scope. Anything else is refused at class creation. A required member
has no default; an optional one is `| UNSET = UNSET`, so absence stays
distinct from a JSON `null`, and a member that means something when
absent is read that way where it is used, not defaulted.

Everything finer than a type -- a bound, a rule about one member, members
read together -- is `__post_init__`, in plain code. It collects every
problem it finds and raises once; `coerce` catches the same error and
reports the problems in the document instead. `problem(loc, message,
kind)` returns a *one-element tuple*, so several are collected with
`found.extend(problem(...))` and raised as `MetadataValidationError(found)`;
pass `kind="invalid_value"` for a value rule, since the default names a
type mismatch. `__post_init__` runs only on an entity whose members all
read: a member of the wrong type is reported and the entity is not built.

**What an entity answers for itself**, beyond its fields. `to_json`,
`canonical` and `coerce` are written once in the base; an entity whose
own members have two spellings that mean the same overrides
`simplified`, which `canonical` calls (overriding `canonical` itself is
refused). Then, by kind:

- Every entity: `identifier`, the name it is registered under. A family
  -- one class for every `acme.fixedN` -- overrides `accepts(name)` and
  keeps the name in a field marked `Annotated[str, FROM_NAME]`, which
  `coerce` fills from the envelope.
- A codec: `kind`. An `array_array` codec must define
  `transition(incoming: ArrayParts) -> ArrayParts | None` -- return
  `incoming` if it leaves the array's shape, grid and data type alone,
  or the parts it hands the next codec -- and may define
  `incoming_problems(incoming)` for what it cannot take. `variable_size`
  says its output length is not fixed. A `bytes_bytes` or `array_bytes`
  codec defines neither.
- A data type: `scalar_storage`, one of `StorageClass` (the `bytes`
  codec asks it whether an endianness is needed), and
  `fill_value_problems(value, loc)`, which judges a document's
  `fill_value`; left undefined, every fill value is accepted. The
  families `IntegerDataType`, `FloatDataType`, `ComplexDataType` and
  `NumpyTimeDataType` carry those for the types they cover; a family of
  your own is a subclass declared with `base=True`, which owes nothing
  itself and passes its class variables down.
- A chunk grid: `grid(array_shape)` and `shape_problems`; see
  `ChunkGridEntity`.

The defaults fail closed for the package's own sake, so the ones an
author would otherwise miss -- an `array_array` codec without a
`transition`, a data type with a `scalar_storage` outside the listed
values, a nested field without `Opaque`, a class without `@dataclass`
(caught at registration, the first place that can see it) -- are refused
with a message that says what to write.

**Naming the JSON type.** `CodecEntity[AcmeLz4Metadata]` types `to_json`
as your own TypedDict rather than as any metadata field. The shape is
`{name: Literal["acme.lz4"], configuration: AcmeLz4Configuration,
must_understand: NotRequired[bool]}`, in a union with the name literal
only if no member is required; class creation holds it to the entity
key by key (names it accepts, the members as configuration keys with
the members' requiredness), and the tests hold the value types.

Two complete extensions written against this module alone, as tests:
`tests/v3/test_acme_affine.py` (an `array_array` codec with a number, an
optional member and a nested data type) and
`tests/v3/test_acme_decimal.py` (a configured data type with a
fill-value rule).

A name in no scope is not rejected -- that is what extension openness
means -- so registering yours is how you get it judged rather than waved
through. `CORE` is what the specification defines; `CORE_AND_EXTENSIONS`
adds the `zarr-extensions` registry; `extended_with` adds yours.

One known friction, under mypy only. An entity's `to_json` returns its
own object TypedDict, and mypy does not accept that where a
`ZarrV3MetadataFieldJSON` is wanted: it reads every TypedDict as
`Mapping[str, object]`, never as the `Mapping[str, JSONValue]` the
envelope declares (python/mypy#8994, python/mypy#18439 -- mypy lacks
PEP 728, which every TypedDict here relies on). The conversion is sound
and the annotation stays; a consumer under mypy casts at the one place
it puts an entity's JSON into a document. Pyright accepts it.
"""

from __future__ import annotations

from zarr_metadata._common import JSONValue
from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import (
    MetadataValidationError,
    ProblemKind,
    ValidationProblem,
)
from zarr_metadata.v3._chain import chain_problems
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._document import ArrayDocumentV3
from zarr_metadata.v3._entity import (
    CHUNK_GRID,
    CHUNK_KEY_ENCODING,
    CODECS,
    DATA_TYPE,
    FROM_NAME,
    STORAGE_TRANSFORMERS,
    ChunkGridEntity,
    CodecEntity,
    CodecKind,
    Coerced,
    DataTypeEntity,
    ExtensionPointField,
    Loc,
    MetadataEntity,
    Opaque,
    StorageClass,
    is_integer,
    named_configuration,
    problem,
    within,
)
from zarr_metadata.v3._parts import ArrayParts, ChunkGrid, Extents
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS, Context, EntityTables
from zarr_metadata.v3.data_type._families import (
    ComplexDataType,
    FloatDataType,
    IntegerDataType,
    NumpyTimeDataType,
)

__all__ = [
    "CHUNK_GRID",
    "CHUNK_KEY_ENCODING",
    "CODECS",
    "CORE",
    "CORE_AND_EXTENSIONS",
    "DATA_TYPE",
    "FROM_NAME",
    "STORAGE_TRANSFORMERS",
    "UNSET",
    "ArrayDocumentV3",
    "ArrayParts",
    "ChunkGrid",
    "ChunkGridEntity",
    "CodecEntity",
    "CodecKind",
    "Coerced",
    "ComplexDataType",
    "Context",
    "DataTypeEntity",
    "EntityTables",
    "ExtensionPointField",
    "Extents",
    "FloatDataType",
    "IntegerDataType",
    "JSONValue",
    "Loc",
    "MetadataEntity",
    "MetadataValidationError",
    "NumpyTimeDataType",
    "Opaque",
    "ProblemKind",
    "StorageClass",
    "ValidationProblem",
    "ZarrV3MetadataFieldJSON",
    "chain_problems",
    "is_integer",
    "named_configuration",
    "problem",
    "within",
]
