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

    from zarr_metadata.v3.entity import ArrayBytesCodec, ArrayDocumentV3, CodecEntity

    array = ArrayDocumentV3.from_json(json.loads(raw))   # or raises
    array.parts.grid.rank
    for codec in array.codecs:
        if isinstance(codec, CodecEntity):
            isinstance(codec, ArrayBytesCodec)   # its pipeline position is its base class
        else:
            codec.json, codec.reason    # 'out_of_scope': resolve it yourself

**What comes back.** Problems, not exceptions, wherever a document is
being judged rather than demanded. `zarr_metadata.rules.validate_array_metadata_v3(document, context=...)`
returns a tuple of `ValidationProblem(loc, message, kind)`, `kind` one of
`ProblemKind`, each `loc` indexing into the document:
`("codecs", 1, "configuration", "level")`. `SCOPE.coerce(CodecEntity, entry)`
reads one metadata field as an entity of that kind and returns
`(entity, problems)` where `entity` is the entity or an `Opaque` -- never
`None` -- with `loc` relative to the entry: `("configuration", "level")`. An entity's own
`coerce(value, context)` returns `(entity or None, problems)`; that is
`Coerced`. Constructing an entity by hand raises `MetadataValidationError`
with `loc` relative to the configuration: `("level",)`.

**Writing an extension.** Subclass the kind of thing it is -- a codec's
kind (`ArrayArrayCodec`, `ArrayBytesCodec`, `BytesBytesCodec`),
`DataTypeEntity`, `ChunkGridEntity`, `ChunkKeyEncodingEntity` or
`StorageTransformerEntity`; declare the configuration as dataclass
fields; write every rule finer than a type as a function of the
instance that yields problems, and bind it as `problems`; add the class
to a scope. Complete; runnable given a `document`:

    from collections.abc import Iterator
    from dataclasses import dataclass
    from typing import ClassVar, Literal, NotRequired

    from typing_extensions import TypedDict

    from zarr_metadata.rules import validate_array_metadata_v3
    from zarr_metadata.v3.entity import (
        CORE_AND_EXTENSIONS,
        UNSET,
        BytesBytesCodec,
        ValidationProblem,
    )

    class AcmeLz4Configuration(TypedDict, closed=True):
        acceleration: NotRequired[int]

    class AcmeLz4Object(TypedDict, closed=True):
        name: Literal["acme.lz4"]
        configuration: AcmeLz4Configuration

    def acme_lz4_problems(codec: "AcmeLz4Codec", /) -> Iterator[ValidationProblem]:
        if codec.acceleration is not UNSET and not 1 <= codec.acceleration <= 65537:
            yield ValidationProblem(
                ("acceleration",),
                f"expected an integer in [1, 65537], got {codec.acceleration}",
                "invalid_value",
            )

    @dataclass(frozen=True)  # the fields are the schema; frozen, so an entity is a value
    class AcmeLz4Codec(BytesBytesCodec):
        acceleration: int | UNSET = UNSET  # optional: absent reads as UNSET

        identifier: ClassVar[str] = "acme.lz4"
        variable_size: ClassVar[bool] = True  # a compressor: its output length is not fixed
        problems = acme_lz4_problems

        def to_json(self) -> AcmeLz4Object | Literal["acme.lz4"]:
            if self.acceleration is UNSET:
                return "acme.lz4"
            return {"name": "acme.lz4", "configuration": {"acceleration": self.acceleration}}

    SCOPE = CORE_AND_EXTENSIONS.extended_with(AcmeLz4Codec)
    validate_array_metadata_v3(document, context=SCOPE)

The fields are the only place the shape is written. Which members exist,
which may be left out (the type admits `UNSET`), and how each one is
type-checked are all read off the annotations, and the shapes are the
ones JSON takes: `int`, `float` (any JSON number), `bool`, `str`,
`JSONValue`, a `Literal` of names, `tuple[T, ...]` or `tuple[T1, T2]`, a
TypedDict or dataclass record, `Mapping[str, V]`, a `NewType`, and a
nested entity, always as `inner: CodecEntity | Opaque`, because that is
what the field holds when the inner name is out of scope -- at any
depth, as an array's element or a record's field, and read in the scope
the containing entity is read in. Anything else is refused at class
creation. A required member
has no default; an optional one is `| UNSET = UNSET`, so absence stays
distinct from a JSON `null`, and a member that means something when
absent is read that way where it is used, not defaulted.

Everything finer than a type -- a bound, a rule about one member, members
read together -- is a function of the instance that yields
`ValidationProblem(loc, message, kind)` as it finds each, in plain
code, bound on the class as `problems`. Locations are relative to the
configuration, and `kind` is `"invalid_value"` for a value rule. The
constructor stops at the first problem it yields, so
`AcmeLz4Codec(acceleration=0)` raises `MetadataValidationError`;
`coerce` runs it to the end and reports every problem in the document.
It runs only on an entity whose members all read: a member of the wrong
type is reported and the entity is not built.

**What an entity answers for itself**, beyond its fields. `to_json`,
abstract: the entity as a document writes it, as a literal of its own
TypedDict, which pyright holds to that type -- the bare name when every
member is absent, the object otherwise, a contained entity through its
own `to_json`. `canonical`, the entity in its simplest equivalent form:
the entity itself by default, overridden where two spellings of its
members mean the same, and in an entity that contains entities to put
those in canonical form -- `replace(self, inner=self.inner.canonical())`.
An `Opaque` answers both as well, with the JSON it kept and with itself,
so a field typed `CodecEntity | Opaque` is written and simplified without
asking which it holds. `coerce` is written once in the base. Then, by
kind:

- Every entity: `identifier`, the name it is registered under. A family
  -- one class for every `acme.fixedN` -- overrides `accepts(name)` and
  keeps the name in a field marked `Annotated[str, FROM_NAME]`, which
  `coerce` fills from the envelope.
- A codec: its kind is its base class. An `ArrayArrayCodec` defines
  `transition(incoming: ArrayParts) -> ArrayParts | None` -- abstract:
  return `incoming` if it leaves the array's shape, grid and data type
  alone, or the parts it hands the next codec -- and any codec may define
  `incoming_problems(incoming)` for what it cannot take. Every codec
  declares `variable_size`, whether its output length depends on its
  input, which is what keeps a compressor out of a shard's index.
- A data type: `scalar_storage`, one of `StorageClass` (the `bytes`
  codec asks it whether an endianness is needed), and
  `fill_value_problems(value, loc)`, abstract: it judges a document's
  `fill_value`, and a type that accepts any says so with `return ()`.
  The families `IntegerDataType`, `FloatDataType`, `ComplexDataType` and
  `NumpyTimeDataType` carry both for the types they cover; a family of
  your own is a plain subclass that is never registered itself, and
  passes its class variables down.
- A chunk grid: `grid(array_shape)`, abstract, and `shape_problems`; see
  `ChunkGridEntity`.

Registration is the one moment an entity is refused, with a message
that says what to write: a class without `@dataclass`, a codec
subclassing `CodecEntity` instead of a kind, a field whose annotation is
not a shape JSON takes -- a nested entity without `Opaque` among them --
a `__post_init__` of the entity's own, a class variable a base
annotates and nothing sets, and what a kind leaves abstract. Everything
else an author could get wrong, pyright says in the editor: the fields,
the class variables and the kind's abstract methods are ordinary typed
Python. A scope reads what a class is off the class: its kind is its
base, its key is its `identifier`, so `extended_with` takes the classes
and nothing can be misfiled.

**Naming the JSON type.** The return annotation of `to_json` -- above,
`AcmeLz4Object | Literal["acme.lz4"]` -- is the entity's own JSON type,
narrower than the `ZarrV3MetadataFieldJSON` the base declares, and
pyright checks the literal returned against it: a key it does not
declare, a required one left out, a value of the wrong type is a static
error.

Two complete extensions written against this module alone, as tests:
`tests/v3/test_acme_affine.py` (an `array_array` codec with a number, an
optional member and a nested data type) and
`tests/v3/test_acme_decimal.py` (a configured data type with a
fill-value rule).

A name in no scope is not rejected -- that is what extension openness
means -- so registering yours is how you get it judged rather than waved
through. `CORE` is what the specification defines; `CORE_AND_EXTENSIONS`
adds the `zarr-extensions` registry; `extended_with(*classes)` adds yours.

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
    FROM_NAME,
    ArrayArrayCodec,
    ArrayBytesCodec,
    BytesBytesCodec,
    ChunkGridEntity,
    ChunkKeyEncodingEntity,
    CodecEntity,
    Coerced,
    DataTypeEntity,
    Loc,
    MetadataEntity,
    Opaque,
    StorageClass,
    StorageTransformerEntity,
    is_integer,
    named_configuration,
    problem,
    within,
)
from zarr_metadata.v3._parts import ArrayParts, ChunkGrid, Extents
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3.data_type._families import (
    ComplexDataType,
    FloatDataType,
    IntegerDataType,
    NumpyTimeDataType,
)

__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "FROM_NAME",
    "UNSET",
    "ArrayArrayCodec",
    "ArrayBytesCodec",
    "ArrayDocumentV3",
    "ArrayParts",
    "BytesBytesCodec",
    "ChunkGrid",
    "ChunkGridEntity",
    "ChunkKeyEncodingEntity",
    "CodecEntity",
    "Coerced",
    "ComplexDataType",
    "Context",
    "DataTypeEntity",
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
    "StorageTransformerEntity",
    "ValidationProblem",
    "ZarrV3MetadataFieldJSON",
    "chain_problems",
    "is_integer",
    "named_configuration",
    "problem",
    "within",
]
