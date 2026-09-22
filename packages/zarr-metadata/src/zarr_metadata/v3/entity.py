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
returns a tuple of `ValidationProblem(loc, message, kind)`, each `loc`
indexing into the document:
`("codecs", 1, "configuration", "level")`, and `kind` one of
`invalid_type`, `invalid_value`, `missing_key`, `unknown_key` and
`invalid_json`. `SCOPE.coerce(CodecEntity, entry)` reads one metadata
field as an entity of that kind and returns `(entity, problems)` where
`entity` is the entity or an `Opaque` -- never `None` -- with `loc`
relative to the entry: `("configuration", "level")`. An entity's own
`coerce(value, context)` returns `(entity or None, problems)`; that is
`Coerced`. Constructing an entity by hand raises `MetadataValidationError`
with `loc` relative to the configuration: `("level",)`.

**Writing an extension.** Subclass the kind of thing it is -- a codec's
kind (`ArrayArrayCodec`, `ArrayBytesCodec`, `BytesBytesCodec`),
`DataTypeEntity`, `ChunkGridEntity`, `ChunkKeyEncodingEntity` or
`StorageTransformerEntity`; declare its configuration as a frozen
`Configuration` of its members, with every rule finer than a type in
its `problems`, and name it in the entity's one field, `configuration`;
add the class to a scope. An entity of a bare name defaults the field
to the empty record: `configuration: Configuration =
field(default_factory=Configuration)`. Complete, and runnable as written:

    from collections.abc import Iterator
    from dataclasses import dataclass
    from typing import ClassVar

    from zarr_metadata.rules import validate_array_metadata_v3
    from zarr_metadata.v3.entity import (
        CORE_AND_EXTENSIONS,
        UNSET,
        BytesBytesCodec,
        Configuration,
            ValidationProblem,
    )

    @dataclass(frozen=True)  # the fields are the schema; frozen, so a configuration is a value
    class AcmeLz4Options(Configuration):
        acceleration: int | UNSET = UNSET  # optional: absent reads as UNSET

        def problems(self) -> Iterator[ValidationProblem]:
            if self.acceleration is not UNSET and not 1 <= self.acceleration <= 65537:
                yield ValidationProblem(
                    ("acceleration",),
                    f"expected an integer in [1, 65537], got {self.acceleration}",
                    "invalid_value",
                )

    @dataclass(frozen=True)
    class AcmeLz4Codec(BytesBytesCodec):
        configuration: AcmeLz4Options   # the shape of the metadata: a name, and a configuration

        identifier: ClassVar[str] = "acme.lz4"
        variable_size: ClassVar[bool] = True  # a compressor: its output length is not fixed

    SCOPE = CORE_AND_EXTENSIONS.extended_with(AcmeLz4Codec)
    document = {
        "zarr_format": 3, "node_type": "array", "shape": [8], "data_type": "uint8",
        "fill_value": 0, "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8]}},
        "chunk_key_encoding": "default",
        "codecs": ["bytes", {"name": "acme.lz4", "configuration": {"acceleration": 3}}],
    }
    assert validate_array_metadata_v3(document, context=SCOPE) == ()

An entity has the shape of its metadata: a name, which is the class,
and a configuration, which is a record dataclass named in the one field
`configuration`; an entity of a bare name defaults it to the empty
`Configuration`, since the spec makes an absent configuration and an
empty one the same. The record's fields are the
one place the entity's members are declared; the public `*Configuration`
TypedDict beside it declares the JSON, and a test holds the two to the
same keys. Which members exist, which may be left out (the type admits
`UNSET`), how each one is type-checked, and how each is written back
are all read off the annotations, and the shapes are the ones JSON
takes: `int`, `float` (any JSON number), `bool`, `str`,
`JSONValue`, a `Literal` of names, `tuple[T, ...]` or `tuple[T1, T2]`, a
TypedDict or dataclass record, `Mapping[str, V]`, a `NewType`, and a
nested entity, always as `inner: CodecEntity | Opaque`, because that is
what the field holds when the inner name is out of scope -- at any
depth, as an array's element or a record's field, and read in the scope
the containing entity is read in. Anything else is refused at
registration. A required member
has no default; an optional one is `| UNSET = UNSET`, so absence stays
distinct from a JSON `null`, and a member that means something when
absent is read that way where it is used, not defaulted. Only `| UNSET`
makes a member optional to a document: a plain default serves hand
construction, and a document must still write the member. A member is
read as `codec.configuration.acceleration`, the shape the metadata has;
nothing lifts it to the entity. `with_configuration(**changes)`
is the entity with members of its configuration replaced, checked as
any construction is.

Everything finer than a type -- a bound, a rule about one member, members
read together -- is the record's `problems`, which yields
`ValidationProblem(loc, message, kind)` as it finds each, in plain
code. Locations are relative to the configuration, and `kind` is
`"invalid_value"` for a value rule. The entity's constructor stops at
the first problem it yields, so `AcmeLz4Codec(AcmeLz4Options(acceleration=0))`
raises `MetadataValidationError`; `coerce` runs it to the end and
reports every problem in the document; a reader with a record asks
`options.problems()` directly and stops or collects. It runs only on a
configuration whose members all read: a member of the wrong type is
reported and the entity is not built. A family's rule about its name --
`r<N>` a multiple of 8 -- is the entity's `name_problems(name)`, a
classmethod, located on the entity.

**What an entity answers for itself**, beyond its configuration. `to_json` is
written once in the base, from the record: the bare name when every
member is absent, the object otherwise, a contained entity through its
own `to_json`; an entity whose JSON is not its fields overrides it, and
none in the package does. `ArrayDocumentV3.to_json` puts each envelope
back as the document spelled it, so a document read and written comes
out as it went in. `canonical`, the entity in its simplest equivalent form:
the entity itself by default, overridden where two spellings of its
members mean the same, and in an entity that contains entities to put
those in canonical form -- `self.with_configuration(inner=self.inner.canonical())`.
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
  alone, the parts it hands the next codec, or None when the metadata
  cannot say -- and any codec may define `incoming_problems(incoming:
  ArrayParts | None) -> tuple[ValidationProblem, ...]` for what it
  cannot take, where None is an array the chain lost track of and the
  answer to it is nothing. Every codec declares `variable_size`, whether
  its output length depends on its input, which is what keeps a
  compressor out of a shard's index.
- A data type: `scalar_storage`, one of `StorageClass` --
  `"single_byte"`, `"multi_byte"`, `"variable_length"` -- which the
  method `storage_class()` answers (the `bytes` codec asks it whether an
  endianness is needed), and `fill_value_problems(value, loc) ->
  tuple[ValidationProblem, ...]`, abstract: it judges a document's
  `fill_value`, and a type that accepts any says so with `return ()`.
  These composition hooks return tuples; a record's `problems` yields.
  The families `IntegerDataType`, `FloatDataType`, `ComplexDataType` and
  `NumpyTimeDataType` carry both for the types they cover; a family of
  your own is a plain subclass that is never registered itself, and
  passes its class variables down.
- A chunk grid: `grid(array_shape)`, abstract, and `shape_problems`; see
  `ChunkGridEntity`.
- A family, one class for many names: `identifier` is an invented key no
  document writes, `accepts(name)` says which names are its own, a field
  marked `Annotated[str, FROM_NAME]` keeps the name as written, and
  `name_problems(name)`, a classmethod, holds any rule about it.

Registration is the one moment an entity is refused, with a message
that says what to write: a class without `@dataclass`, a codec
subclassing `CodecEntity` instead of a kind, a field other than
`configuration` and a carried name, a configuration that is not a
`Configuration` record, a
member whose annotation is not a shape JSON takes
-- a nested entity without `Opaque` among them --
a `__post_init__` of the entity's own, a class variable a base
annotates and nothing sets, and what a kind leaves abstract. What is
left, pyright says in the editor: a member of the wrong type, a
`canonical` returning something else, a hook with the wrong signature.
A scope reads what a class is off the class: its kind is its base, its
key is its `identifier`, so `extended_with` takes the classes and
nothing can be misfiled -- and a class whose `identifier` the scope
already has takes the name over, so registering your own `"gzip"`
replaces the package's reading of it. `Context.of(*classes)` is a scope
of exactly those.

Two complete extensions written against this module alone, as tests in
the repository:
`tests/v3/test_acme_affine.py` (an `array_array` codec with a number, an
optional member and a nested data type) and
`tests/v3/test_acme_decimal.py` (a configured data type with a
fill-value rule).

A name in no scope is not rejected -- that is what extension openness
means -- so registering yours is how you get it judged rather than waved
through. `CORE` is what the specification defines; `CORE_AND_EXTENSIONS`
adds the `zarr-extensions` registry; `extended_with(*classes)` adds yours.
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
    Configuration,
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
    "Configuration",
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
