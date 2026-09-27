"""Metadata fields read against their definitions: the public door.

Every Zarr v3 extension point -- codecs, data types, chunk grids, chunk
key encodings, storage transformers -- is a metadata field: a name, and
a configuration whose JSON the extension defines. A **definition** says
what that JSON is and what is allowed in it. It is a value, not a class
to subclass:

- `name`, the name the metadata carries;
- `configuration`, the TypedDict the configuration's JSON is, which is
  the one declaration of it: the type checker is compiled from it, and a
  checked configuration has it as its static type. It reads as the
  typing spec defines a TypedDict -- `total`, `Required`, `NotRequired`,
  `closed` and `extra_items` mean what they mean to a type checker,
  whether or not its module postpones annotations;
- `rules`, a function over that TypedDict yielding what the spec
  disallows -- a bound, members read together -- located in the
  configuration.

What kind of metadata a definition defines is its type: `CodecDefinition`
(with the codec's `kind`, and its `size`: whether the size of what it gives
out is fixed by the size of what it is handed, `"static"`, or depends on
the values, `"dynamic"`), `DataTypeDefinition`, `ChunkGridDefinition`,
`ChunkKeyEncodingDefinition`, `StorageTransformerDefinition`.

**Reading JSON.** Three steps, each feeding the next, and each usable on
its own by a caller that holds nothing but JSON:

1. `check(value, SomeTypedDict)`, from `zarr_metadata.typed_json` and
   here too, type-checks JSON against a TypedDict and needs nothing else:
   a value of the TypedDict or None, and every problem, each located. The
   value holds what the TypedDict admits and nothing else. A member typed
   with a field alias is checked as the JSON a metadata field is.
2. `definition.judge(configuration)` is the check, each nested field's
   envelope judged -- a stray member, a `must_understand` of `false` --
   and then the rules, for one configuration:
   `GZIP_CODEC.judge({"level": 12})`.
3. `resolve(field, CodecDefinition, CORE_AND_EXTENSIONS)` reads a whole
   field in a scope: its envelope judged, its name related to a
   definition, its configuration judged, and each nested field read the
   same way. It returns `Resolved` -- the field's JSON, its
   `resolution`, the definition and the checked configuration -- and
   every problem. A name nothing in scope claims is `out_of_scope`:
   left unjudged, which is what keeps the format open. A field is read
   as one of the five kinds, with or without type arguments;
   `resolve(field, Definition, scope)` is a `TypeError`, since nothing
   is filed under it. `configuration_of(resolved, GZIP_CODEC)` is the
   configuration typed as that definition's TypedDict, when it read it.

    from zarr_metadata.v3.codec.gzip import GZIP_CODEC
    from zarr_metadata.v3.definition import (
        CORE_AND_EXTENSIONS,
        CodecDefinition,
        configuration_of,
        resolve,
    )

    resolved, problems = resolve({"name": "gzip", "configuration": {"level": 12}},
                                 CodecDefinition, CORE_AND_EXTENSIONS)
    resolved.resolution     # 'invalid'
    problems[0].loc         # ('configuration', 'level')

    resolved, problems = resolve({"name": "gzip", "configuration": {"level": 5}},
                                 CodecDefinition, CORE_AND_EXTENSIONS)
    configuration_of(resolved, GZIP_CODEC)   # {'level': 5}, a GzipCodecConfiguration

Problems are values, not exceptions: `ValidationProblem(loc, message,
kind)`, with `kind` one of `invalid_type`, `invalid_value`,
`missing_key`, `unknown_key` and `invalid_json`. A field with an
unknown key is still read -- the key reported, the configuration judged
without it -- so a consumer that tolerates one filters by kind and uses
what was read; the field is valid only when there is no problem at all.

**Writing an extension.** A TypedDict, a function for its rules, and a
definition; then a scope that holds it. The TypedDict is a
`typing_extensions.TypedDict`: `closed` and `extra_items` are PEP 728's,
which `typing.TypedDict` does not take on the versions this package
supports. The rules are handed the configuration and the fields it holds
as the scope read them: a field that is read keeps what it read inside it
as `Resolved.nested`, a `Nested` mapping by where each sits, so a
struct's rules reach its field types. `judge`, which reads in no scope,
hands them none.

    from collections.abc import Iterator

    from typing_extensions import TypedDict

    from zarr_metadata.v3.definition import (
        CORE_AND_EXTENSIONS,
        CodecDefinition,
        Nested,
        ValidationProblem,
    )


    class AcmeLz4Configuration(TypedDict, closed=True):
        acceleration: int


    def acme_lz4_rules(
        configuration: AcmeLz4Configuration, nested: Nested
    ) -> Iterator[ValidationProblem]:
        if configuration["acceleration"] < 1:
            yield ValidationProblem(("acceleration",), "expected an integer >= 1", "invalid_value")


    ACME_LZ4 = CodecDefinition(
        name="acme.lz4",
        configuration=AcmeLz4Configuration,
        kind="bytes_bytes",
        size="dynamic",
        rules=acme_lz4_rules,
    )
    SCOPE = CORE_AND_EXTENSIONS.extended_with(ACME_LZ4)

A scope reads whole documents as well as fields:
`validate_array_metadata_v3(document, context=SCOPE)`, from
`zarr_metadata.model`, reads each extension point of a v3 array document
through the definitions in `SCOPE`, and so do the model's `from_json` and
`from_key_value`: a fill value is judged against the data type it names,
by that data type's definition, and the chunk grid against the shape, by
the grid's definition. No rule here reads a codec against the array it
is handed.

The TypedDict says what a key it does not declare is: with
`closed=True`, a problem, as above; with `extra_items=`, a key holding
that type; with `closed=False`, anything at all. One that says none of
these is open by default, and would take a misspelled key without a
word, so a definition refuses it. Its members are the shapes JSON takes:
`int`, `float`, `bool`, `str`, `None`, `JSONValue`, a `Literal`,
`tuple[T, ...]` and `tuple[T1, T2]`, a union, a TypedDict,
`Mapping[str, V]`, a `NewType` and a type alias.

A member holding another metadata field is annotated with the field alias
of its kind -- a shard's `codecs: tuple[CodecField, ...]` -- and read in
the scope its field is read in. A member that takes codecs of static size
only is annotated `StaticCodecField` -- a shard's `index_codecs`, since a
reader finds the index by a size it knows before reading it -- and a codec
of dynamic size there is a problem at its place, where the field is read
in a scope; a name nothing claims is left unjudged, its size unknown with
the rest of it. `ZarrV3MetadataFieldJSON` is the same
JSON, but checks as JSON and nothing more, so a definition refuses a
member typed with it. An extension with nothing to configure takes
`EmptyConfiguration`, and is written as its bare name.

A data type also says what its fill value is: `fill_value`, the JSON
shape of one as an annotation the checker reads -- `Int8FillValue` -- and
`fill_value_rules`, a function yielding what the spec disallows in a fill
value of that shape: an integer out of range, a hex string of another
width. The rules are handed the configuration, the fields it holds as the
scope read them, and the typed fill value, so a struct judges each
field's fill value by that field's own type.
`fill_value_problems(data_type, value)` judges a fill value against a data
type field the scope read; one nothing in scope claims leaves it unjudged.
A data type that says nothing of its fill value takes any JSON.

A chunk grid says which arrays it fits: `shape_rules`, a function
yielding what the spec disallows in a grid of its configuration over an
array of a given shape -- a dimension with no chunk length, chunks that
fall short of one -- located in the configuration. A grid that says
nothing of the shape fits every one.

Raw bits are the one data type whose name carries its configuration: a
document writes `r` and the size in bits, and `r16` reads as `r*`, as the
specification's table writes raw bits, with `{"bits": 16}`. A reader that
reads raw bits its own way defines `r*`; a data type named `r16` is
refused, since that name reads as `r*`. `r*` itself is notation, and a
document that writes it names nothing in any scope.

**The simplest spelling.** `canonicalize(field, kind, scope)` gives a
field without problems in its simplest equivalent spelling: each nested
field in its own simplest spelling, then the definition's `canonical` --
blosc drops a `typesize` that `noshuffle` ignores, a rectilinear grid
run-length encodes its chunk shapes -- and the envelope in the fewest
words; raw bits write their size back into the name, in decimal, so
`r008` is `r8`. A field with any problem, an unknown key included, has
none: a simpler spelling of it would erase what its author wrote. What
`canonical` gives is judged again: one that does not hold is a
`ValueError`, a fault in the definition.

A definition checks itself when it is built, and each of these is a
`TypeError` saying what is wrong: a `configuration` that is not a
TypedDict, says nothing of the keys it does not declare, or has a member
no checker reads, named down to the TypedDict that holds it; a `name`
that is not a string; a member declared as a function -- `rules`,
`canonical`, `fill_value_rules`, `shape_rules` -- that is not one; a data
type's `fill_value` no checker reads; a codec `kind` that is not one of
the three, or a `size` that is not `"static"` or `"dynamic"`; a data type
named as raw bits of one size are written. A scope refuses a definition
of no kind. Nothing happens at class creation.
"""

from zarr_metadata._common import JSONValue
from zarr_metadata._json import MetadataValidationError, ProblemKind, ValidationProblem
from zarr_metadata._typed_json import Loc, check
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._definition import (
    ChunkGridDefinition,
    ChunkGridField,
    ChunkKeyEncodingDefinition,
    ChunkKeyEncodingField,
    CodecDefinition,
    CodecField,
    CodecKind,
    CodecSize,
    DataTypeDefinition,
    DataTypeField,
    Definition,
    EmptyConfiguration,
    Nested,
    Resolution,
    Resolved,
    StaticCodecField,
    StorageTransformerDefinition,
    StorageTransformerField,
    Unread,
    canonicalize,
    configuration_of,
    fill_value_problems,
    resolve,
)
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS, Context

__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "ChunkGridDefinition",
    "ChunkGridField",
    "ChunkKeyEncodingDefinition",
    "ChunkKeyEncodingField",
    "CodecDefinition",
    "CodecField",
    "CodecKind",
    "CodecSize",
    "Context",
    "DataTypeDefinition",
    "DataTypeField",
    "Definition",
    "EmptyConfiguration",
    "JSONValue",
    "Loc",
    "MetadataValidationError",
    "Nested",
    "ProblemKind",
    "Resolution",
    "Resolved",
    "StaticCodecField",
    "StorageTransformerDefinition",
    "StorageTransformerField",
    "Unread",
    "ValidationProblem",
    "ZarrV3MetadataFieldJSON",
    "canonicalize",
    "check",
    "configuration_of",
    "fill_value_problems",
    "resolve",
]
