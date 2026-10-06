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
  whether or not its module postpones annotations -- and a member's type
  carries its bounds, as pydantic reads them: a gzip `level` is
  `Annotated[int, Interval(ge=0, le=9)]`;
- `rules`, a function over that TypedDict yielding what the spec
  disallows that a type cannot say -- members read together -- located
  in the configuration.

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
   same way. It returns what the scope made of the field, and every
   problem: `Read` by the definition that claims its name, with the
   configuration it checked and allowed and the fields it holds, each
   read the same way; `Unclaimed`, a name nothing in scope claims, left
   unjudged, which is what keeps the format open; or `Refused`, whose
   problems say why. Each has the field's JSON, the `name` it is
   written with, the kind it was read as, `read_as`, the `definition`
   that claims its name -- None for `Unclaimed` -- and the fields it
   holds as the scope read them, `nested`. The two a model holds, `Read`
   and `Unclaimed`, have a `configuration` and `to_json()`, the field as
   a document writes it; two of them are equal when they read the same,
   however each was spelled. `Resolved` is the three, for `match`. A
   field is read as one of the five kinds, with or without type
   arguments;
   `resolve(field, Definition, scope)` is a `TypeError`, since nothing
   is filed under it. `configuration_of(resolved, GZIP_CODEC)` is the
   configuration typed as that definition's TypedDict, when it read it.
   `fields_of(resolved)` gives the field and each field it holds, with
   where each sits. A whole v3 array document is read by
   `read_array_metadata_v3`, in `zarr_metadata.model`.

    from zarr_metadata.v3.codec.gzip import GZIP_CODEC
    from zarr_metadata.v3.definition import (
        CORE_AND_EXTENSIONS,
        CodecDefinition,
        configuration_of,
        resolve,
    )

    resolved, problems = resolve({"name": "gzip", "configuration": {"level": 12}},
                                 CodecDefinition, CORE_AND_EXTENSIONS)
    resolved                # Refused(..., definition=CodecDefinition(name='gzip'), ...)
    problems[0].loc         # ('configuration', 'level')
    problems[0].input       # 12
    dict(problems[0].ctx)   # {'ge': 0, 'le': 9}

    resolved, problems = resolve({"name": "gzip", "configuration": {"level": 5}},
                                 CodecDefinition, CORE_AND_EXTENSIONS)
    configuration_of(resolved, GZIP_CODEC)   # {'level': 5}, a GzipCodecConfiguration

Problems are values, not exceptions: `ValidationProblem(loc, message,
kind)`, with `kind` one of `invalid_type`, `invalid_value`,
`missing_key`, `unknown_key` and `invalid_json`. A field with an
unknown key is still read -- the key reported, the configuration judged
without it -- so a consumer that tolerates one filters by kind and uses
what was read; the field is valid only when there is no problem at all.
Each problem carries what its message says as data, as pydantic's errors
and zod's issues do: `input`, what was found at `loc` -- the `12` above --
and `ctx`, what was expected, where that is more than a type: the bounds
`{"ge": 0, "le": 9}`, or the values of a closed set.

**Writing an extension.** A TypedDict, which says what the
configuration's JSON is, bounds and all; a function for the rules a type
cannot say; and a definition; then a scope that holds it. The TypedDict
is a `typing_extensions.TypedDict`: `closed` and `extra_items` are PEP
728's, which `typing.TypedDict` does not take on the versions this
package supports. The rules are handed the configuration and the fields
it holds as the scope read them: a field that is read keeps what it read
inside it as `Read.nested`, a `Nested` mapping by where each sits, so a
struct's rules reach its field types. `judge`, which reads in no scope,
hands them none. A rule's message shows a value as the package's own
messages do, as JSON, with `shown`: `null`, `[1, 2]`, `"C"`. A rule
reports where a problem is; what is found there is the problem's
`input` without the rule saying so. Define each function at a module's
top level: a model holds the definitions that read its fields, so it
pickles, and compares equal once loaded, only when they do -- a lambda
or a closure does not pickle, and a `functools.partial` pickles but
compares unequal to itself loaded.

    from collections.abc import Iterator
    from typing import Annotated, NotRequired

    from annotated_types import Ge
    from typing_extensions import TypedDict

    from zarr_metadata.v3.definition import (
        CORE_AND_EXTENSIONS,
        CodecDefinition,
        Nested,
        ValidationProblem,
    )


    class AcmeLz4Configuration(TypedDict, closed=True):
        acceleration: Annotated[int, Ge(1)]
        dictionary: NotRequired[str]
        dictionary_size: NotRequired[Annotated[int, Ge(1)]]


    def acme_lz4_rules(
        configuration: AcmeLz4Configuration, nested: Nested
    ) -> Iterator[ValidationProblem]:
        if "dictionary" in configuration and "dictionary_size" not in configuration:
            yield ValidationProblem(
                ("dictionary_size",), "a dictionary needs its size", "missing_key"
            )


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
by that data type's definition, the chunk grid against the shape, by
the grid's definition, and the codecs as a pipeline, each by its
definition against the chunk it is handed.

The TypedDict says what a key it does not declare is: with
`closed=True`, a problem, as above; with `extra_items=`, a key holding
that type; with `closed=False`, anything at all. One that says none of
these is open by default, and would take a misspelled key without a
word, so a definition refuses it. Its members are the shapes JSON takes:
`int`, `float`, `bool`, `str`, `None`, `JSONValue`, a `Literal`,
`tuple[T, ...]` and `tuple[T1, T2]`, a union, a TypedDict,
`Mapping[str, V]`, a `NewType` and a type alias. A number's type may
carry bounds, as annotated-types spells them and pydantic reads them --
`Gt`, `Ge`, `Lt`, `Le` and `Interval`, one from each side -- at any
depth: `tuple[Annotated[int, Ge(1)], ...]` bounds each element. A value
out of them is a problem, `invalid_value`, whose message says what the
type admits, "expected an integer >= 1, got 0", and whose `ctx` holds
the bounds. The rules are asked only of a configuration within its
bounds, so a rule relies on them, as pydantic's after-validators and
zod's refinements do: until a value out of bounds is fixed, it is the
one problem reported of the configuration. `Annotated` may also carry a
note, a string or a `Doc`. Any other metadata -- a `MinLen`, a
`Predicate`, pydantic's `Field` -- is refused when the definition is
built, since a type the checker does not hold its values to would say
what is not so.

A member holding another metadata field is annotated with the field alias
of its kind -- a shard's `codecs: tuple[CodecField, ...]` -- and read in
the scope its field is read in. What is wrong with the field it holds is
that field's own, reported where it sits: the field holding it is still
read, as a document holding it would be. A member that takes codecs of
static size only is annotated `StaticCodecField` -- a shard's
`index_codecs`, since a reader finds the index by a size it knows before
reading it -- and a codec of dynamic size there is a problem at its
place, where the field is read in a scope; a name nothing claims is left
unjudged, its size unknown with the rest of it. `ZarrV3MetadataFieldJSON`
is the same JSON, but checks as JSON and nothing more, so a definition
refuses a member typed with it. An extension with nothing to configure
takes `EmptyConfiguration`, and is written with its name alone.

A data type also says what its fill value is: `fill_value`, the JSON
shape of one as an annotation the checker reads -- `Int8FillValue`,
whose type carries the range -- and `fill_value_rules`, a function
yielding what the spec disallows in a fill value of that shape that the
type cannot say: a hex string of another width, a number of byte values
the size does not take. The rules are handed the configuration, the fields it holds as the
scope read them, and the typed fill value, so a struct judges each
field's fill value by that field's own type.
`fill_value_problems(data_type, value)` judges a fill value against a data
type field the scope read; one nothing in scope claims leaves it unjudged.
A data type that says nothing of its fill value takes any JSON.
A fill value may be spelled more ways than one -- `"NaN"` and
`"0x7fc00000"` are one `float32` -- so a data type says which spelling
is its value's own: `fill_value_canonical`, handed what the rules are
handed and a fill value they allow. Two fill values are one value of the
type exactly when their canonical spellings are written alike: the same
JSON, as `json.dumps` writes it, which `==` is not -- it takes `-0.0`,
a `float32` of its own, for `0.0`. `canonical_fill_value(data_type,
value)` spells one, and gives `UNSET` for a fill value with a problem; a
data type that says nothing of it spells each value as written.

A data type says how its values are stored, too: `storage`, a function
of its configuration and the fields it holds, giving a `StorageClass` --
in single bytes, in several bytes at a time, or each in as many as it
needs. A struct's is its fields'. `storage_of(data_type)` asks it of a
data type field the scope read: the `bytes` codec takes an `endian` for
numbers of several bytes, and a struct refuses a field whose values vary
in size. A data type that says nothing of it leaves it unknown.

A chunk grid says which arrays it fits: `shape_rules`, a function
yielding what the spec disallows in a grid of its configuration over an
array of a given shape -- a dimension with no chunk length, chunks that
fall short of one -- located in the configuration. A grid that says
nothing of the shape fits every one. It also says the lengths its chunks
take along each axis of an array it fits, `chunk_lengths`: a set per
axis, since a rectilinear grid's chunks differ. `chunk_grid_lengths(grid,
shape)` gives both of a chunk grid field the scope read: an entry for
each dimension of the shape, None where nothing says the lengths.

A codec is judged against what it is handed. The array hands its first
codec a `Chunk`: the lengths of its grid's chunks along each of the
array's dimensions, and its data type field, with None for what nothing
says. A codec handed an array says what the spec disallows in it handed
a chunk: `chunk_rules`, located in its configuration -- a `transpose`
whose `order` has another number of axes. An array -> array codec says
what it hands the next, whatever its chunk rules found: `transition` --
`transpose` permutes the axes. `read_pipeline(codecs, chunk)` reads codec
fields the scope read as a pipeline: their order -- array -> array
codecs, one array -> bytes codec, bytes -> bytes codecs -- and then each
against the chunk it is handed, giving each codec's `Stage` with that
chunk. A codec that holds pipelines of its own says what each is
handed: `pipelines`, by the member of its configuration that holds each
-- a shard's inner codecs its inner chunks, its index codecs the shard
index -- and each is read the same way, its stages kept as the codec's
`Stage.inner`.
Nothing is guessed: the codec after one the scope did not read, or after
one that says nothing of what it hands on, is handed a chunk nothing is
known of, `Chunk()`, which is refused nothing; a codec after that hands
on only what it says of its own accord.

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
words every reader takes: a data type with nothing to configure is its
bare name, any other field an object, `{"name": ...}`, as a Zarr v3.0
reader takes no short-hand name in `codecs`; raw bits write their size
back into the name, in decimal, so `r008` is `r8`. A field with any
problem, an unknown key included, has none: a simpler spelling of it
would erase what its author wrote. What `canonical` gives is judged
again: one that does not hold is a `ValueError`, a fault in the
definition. `canonical_of(resolved, problems)` spells a field a scope
has read already, given its problems -- as `resolve` gives them, or
`with_problems` gives each field of a reading -- without reading it
again, and gives what `canonicalize` gives: None for a field with a
problem.

**JSON Schema.** `field_json_schema(CodecDefinition, SCOPE)` writes the
fields of one kind a scope reads as a JSON Schema, draft 2020-12, for a
validator in another language or an editor: each definition's field --
its name, its configuration as its TypedDict says, bounds and all, a
`must_understand` of `true`, and its bare name when it needs no
configuration -- and a name nothing in scope claims, with any
configuration. A field a configuration holds is written in the same
scope, and a member taking codecs of static size only takes those. The
rules are not in it, so a field it accepts may still have a problem;
one `resolve` reads without a problem, it accepts, as JSON: arrays as
lists, as a parser gives them. Each configuration
TypedDict, and each field alias, is written once, in `$defs`, under its
name. `node_metadata_json_schema_v3`, in `zarr_metadata.model`, writes a
whole `zarr.json`, its fill value held to its data type's.

**Scopes as values.** Two scopes are equal when they file the same
definitions, and equal scopes hash alike. `claims_of(fields_of(field))`
says what a reading claimed of each name -- the definition that read it,
or None -- keyed as the scope files it, `r16` under `r*`.
`refines(field, other)` orders two readings of a field by information: a
name nothing claimed, read by a definition, is a gain; the reverse a
loss; one name read by two definitions a conflict.
`scope.disagreements(claims)` says where a scope would read a reading
otherwise, and `Context.joined(*scopes)` is the least scope above each,
or a `ScopeConflictError` naming each name filed two ways; `extended_with`
remains the way to take a name over on purpose.

A definition checks itself when it is built, and each of these is a
`TypeError` saying what is wrong: a `configuration` that is not a
TypedDict, says nothing of the keys it does not declare, or has a member
no checker reads, named down to the TypedDict that holds it; a `name`
that is not a string; a member declared as a function -- `rules`,
`canonical`, `fill_value_rules`, `storage`, `shape_rules`,
`chunk_lengths`, `chunk_rules`, `transition`, `pipelines` -- that is not
one; a data type's `fill_value` no checker reads, or one holding a
metadata field, which a value of the data type never is; a codec `kind`
that is not one of the three, or a `size` that is not `"static"` or
`"dynamic"`; a function no codec of its kind is asked -- chunk rules or
pipelines of a bytes -> bytes codec, which is handed bytes, or a
`transition` of a codec that hands on bytes; a data type named as raw
bits of one size are written. A scope refuses a definition of no kind.
Nothing happens at class creation.
"""

from zarr_metadata._common import JSONValue
from zarr_metadata._json import MetadataValidationError, ProblemKind, ValidationProblem, shown
from zarr_metadata._typed_json import Loc, check
from zarr_metadata.v3._common import (
    ChunkGridField,
    ChunkKeyEncodingField,
    CodecField,
    DataTypeField,
    StaticCodecField,
    StorageTransformerField,
    ZarrV3MetadataFieldJSON,
)
from zarr_metadata.v3._definition import (
    Chunk,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    CodecKind,
    CodecSize,
    DataTypeDefinition,
    Definition,
    EmptyConfiguration,
    Lengths,
    Nested,
    Read,
    Refused,
    Resolved,
    StorageClass,
    StorageTransformerDefinition,
    Unclaimed,
    WithFillValue,
    canonical_fill_value,
    canonical_of,
    canonicalize,
    chunk_grid_lengths,
    configuration_of,
    field_json_schema,
    fields_of,
    fill_value_problems,
    resolve,
    storage_of,
    with_problems,
)
from zarr_metadata.v3._pipeline import Stage, read_pipeline
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS, Context
from zarr_metadata.v3._scope import (
    ClaimKey,
    Claims,
    Conflict,
    Disagreements,
    ScopeConflictError,
    claim_key,
    claims_of,
    refines,
)

__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "Chunk",
    "ChunkGridDefinition",
    "ChunkGridField",
    "ChunkKeyEncodingDefinition",
    "ChunkKeyEncodingField",
    "ClaimKey",
    "Claims",
    "CodecDefinition",
    "CodecField",
    "CodecKind",
    "CodecSize",
    "Conflict",
    "Context",
    "DataTypeDefinition",
    "DataTypeField",
    "Definition",
    "Disagreements",
    "EmptyConfiguration",
    "JSONValue",
    "Lengths",
    "Loc",
    "MetadataValidationError",
    "Nested",
    "ProblemKind",
    "Read",
    "Refused",
    "Resolved",
    "ScopeConflictError",
    "Stage",
    "StaticCodecField",
    "StorageClass",
    "StorageTransformerDefinition",
    "StorageTransformerField",
    "Unclaimed",
    "ValidationProblem",
    "WithFillValue",
    "ZarrV3MetadataFieldJSON",
    "canonical_fill_value",
    "canonical_of",
    "canonicalize",
    "check",
    "chunk_grid_lengths",
    "claim_key",
    "claims_of",
    "configuration_of",
    "field_json_schema",
    "fields_of",
    "fill_value_problems",
    "read_pipeline",
    "refines",
    "resolve",
    "shown",
    "storage_of",
    "with_problems",
]
