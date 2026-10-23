"""Reading metadata fields against their definitions: the public entry point.

Every Zarr v3 extension point (codecs, data types, chunk grids, chunk
key encodings, storage transformers) is a metadata field: a name, and
a configuration whose JSON the extension defines. A **definition**
declares what that JSON is and what it may contain. A definition is a
value, not a class to subclass. It has:

- `name`, the name the metadata uses;
- `configuration`, the TypedDict for the configuration's JSON. This is
  the single declaration of the JSON: the type checker is compiled from
  it, and a checked configuration has it as its static type. It is
  interpreted as the typing spec defines a TypedDict: `total`,
  `Required`, `NotRequired`, `closed` and `extra_items` mean what they
  mean to a type checker, whether or not its module postpones
  annotations. A member's type carries its bounds, in the same way
  pydantic reads them: a gzip `level` is
  `Annotated[int, Interval(ge=0, le=9)]`;
- `rules`, a function over that TypedDict that yields the spec
  violations a type cannot express (such as constraints between
  members), located in the configuration.

The kind of metadata a definition defines is its type: `CodecDefinition`
(with the codec's `kind`, and its `size`: `"static"` if the output size
is determined by the input size, or `"dynamic"` if it depends on the
values), `DataTypeDefinition`, `ChunkGridDefinition`,
`ChunkKeyEncodingDefinition`, `StorageTransformerDefinition`.

**Reading JSON.** There are three steps. Each feeds the next, and each
can be used on its own by a caller that has only JSON:

1. `check(value, SomeTypedDict)`, from `zarr_metadata.typed_json`,
   type-checks JSON against a TypedDict and needs nothing else. It
   returns a value of the TypedDict or None, and every problem, each
   with a location. The value contains only what the TypedDict admits. A
   member typed with a field alias is checked as metadata field JSON.
2. `definition.read_configuration(configuration)` returns the
   configuration as that definition reads it. It type-checks the
   configuration, checks the envelope of each nested field (an extra
   member, a `must_understand` of `false`), and then runs the rules, for
   one configuration: `GZIP_CODEC.read_configuration({"level": 12})`.
3. `resolve(field, CodecDefinition, CORE_AND_EXTENSIONS)` reads a whole
   field in a scope. It checks the envelope, maps the name to a
   definition, checks the configuration, and reads each nested field the
   same way. It returns the result of the reading and every problem. The
   result is one of three types. `AcceptedField` is a field read by the
   definition that claims its name, with the configuration it checked
   and allowed, and its nested fields, each read the same way.
   `UnclaimedField` is a name that nothing in scope claims. It is not
   checked, which is what keeps the format open. `RefusedField` is a
   field with problems that explain why it was rejected. Each type has the
   field's JSON, the `name` it is written with, the kind it was read as
   (`read_as`), the `definition` that claims its name (None for
   `UnclaimedField`), and its nested fields as the scope read them
   (`nested`). The two types a model can hold, `AcceptedField` and
   `UnclaimedField`, have a `configuration` and `to_json()`, which
   returns the field as a document writes it. Two of them are equal when
   they read the same, however each was spelled. `ResolvedField` is the
   union of the three, for use with `match`. A field is read as one of
   the five kinds, with or without type arguments.
   `resolve(field, Definition, scope)` raises `TypeError`, because
   nothing is filed under `Definition`. A whole v3 array document is
   read by `read_array_metadata_v3`, in `zarr_metadata.model`. Its
   reading returns each field with its location in the document.

    from zarr_metadata.v3.codec.gzip import GZIP_CODEC
    from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS, CodecDefinition, resolve

    resolved, problems = resolve({"name": "gzip", "configuration": {"level": 12}},
                                 CodecDefinition, CORE_AND_EXTENSIONS)
    resolved                # RefusedField(..., definition=CodecDefinition(name='gzip'), ...)
    problems[0].loc         # ('configuration', 'level')
    problems[0].input       # 12
    dict(problems[0].ctx)   # {'ge': 0, 'le': 9}

    resolved, problems = resolve({"name": "gzip", "configuration": {"level": 5}},
                                 CodecDefinition, CORE_AND_EXTENSIONS)
    resolved.definition is GZIP_CODEC        # True
    resolved.configuration                   # {'level': 5}

Problems are values, not exceptions: `ValidationProblem(loc, message,
kind)`, with `kind` one of `invalid_type`, `invalid_value`,
`missing_key`, `unknown_key` and `invalid_json`. A field with an
unknown key is still read: the key is reported, and the configuration
is checked without it. So a consumer that tolerates unknown keys can
filter problems by kind and use the result. The field is valid only
when there are no problems at all. Each problem stores the content of
its message as data, as pydantic's errors and zod's issues do: `input`
is the value found at `loc` (the `12` above), and `ctx` is what was
expected, when that is more than a type: the bounds
`{"ge": 0, "le": 9}`, or the values of a closed set.

**Writing an extension.** Write a TypedDict that declares the
configuration's JSON, including bounds; a function for the rules a type
cannot express; and a definition. Then build a scope that contains the
definition. The TypedDict must be a `typing_extensions.TypedDict`,
because `closed` and `extra_items` come from PEP 728, and
`typing.TypedDict` does not accept them on the Python versions this
package supports. The rules receive the configuration and its nested
fields as the scope read them. A field that is read stores its nested
fields as `AcceptedField.nested`, a `Nested` mapping keyed by location,
so a struct's rules can access its field types. `read_configuration`
reads in no scope, so it passes no nested fields. A rule's message
should show a value as JSON, as the package's own messages do, with
`shown`: `null`, `[1, 2]`, `"C"`. A rule reports the location of a
problem. The value at that location becomes the problem's `input`
automatically. Define each function at module top level. A model holds
the definitions that read its fields, so the model can be pickled, and
compares equal after loading, only if its definitions can. A lambda or
a closure cannot be pickled, and a `functools.partial` can be pickled
but does not compare equal to its loaded copy.

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

A scope reads whole documents as well as fields.
`validate_array_metadata_v3(document, context=SCOPE)`, from
`zarr_metadata.model`, reads each extension point of a v3 array document
through the definitions in `SCOPE`, and so do the model's `from_json` and
`from_key_value`. The fill value is checked against the data type it
names, by that data type's definition. The chunk grid is checked against
the shape, by the grid's definition. The codecs are checked as a
pipeline, each by its definition against the chunk it receives.

The TypedDict declares how keys it does not declare are handled. With
`closed=True`, such a key is a problem, as above. With `extra_items=`,
such a key must hold that type. With `closed=False`, anything is
allowed. A TypedDict that declares none of these is open by default and
would accept a misspelled key silently, so a definition rejects it. Its
members must be JSON shapes: `int`, `float`, `bool`, `str`, `None`,
`JSONValue`, a `Literal`, `tuple[T, ...]` and `tuple[T1, T2]`, a union,
a TypedDict, `Mapping[str, V]`, a `NewType` and a type alias. A number's
type may carry bounds, written with annotated-types and read the way
pydantic reads them: `Gt`, `Ge`, `Lt`, `Le` and `Interval`, at most one
from each side, at any depth. For example,
`tuple[Annotated[int, Ge(1)], ...]` bounds each element. A value out of
bounds is an `invalid_value` problem. Its message says what the type
admits ("expected an integer >= 1, got 0"), and its `ctx` holds the
bounds. The rules run only on a configuration within its bounds, so a
rule can rely on them, as pydantic's after-validators and zod's
refinements do. Until a value out of bounds is fixed, it is the only
problem reported for the configuration. `Annotated` may also carry a
note, as a string or a `Doc`. Any other metadata (a `MinLen`, a
`Predicate`, pydantic's `Field`) is rejected when the definition is
built, because a type whose constraint the checker does not enforce
would give false information.

A member that holds another metadata field is annotated with the field
alias of its kind (a shard's `codecs: tuple[ZarrV3CodecField, ...]`) and
is read in the same scope as its parent field. A problem with the nested
field belongs to that field and is reported at its location. The parent
field is still read, as a document containing the nested field would be.
A member that accepts only codecs of static size is annotated
`ZarrV3StaticCodecField`. A shard's `index_codecs` uses this, because a
reader locates the index from a size it knows before reading it. A codec
of dynamic size in such a member is a problem at its location, when the
field is read in a scope. A name that nothing claims is not checked, and
its size is unknown, like everything else about it.
`ZarrV3MetadataFieldJSON` describes the same JSON, but it is checked
only as JSON, so a definition rejects a member typed with it. An
extension with nothing to configure uses `EmptyConfiguration`, and is
written with its name only.

A data type also declares its fill value. `fill_value` is the JSON
shape of a fill value, as an annotation the checker reads (for example
`Int8FillValue`, whose type carries the range). `fill_value_rules` is a
function that yields the spec violations in a fill value of that shape
that the type cannot express, such as a hex string of the wrong width,
or a number of byte values that does not match the size. The rules
receive the configuration, the nested fields as the scope read them,
and the typed fill value, so a struct checks each field's fill value
against that field's own type.
`fill_value_problems(data_type, value)` checks a fill value against a
data type field the scope read. If nothing in scope claims the data
type, the fill value is not checked. A data type that does not declare
its fill value accepts any JSON.
A fill value can have more than one spelling (`"NaN"` and `"0x7fc00000"`
are the same `float32`), so a data type declares the canonical spelling
of each value with `fill_value_canonical`. It receives the same
arguments as the rules, plus a fill value the rules allow. Two fill
values are the same value of the type exactly when their canonical
spellings are written the same way: the same JSON, as `json.dumps`
writes it. `==` is not the same test, because it treats `-0.0` (a
distinct `float32`) as equal to `0.0`.
`canonical_fill_value(data_type, value)` returns the canonical spelling,
or `UNSET` for a fill value with a problem. A data type that does not
declare `fill_value_canonical` keeps each value as written.

A data type also declares how its values are stored. `storage` is a
function of the configuration and the nested fields that returns a
`StorageClass`: single bytes, several bytes at a time, or each value in
as many bytes as it needs. A struct's storage depends on its fields.
`storage_of(data_type)` calls it for a data type field the scope read.
The `bytes` codec requires an `endian` for multi-byte numbers, and a
struct rejects a field whose values vary in size. A data type that does
not declare storage leaves it unknown.

A chunk grid declares which array shapes it fits with `shape_rules`, a
function that yields the spec violations in a grid with its
configuration over an array of a given shape (a dimension with no chunk
length, or chunks that do not cover the dimension), located in the
configuration. A grid that does not declare shape rules fits every
shape. It also declares the lengths of its chunks along each axis of an
array it fits, with `chunk_lengths`: a set per axis, because a
rectilinear grid's chunks differ in length. A reading stores both
results for the grid it read: an entry for each dimension of the shape,
with None where the lengths are unknown.

A codec is checked against its input. The array passes its first codec
a `Chunk`: the lengths of the grid's chunks along each of the array's
dimensions, and the array's data type field, with None for unknown
parts. A codec that receives an array declares the spec violations for
a given chunk with `chunk_rules`, located in its configuration (for
example, a `transpose` whose `order` has the wrong number of axes). An
array -> array codec declares what it passes to the next codec with
`transition`, whatever its chunk rules found. For example, `transpose`
permutes the axes. A reading reads the codec fields as a pipeline. It
checks their order (array -> array codecs, one array -> bytes codec,
bytes -> bytes codecs), then checks each codec against the chunk it
receives, and records a `Stage` for each codec with that chunk. A codec
that contains its own pipelines declares what each one receives with
`pipelines`, keyed by the configuration member that holds each
pipeline. For example, a shard's inner codecs receive its inner chunks,
and its index codecs receive the shard index. Each pipeline is read the
same way, and its stages are stored as the codec's `Stage.inner`.
The reader does not guess. After a codec the scope did not read, or
after a codec that does not declare its output, the next codec receives
a chunk with nothing known about it, `Chunk()`, which never causes a
problem. A later codec can still declare its own output.

Raw bits are the only data type whose name encodes its configuration. A
document writes `r` followed by the size in bits, and `r16` is read as
`r*` (the notation in the specification's table of data types) with
`{"bits": 16}`. A reader that wants to read raw bits its own way defines
`r*`. A data type named `r16` is rejected, because that name is read as
`r*`. `r*` itself is notation, and a document that uses it as a name
refers to nothing in any scope.

**The simplest spelling.** A field without problems has a simplest
equivalent spelling, and fields are compared by it. Each nested field is
written in its own simplest spelling, then the definition's `canonical`
is applied (blosc drops a `typesize` that `noshuffle` ignores, and a
rectilinear grid run-length encodes its chunk shapes). The envelope
uses the shortest form every reader accepts: a data type with nothing to
configure is its bare name, and any other field is an object,
`{"name": ...}`, because a Zarr v3.0 reader does not accept a short-hand
name in `codecs`. Raw bits write their size back into the name, in
decimal, so `r008` becomes `r8`. A field with any problem, including an
unknown key, has no simplest spelling, because a simpler spelling would
erase what its author wrote. The result of `canonical` is checked again,
and an invalid result raises `ValueError`, which indicates a bug in the
definition. `canonical_fill_value` writes a fill value in the same way,
using the data type that read it.

**JSON Schema.** `node_metadata_json_schema_v3`, in `zarr_metadata.model`,
writes a whole `zarr.json` as a JSON Schema (draft 2020-12), for a
validator in another language or for an editor. Its fields are written
as the scope reads them. For each definition, the schema accepts its
field (its name, its configuration as its TypedDict declares it,
including bounds, a `must_understand` of `true`, and its bare name when
it needs no configuration). It also accepts a name that nothing in scope
claims, with any configuration. A nested field is written in the same
scope, and a member that accepts only codecs of static size accepts
only those. The rules are not in the schema, so a field it accepts may
still have a problem. Any field that `resolve` reads without a problem
is accepted by the schema, as JSON (arrays as lists, as a parser
returns them). Each configuration TypedDict and each field alias is
written once, in `$defs`, under its name. The fill value must match its
data type's fill value schema.

**Scopes as values.** A scope reads documents of one Zarr format: the
`format` declared by every kind it files. `ZarrV3Context` is the type
of a v3 scope, and `ZarrV2Context` is the type of a v2 scope. A reader
rejects a scope of the other format with `TypeError`. A scope that files
nothing has no format and can be used by either reader. Two scopes are
equal when they file the same definitions, and equal scopes have equal
hashes. `Context.joined(*scopes)` returns the smallest scope that
contains each of them, or raises a `ScopeConflictError` naming each name
filed with two different definitions. To replace a definition on
purpose, use `extended_with`. A model's `refined_in` moves it to a scope
that claims more names and contradicts nothing. `refines` orders two
models by how much they know. A name that was unclaimed and is now read
by a definition is a gain. The reverse is a loss. One name read by two
different definitions is a conflict.

A definition checks itself when it is built. Each of these raises a
`TypeError` that says what is wrong: a `configuration` that is not a
TypedDict, that does not declare how undeclared keys are handled, or
that has a member no checker reads (the error names the TypedDict that
contains it); a `name` that is not a string; a member declared as a
function (`rules`, `canonical`, `fill_value_rules`, `storage`,
`shape_rules`, `chunk_lengths`, `chunk_rules`, `transition`,
`pipelines`) that is not a function; a data type's `fill_value` that no
checker reads, or that contains a metadata field, which a value of the
data type never does; a codec `kind` that is not one of the three, or a
`size` that is not `"static"` or `"dynamic"`; a function that is never
called for a codec of its kind (chunk rules or pipelines of a
bytes -> bytes codec, which receives bytes, or a `transition` of a codec
that outputs bytes); a data type named the way raw bits of one size are
written. A scope rejects a definition of no kind, and a kind is
declared with its format, `class Tag(Definition[C], kind=True,
format=3)`. Nothing else happens at class creation.
"""

from zarr_metadata._common import JSONValue
from zarr_metadata._json import MetadataValidationError, ProblemKind, ValidationProblem, shown
from zarr_metadata._typed_json import Loc
from zarr_metadata.v3._common import (
    ZarrV3ChunkGridField,
    ZarrV3ChunkKeyEncodingField,
    ZarrV3CodecField,
    ZarrV3DataTypeField,
    ZarrV3StaticCodecField,
    ZarrV3StorageTransformerField,
)
from zarr_metadata.v3._definition import (
    AcceptedField,
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
    RefusedField,
    ResolvedField,
    StorageClass,
    StorageTransformerDefinition,
    UnclaimedField,
    canonical_fill_value,
    fill_value_problems,
    resolve,
    storage_of,
)
from zarr_metadata.v3._pipeline import Stage
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS, Context, ZarrV3Context
from zarr_metadata.v3._scope import (
    ClaimKey,
    Claims,
    Conflict,
    Disagreements,
    ScopeConflictError,
)

__all__ = [
    "CORE",
    "CORE_AND_EXTENSIONS",
    "AcceptedField",
    "Chunk",
    "ChunkGridDefinition",
    "ChunkKeyEncodingDefinition",
    "ClaimKey",
    "Claims",
    "CodecDefinition",
    "CodecKind",
    "CodecSize",
    "Conflict",
    "Context",
    "DataTypeDefinition",
    "Definition",
    "Disagreements",
    "EmptyConfiguration",
    "JSONValue",
    "Lengths",
    "Loc",
    "MetadataValidationError",
    "Nested",
    "ProblemKind",
    "RefusedField",
    "ResolvedField",
    "ScopeConflictError",
    "Stage",
    "StorageClass",
    "StorageTransformerDefinition",
    "UnclaimedField",
    "ValidationProblem",
    "ZarrV3ChunkGridField",
    "ZarrV3ChunkKeyEncodingField",
    "ZarrV3CodecField",
    "ZarrV3Context",
    "ZarrV3DataTypeField",
    "ZarrV3StaticCodecField",
    "ZarrV3StorageTransformerField",
    "canonical_fill_value",
    "fill_value_problems",
    "resolve",
    "shown",
    "storage_of",
]
