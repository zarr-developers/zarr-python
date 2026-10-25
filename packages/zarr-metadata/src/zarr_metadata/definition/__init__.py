"""Reading metadata fields against their definitions, in either Zarr format.

Every extension point of a Zarr document (a v3 codec, data type, chunk
grid, chunk key encoding or storage transformer; a v2 dtype or numcodecs
codec) is a metadata field: a name, and a configuration whose JSON the
extension defines. A **definition** declares what that JSON is and what
it may contain. A definition is a value, not a class to subclass. It has:

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

The kind of metadata a definition defines is its type. The v3 kinds are
in `zarr_metadata.v3.definition`, and the v2 kinds are in
`zarr_metadata.v2.definition`. This module holds what both formats share:
`Definition` and `WithFillValue`, which the kinds derive from; `resolve`,
which reads a field in a scope; the three results of a reading; and
`Context`, the protocol of a scope.

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
   union of the three, for use with `match`.
   `configuration_of(field, GZIP_CODEC)` returns the configuration of a
   field that `GZIP_CODEC` read, typed as its TypedDict
   (`GzipCodecConfiguration`), and None for a field that another
   definition read or that nothing claims. A field is read as one of
   the kinds of its format, with or without type arguments.
   `resolve(field, Definition, context)` raises `TypeError`, because
   nothing is filed under `Definition`. A whole v3 array document is
   read by `read_array_metadata_v3`, in `zarr_metadata.model`. Its
   reading returns each field with its location in the document.

    from zarr_metadata.definition import resolve
    from zarr_metadata.v3.codec.gzip import GZIP_CODEC
    from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS, CodecDefinition

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
`zarr_metadata.problems.shown`: `null`, `[1, 2]`, `"C"`. A rule reports the location of a
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

    from zarr_metadata import ValidationProblem
    from zarr_metadata.definition import Nested
    from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS, CodecDefinition


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
    CONTEXT = CORE_AND_EXTENSIONS.update(codecs=[*CORE_AND_EXTENSIONS.codecs.values(), ACME_LZ4])

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

**Scopes as values.** A scope reads documents of one Zarr format.
`ZarrV3Context`, in `zarr_metadata.v3.definition`, and `ZarrV2Context`,
in `zarr_metadata.v2.definition`, are the scopes of each format. They
are separate classes, and `Context` is the protocol both implement: the
Zarr format a scope reads, and `claimant`, which says which definition
of a kind reads a name. That is all `resolve` needs, so any object that
provides these two can be a scope. A scope has one slot for each kind of
field its format has, named like the document member that holds such
fields: `ZarrV3Context(codecs=[...], data_types=[...])` builds one, and
`context.codecs["gzip"]` reads one slot. A type checker rejects a
definition in the wrong slot, and a scope of one format given to a
reader of the other. The constructor and the reader also reject them at
run time, with `TypeError`. Two scopes are equal when they are of one
format and file the same definitions, and equal scopes have equal
hashes. `context.union(*others)` returns the smallest scope that
contains each of them, or raises a `ScopeConflictError` naming each name
filed with two different definitions. `update` replaces the
definitions in the slots it is given and keeps the others, and takes a
slot as definitions or as a mapping by name, so
`context.update(codecs={**context.codecs, "gzip": MY_GZIP})` replaces
one definition on purpose. A model's `refined_in` moves it to a scope
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
written; an extra field named after a member the spec defines, whose
`finite` is not a boolean, or whose configuration does not declare
`must_understand`. Building a definition files it nowhere: only a scope
built from it does.
"""

from zarr_metadata.definition._definition import (
    AcceptedField,
    Context,
    Definition,
    EmptyConfiguration,
    Nested,
    RefusedField,
    ResolvedField,
    UnclaimedField,
    WithFillValue,
    canonical_fill_value,
    canonical_of,
    configuration_of,
    fill_value_problems,
    resolve,
    with_problems,
)
from zarr_metadata.definition._scope import (
    ClaimKey,
    Conflict,
    ScopeConflictError,
)

__all__ = [
    "AcceptedField",
    "ClaimKey",
    "Conflict",
    "Context",
    "Definition",
    "EmptyConfiguration",
    "Nested",
    "RefusedField",
    "ResolvedField",
    "ScopeConflictError",
    "UnclaimedField",
    "WithFillValue",
    "canonical_fill_value",
    "canonical_of",
    "configuration_of",
    "fill_value_problems",
    "resolve",
    "with_problems",
]
