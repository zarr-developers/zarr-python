"""JSON Schemas of what the package reads: a TypedDict, a field in a scope, a `zarr.json`.

A schema is held to the reader it is written from: what the reader finds
nothing wrong with, the schema accepts, and what the type says -- a
bound, a key a closed TypedDict does not declare, a name a scope claims
-- the schema says too. What only a rule says, the schema does not.
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, NewType, NotRequired, cast

import pytest
from annotated_types import Ge, Gt, Interval, Le, Lt, MinLen
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator
from typing_extensions import Doc, TypeAliasType, TypedDict

from tests.v3.test_every_definition import CASES, KINDS
from zarr_metadata._common import JSONValue
from zarr_metadata._typed_json import Schemas
from zarr_metadata.model import node_metadata_json_schema_v3, validate_node_metadata_v3
from zarr_metadata.typed_json import check, json_schema
from zarr_metadata.v3.codec.gzip import GzipCodecConfiguration
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    Context,
    DataTypeDefinition,
    Definition,
    StorageTransformerDefinition,
    field_json_schema,
    resolve,
)

DIALECT = "https://json-schema.org/draft/2020-12/schema"

# Built at run time, where a type checker reads each special form's
# arguments as it would in an annotation.
_annotated: Any = Annotated
_new_type: Callable[[str, object], object] = cast("Any", NewType)
_alias_type: Callable[[str, object], object] = cast("Any", TypeAliasType)

Level = TypeAliasType("Level", Annotated[int, Interval(ge=0, le=9)])
Tree = TypeAliasType("Tree", "int | tuple[Tree, ...]")
Name = NewType("Name", str)
Count = _new_type("Count", Annotated[int, Ge(0)])


class Inner(TypedDict, closed=True):
    x: int


class Open(TypedDict, closed=False):
    x: NotRequired[int]


class Extra(TypedDict, extra_items=int):
    x: str


class ExtraJSON(TypedDict, extra_items=JSONValue):
    x: str


class Node(TypedDict, closed=True):
    children: tuple[Node, ...]


def _closed(name: str, annotations: dict[str, object]) -> type:
    """A closed TypedDict of these required keys, made here, so its annotations resolve in this module."""
    base: object = TypedDict
    namespace = {"__annotations__": annotations, "__module__": __name__}
    return types.new_class(name, (base,), {"closed": True}, lambda body: body.update(namespace))


def _holding(annotation: object) -> type:
    """A closed TypedDict of one required key, `value`, holding `annotation`."""
    return _closed("Holder", {"value": annotation})


# Another class of the name `Inner`, which the schema tells from the first.
Both = _closed("Both", {"first": Inner, "second": _closed("Inner", {"y": str})})


def _held(member: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """The schema of `_holding` an annotation whose schema is `member`."""
    schema: dict[str, Any] = {
        "$schema": DIALECT,
        "type": "object",
        "properties": {"value": member},
        "required": ["value"],
        "additionalProperties": False,
    }
    return schema if defs is None else {**schema, "$defs": defs}


INNER = {
    "type": "object",
    "properties": {"x": {"type": "integer"}},
    "required": ["x"],
    "additionalProperties": False,
}

SHAPES: list[tuple[str, type, dict[str, Any]]] = [
    ("int", _holding(int), _held({"type": "integer"})),
    ("float", _holding(float), _held({"type": "number"})),
    ("bool", _holding(bool), _held({"type": "boolean"})),
    ("str", _holding(str), _held({"type": "string"})),
    ("null", _holding(None), _held({"type": "null"})),
    ("json", _holding(JSONValue), _held({})),
    ("one-value", _holding(Literal["a"]), _held({"const": "a"})),
    # Sorted as the checker sorts them, so the order written does not show.
    ("values", _holding(Literal["b", "a", 1]), _held({"enum": ["a", "b", 1]})),
    ("array", _holding(tuple[int, ...]), _held({"type": "array", "items": {"type": "integer"}})),
    ("array-of-json", _holding(tuple[JSONValue, ...]), _held({"type": "array"})),
    (
        "pair",
        _holding(tuple[int, str]),
        _held(
            {
                "type": "array",
                "prefixItems": [{"type": "integer"}, {"type": "string"}],
                "items": False,
                "minItems": 2,
            }
        ),
    ),
    ("empty-array", _holding(tuple[()]), _held({"type": "array", "maxItems": 0})),
    (
        "union",
        _holding(int | None),
        _held({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
    ),
    (
        "mapping",
        _holding(Mapping[str, int]),
        _held({"type": "object", "additionalProperties": {"type": "integer"}}),
    ),
    ("mapping-of-json", _holding(Mapping[str, JSONValue]), _held({"type": "object"})),
    ("newtype", _holding(Name), _held({"type": "string"})),
    (
        "interval",
        _holding(Annotated[int, Interval(ge=0, le=9)]),
        _held({"type": "integer", "minimum": 0, "maximum": 9}),
    ),
    (
        "exclusive-bounds",
        _holding(Annotated[float, Gt(0), Lt(1)]),
        _held({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}),
    ),
    (
        "doc",
        _holding(Annotated[int, Ge(0), Doc("a count")]),
        _held({"type": "integer", "description": "a count", "minimum": 0}),
    ),
    # A note that is no `Doc` says nothing a schema writes.
    ("note", _holding(Annotated[str, "a note"]), _held({"type": "string"})),
    # A value is held to its type's bounds and to its `NewType`'s: of two
    # of one keyword, the stricter.
    (
        "bounds-on-a-bounded-newtype",
        _holding(_annotated[Count, Ge(-5), Lt(9)]),
        _held({"type": "integer", "minimum": 0, "exclusiveMaximum": 9}),
    ),
    (
        "alias",
        _holding(Level),
        _held(
            {"$ref": "#/$defs/Level"},
            {"Level": {"type": "integer", "minimum": 0, "maximum": 9}},
        ),
    ),
    (
        "alias-holding-itself",
        _holding(Tree),
        _held(
            {"$ref": "#/$defs/Tree"},
            {
                "Tree": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "array", "items": {"$ref": "#/$defs/Tree"}},
                    ]
                }
            },
        ),
    ),
    ("typeddict", _holding(Inner), _held({"$ref": "#/$defs/Inner"}, {"Inner": INNER})),
    (
        "open",
        Open,
        {"$schema": DIALECT, "type": "object", "properties": {"x": {"type": "integer"}}},
    ),
    (
        "extra-items",
        Extra,
        {
            "$schema": DIALECT,
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": {"type": "integer"},
        },
    ),
    (
        "extra-items-of-json",
        ExtraJSON,
        {
            "$schema": DIALECT,
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    ),
    # A TypedDict that holds itself is referred to, not written in place.
    (
        "holding-itself",
        Node,
        {
            "$schema": DIALECT,
            "$ref": "#/$defs/Node",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
                    },
                    "required": ["children"],
                    "additionalProperties": False,
                }
            },
        },
    ),
    (
        "one-name-two-classes",
        Both,
        {
            "$schema": DIALECT,
            "type": "object",
            "properties": {
                "first": {"$ref": "#/$defs/Inner"},
                "second": {"$ref": "#/$defs/Inner2"},
            },
            "required": ["first", "second"],
            "additionalProperties": False,
            "$defs": {
                "Inner": INNER,
                "Inner2": {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                    "required": ["y"],
                    "additionalProperties": False,
                },
            },
        },
    ),
    (
        "a-definition-s-configuration",
        GzipCodecConfiguration,
        {
            "$schema": DIALECT,
            "type": "object",
            "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 9}},
            "required": ["level"],
            "additionalProperties": False,
        },
    ),
]


@pytest.mark.parametrize(
    ("shape", "expected"), [case[1:] for case in SHAPES], ids=[case[0] for case in SHAPES]
)
def test_json_schema_writes_what_check_reads(shape: type, expected: dict[str, Any]) -> None:
    schema = json_schema(shape)
    assert schema == expected
    Draft202012Validator.check_schema(schema)
    assert json.loads(json.dumps(schema)) == schema


def test_json_schema_refuses_what_is_not_a_typeddict() -> None:
    with pytest.raises(TypeError, match="is not a TypedDict"):
        json_schema(int)


def test_json_schema_refuses_what_check_cannot_read() -> None:
    unread = _holding(bytes)
    with pytest.raises(TypeError) as raised:
        check({}, unread)
    with pytest.raises(TypeError, match=str(raised.value)):
        json_schema(unread)


def test_error_a_schema_that_fails_to_write_leaves_nothing_behind() -> None:
    # Reserved before it is written, so that it can refer to itself, and
    # given up when the writing fails, so that nothing refers to an empty
    # schema, which takes anything.
    schemas = Schemas()
    unread = _closed("Unread", {"value": bytes})
    with pytest.raises(TypeError, match="is not a shape JSON takes"):
        schemas.of(unread)
    assert schemas.document({}) == {"$schema": DIALECT}
    with pytest.raises(TypeError, match="is not a shape JSON takes"):
        schemas.of(unread)


def test_json_schema_refuses_metadata_check_does_not_hold_a_value_to() -> None:
    with pytest.raises(TypeError, match="is not a constraint the checker reads"):
        json_schema(_holding(Annotated[tuple[int, ...], MinLen(1)]))


_PROPERTY = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])

_MARKERS: dict[str, Callable[[int], object]] = {"ge": Ge, "gt": Gt, "le": Le, "lt": Lt}
_LOW = st.none() | st.tuples(st.sampled_from(("ge", "gt")), st.integers(-3, 3))
_HIGH = st.none() | st.tuples(st.sampled_from(("le", "lt")), st.integers(-3, 3))
_LAYERS = st.lists(
    st.tuples(st.sampled_from(("annotated", "newtype", "alias")), _LOW, _HIGH),
    min_size=1,
    max_size=3,
)
_NUMBERS = st.integers(-5, 5) | st.floats(-5, 5, allow_nan=False) | st.sampled_from((True, "1"))


def _integral_as_int(value: object) -> object:
    """`value` as JSON Schema reads a number: `1.0` is the integer 1."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


@_PROPERTY
@given(base=st.sampled_from((int, float)), layers=_LAYERS, values=st.lists(_NUMBERS, max_size=8))
def test_bounds_in_layers_are_written_as_check_holds_a_value_to_them(
    base: type, layers: list[tuple[str, object, object]], values: list[object]
) -> None:
    # A number's bounds, on it, on a `NewType` of it, on an alias of it,
    # layer on layer: the schema accepts what `check` does, a number with no
    # fraction read as the integer it equals.
    annotation: object = base
    for index, (how, low, high) in enumerate(layers):
        sides = [cast("tuple[str, int]", side) for side in (low, high) if side is not None]
        bounds = [_MARKERS[name](bound) for name, bound in sides]
        bounded = _annotated[(annotation, *bounds)] if len(bounds) != 0 else annotation
        if how == "newtype":
            annotation = _new_type(f"Bounded{index}", bounded)
        elif how == "alias":
            annotation = _alias_type(f"Bounded{index}", bounded)
        else:
            annotation = bounded
    holder = _holding(annotation)
    try:
        schema = json_schema(holder)
    except TypeError:
        # A second bound from one side, which neither reads.
        event("refused")
        with pytest.raises(TypeError):
            check({}, holder)
        return
    validator = Draft202012Validator(schema)
    for value in values:
        accepted = check({"value": _integral_as_int(value)}, holder)[1] == ()
        event("accepted" if accepted else "refused a value")
        assert validator.is_valid(cast("Any", {"value": value})) == accepted, (value, schema)


# --- values changed in one place --------------------------------------------

_GONE = object()
_NAMES = ("r16", "r16\n", "r*", "gzip", "bytes", "crc32c", "int8", "regular", "default", "acme.x")
_KEYS = ("name", "configuration", "must_understand", "level", "x")
_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(-3, 300)
    | st.floats(-3, 3, allow_nan=False)
    | st.sampled_from(_NAMES)
    | st.text(max_size=3)
)
_VALUES = st.recursive(
    _SCALARS,
    lambda inner: (
        st.lists(inner, max_size=2)
        | st.dictionaries(st.sampled_from(_KEYS) | st.text(max_size=2), inner, max_size=2)
    ),
    max_leaves=4,
)


def _places(value: object, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """Every place in `value`: itself, and each member or entry inside it, depth first."""
    found = [path]
    if isinstance(value, dict):
        for key, entry in cast("dict[str, object]", value).items():
            found += _places(entry, (*path, key))
    elif isinstance(value, list):
        for index, entry in enumerate(cast("list[object]", value)):
            found += _places(entry, (*path, index))
    return found


def _at(value: object, path: tuple[str | int, ...]) -> object:
    for step in path:
        value = cast("Any", value)[step]
    return value


def _put(value: object, path: tuple[str | int, ...], new: object) -> object:
    """`value` with what sits at `path` replaced by `new`, or removed when `new` is `_GONE`."""
    if len(path) == 0:
        return new
    step, rest = path[0], path[1:]
    copy: Any = (
        dict(cast("dict[str, object]", value))
        if isinstance(value, dict)
        else list(cast("list[object]", value))
    )
    if len(rest) == 0 and new is _GONE:
        del copy[step]
    else:
        copy[step] = _put(copy[step], rest, new)
    return copy


@st.composite
def _changed(draw: st.DrawFn, value: object) -> object:
    """`value` changed in one or two places: something replaced, dropped or added, a string with a character more, or a field renamed."""
    for _ in range(draw(st.integers(1, 2))):
        path = draw(st.sampled_from(_places(value)))
        here = _at(value, path)
        how = draw(st.sampled_from(("replace", "drop", "add", "extend", "rename")))
        if how == "drop" and len(path) != 0:
            value = _put(value, path, _GONE)
        elif how == "add" and isinstance(here, dict):
            members = cast("dict[str, object]", here)
            value = _put(value, path, {**members, draw(st.sampled_from(_KEYS)): draw(_VALUES)})
        elif how == "add" and isinstance(here, list):
            value = _put(value, path, [*cast("list[object]", here), draw(_VALUES)])
        elif how == "extend" and isinstance(here, str):
            value = _put(value, path, here + draw(st.sampled_from(("\n", " ", "0", "x"))))
        elif how == "rename" and isinstance(here, dict) and "name" in here:
            value = _put(value, (*path, "name"), draw(st.sampled_from(_NAMES)))
        elif how == "rename" and isinstance(here, str):
            value = _put(value, path, draw(st.sampled_from(_NAMES)))
        else:
            value = _put(value, path, draw(_VALUES))
    return value


_SCOPES = (CORE, CORE_AND_EXTENSIONS, Context.of())
_FIELD_BASES: tuple[tuple[str, object], ...] = (
    *CASES,
    # Names nothing claims, whose configuration nothing judges.
    ("codecs:acme.x", {"name": "acme.x", "configuration": {"x": [1]}}),
    ("data_type:acme.t", {"name": "acme.t", "configuration": {"bits": 8}}),
    ("chunk_grid:acme.g", {"name": "acme.g", "configuration": {"chunk_shape": [0]}}),
)
_FIELD_SCHEMAS: dict[tuple[type[Definition[Any]], int], Draft202012Validator] = {}


def _field_validator(kind: type[Definition[Any]], scope: int) -> Draft202012Validator:
    if (kind, scope) not in _FIELD_SCHEMAS:
        schema = field_json_schema(kind, _SCOPES[scope])
        _FIELD_SCHEMAS[(kind, scope)] = Draft202012Validator(schema)
    return _FIELD_SCHEMAS[(kind, scope)]


# --- a field in a scope -----------------------------------------------------

_INDEXED = {"chunk_shape": [2], "codecs": ["bytes"]}

# Each field, the kind and scope it is read in, whether the schema accepts
# it, and whether the scope reads it without a problem. Where the two
# differ, a rule found what the schema cannot say.
FIELDS: list[tuple[str, object, type[Definition[Any]], Context, bool, bool]] = [
    ("object", {"name": "gzip", "configuration": {"level": 5}}, CodecDefinition, CORE, True, True),
    (
        "out-of-bounds",
        {"name": "gzip", "configuration": {"level": 12}},
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    (
        "unknown-key",
        {"name": "gzip", "configuration": {"level": 5, "window": 15}},
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    ("bare-name-needing-a-configuration", "gzip", CodecDefinition, CORE, False, False),
    ("bare-name", "bytes", CodecDefinition, CORE, True, True),
    (
        "understood",
        {"name": "gzip", "configuration": {"level": 5}, "must_understand": True},
        CodecDefinition,
        CORE,
        True,
        True,
    ),
    (
        "not-understood",
        {"name": "gzip", "configuration": {"level": 5}, "must_understand": False},
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    (
        "stray-member",
        {"name": "gzip", "configuration": {"level": 5}, "version": 2},
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    (
        "unclaimed",
        {"name": "zfpy", "configuration": {"mode": 4}},
        CodecDefinition,
        CORE,
        True,
        True,
    ),
    ("unclaimed-bare", "zfpy", CodecDefinition, CORE, True, True),
    (
        "unclaimed-not-understood",
        {"name": "zfpy", "must_understand": False},
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    # Nothing in an empty scope claims gzip, so nothing judges it.
    (
        "out-of-scope",
        {"name": "gzip", "configuration": {"level": 12}},
        CodecDefinition,
        Context.of(),
        True,
        True,
    ),
    (
        "a-rule",
        {
            "name": "blosc",
            "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle", "blocksize": 0},
        },
        CodecDefinition,
        CORE,
        True,
        False,
    ),
    (
        "static-index-codec",
        {
            "name": "sharding_indexed",
            "configuration": {**_INDEXED, "index_codecs": ["bytes", "crc32c"]},
        },
        CodecDefinition,
        CORE,
        True,
        True,
    ),
    (
        "dynamic-index-codec",
        {
            "name": "sharding_indexed",
            "configuration": {
                **_INDEXED,
                "index_codecs": [{"name": "gzip", "configuration": {"level": 1}}],
            },
        },
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    (
        "unclaimed-index-codec",
        {
            "name": "sharding_indexed",
            "configuration": {**_INDEXED, "index_codecs": ["bytes", "acme.sum"]},
        },
        CodecDefinition,
        CORE,
        True,
        True,
    ),
    (
        "inner-codec-out-of-bounds",
        {
            "name": "sharding_indexed",
            "configuration": {
                **_INDEXED,
                "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 12}}],
                "index_codecs": ["bytes"],
            },
        },
        CodecDefinition,
        CORE,
        False,
        False,
    ),
    ("raw-bits", "r16", DataTypeDefinition, CORE, True, True),
    ("raw-bits-object", {"name": "r16", "configuration": {}}, DataTypeDefinition, CORE, True, True),
    # What a raw-bits name carries is not written beside it.
    (
        "raw-bits-configured",
        {"name": "r16", "configuration": {"bits": 16}},
        DataTypeDefinition,
        CORE,
        False,
        False,
    ),
    ("raw-bits-of-a-size-the-spec-refuses", "r12", DataTypeDefinition, CORE, True, False),
    ("raw-bits-notation", "r*", DataTypeDefinition, CORE, True, True),
    # Matched to the end of the name, as the package matches it, so a
    # validator that matches as Python does takes no final newline for it:
    # a name nothing claims, which any configuration goes with.
    (
        "raw-bits-and-a-newline",
        {"name": "r16\n", "configuration": {"x": 1}},
        DataTypeDefinition,
        CORE,
        True,
        True,
    ),
    (
        "struct-field-out-of-bounds",
        {
            "name": "struct",
            "configuration": {
                "fields": [
                    {
                        "name": "t",
                        "data_type": {
                            "name": "numpy.datetime64",
                            "configuration": {"unit": "s", "scale_factor": 0},
                        },
                    }
                ]
            },
        },
        DataTypeDefinition,
        CORE_AND_EXTENSIONS,
        False,
        False,
    ),
    (
        "grid-out-of-bounds",
        {"name": "regular", "configuration": {"chunk_shape": [0]}},
        ChunkGridDefinition,
        CORE,
        False,
        False,
    ),
    (
        "encoding",
        {"name": "v2", "configuration": {"separator": "/"}},
        ChunkKeyEncodingDefinition,
        CORE,
        True,
        True,
    ),
    ("transformer", {"name": "acme.cache"}, StorageTransformerDefinition, CORE, True, True),
]


@pytest.mark.parametrize(
    ("field", "kind", "scope", "accepted", "clean"),
    [case[1:] for case in FIELDS],
    ids=[case[0] for case in FIELDS],
)
def test_field_json_schema_says_what_the_scope_reads(
    field: object, kind: type[Definition[Any]], scope: Context, accepted: bool, clean: bool
) -> None:
    schema = field_json_schema(kind, scope)
    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(cast("Any", field)) == accepted
    assert (resolve(field, kind, scope)[1] == ()) == clean


@pytest.mark.parametrize(
    ("key", "field"), CASES, ids=[f"{key}:{index}" for index, (key, _) in enumerate(CASES)]
)
def test_every_example_field_is_one_its_schema_accepts(key: str, field: object) -> None:
    schema = field_json_schema(KINDS[key.split(":")[0]], CORE_AND_EXTENSIONS)
    assert Draft202012Validator(schema).is_valid(cast("Any", field))


@_PROPERTY
@given(st.data())
def test_a_field_read_without_a_problem_is_one_its_schema_accepts(data: st.DataObject) -> None:
    # Each example changed in one place, in one of three scopes: whatever
    # the scope still reads without a problem, the schema accepts.
    key, example = data.draw(st.sampled_from(_FIELD_BASES), label="example")
    kind = KINDS[key.split(":")[0]]
    scope = data.draw(st.sampled_from(range(len(_SCOPES))), label="scope")
    field = data.draw(_changed(json.loads(json.dumps(example))), label="field")
    read = resolve(field, kind, _SCOPES[scope])[1] == ()
    event("read without a problem" if read else "a problem")
    if read:
        assert _field_validator(kind, scope).is_valid(cast("Any", field))


def test_field_json_schema_refuses_what_is_not_a_kind() -> None:
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        field_json_schema(Definition, CORE)


# --- a zarr.json ------------------------------------------------------------

ARRAY: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4, 4],
    "data_type": "int8",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
    "chunk_key_encoding": {"name": "default"},
    "fill_value": 0,
    "codecs": [{"name": "bytes"}, {"name": "gzip", "configuration": {"level": 5}}],
}
FLOATS: dict[str, Any] = {
    **ARRAY,
    "data_type": "float32",
    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
}
GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group"}


def _consolidated(**metadata: object) -> dict[str, Any]:
    return {
        **GROUP,
        "consolidated_metadata": {"kind": "inline", "must_understand": False, "metadata": metadata},
    }


# Each document, whether the schema accepts it, and whether
# `validate_node_metadata_v3` finds nothing wrong with it.
DOCUMENTS: list[tuple[str, dict[str, Any], bool, bool]] = [
    ("array", ARRAY, True, True),
    ("group", {**GROUP, "attributes": {"a": [1, None]}}, True, True),
    ("fill-value-out-of-range", {**ARRAY, "fill_value": 300}, False, False),
    ("fill-value-of-the-wrong-type", {**ARRAY, "fill_value": "0"}, False, False),
    ("float-fill-value", {**FLOATS, "fill_value": "NaN"}, True, True),
    ("hex-fill-value", {**FLOATS, "fill_value": "0x7fc00000"}, True, True),
    # A string that is no float's is a rule's to find.
    ("fill-value-a-rule-refuses", {**FLOATS, "fill_value": "nan"}, True, False),
    ("unclaimed-data-type", {**ARRAY, "data_type": "acme.int7", "fill_value": "any"}, True, True),
    (
        "a-name-raw-bits-and-a-newline",
        {**ARRAY, "data_type": "r16\n", "fill_value": "any"},
        True,
        True,
    ),
    ("raw-bits", {**ARRAY, "data_type": "r16", "fill_value": [0, 255]}, True, True),
    (
        "raw-bits-byte-out-of-range",
        {**ARRAY, "data_type": "r16", "fill_value": [0, 256]},
        False,
        False,
    ),
    (
        "codec-not-understood",
        {**ARRAY, "codecs": [{"name": "bytes", "must_understand": False}]},
        False,
        False,
    ),
    (
        "codec-out-of-bounds",
        {**ARRAY, "codecs": [{"name": "bytes"}, {"name": "gzip", "configuration": {"level": 12}}]},
        False,
        False,
    ),
    ("negative-shape", {**ARRAY, "shape": [-1, 4]}, False, False),
    ("wrong-format", {**ARRAY, "zarr_format": 2}, False, False),
    (
        "no-node-type",
        {key: value for key, value in ARRAY.items() if key != "node_type"},
        False,
        False,
    ),
    ("an-extension-field", {**ARRAY, "acme": {"must_understand": False}}, True, True),
    # What members read together say is the rules'.
    ("names-for-another-rank", {**ARRAY, "dimension_names": ["x"]}, True, False),
    (
        "codecs-out-of-order",
        {**ARRAY, "codecs": [{"name": "gzip", "configuration": {"level": 5}}, "bytes"]},
        True,
        False,
    ),
    ("consolidated", _consolidated(a=ARRAY, b=GROUP), True, True),
    ("consolidated-null", {**GROUP, "consolidated_metadata": None}, True, True),
    ("consolidated-bad-array", _consolidated(a={**ARRAY, "fill_value": 300}), False, False),
    (
        "consolidated-within-consolidated",
        _consolidated(b=_consolidated(c={**ARRAY, "fill_value": 300})),
        False,
        False,
    ),
    (
        "consolidated-kind",
        {
            **GROUP,
            "consolidated_metadata": {"kind": "sidecar", "must_understand": False, "metadata": {}},
        },
        False,
        False,
    ),
]

NODE_SCHEMA = node_metadata_json_schema_v3()


@pytest.mark.parametrize(
    ("document", "accepted", "clean"),
    [case[1:] for case in DOCUMENTS],
    ids=[case[0] for case in DOCUMENTS],
)
def test_node_metadata_json_schema_says_what_a_zarr_json_holds(
    document: dict[str, Any], accepted: bool, clean: bool
) -> None:
    assert Draft202012Validator(NODE_SCHEMA).is_valid(document) == accepted
    assert (validate_node_metadata_v3(document) == ()) == clean


_BASES: tuple[dict[str, Any], ...] = (
    ARRAY,
    FLOATS,
    {**ARRAY, "data_type": "r16", "fill_value": [0, 0]},
    {
        **ARRAY,
        "codecs": [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [1, 1],
                    "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 1}}],
                    "index_codecs": ["bytes", "crc32c"],
                },
            }
        ],
    },
    _consolidated(a=ARRAY, b=GROUP),
)
_NODE_SCHEMAS: dict[int, Draft202012Validator] = {}


def _node_validator(scope: int) -> Draft202012Validator:
    if scope not in _NODE_SCHEMAS:
        _NODE_SCHEMAS[scope] = Draft202012Validator(
            node_metadata_json_schema_v3(context=_SCOPES[scope])
        )
    return _NODE_SCHEMAS[scope]


@_PROPERTY
@given(st.data())
def test_a_document_read_without_a_problem_is_one_its_schema_accepts(data: st.DataObject) -> None:
    # Each document changed in one place, in one of three scopes: whatever
    # `validate_node_metadata_v3` finds nothing wrong with, the schema accepts.
    base = data.draw(st.sampled_from(_BASES), label="base")
    scope = data.draw(st.sampled_from(range(len(_SCOPES))), label="scope")
    document = data.draw(_changed(json.loads(json.dumps(base))), label="document")
    read = validate_node_metadata_v3(document, context=_SCOPES[scope]) == ()
    event("read without a problem" if read else "a problem")
    if read:
        assert _node_validator(scope).is_valid(cast("Any", document))


@pytest.mark.parametrize(
    "scope",
    [CORE, CORE_AND_EXTENSIONS, Context.of()],
    ids=["core", "core-and-extensions", "empty"],
)
def test_every_schema_is_a_json_schema_written_the_same_each_time(scope: Context) -> None:
    schema = node_metadata_json_schema_v3(context=scope)
    Draft202012Validator.check_schema(schema)
    assert json.loads(json.dumps(schema)) == schema
    assert schema == node_metadata_json_schema_v3(context=scope)
    defs = cast("dict[str, JSONValue]", schema["$defs"])
    assert {"ZarrV3ArrayMetadataJSON", "ZarrV3GroupMetadataJSON"} <= defs.keys()
    for kind in (
        CodecDefinition,
        DataTypeDefinition,
        ChunkGridDefinition,
        ChunkKeyEncodingDefinition,
        StorageTransformerDefinition,
    ):
        Draft202012Validator.check_schema(field_json_schema(kind, scope))
