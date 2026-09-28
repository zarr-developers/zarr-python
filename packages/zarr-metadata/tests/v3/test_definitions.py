"""One metadata field, read against its definition: checked, judged, and read in a scope."""

from __future__ import annotations

import copy
import math
import pickle
from collections.abc import (
    Mapping,  # noqa: TC003 - a TypedDict's annotations are evaluated at run time
)
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Final, NotRequired, cast

import pytest
from annotated_types import Ge, Predicate
from typing_extensions import TypedDict

from zarr_metadata.model import validate_array_metadata_v3
from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC, GzipCodecConfiguration
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    CodecField,
    Context,
    DataTypeDefinition,
    Definition,
    EmptyConfiguration,
    JSONValue,
    Nested,
    Read,
    Refused,
    StorageTransformerDefinition,
    Unclaimed,
    ValidationProblem,
    ZarrV3MetadataFieldJSON,
    check,
    configuration_of,
    resolve,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from decimal import Decimal


class AcmeStackConfiguration(TypedDict, closed=True):
    """A third party's container: a codec holding a pipeline of codecs."""

    codecs: tuple[CodecField, ...]


def acme_stack_rules(
    configuration: AcmeStackConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    # A rule may read a nested field's name: it is asked only of a
    # configuration whose nested envelopes are sound.
    for index, codec in enumerate(configuration["codecs"]):
        if (codec if isinstance(codec, str) else codec["name"]) == "acme.stack":
            yield ValidationProblem(
                ("codecs", index), "a stack does not hold itself", "invalid_value"
            )
        # And what the scope read of it: a codec of another kind is refused,
        # one nothing in scope claims left be.
        inner = nested.get(("codecs", index))
        definition = inner.definition if isinstance(inner, (Read, Refused)) else None
        if isinstance(definition, CodecDefinition) and definition.kind != "bytes_bytes":
            yield ValidationProblem(
                ("codecs", index), "a stack holds bytes -> bytes codecs", "invalid_value"
            )


ACME_STACK = CodecDefinition(
    name="acme.stack",
    configuration=AcmeStackConfiguration,
    kind="bytes_bytes",
    size="dynamic",
    rules=acme_stack_rules,
)


class AcmeLevelConfiguration(TypedDict, closed=True):
    """Under postponed annotations, as in this module, `NotRequired` is written as a string."""

    level: NotRequired[int]


class AcmeTotalConfiguration(TypedDict, total=False, closed=True):
    level: int


class AcmeTreeConfiguration(TypedDict, closed=True):
    """A configuration that holds itself, as JSON can to any depth."""

    label: str
    children: tuple[AcmeTreeConfiguration, ...]


class ByCodec(TypedDict, closed=True):
    codec: CodecField
    weight: int


class Raw(TypedDict, closed=True):
    codec: Mapping[str, JSONValue]
    note: str


class AcmeFallbackConfiguration(TypedDict, closed=True):
    """A union whose branches disagree on whether `codec` is a metadata field."""

    fallback: ByCodec | Raw


class AcmeRoutesConfiguration(TypedDict, extra_items=CodecField):
    """Every key a route, holding the codec that takes it."""


ACME_BOUNDS: Final = {"level": (0, 9), "window": (9, 15)}


class AcmeBoundedConfiguration(TypedDict, closed=True):
    level: int
    window: NotRequired[int]


def acme_bounded_rules(
    configuration: AcmeBoundedConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    # Every key it meets is one its TypedDict declares.
    for key, value in cast("Mapping[str, int]", configuration).items():
        low, high = ACME_BOUNDS[key]
        if not low <= value <= high:
            yield ValidationProblem((key,), f"expected {low} to {high}", "invalid_value")


class AcmePairedConfiguration(TypedDict, closed=True):
    first: NotRequired[int]
    second: NotRequired[int]


def acme_paired_rules(
    configuration: AcmePairedConfiguration, nested: Nested
) -> Iterator[ValidationProblem]:
    if ("first" in configuration) != ("second" in configuration):
        yield ValidationProblem((), "expected first and second, or neither", "invalid_value")


ACME_LEVEL = CodecDefinition(
    name="acme.level", configuration=AcmeLevelConfiguration, kind="bytes_bytes", size="dynamic"
)
ACME_TOTAL = CodecDefinition(
    name="acme.total", configuration=AcmeTotalConfiguration, kind="bytes_bytes", size="dynamic"
)
ACME_TREE = CodecDefinition(
    name="acme.tree", configuration=AcmeTreeConfiguration, kind="bytes_bytes", size="dynamic"
)
ACME_FALLBACK = CodecDefinition(
    name="acme.fallback",
    configuration=AcmeFallbackConfiguration,
    kind="bytes_bytes",
    size="dynamic",
)
ACME_ROUTES = CodecDefinition(
    name="acme.routes", configuration=AcmeRoutesConfiguration, kind="bytes_bytes", size="dynamic"
)
ACME_BOUNDED = CodecDefinition(
    name="acme.bounded",
    configuration=AcmeBoundedConfiguration,
    kind="bytes_bytes",
    size="dynamic",
    rules=acme_bounded_rules,
)
ACME_PAIRED = CodecDefinition(
    name="acme.paired",
    configuration=AcmePairedConfiguration,
    kind="bytes_bytes",
    size="dynamic",
    rules=acme_paired_rules,
)

SCOPE = CORE_AND_EXTENSIONS.extended_with(
    ACME_STACK,
    ACME_LEVEL,
    ACME_TOTAL,
    ACME_TREE,
    ACME_FALLBACK,
    ACME_ROUTES,
    ACME_BOUNDED,
    ACME_PAIRED,
)


INT8: Final = Read(json="int8", name="int8", definition=INT8_DATA_TYPE, configuration={})
"""An `int8` field as a scope reads it: by its definition, holding nothing inside."""


def _locs(problems: tuple[ValidationProblem, ...]) -> list[tuple[tuple[str | int, ...], str]]:
    return [(found.loc, found.kind) for found in problems]


def test_core_is_a_subset_of_core_and_extensions() -> None:
    assert set(CORE.definitions()) <= set(CORE_AND_EXTENSIONS.definitions())


def test_a_read_field_keeps_the_fields_it_read_inside() -> None:
    # By where each sits in the configuration, as the scope read it.
    shard = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": [2],
            "codecs": ["bytes"],
            "index_codecs": ["bytes", "crc32c"],
        },
    }
    resolved, _ = resolve(shard, CodecDefinition, CORE_AND_EXTENSIONS)
    assert isinstance(resolved, Read)
    assert {loc: inner.json for loc, inner in resolved.nested.items()} == {
        ("codecs", 0): "bytes",
        ("index_codecs", 0): "bytes",
        ("index_codecs", 1): "crc32c",
    }
    cast_value = {"name": "cast_value", "configuration": {"data_type": "int8"}}
    resolved, _ = resolve(cast_value, CodecDefinition, CORE_AND_EXTENSIONS)
    assert isinstance(resolved, Read)
    assert resolved.nested[("data_type",)] == INT8
    # A field holding none has nothing inside, and one nothing in scope
    # claims holds no field at all; one its check or rules refuse keeps
    # what it read.
    assert resolve("int8", DataTypeDefinition, CORE)[0] == INT8
    unclaimed = {"name": "acme.cast", "configuration": {"data_type": "int8"}}
    assert isinstance(resolve(unclaimed, CodecDefinition, CORE_AND_EXTENSIONS)[0], Unclaimed)
    refused = {"name": "cast_value", "configuration": {"data_type": "int8", "rounding": 1}}
    resolved, _ = resolve(refused, CodecDefinition, CORE_AND_EXTENSIONS)
    assert isinstance(resolved, Refused)
    assert resolved.nested[("data_type",)] == INT8


@pytest.mark.parametrize(
    ("field", "kind", "name"),
    [
        ("int8", DataTypeDefinition, "int8"),
        ({"name": "gzip", "configuration": {"level": 1}}, CodecDefinition, "gzip"),
        # Raw bits: the name written, not the one its definition is filed under.
        ("r16", DataTypeDefinition, "r16"),
        ({"name": "acme.codec"}, CodecDefinition, "acme.codec"),
        ({"configuration": {}}, CodecDefinition, None),
        (5, CodecDefinition, None),
    ],
    ids=["bare-name", "object", "raw-bits", "out-of-scope", "no-name", "not-a-field"],
)
def test_a_reading_says_the_name_the_field_was_written_with(
    field: JSONValue, kind: type[Definition[Any]], name: str | None
) -> None:
    assert resolve(field, kind, CORE_AND_EXTENSIONS)[0].name == name


KINDLESS: Final = Definition(name="acme.kindless", configuration=Empty)
"""A definition of no kind, which no scope files."""


@pytest.mark.parametrize(
    ("built", "read_as"),
    [
        (INT8, DataTypeDefinition),
        # A name read by the definition filed under another, as raw bits are.
        (
            Read(
                json="r16", name="r16", definition=RAW_BYTES_DATA_TYPE, configuration={"bits": 16}
            ),
            DataTypeDefinition,
        ),
        # Type arguments dropped, as `resolve` drops them.
        (
            Unclaimed(json="acme.t", name="acme.t", read_as=DataTypeDefinition[Any]),
            DataTypeDefinition,
        ),
        (
            Refused(
                json="int8",
                name="int8",
                read_as=DataTypeDefinition[Any],
                definition=INT8_DATA_TYPE,
            ),
            DataTypeDefinition,
        ),
        # Not a field, so named nothing and claimed by nothing.
        (Refused(json=5, name=None, read_as=CodecDefinition), CodecDefinition),
    ],
    ids=["read", "raw-bits", "unclaimed", "refused", "refused-nameless"],
)
def test_a_field_built_by_hand_is_of_the_kind_it_says(
    built: Read[Any] | Unclaimed | Refused[Any], read_as: type[Definition[Any]]
) -> None:
    assert built.read_as is read_as


@pytest.mark.parametrize("definition", [None, KINDLESS], ids=["none", "kindless"])
def test_error_a_field_read_by_hand_by_what_is_not_a_definition_of_a_kind(
    definition: Definition[Any] | None,
) -> None:
    with pytest.raises(TypeError, match="a field read is read by a definition of a kind"):
        Read(
            json="acme.kindless",
            name="acme.kindless",
            definition=definition,  # pyright: ignore[reportArgumentType]
            configuration={},
        )


@pytest.mark.parametrize("name", ["int16", "r16"])
def test_error_a_field_read_by_hand_by_a_definition_filed_under_another_name(name: str) -> None:
    with pytest.raises(
        TypeError, match=f"a field named '{name}' is read by the definition filed under it"
    ):
        Read(json=name, name=name, definition=INT8_DATA_TYPE, configuration={})


def test_error_a_field_refused_by_hand_by_a_definition_of_another_kind() -> None:
    with pytest.raises(
        TypeError, match="read as a DataTypeDefinition is read by one, got CodecDefinition"
    ):
        Refused(json="gzip", name="gzip", read_as=DataTypeDefinition, definition=GZIP_CODEC)


@pytest.mark.parametrize("name", ["int16", None])
def test_error_a_field_refused_by_hand_by_a_definition_filed_under_another_name(
    name: str | None,
) -> None:
    with pytest.raises(
        TypeError, match=f"a field named {name!r} is read by the definition filed under it"
    ):
        Refused(json=name, name=name, read_as=DataTypeDefinition, definition=INT8_DATA_TYPE)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Unclaimed(json="acme.t", name="acme.t", read_as=Definition),
        lambda: Refused(json="acme.t", name="acme.t", read_as=Definition),
    ],
    ids=["unclaimed", "refused"],
)
def test_error_a_field_built_by_hand_of_no_kind(build: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        build()


def test_error_a_field_nothing_claims_built_by_hand_without_a_name() -> None:
    with pytest.raises(TypeError, match="a field nothing in scope claims is named"):
        Unclaimed(json=5, name=None, read_as=CodecDefinition)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("field", "kind", "written"),
    [
        # A data type with nothing to configure by its bare name, however
        # it was written; every other field an object.
        ("int8", DataTypeDefinition, "int8"),
        (
            {"name": "int8", "configuration": {}, "must_understand": True},
            DataTypeDefinition,
            "int8",
        ),
        ("crc32c", CodecDefinition, {"name": "crc32c"}),
        ("default", ChunkKeyEncodingDefinition, {"name": "default"}),
        (
            {"name": "regular", "configuration": {"chunk_shape": [2, 3]}},
            ChunkGridDefinition,
            {"name": "regular", "configuration": {"chunk_shape": (2, 3)}},
        ),
        # Each field a configuration holds written the same way.
        (
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [2],
                    "codecs": ["bytes"],
                    "index_codecs": ["bytes", {"name": "crc32c", "configuration": {}}],
                },
            },
            CodecDefinition,
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": (2,),
                    "codecs": ({"name": "bytes"},),
                    "index_codecs": ({"name": "bytes"}, {"name": "crc32c"}),
                },
            },
        ),
        (
            {"name": "cast_value", "configuration": {"data_type": {"name": "int8"}}},
            CodecDefinition,
            {"name": "cast_value", "configuration": {"data_type": "int8"}},
        ),
        # One nothing in scope claims inside it too; one refused as written.
        (
            {
                "name": "acme.stack",
                "configuration": {
                    "codecs": [
                        "zfpy",
                        {"name": "gzip", "configuration": {"level": 12}, "must_understand": True},
                    ]
                },
            },
            CodecDefinition,
            {
                "name": "acme.stack",
                "configuration": {
                    "codecs": (
                        {"name": "zfpy"},
                        {"name": "gzip", "configuration": {"level": 12}, "must_understand": True},
                    )
                },
            },
        ),
        # A name that carries its configuration is written alone.
        ("r16", DataTypeDefinition, "r16"),
        ({"name": "r16", "configuration": {}}, DataTypeDefinition, "r16"),
        # Nothing in scope claims it: its configuration as written.
        ("zfpy", CodecDefinition, {"name": "zfpy"}),
        (
            {"name": "zfpy", "configuration": {"x": [1]}},
            CodecDefinition,
            {"name": "zfpy", "configuration": {"x": (1,)}},
        ),
        ({"name": "acme.t", "configuration": {}}, DataTypeDefinition, "acme.t"),
    ],
    ids=[
        "data-type",
        "data-type-spelled-out",
        "codec",
        "chunk-key-encoding",
        "configured",
        "nested",
        "nested-data-type",
        "nested-unclaimed-and-refused",
        "raw-bits",
        "raw-bits-spelled-out",
        "unclaimed-codec",
        "unclaimed-configured",
        "unclaimed-data-type",
    ],
)
def test_a_field_is_written_as_every_reader_takes_it(
    field: JSONValue, kind: type[Definition[Any]], written: JSONValue
) -> None:
    resolved, _ = resolve(field, kind, SCOPE)
    assert isinstance(resolved, (Read, Unclaimed))
    assert resolved.to_json() == written


@pytest.mark.parametrize(
    ("field", "kind", "variant", "configuration", "problems"),
    [
        (
            {"name": "gzip", "configuration": {"level": 5}},
            CodecDefinition,
            Read,
            {"level": 5},
            [],
        ),
        ("crc32c", CodecDefinition, Read, {}, []),
        ({"name": "crc32c"}, CodecDefinition, Read, {}, []),
        ({"name": "crc32c", "configuration": {}}, CodecDefinition, Read, {}, []),
        ({"name": "crc32c", "must_understand": True}, CodecDefinition, Read, {}, []),
        ("bytes", CodecDefinition, Read, {}, []),
        (
            {"name": "bytes", "configuration": {"endian": "big"}},
            CodecDefinition,
            Read,
            {"endian": "big"},
            [],
        ),
        (
            {"name": "regular", "configuration": {"chunk_shape": [2, 3]}},
            ChunkGridDefinition,
            Read,
            {"chunk_shape": (2, 3)},
            [],
        ),
        # An extent of 0 is right on a dimension of length 0, which only the
        # array's shape can tell.
        (
            {"name": "regular", "configuration": {"chunk_shape": [0, 3]}},
            ChunkGridDefinition,
            Read,
            {"chunk_shape": (0, 3)},
            [],
        ),
        # An unknown key is survivable: reported, left out of the
        # configuration, and the field still read.
        (
            {"name": "gzip", "configuration": {"level": 5, "extra": 1}},
            CodecDefinition,
            Read,
            {"level": 5},
            [(("configuration", "extra"), "unknown_key")],
        ),
        # Nothing the rules could meet but what the TypedDict declares.
        (
            {"name": "acme.bounded", "configuration": {"level": 5, "windw": 10}},
            CodecDefinition,
            Read,
            {"level": 5},
            [(("configuration", "windw"), "unknown_key")],
        ),
        # A key that is not required, written as a string under postponed
        # annotations, and every key of a `total=False` TypedDict.
        ("acme.level", CodecDefinition, Read, {}, []),
        ("acme.total", CodecDefinition, Read, {}, []),
        # A kind with type arguments is that kind.
        (
            {"name": "gzip", "configuration": {"level": 5}},
            CodecDefinition[Any],
            Read,
            {"level": 5},
            [],
        ),
        (
            {
                "name": "acme.tree",
                "configuration": {"label": "a", "children": [{"label": "b", "children": []}]},
            },
            CodecDefinition,
            Read,
            {"label": "a", "children": ({"label": "b", "children": ()},)},
            [],
        ),
        # Only the union branch that read has its nested fields read: here
        # `codec` is plain JSON, not a gzip missing its configuration.
        (
            {
                "name": "acme.fallback",
                "configuration": {"fallback": {"codec": {"name": "gzip"}, "note": "x"}},
            },
            CodecDefinition,
            Read,
            {"fallback": {"codec": {"name": "gzip"}, "note": "x"}},
            [],
        ),
        # `extra_items` of a field alias: every other key holds a codec,
        # held as a document writes it.
        (
            {"name": "acme.routes", "configuration": {"fast": "crc32c"}},
            CodecDefinition,
            Read,
            {"fast": {"name": "crc32c"}},
            [],
        ),
        # Nothing in scope claims it: left unjudged, not refused.
        ({"name": "zfpy", "configuration": {"x": 1}}, CodecDefinition, Unclaimed, None, []),
        # A nested field is read in the same scope, and held as a document
        # writes it; one out of scope is left be.
        (
            {"name": "acme.stack", "configuration": {"codecs": ["crc32c", "zfpy"]}},
            CodecDefinition,
            Read,
            {"codecs": ({"name": "crc32c"}, {"name": "zfpy"})},
            [],
        ),
    ],
    ids=[
        "gzip",
        "bare-name",
        "object-without-configuration",
        "empty-configuration",
        "must-understand",
        "bytes-bare",
        "bytes-endian",
        "regular-grid",
        "regular-grid-zero-extent",
        "unknown-key",
        "unknown-key-before-the-rules",
        "not-required-postponed",
        "total-false",
        "kind-with-type-arguments",
        "recursive",
        "union-branch-that-read",
        "extra-items-of-fields",
        "out-of-scope",
        "nested",
    ],
)
def test_a_field_is_read_in_scope(
    field: object,
    kind: type[Definition[Any]],
    variant: type,
    configuration: dict[str, object] | None,
    problems: list[tuple[tuple[str | int, ...], str]],
) -> None:
    resolved, found = resolve(field, kind, SCOPE)
    assert type(resolved) is variant
    assert (resolved.configuration if isinstance(resolved, Read) else None) == configuration
    assert _locs(found) == problems


def test_error_a_rule_refuses_a_value() -> None:
    resolved, found = resolve(
        {"name": "gzip", "configuration": {"level": 12}}, CodecDefinition, SCOPE
    )
    assert isinstance(resolved, Refused)
    assert resolved.definition is GZIP_CODEC
    assert _locs(found) == [(("configuration", "level"), "invalid_value")]


def test_error_a_member_of_the_wrong_type_is_not_asked_of_the_rules() -> None:
    resolved, found = resolve(
        {"name": "gzip", "configuration": {"level": "5"}}, CodecDefinition, SCOPE
    )
    assert isinstance(resolved, Refused)
    assert _locs(found) == [(("configuration", "level"), "invalid_type")]


def test_error_a_required_configuration_is_missing() -> None:
    resolved, found = resolve("gzip", CodecDefinition, SCOPE)
    assert isinstance(resolved, Refused)
    assert _locs(found) == [(("configuration",), "missing_key")]


def test_error_the_configuration_is_not_an_object() -> None:
    # Refused, and still claimed by the definition its name names.
    resolved, found = resolve({"name": "gzip", "configuration": 5}, CodecDefinition, SCOPE)
    assert resolved == Refused(
        json={"name": "gzip", "configuration": 5},
        name="gzip",
        read_as=CodecDefinition,
        definition=GZIP_CODEC,
    )
    assert [found.loc for found in found] == [("configuration",)]


def test_error_must_understand_false_is_refused() -> None:
    # The envelope's problem, reported with the field; the configuration
    # was read, so a later layer can still judge the codec.
    resolved, found = resolve({"name": "crc32c", "must_understand": False}, CodecDefinition, SCOPE)
    assert isinstance(resolved, Read)
    assert _locs(found) == [(("must_understand",), "invalid_value")]


@pytest.mark.parametrize(
    ("member", "kind"),
    [
        ("data_type", DataTypeDefinition),
        ("chunk_grid", ChunkGridDefinition),
        ("chunk_key_encoding", ChunkKeyEncodingDefinition),
        ("codecs", CodecDefinition),
        ("storage_transformers", StorageTransformerDefinition),
    ],
)
def test_error_must_understand_false_is_refused_wherever_the_model_refuses_it(
    member: str, kind: type[Definition[Any]]
) -> None:
    # One reading of the spec, applied by two readers: a field read here,
    # and the model's validator. If either changes alone, this fails.
    field: dict[str, JSONValue] = {"name": "acme.example", "must_understand": False}
    listed = member in ("codecs", "storage_transformers")
    document: dict[str, object] = dict(ZarrV3ArrayMetadata.create_default().to_json())
    document[member] = [field] if listed else field
    at = (member, 0) if listed else (member,)
    by_model = [
        (problem.loc[len(at) :], problem.kind)
        for problem in validate_array_metadata_v3(document)
        if problem.loc[: len(at)] == at
    ]
    assert _locs(resolve(field, kind, CORE_AND_EXTENSIONS)[1]) == by_model
    assert by_model == [(("must_understand",), "invalid_value")]


def _ruled_by(
    rules: Callable[[GzipCodecConfiguration, Nested], Iterable[ValidationProblem]],
) -> Context:
    lying = replace(GZIP_CODEC, name="acme.gzip", rules=rules)
    return CORE.extended_with(lying)


def test_error_a_rule_that_yields_something_else() -> None:
    def rules(configuration: GzipCodecConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
        yield "level is too high"  # pyright: ignore[reportReturnType]

    with pytest.raises(TypeError, match="'acme.gzip': its rules yield ValidationProblem values"):
        resolve(
            {"name": "acme.gzip", "configuration": {"level": 1}},
            CodecDefinition,
            _ruled_by(rules),
        )


def test_error_a_rule_that_raises_says_whose_it_is() -> None:
    def rules(configuration: GzipCodecConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
        yield from ({}[configuration["level"]],)

    with pytest.raises(KeyError) as raised:
        resolve(
            {"name": "acme.gzip", "configuration": {"level": 1}},
            CodecDefinition,
            _ruled_by(rules),
        )
    assert raised.value.__notes__ == [
        "raised by the rules of 'acme.gzip', reading ('configuration',)"
    ]


def test_error_a_rule_that_returns_none_says_whose_it_is() -> None:
    # A plain function that forgot to yield: nothing to iterate.
    def rules(configuration: GzipCodecConfiguration, nested: Nested) -> None:
        return None

    with pytest.raises(TypeError, match="not iterable") as raised:
        resolve(
            {"name": "acme.gzip", "configuration": {"level": 1}},
            CodecDefinition,
            _ruled_by(rules),  # pyright: ignore[reportArgumentType]
        )
    assert raised.value.__notes__ == [
        "raised by the rules of 'acme.gzip', reading ('configuration',)"
    ]


def test_error_null_is_not_a_field() -> None:
    # JSON's null is JSON: refined, it is None with no problem, which is
    # not a verdict. Read as a field or checked as a configuration, it is
    # a value of the wrong type.
    resolved, found = resolve(None, CodecDefinition, SCOPE)
    assert (resolved, _locs(found)) == (
        Refused(json=None, name=None, read_as=CodecDefinition),
        [((), "invalid_type")],
    )
    configuration, found = GZIP_CODEC.judge(None)
    assert (configuration, _locs(found)) == (None, [((), "invalid_type")])


def test_error_a_value_that_is_not_json() -> None:
    # Held as None, not JSON; its name still says what claims it.
    resolved, found = resolve(
        {"name": "gzip", "configuration": {"level": math.nan}}, CodecDefinition, SCOPE
    )
    assert resolved == Refused(
        json=None, name="gzip", read_as=CodecDefinition, definition=GZIP_CODEC
    )
    assert _locs(found) == [(("configuration", "level"), "invalid_value")]


def test_error_a_value_that_is_not_a_field() -> None:
    resolved, found = resolve(5, CodecDefinition, SCOPE)
    assert resolved == Refused(json=5, name=None, read_as=CodecDefinition)
    assert len(found) == 1


def test_error_a_regular_grid_extent_is_negative() -> None:
    resolved, found = resolve(
        {"name": "regular", "configuration": {"chunk_shape": [2, -1]}}, ChunkGridDefinition, SCOPE
    )
    assert isinstance(resolved, Refused)
    assert _locs(found) == [(("configuration", "chunk_shape", 1), "invalid_value")]


def test_error_a_nested_field_is_judged_where_it_sits() -> None:
    # Its problem is its own: the field holding it is read, as a document
    # holding it would be.
    field = {
        "name": "acme.stack",
        "configuration": {"codecs": ["crc32c", {"name": "gzip", "configuration": {"level": 12}}]},
    }
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert isinstance(resolved, Read)
    assert isinstance(resolved.nested[("codecs", 1)], Refused)
    assert _locs(found) == [
        (("configuration", "codecs", 1, "configuration", "level"), "invalid_value")
    ]


def test_error_a_nested_field_in_extra_items_is_judged_where_it_sits() -> None:
    field = {
        "name": "acme.routes",
        "configuration": {"slow": {"name": "gzip", "configuration": {"level": 12}}},
    }
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert isinstance(resolved, Read)
    assert isinstance(resolved.nested[("slow",)], Refused)
    assert _locs(found) == [(("configuration", "slow", "configuration", "level"), "invalid_value")]


def test_error_a_nested_envelope_s_problem_is_its_own_and_the_rules_are_asked() -> None:
    # A `must_understand` of false inside, as in a document, is reported
    # where it sits, and the stack is read.
    unread = {"name": "crc32c", "must_understand": False}
    resolved, found = resolve(
        {"name": "acme.stack", "configuration": {"codecs": [unread]}}, CodecDefinition, SCOPE
    )
    assert isinstance(resolved, Read)
    assert _locs(found) == [(("configuration", "codecs", 0, "must_understand"), "invalid_value")]
    # Its rules are asked all the same: one refuses the array -> bytes
    # codec it holds, which is the stack's own problem.
    resolved, found = resolve(
        {"name": "acme.stack", "configuration": {"codecs": [unread, "bytes"]}},
        CodecDefinition,
        SCOPE,
    )
    assert isinstance(resolved, Refused)
    assert _locs(found) == [
        (("configuration", "codecs", 1), "invalid_value"),
        (("configuration", "codecs", 0, "must_understand"), "invalid_value"),
    ]


def test_error_a_container_rule_is_not_asked_of_a_malformed_nested_field() -> None:
    # The stack's rule reads each nested field's name; one with no name is
    # reported where it sits, and the rule is not asked, as `judge` would not.
    field = {"name": "acme.stack", "configuration": {"codecs": [{"configuration": {}}]}}
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert isinstance(resolved, Refused)
    assert _locs(found) == [(("configuration", "codecs", 0, "name"), "invalid_type")]
    assert ACME_STACK.judge(field["configuration"])[0] is None


def test_error_a_rule_about_the_whole_configuration_lands_on_it() -> None:
    # A rule reports relative to the configuration: an empty location is
    # the configuration, judged alone or read in a field.
    _, judged = ACME_PAIRED.judge({"first": 1})
    _, read = resolve(
        {"name": "acme.paired", "configuration": {"first": 1}}, CodecDefinition, SCOPE
    )
    assert _locs(judged) == [((), "invalid_value")]
    assert _locs(read) == [(("configuration",), "invalid_value")]


def test_error_a_rule_reads_the_fields_the_configuration_holds_as_the_scope_read_them() -> None:
    # `bytes` is an array -> bytes codec, which the stack's rule refuses
    # from what the scope read; `judge` reads in no scope, so its rule sees
    # nothing read.
    field = {"name": "acme.stack", "configuration": {"codecs": ["crc32c", "bytes"]}}
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert isinstance(resolved, Refused)
    assert _locs(found) == [(("configuration", "codecs", 1), "invalid_value")]
    assert ACME_STACK.judge(field["configuration"])[1] == ()


def test_error_a_nested_member_is_not_a_field() -> None:
    resolved, found = resolve(
        {"name": "acme.stack", "configuration": {"codecs": [5]}}, CodecDefinition, SCOPE
    )
    assert isinstance(resolved, Refused)
    assert _locs(found) == [(("configuration", "codecs", 0), "invalid_type")]


def test_check_needs_nothing_but_the_value_and_a_typeddict() -> None:
    # The first step: JSON against a TypedDict. A nested field is an
    # envelope here, so no scope is needed and no name is judged.
    assert check({"level": 5}, GzipCodecConfiguration) == ({"level": 5}, ())
    stack, problems = check({"codecs": ["gzip", {"name": "anything"}]}, AcmeStackConfiguration)
    assert (stack, problems) == ({"codecs": ("gzip", {"name": "anything"})}, ())
    # The check and the reading agree on which union branch a value is.
    fallback = {"fallback": {"codec": {"name": "gzip"}, "note": "x"}}
    assert check(fallback, AcmeFallbackConfiguration) == (fallback, ())


def test_check_reads_a_whole_array_document() -> None:
    # A null in `dimension_names`, and a top-level key the document does
    # not declare, typed by its `extra_items`.
    document = {
        **ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json(),
        "dimension_names": [None],
        "acme": {"must_understand": False},
    }
    typed, problems = check(document, ZarrV3ArrayMetadataJSON)
    assert problems == ()
    assert typed is not None
    assert typed.get("dimension_names") == (None,)


def test_configuration_of_types_what_its_definition_read() -> None:
    resolved, _ = resolve({"name": "gzip", "configuration": {"level": 5}}, CodecDefinition, SCOPE)
    assert configuration_of(resolved, GZIP_CODEC) == {"level": 5}
    assert configuration_of(resolved, ACME_LEVEL) is None


def test_error_check_locates_what_is_not_the_typeddict() -> None:
    typed, problems = check({"level": "5", "extra": 1}, GzipCodecConfiguration)
    assert typed is None
    assert sorted(_locs(problems)) == [(("extra",), "unknown_key"), (("level",), "invalid_type")]


def test_error_check_judges_a_nested_envelope() -> None:
    typed, problems = check(
        {"codecs": [{"name": "gzip", "configuration": 3}]}, AcmeStackConfiguration
    )
    assert typed is None
    assert [found.loc for found in problems] == [("codecs", 0, "configuration")]


def test_judge_is_the_check_and_then_the_rules() -> None:
    # A caller holding one configuration: the rules are asked only of a
    # configuration that type-checked, so they never meet a wrong type.
    assert GZIP_CODEC.judge({"level": 5}) == ({"level": 5}, ())
    refused, problems = GZIP_CODEC.judge({"level": 12})
    assert (refused, _locs(problems)) == (None, [(("level",), "invalid_value")])
    mistyped, problems = GZIP_CODEC.judge({"level": "x"})
    assert (mistyped, _locs(problems)) == (None, [(("level",), "invalid_type")])


def test_a_scope_takes_a_name_over() -> None:
    strict = CodecDefinition(
        name="gzip",
        configuration=GzipCodecConfiguration,
        kind="bytes_bytes",
        size="dynamic",
        rules=lambda configuration, nested: (
            [ValidationProblem(("level",), "level 0 stores uncompressed", "invalid_value")]
            if configuration["level"] == 0
            else []
        ),
    )
    scope = CORE_AND_EXTENSIONS.extended_with(strict)
    _, found = resolve({"name": "gzip", "configuration": {"level": 0}}, CodecDefinition, scope)
    assert _locs(found) == [(("configuration", "level"), "invalid_value")]
    assert scope.claimant(ChunkGridDefinition, "regular") is REGULAR_CHUNK_GRID


def test_a_scope_is_shown_by_how_many_definitions_it_holds() -> None:
    # Short, as a validator's default argument is shown by `help`.
    assert repr(CORE) == f"Context(<{len(CORE.definitions())} definitions>)"


def test_a_scope_pickles_as_its_definitions_and_copies_as_itself() -> None:
    # A model holds the scope it was read in, and goes to another process with it.
    again = pickle.loads(pickle.dumps(CORE_AND_EXTENSIONS))
    assert again.definitions() == CORE_AND_EXTENSIONS.definitions()
    assert copy.copy(CORE) is CORE
    assert copy.deepcopy(CORE) is CORE


LE: Final = {"name": "bytes", "configuration": {"endian": "little"}}
SHARD: Final = {"chunk_shape": [2], "codecs": [LE], "index_location": "end"}


@pytest.mark.parametrize(
    ("one", "other", "kind", "equal"),
    [
        ("uint8", {"name": "uint8"}, DataTypeDefinition, True),
        ({"name": "bytes"}, {"name": "bytes", "configuration": {}}, CodecDefinition, True),
        (
            {"name": "gzip", "configuration": {"level": 1}},
            {"name": "gzip", "configuration": {"level": 1}, "must_understand": True},
            CodecDefinition,
            True,
        ),
        ("acme.codec", {"name": "acme.codec", "configuration": {}}, CodecDefinition, True),
        (
            {
                "name": "sharding_indexed",
                "configuration": {**SHARD, "index_codecs": [LE, "crc32c"]},
            },
            {
                "name": "sharding_indexed",
                "configuration": {**SHARD, "index_codecs": [LE, {"name": "crc32c"}]},
            },
            CodecDefinition,
            True,
        ),
        (
            {"name": "gzip", "configuration": {"level": 1}},
            {"name": "gzip", "configuration": {"level": 2}},
            CodecDefinition,
            False,
        ),
        (
            {"name": "acme.codec", "configuration": {"a": 1}},
            {"name": "acme.codec", "configuration": {"a": 2}},
            CodecDefinition,
            False,
        ),
        ("int8", "uint8", DataTypeDefinition, False),
    ],
    ids=[
        "bare-or-object",
        "no-or-empty-configuration",
        "must-understand-true-or-absent",
        "unclaimed-bare-or-object",
        "a-field-it-holds-bare-or-object",
        "another-configuration",
        "unclaimed-another-configuration",
        "another-name",
    ],
)
def test_two_fields_are_equal_when_they_read_the_same(
    one: JSONValue, other: JSONValue, kind: type[Definition[Any]], equal: bool
) -> None:
    """However each was spelled; each is written as it reads, and equal fields are written the same."""
    first, _ = resolve(one, kind, CORE_AND_EXTENSIONS)
    second, _ = resolve(other, kind, CORE_AND_EXTENSIONS)
    assert isinstance(first, (Read, Unclaimed))
    assert isinstance(second, (Read, Unclaimed))
    assert (first == second) is equal
    assert (first.to_json() == second.to_json()) is equal


def test_a_field_copied_or_pickled_is_read_by_a_definition_equal_to_its_own() -> None:
    field, _ = resolve({"name": "gzip", "configuration": {"level": 1}}, CodecDefinition, CORE)
    for again in (pickle.loads(pickle.dumps(field)), copy.deepcopy(field)):
        assert again == field
        assert configuration_of(again, GZIP_CODEC) == {"level": 1}


def test_a_definition_is_shown_by_its_kind_and_name() -> None:
    # Short, as a reading that holds it shows it.
    assert repr(GZIP_CODEC) == "CodecDefinition(name='gzip')"
    assert repr(INT8_DATA_TYPE) == "DataTypeDefinition(name='int8')"


class Unreadable(TypedDict, closed=True):
    members: set[int]


class Loose(TypedDict):
    level: int


class AcmePlainConfiguration(TypedDict, closed=True):
    inner: ZarrV3MetadataFieldJSON


class AcmeDecimal(TypedDict, closed=True):
    value: Decimal  # a name the type checker sees, and the running module does not


class AcmeUnresolvedConfiguration(TypedDict, closed=True):
    inner: AcmeDecimal


@dataclass(frozen=True, kw_only=True, slots=True)
class AcmeCodecDefinition(CodecDefinition[Any]):
    """A reader's own kind of codec, under which no scope files anything."""


def test_error_a_definition_configuration_is_a_typeddict() -> None:
    with pytest.raises(TypeError, match="give the TypedDict"):
        CodecDefinition(name="acme.bad", configuration=dict, kind="bytes_bytes", size="dynamic")


def test_error_a_definition_member_is_a_shape_json_takes() -> None:
    with pytest.raises(TypeError, match="Unreadable: members is not a shape JSON takes"):
        CodecDefinition(
            name="acme.bad", configuration=Unreadable, kind="bytes_bytes", size="dynamic"
        )


def test_error_a_configuration_says_what_its_other_keys_are() -> None:
    # Open by default, it would take a misspelled key without a word.
    with pytest.raises(TypeError, match="Loose says nothing of the keys it does not declare"):
        CodecDefinition(name="acme.loose", configuration=Loose, kind="bytes_bytes", size="dynamic")


def test_error_a_member_typed_as_plain_field_json_is_refused() -> None:
    # It checks as JSON, and its name would never be related to a definition.
    with pytest.raises(TypeError, match="annotate it with the field alias of its kind"):
        CodecDefinition(
            name="acme.plain",
            configuration=AcmePlainConfiguration,
            kind="bytes_bytes",
            size="dynamic",
        )


def test_error_a_configuration_whose_annotations_do_not_resolve() -> None:
    # Named down to the TypedDict that holds the annotation.
    with pytest.raises(TypeError, match="AcmeDecimal: name 'Decimal' is not defined"):
        CodecDefinition(
            name="acme.unresolved",
            configuration=AcmeUnresolvedConfiguration,
            kind="bytes_bytes",
            size="dynamic",
        )


class AcmeUncheckedConfiguration(TypedDict, closed=True):
    digits: Annotated[str, Predicate(str.isdigit)]


def test_error_a_configuration_with_a_constraint_the_checker_does_not_read() -> None:
    # A bound the checker did not hold a value to would say what is not so.
    with pytest.raises(
        TypeError,
        match=r"AcmeUncheckedConfiguration.digits: Predicate\(str.isdigit\) is not a constraint",
    ):
        CodecDefinition(
            name="acme.unchecked",
            configuration=AcmeUncheckedConfiguration,
            kind="bytes_bytes",
            size="dynamic",
        )


def test_error_a_fill_value_with_a_constraint_its_type_cannot_take() -> None:
    with pytest.raises(TypeError, match="fill_value: ge: a bound is on a number, and a string"):
        DataTypeDefinition(
            name="acme.bounded", configuration=EmptyConfiguration, fill_value=Annotated[str, Ge(0)]
        )


def test_error_a_definition_name_is_a_string() -> None:
    with pytest.raises(TypeError, match="a definition's name is a string"):
        CodecDefinition(name=5, configuration=Empty, kind="bytes_bytes", size="dynamic")  # pyright: ignore[reportArgumentType]


def test_error_a_codec_kind_is_one_of_three() -> None:
    with pytest.raises(TypeError, match="kind is one of"):
        CodecDefinition(name="acme.k", configuration=Empty, kind="bytes_to_array", size="dynamic")  # pyright: ignore[reportArgumentType]


def test_error_a_data_type_is_named_as_raw_bits_of_one_size_are_written() -> None:
    # `r16` reads as `r*`, so a definition filed under it would read nothing.
    with pytest.raises(TypeError, match="to read raw bits your own way, define 'r\\*'"):
        DataTypeDefinition(name="r16", configuration=Empty)


def test_error_a_data_type_fill_value_no_checker_reads() -> None:
    with pytest.raises(TypeError, match="'acme.set': fill_value: "):
        DataTypeDefinition(name="acme.set", configuration=Empty, fill_value=set[int])


@pytest.mark.parametrize(
    ("kind", "member"),
    [
        (DataTypeDefinition, "rules"),
        (DataTypeDefinition, "canonical"),
        (DataTypeDefinition, "fill_value_rules"),
        (DataTypeDefinition, "storage"),
        (ChunkGridDefinition, "shape_rules"),
        (ChunkGridDefinition, "chunk_lengths"),
        (CodecDefinition, "chunk_rules"),
        (CodecDefinition, "transition"),
        (CodecDefinition, "pipelines"),
    ],
)
def test_error_a_function_member_that_is_not_a_function(
    kind: type[Definition[Any]], member: str
) -> None:
    # Each member a definition's annotations declare a `Callable`.
    codec = {"kind": "array_array", "size": "static"} if kind is CodecDefinition else {}
    with pytest.raises(TypeError, match=f"'acme.t': {member} is a function, got 'none'"):
        kind(name="acme.t", configuration=Empty, **codec, **{member: "none"})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("kind", [Definition, AcmeCodecDefinition])
def test_error_a_field_is_read_as_a_kind_of_metadata(kind: type[Definition[Any]]) -> None:
    # Nothing is filed under either, so the field would go unjudged.
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        resolve({"name": "gzip", "configuration": {"level": 99}}, kind, SCOPE)


def test_error_a_scope_refuses_a_definition_of_no_kind() -> None:
    with pytest.raises(TypeError, match="a definition of no kind"):
        Context.of(Definition(name="acme.kindless", configuration=Empty))
