"""One metadata field, read against its definition: checked, judged, and read in a scope."""

from __future__ import annotations

import math
import re
from collections.abc import (
    Mapping,  # noqa: TC003 - a TypedDict's annotations are evaluated at run time
)
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final, NotRequired, cast

import pytest
from typing_extensions import TypedDict

from zarr_metadata.model import validate_array_metadata_v3
from zarr_metadata.model._array import ZarrV3ArrayMetadata
from zarr_metadata.v3.array import ZarrV3ArrayMetadataJSON
from zarr_metadata.v3.chunk_grid.regular import REGULAR_CHUNK_GRID
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC, GzipCodecConfiguration
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
    JSONValue,
    StorageTransformerDefinition,
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


def acme_stack_rules(configuration: AcmeStackConfiguration) -> Iterator[ValidationProblem]:
    # A rule may read a nested field's name: it is asked only of a
    # configuration whose nested envelopes are sound.
    for index, codec in enumerate(configuration["codecs"]):
        if (codec if isinstance(codec, str) else codec["name"]) == "acme.stack":
            yield ValidationProblem(
                ("codecs", index), "a stack does not hold itself", "invalid_value"
            )


ACME_STACK = CodecDefinition(
    name="acme.stack",
    configuration=AcmeStackConfiguration,
    kind="bytes_bytes",
    rules=acme_stack_rules,
)

ACME_FIXED = re.compile(r"acme\.fixed(\d+)")


def acme_fixed_name_rules(name: str) -> Iterator[ValidationProblem]:
    match = ACME_FIXED.fullmatch(name)
    if match is not None and int(match.group(1)) % 8 != 0:
        yield ValidationProblem((), "expected a width that is a multiple of 8", "invalid_value")


ACME_FIXED_TYPE = DataTypeDefinition(
    name="acme.fixed<N>",
    configuration=Empty,
    names=lambda name: ACME_FIXED.fullmatch(name) is not None,
    name_rules=acme_fixed_name_rules,
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


def acme_bounded_rules(configuration: AcmeBoundedConfiguration) -> Iterator[ValidationProblem]:
    # Every key it meets is one its TypedDict declares.
    for key, value in cast("Mapping[str, int]", configuration).items():
        low, high = ACME_BOUNDS[key]
        if not low <= value <= high:
            yield ValidationProblem((key,), f"expected {low} to {high}", "invalid_value")


class AcmePairedConfiguration(TypedDict, closed=True):
    first: NotRequired[int]
    second: NotRequired[int]


def acme_paired_rules(configuration: AcmePairedConfiguration) -> Iterator[ValidationProblem]:
    if ("first" in configuration) != ("second" in configuration):
        yield ValidationProblem((), "expected first and second, or neither", "invalid_value")


ACME_LEVEL = CodecDefinition(
    name="acme.level", configuration=AcmeLevelConfiguration, kind="bytes_bytes"
)
ACME_TOTAL = CodecDefinition(
    name="acme.total", configuration=AcmeTotalConfiguration, kind="bytes_bytes"
)
ACME_TREE = CodecDefinition(
    name="acme.tree", configuration=AcmeTreeConfiguration, kind="bytes_bytes"
)
ACME_FALLBACK = CodecDefinition(
    name="acme.fallback", configuration=AcmeFallbackConfiguration, kind="bytes_bytes"
)
ACME_ROUTES = CodecDefinition(
    name="acme.routes", configuration=AcmeRoutesConfiguration, kind="bytes_bytes"
)
ACME_BOUNDED = CodecDefinition(
    name="acme.bounded",
    configuration=AcmeBoundedConfiguration,
    kind="bytes_bytes",
    rules=acme_bounded_rules,
)
ACME_PAIRED = CodecDefinition(
    name="acme.paired",
    configuration=AcmePairedConfiguration,
    kind="bytes_bytes",
    rules=acme_paired_rules,
)

SCOPE = CORE_AND_EXTENSIONS.extended_with(
    ACME_STACK,
    ACME_FIXED_TYPE,
    ACME_LEVEL,
    ACME_TOTAL,
    ACME_TREE,
    ACME_FALLBACK,
    ACME_ROUTES,
    ACME_BOUNDED,
    ACME_PAIRED,
)


def _locs(problems: tuple[ValidationProblem, ...]) -> list[tuple[tuple[str | int, ...], str]]:
    return [(found.loc, found.kind) for found in problems]


def test_core_is_a_subset_of_core_and_extensions() -> None:
    assert set(CORE.definitions()) <= set(CORE_AND_EXTENSIONS.definitions())


@pytest.mark.parametrize(
    ("field", "kind", "resolution", "configuration", "problems"),
    [
        (
            {"name": "gzip", "configuration": {"level": 5}},
            CodecDefinition,
            "read",
            {"level": 5},
            [],
        ),
        ("crc32c", CodecDefinition, "read", {}, []),
        ({"name": "crc32c"}, CodecDefinition, "read", {}, []),
        ({"name": "crc32c", "configuration": {}}, CodecDefinition, "read", {}, []),
        ({"name": "crc32c", "must_understand": True}, CodecDefinition, "read", {}, []),
        ("bytes", CodecDefinition, "read", {}, []),
        (
            {"name": "bytes", "configuration": {"endian": "big"}},
            CodecDefinition,
            "read",
            {"endian": "big"},
            [],
        ),
        (
            {"name": "regular", "configuration": {"chunk_shape": [2, 3]}},
            ChunkGridDefinition,
            "read",
            {"chunk_shape": (2, 3)},
            [],
        ),
        # An extent of 0 is right on a dimension of length 0, which only the
        # array's shape can tell.
        (
            {"name": "regular", "configuration": {"chunk_shape": [0, 3]}},
            ChunkGridDefinition,
            "read",
            {"chunk_shape": (0, 3)},
            [],
        ),
        # An unknown key is survivable: reported, left out of the
        # configuration, and the field still read.
        (
            {"name": "gzip", "configuration": {"level": 5, "extra": 1}},
            CodecDefinition,
            "read",
            {"level": 5},
            [(("configuration", "extra"), "unknown_key")],
        ),
        # Nothing the rules could meet but what the TypedDict declares.
        (
            {"name": "acme.bounded", "configuration": {"level": 5, "windw": 10}},
            CodecDefinition,
            "read",
            {"level": 5},
            [(("configuration", "windw"), "unknown_key")],
        ),
        # A key that is not required, written as a string under postponed
        # annotations, and every key of a `total=False` TypedDict.
        ("acme.level", CodecDefinition, "read", {}, []),
        ("acme.total", CodecDefinition, "read", {}, []),
        # A kind with type arguments is that kind.
        (
            {"name": "gzip", "configuration": {"level": 5}},
            CodecDefinition[Any],
            "read",
            {"level": 5},
            [],
        ),
        (
            {
                "name": "acme.tree",
                "configuration": {"label": "a", "children": [{"label": "b", "children": []}]},
            },
            CodecDefinition,
            "read",
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
            "read",
            {"fallback": {"codec": {"name": "gzip"}, "note": "x"}},
            [],
        ),
        # `extra_items` of a field alias: every other key holds a codec.
        (
            {"name": "acme.routes", "configuration": {"fast": "crc32c"}},
            CodecDefinition,
            "read",
            {"fast": "crc32c"},
            [],
        ),
        # Nothing in scope claims it: left unjudged, not refused.
        ({"name": "zfpy", "configuration": {"x": 1}}, CodecDefinition, "out_of_scope", None, []),
        # A nested field is read in the same scope; one out of scope is left be.
        (
            {"name": "acme.stack", "configuration": {"codecs": ["crc32c", "zfpy"]}},
            CodecDefinition,
            "read",
            {"codecs": ("crc32c", "zfpy")},
            [],
        ),
        # A family claims many names.
        ("acme.fixed16", DataTypeDefinition, "read", {}, []),
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
        "family",
    ],
)
def test_a_field_is_read_in_scope(
    field: object,
    kind: type[Definition[Any]],
    resolution: str,
    configuration: dict[str, object] | None,
    problems: list[tuple[tuple[str | int, ...], str]],
) -> None:
    resolved, found = resolve(field, kind, SCOPE)
    assert resolved.resolution == resolution
    assert resolved.configuration == configuration
    assert _locs(found) == problems


def test_error_a_rule_refuses_a_value() -> None:
    resolved, found = resolve(
        {"name": "gzip", "configuration": {"level": 12}}, CodecDefinition, SCOPE
    )
    assert resolved.resolution == "invalid"
    assert resolved.configuration is None
    assert _locs(found) == [(("configuration", "level"), "invalid_value")]


def test_error_a_member_of_the_wrong_type_is_not_asked_of_the_rules() -> None:
    resolved, found = resolve(
        {"name": "gzip", "configuration": {"level": "5"}}, CodecDefinition, SCOPE
    )
    assert resolved.resolution == "invalid"
    assert _locs(found) == [(("configuration", "level"), "invalid_type")]


def test_error_a_required_configuration_is_missing() -> None:
    resolved, found = resolve("gzip", CodecDefinition, SCOPE)
    assert resolved.resolution == "invalid"
    assert _locs(found) == [(("configuration",), "missing_key")]


def test_error_the_configuration_is_not_an_object() -> None:
    resolved, found = resolve({"name": "gzip", "configuration": 5}, CodecDefinition, SCOPE)
    assert resolved.resolution == "invalid"
    assert [found.loc for found in found] == [("configuration",)]


def test_error_must_understand_false_is_refused() -> None:
    # The envelope's problem, reported with the field; the configuration
    # was read, so a later layer can still judge the codec.
    resolved, found = resolve({"name": "crc32c", "must_understand": False}, CodecDefinition, SCOPE)
    assert resolved.resolution == "read"
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
    rules: Callable[[GzipCodecConfiguration], Iterable[ValidationProblem]],
) -> Context:
    lying = replace(GZIP_CODEC, name="acme.gzip", rules=rules)
    return CORE.extended_with(lying)


def test_error_a_rule_that_yields_something_else() -> None:
    def rules(configuration: GzipCodecConfiguration) -> Iterator[ValidationProblem]:
        yield "level is too high"  # pyright: ignore[reportReturnType]

    with pytest.raises(TypeError, match="'acme.gzip': its rules yield ValidationProblem values"):
        resolve(
            {"name": "acme.gzip", "configuration": {"level": 1}},
            CodecDefinition,
            _ruled_by(rules),
        )


def test_error_a_rule_that_raises_says_whose_it_is() -> None:
    def rules(configuration: GzipCodecConfiguration) -> Iterator[ValidationProblem]:
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
    def rules(configuration: GzipCodecConfiguration) -> None:
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
    assert (resolved.resolution, _locs(found)) == ("invalid", [((), "invalid_type")])
    configuration, found = GZIP_CODEC.judge(None)
    assert (configuration, _locs(found)) == (None, [((), "invalid_type")])


def test_error_a_value_that_is_not_json() -> None:
    resolved, found = resolve(
        {"name": "gzip", "configuration": {"level": math.nan}}, CodecDefinition, SCOPE
    )
    assert (resolved.json, resolved.resolution) == (None, "invalid")
    assert _locs(found) == [(("configuration", "level"), "invalid_value")]


def test_error_a_value_that_is_not_a_field() -> None:
    resolved, found = resolve(5, CodecDefinition, SCOPE)
    assert resolved.resolution == "invalid"
    assert len(found) == 1


def test_error_a_regular_grid_extent_is_negative() -> None:
    resolved, found = resolve(
        {"name": "regular", "configuration": {"chunk_shape": [2, -1]}}, ChunkGridDefinition, SCOPE
    )
    assert resolved.resolution == "invalid"
    assert _locs(found) == [(("configuration", "chunk_shape", 1), "invalid_value")]


def test_error_a_nested_field_is_judged_where_it_sits() -> None:
    field = {
        "name": "acme.stack",
        "configuration": {"codecs": ["crc32c", {"name": "gzip", "configuration": {"level": 12}}]},
    }
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert resolved.resolution == "invalid"
    assert _locs(found) == [
        (("configuration", "codecs", 1, "configuration", "level"), "invalid_value")
    ]


def test_error_a_nested_field_in_extra_items_is_judged_where_it_sits() -> None:
    field = {
        "name": "acme.routes",
        "configuration": {"slow": {"name": "gzip", "configuration": {"level": 12}}},
    }
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert resolved.resolution == "invalid"
    assert _locs(found) == [(("configuration", "slow", "configuration", "level"), "invalid_value")]


def test_error_a_container_rule_is_not_asked_of_a_malformed_nested_field() -> None:
    # The stack's rule reads each nested field's name; one with no name is
    # reported where it sits, and the rule is not asked, as `judge` would not.
    field = {"name": "acme.stack", "configuration": {"codecs": [{"configuration": {}}]}}
    resolved, found = resolve(field, CodecDefinition, SCOPE)
    assert resolved.resolution == "invalid"
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


def test_error_a_nested_member_is_not_a_field() -> None:
    resolved, found = resolve(
        {"name": "acme.stack", "configuration": {"codecs": [5]}}, CodecDefinition, SCOPE
    )
    assert resolved.resolution == "invalid"
    assert _locs(found) == [(("configuration", "codecs", 0), "invalid_type")]


def test_error_a_family_judges_the_name_it_claims() -> None:
    resolved, found = resolve("acme.fixed12", DataTypeDefinition, SCOPE)
    assert resolved.resolution == "invalid"
    assert _locs(found) == [((), "invalid_value")]


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
        rules=lambda configuration: (
            [ValidationProblem(("level",), "level 0 stores uncompressed", "invalid_value")]
            if configuration["level"] == 0
            else []
        ),
    )
    scope = CORE_AND_EXTENSIONS.extended_with(strict)
    _, found = resolve({"name": "gzip", "configuration": {"level": 0}}, CodecDefinition, scope)
    assert _locs(found) == [(("configuration", "level"), "invalid_value")]
    assert scope.claimant(ChunkGridDefinition, "regular") is REGULAR_CHUNK_GRID


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
        CodecDefinition(name="acme.bad", configuration=dict, kind="bytes_bytes")


def test_error_a_definition_member_is_a_shape_json_takes() -> None:
    with pytest.raises(TypeError, match="Unreadable: members is not a shape JSON takes"):
        CodecDefinition(name="acme.bad", configuration=Unreadable, kind="bytes_bytes")


def test_error_a_configuration_says_what_its_other_keys_are() -> None:
    # Open by default, it would take a misspelled key without a word.
    with pytest.raises(TypeError, match="Loose says nothing of the keys it does not declare"):
        CodecDefinition(name="acme.loose", configuration=Loose, kind="bytes_bytes")


def test_error_a_member_typed_as_plain_field_json_is_refused() -> None:
    # It checks as JSON, and its name would never be related to a definition.
    with pytest.raises(TypeError, match="annotate it with the field alias of its kind"):
        CodecDefinition(name="acme.plain", configuration=AcmePlainConfiguration, kind="bytes_bytes")


def test_error_a_configuration_whose_annotations_do_not_resolve() -> None:
    # Named down to the TypedDict that holds the annotation.
    with pytest.raises(TypeError, match="AcmeDecimal: name 'Decimal' is not defined"):
        CodecDefinition(
            name="acme.unresolved",
            configuration=AcmeUnresolvedConfiguration,
            kind="bytes_bytes",
        )


def test_error_a_definition_name_is_a_string() -> None:
    with pytest.raises(TypeError, match="a definition's name is a string"):
        CodecDefinition(name=5, configuration=Empty, kind="bytes_bytes")  # pyright: ignore[reportArgumentType]


def test_error_a_codec_kind_is_one_of_three() -> None:
    with pytest.raises(TypeError, match="kind is one of"):
        CodecDefinition(name="acme.k", configuration=Empty, kind="bytes_to_array")  # pyright: ignore[reportArgumentType]


def test_error_a_family_names_its_names_with_a_function() -> None:
    with pytest.raises(TypeError, match="names is a function"):
        CodecDefinition(name="acme.n", configuration=Empty, kind="bytes_bytes", names="acme.n")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("kind", [Definition, AcmeCodecDefinition])
def test_error_a_field_is_read_as_a_kind_of_metadata(kind: type[Definition[Any]]) -> None:
    # Nothing is filed under either, so the field would go unjudged.
    with pytest.raises(TypeError, match="is not a kind of metadata"):
        resolve({"name": "gzip", "configuration": {"level": 99}}, kind, SCOPE)


def test_error_a_scope_refuses_a_definition_of_no_kind() -> None:
    with pytest.raises(TypeError, match="a definition of no kind"):
        Context.of(Definition(name="acme.kindless", configuration=Empty))
