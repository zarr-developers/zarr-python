"""A v3 model is a document and the scope it was read in: the pair, and what follows from it."""

from __future__ import annotations

import copy
import dataclasses
import operator
import pickle
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast

import pytest

from zarr_metadata._json import refine_json
from zarr_metadata.model import (
    UNSET,
    MetadataValidationError,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    read_array_metadata_v3,
    read_group_metadata_v3,
)
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    AcceptedField,
    CodecDefinition,
    Context,
    Nested,
    ScopeConflictError,
    UnclaimedField,
    ValidationProblem,
)

ARRAY: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [4],
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
    "chunk_key_encoding": {"name": "default"},
    "codecs": ["bytes"],
}
MY_BYTES = CodecDefinition(name="bytes", configuration=Empty, kind="array_bytes", size="static")
"""A private `bytes`: another meaning under the core name, which takes no configuration."""
PRIVATE = CORE.extended_with(MY_BYTES)
SPELLED_OUT: dict[str, Any] = {
    **ARRAY,
    "data_type": {"name": "uint8"},
    "codecs": [{"name": "bytes", "configuration": {}}],
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
}


@pytest.mark.parametrize(
    ("document", "context", "expected_context"),
    [
        (ARRAY, None, CORE_AND_EXTENSIONS),
        (ARRAY, CORE, CORE),
        (SPELLED_OUT, Context.of(), Context.of()),
    ],
    ids=["default-scope", "core", "empty-scope"],
)
def test_a_model_is_its_document_read_in_its_scope(
    document: dict[str, Any], context: Context | None, expected_context: Context
) -> None:
    """A model built from a document holds that document, as written and refined, and the scope it was read in, `CORE_AND_EXTENSIONS` when none is given."""
    model = ZarrV3ArrayMetadata(document, context=context)
    assert model.context == expected_context
    assert model.to_json() == refine_json(document)[0]


@pytest.mark.parametrize(
    ("document", "key", "written"),
    [
        (ARRAY, "data_type", "uint8"),
        (SPELLED_OUT, "data_type", {"name": "uint8"}),
        (ARRAY, "codecs", ("bytes",)),
        (SPELLED_OUT, "codecs", ({"name": "bytes", "configuration": {}},)),
    ],
    ids=["bare-data-type", "object-data-type", "bare-codec", "object-codec"],
)
def test_to_json_keeps_the_spelling_the_document_was_written_in(
    document: dict[str, Any], key: str, written: object
) -> None:
    """`to_json` writes each field as the document wrote it, not as a reader would respell it: the model is the document."""
    assert ZarrV3ArrayMetadata(document).to_json()[key] == written


def test_properties_are_what_the_reading_holds() -> None:
    """The typed members -- fields as the scope read them, shape, fill value, attributes -- are views of the reading, read-only."""
    model = ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": [1]}, "acme": 1})
    assert isinstance(model.data_type, AcceptedField)
    assert isinstance(model.codecs[0], AcceptedField)
    assert model.shape == (4,)
    assert model.fill_value == 0
    assert model.dimension_names is UNSET
    assert isinstance(model.attributes, Mapping)
    assert model.attributes == {"a": (1,)}
    assert model.extra_fields == {"acme": 1}
    assert (model.zarr_format, model.node_type) == (3, "array")
    with pytest.raises(TypeError):
        model.attributes["b"] = 1  # pyright: ignore[reportIndexIssue]
    with pytest.raises(AttributeError):
        model.shape = (5,)  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(ZarrV3ArrayMetadata(ARRAY, context=Context.of()).data_type, UnclaimedField)


def test_a_reading_without_problems_builds_the_model_without_reading_again() -> None:
    """`read_array_metadata_v3` hands its reading to the model it builds, so the model's `reading` is that reading, not a second one."""
    reading = read_array_metadata_v3(ARRAY)
    assert reading.metadata is not None
    assert reading.metadata.reading.pipeline is reading.pipeline


def test_error_a_document_with_a_problem_is_refused_at_construction() -> None:
    """The constructor raises `MetadataValidationError` with every problem, as `from_json` does."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata({**ARRAY, "fill_value": 300, "shape": [-1]})
    assert sorted(problem.loc for problem in raised.value.problems) == [
        ("fill_value",),
        ("shape",),
    ]


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(SPELLED_OUT), True),
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), ZarrV3ArrayMetadata(ARRAY), True),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=PRIVATE), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=Context.of()), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata({**ARRAY, "attributes": {}}), True),
    ],
    ids=["spellings", "unused-definitions", "private-bytes", "unclaimed", "empty-attributes"],
)
def test_models_are_equal_by_what_their_documents_mean(
    left: ZarrV3ArrayMetadata, right: ZarrV3ArrayMetadata, equal: bool
) -> None:
    """Two models are equal when every field reads the same by the same definition and the rest is the same JSON: spelling and unused definitions do not matter, a private definition under a core name does, and so does a name read against one left unclaimed."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


@pytest.mark.parametrize(
    "context", [None, CORE, PRIVATE, Context.of()], ids=["default", "core", "private", "empty"]
)
def test_a_model_round_trips_through_its_document_in_its_scope(context: Context | None) -> None:
    """`from_json(m.to_json(), context=m.context) == m`: the document and the scope determine the model, and pickle carries both."""
    model = ZarrV3ArrayMetadata(ARRAY, context=context)
    assert ZarrV3ArrayMetadata(model.to_json(), context=model.context) == model
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded == model
    assert loaded.context == model.context
    assert loaded.to_json() == model.to_json()


def test_error_a_model_whose_scope_does_not_pickle_says_so() -> None:
    """A scope holding a definition with a local function does not pickle, and the model raises the error pickling a `Context` raises."""
    local = CodecDefinition(
        name="acme.c",
        configuration=Empty,
        kind="bytes_bytes",
        size="dynamic",
        rules=lambda configuration, nested: iter(()),
    )
    model = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ["bytes", "acme.c"]}, context=CORE.extended_with(local)
    )
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(model)


FLOAT: dict[str, Any] = {
    **ARRAY,
    "data_type": "float32",
    "fill_value": "0x7fc00000",
    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
}
SHARDED: dict[str, Any] = {
    **ARRAY,
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [1],
                "codecs": ["bytes"],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    "crc32c",
                ],
            },
        }
    ],
}


@pytest.mark.parametrize(
    ("model", "members", "expected"),
    [
        (
            ZarrV3ArrayMetadata(ARRAY),
            {"attributes": {"a": 1}},
            ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": 1}}),
        ),
        (
            ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": 1}}),
            {"attributes": UNSET},
            ZarrV3ArrayMetadata(ARRAY),
        ),
        (
            ZarrV3ArrayMetadata(ARRAY, context=PRIVATE),
            {"attributes": {"a": 1}},
            ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": 1}}, context=PRIVATE),
        ),
        (
            ZarrV3ArrayMetadata(ARRAY),
            {
                "shape": (6,),
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (3,)}},
            },
            ZarrV3ArrayMetadata(
                {
                    **ARRAY,
                    "shape": [6],
                    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
                }
            ),
        ),
    ],
    ids=["set", "unset", "update-keeps-scope", "members-together"],
)
def test_update_reads_new_members_in_the_models_own_scope(
    model: ZarrV3ArrayMetadata, members: dict[str, Any], expected: ZarrV3ArrayMetadata
) -> None:
    """`update` puts JSON members in place of the document's, `UNSET` removing one, and reads the result in the model's own scope, so a model read privately stays private."""
    updated = model.update(**members)
    assert updated == expected
    assert updated.context == model.context


def test_error_update_refuses_a_document_with_a_problem() -> None:
    """`update` raises `MetadataValidationError` when the members make an invalid document, as the constructor does: two dimension names for one dimension."""
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadata(ARRAY).update(dimension_names=("x", "y"))


@pytest.mark.parametrize(
    ("model", "context", "gained"),
    [
        (ZarrV3ArrayMetadata(ARRAY, context=Context.of()), CORE, True),
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), CORE_AND_EXTENSIONS, False),
        (ZarrV3ArrayMetadata(FLOAT, context=Context.of()), CORE, True),
        (ZarrV3ArrayMetadata(SHARDED, context=Context.of()), CORE, True),
    ],
    ids=["gain", "nothing-to-gain", "gain-data-type-with-fill-value", "gain-of-a-shard"],
)
def test_refined_in_moves_a_model_up_the_order(
    model: ZarrV3ArrayMetadata, context: Context, gained: bool
) -> None:
    """`refined_in` reads the document in a scope that claims what this one left unclaimed and contradicts nothing; the result refines the model, keeps its document, and is the model itself when the scope reads nothing otherwise."""
    refined = model.refined_in(context)
    assert refined.context == context
    assert refined.to_json() == model.to_json()
    assert refined.refines(model)
    assert (refined == model) is not gained
    if not gained:
        # Not read again: the reading's pipeline is the one read.
        assert refined.reading.pipeline is model.reading.pipeline


@pytest.mark.parametrize(
    ("model", "context"),
    [
        (ZarrV3ArrayMetadata(ARRAY), PRIVATE),
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), Context.of()),
    ],
    ids=["conflict", "loss"],
)
def test_error_refined_in_refuses_a_conflict_or_a_loss(
    model: ZarrV3ArrayMetadata, context: Context
) -> None:
    """`refined_in` raises `ScopeConflictError` naming each name the scope reads by another definition, or by none."""
    with pytest.raises(ScopeConflictError) as raised:
        model.refined_in(context)
    assert "bytes" in [conflict.key[1] for conflict in raised.value.conflicts]


def test_with_context_reads_the_document_in_any_scope() -> None:
    """`with_context` reads the same document in another scope, whatever that changes: a private `bytes` is read as such, and a scope that claims nothing leaves every name unclaimed."""
    model = ZarrV3ArrayMetadata(ARRAY)
    private = model.with_context(PRIVATE)
    assert private.context == PRIVATE
    assert private.codecs[0].definition == MY_BYTES
    assert isinstance(model.with_context(Context.of()).codecs[0], UnclaimedField)
    assert model.with_context(None).context == CORE_AND_EXTENSIONS


def test_error_with_context_refuses_a_document_the_scope_reads_with_a_problem() -> None:
    """`with_context` raises `MetadataValidationError` when the document has a problem in the new scope: a gzip `level` the core definition refuses."""
    loose = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 12}}]},
        context=Context.of(),
    )
    with pytest.raises(MetadataValidationError):
        loose.with_context(CORE)


@pytest.mark.parametrize(
    ("upper", "lower", "expected"),
    [
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=Context.of()), True),
        (ZarrV3ArrayMetadata(ARRAY, context=Context.of()), ZarrV3ArrayMetadata(ARRAY), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(SPELLED_OUT), True),
        (ZarrV3ArrayMetadata(ARRAY, context=PRIVATE), ZarrV3ArrayMetadata(ARRAY), False),
        (ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": 1}}), ZarrV3ArrayMetadata(ARRAY), False),
        (
            ZarrV3ArrayMetadata(FLOAT, context=CORE),
            ZarrV3ArrayMetadata(FLOAT, context=Context.of()),
            True,
        ),
        (
            ZarrV3ArrayMetadata(SHARDED, context=CORE),
            ZarrV3ArrayMetadata(SHARDED, context=Context.of()),
            True,
        ),
        (
            ZarrV3ArrayMetadata(FLOAT, context=CORE),
            ZarrV3ArrayMetadata({**FLOAT, "fill_value": "banana"}, context=Context.of()),
            False,
        ),
    ],
    ids=[
        "gain",
        "loss",
        "equal",
        "conflict",
        "other-members-differ",
        "fill-value-spelled-by-the-informed-side",
        "gain-of-a-field-holding-fields",
        "fill-value-the-gained-definition-refuses",
    ],
)
def test_refines_orders_models_by_information(
    upper: ZarrV3ArrayMetadata, lower: ZarrV3ArrayMetadata, expected: bool
) -> None:
    """A model refines another when every field refines its counterpart -- the fields a field holds too, so a shard gained is a gain -- and every other member is the same, the fill value compared as the more informed data type spells it; a fill value that definition refuses is no refinement, and no error."""
    assert upper.refines(lower) is expected


GROUP: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {"g": 1}}
CONSOLIDATED: dict[str, Any] = {
    **GROUP,
    "consolidated_metadata": {
        "kind": "inline",
        "must_understand": False,
        "metadata": {
            "a": ARRAY,
            "b": {
                **GROUP,
                "consolidated_metadata": {
                    "kind": "inline",
                    "must_understand": False,
                    "metadata": {"c": SPELLED_OUT},
                },
            },
            "b/c": SPELLED_OUT,
        },
    },
}


def test_a_group_is_its_document_and_scope_and_its_nested_models_share_them() -> None:
    """A group model is the pair, and each document its consolidated metadata holds is a model of the same scope, built from the group's one read, writing its document as written."""
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=CORE)
    held = group.consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    assert set(held.metadata) == {"a", "b", "b/c"}
    for node in held.metadata.values():
        assert node.context is group.context
    assert held.metadata["b/c"].to_json()["data_type"] == {"name": "uint8"}
    assert group.to_json() == refine_json(CONSOLIDATED)[0]
    assert ZarrV3GroupMetadata(GROUP).consolidated_metadata is UNSET


def test_a_group_reads_each_nested_field_once() -> None:
    """Building a group with consolidated metadata asks a definition's rules once per nested field: the models are built from the read, not read again."""
    calls: list[int] = []

    def counted(configuration: object, nested: Nested) -> Iterator[ValidationProblem]:
        calls.append(1)
        return iter(())

    scope = CORE.extended_with(dataclasses.replace(BYTES_CODEC, rules=counted))
    ZarrV3GroupMetadata(CONSOLIDATED, context=scope)
    assert len(calls) == 3  # `a`, `b/c`, and `c` in `b`'s own listing, each one bytes codec


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (ZarrV3GroupMetadata(CONSOLIDATED), ZarrV3GroupMetadata(CONSOLIDATED, context=CORE), True),
        (
            ZarrV3GroupMetadata(CONSOLIDATED),
            ZarrV3GroupMetadata(CONSOLIDATED, context=PRIVATE),
            False,
        ),
        (ZarrV3GroupMetadata(GROUP), ZarrV3GroupMetadata({**GROUP, "attributes": {"g": 2}}), False),
    ],
    ids=["unused-definitions", "private-bytes-inside", "attributes"],
)
def test_groups_are_equal_by_what_their_documents_mean(
    left: ZarrV3GroupMetadata, right: ZarrV3GroupMetadata, equal: bool
) -> None:
    """A group compares by its attributes, extra fields and each nested model's meaning; equal groups hash alike."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


@pytest.mark.parametrize("document", [GROUP, CONSOLIDATED], ids=["group", "consolidated"])
def test_a_group_round_trips_through_its_document_in_its_scope(document: dict[str, Any]) -> None:
    """`from_json(g.to_json(), context=g.context) == g`, and pickle carries the pair."""
    group = ZarrV3GroupMetadata(document, context=CORE)
    assert ZarrV3GroupMetadata(group.to_json(), context=group.context) == group
    assert pickle.loads(pickle.dumps(group)) == group


def test_group_update_with_context_and_refined_in_behave_as_the_arrays_do() -> None:
    """`update` reads in the group's scope and keeps its consolidated metadata unless given; `refined_in` moves every nested model up the order; `with_context` reads all of it in another scope."""
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=Context.of())
    assert group.update(attributes={"g": 2}).consolidated_metadata == group.consolidated_metadata
    refined = group.refined_in(CORE)
    assert refined.refines(group)
    assert refined.context == CORE
    held = refined.consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    array = held.metadata["a"]
    assert isinstance(array, ZarrV3ArrayMetadata)
    assert isinstance(array.codecs[0], AcceptedField)
    with pytest.raises(ScopeConflictError):
        ZarrV3GroupMetadata(CONSOLIDATED).refined_in(PRIVATE)
    private = group.with_context(PRIVATE).consolidated_metadata
    assert isinstance(private, ZarrV3ConsolidatedMetadata)
    private_array = private.metadata["a"]
    assert isinstance(private_array, ZarrV3ArrayMetadata)
    assert private_array.codecs[0].definition == MY_BYTES


def test_consolidated_metadata_reads_on_its_own() -> None:
    """`ZarrV3ConsolidatedMetadata(member, context)` reads the member as a group's read reads it, each document a model of that scope."""
    member = CONSOLIDATED["consolidated_metadata"]
    held = ZarrV3ConsolidatedMetadata(member, context=CORE)
    assert held == ZarrV3GroupMetadata(CONSOLIDATED, context=CORE).consolidated_metadata
    assert held.to_json() == refine_json(member)[0]
    assert ZarrV3ConsolidatedMetadata.from_json(member) == ZarrV3ConsolidatedMetadata(member)


def test_error_a_group_with_a_nested_problem_is_refused_at_the_nested_path() -> None:
    """A nested document's problem is the group's, located under `consolidated_metadata.metadata.<path>`."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3GroupMetadata(
            {
                **CONSOLIDATED,
                "consolidated_metadata": {
                    **CONSOLIDATED["consolidated_metadata"],
                    "metadata": {"a": {**ARRAY, "shape": [-1]}},
                },
            }
        )
    assert raised.value.problems[0].loc == ("consolidated_metadata", "metadata", "a", "shape")


def test_the_held_field_machinery_is_gone() -> None:
    """A v3 model is built only by reading its document, so nothing in the package takes fields read already, or re-reads a model in an empty scope: `NO_SCOPE`, `overlapping` and `held` are gone."""
    import inspect

    import zarr_metadata.model._validation as validation

    assert not hasattr(validation, "NO_SCOPE")
    assert not hasattr(validation, "overlapping")
    assert "held" not in inspect.signature(validation.read_array_v3).parameters


def test_error_refined_in_refuses_a_gain_that_surfaces_a_problem() -> None:
    """A scope that claims a name this one left unclaimed may refuse what was written under it: `refined_in` raises `MetadataValidationError`, the document having a problem in that scope."""
    loose = ZarrV3ArrayMetadata({**ARRAY, "fill_value": "banana"}, context=Context.of())
    with pytest.raises(MetadataValidationError) as raised:
        loose.refined_in(CORE)
    assert [problem.loc for problem in raised.value.problems] == [("fill_value",)]


def test_a_scope_conflict_says_where_each_conflict_sits() -> None:
    """`refined_in` names each conflict with where the field sits in the document, as a problem is located: a loss of `bytes` at `codecs.0`, in the shard too."""
    model = ZarrV3ArrayMetadata(SHARDED, context=CORE)
    with pytest.raises(ScopeConflictError) as raised:
        model.refined_in(Context.of())
    assert sorted((conflict.key[1], conflict.loc) for conflict in raised.value.conflicts) == [
        ("bytes", ("codecs", 0, "configuration", "codecs", 0)),
        ("bytes", ("codecs", 0, "configuration", "index_codecs", 0)),
        ("crc32c", ("codecs", 0, "configuration", "index_codecs", 1)),
        ("default", ("chunk_key_encoding",)),
        ("regular", ("chunk_grid",)),
        ("sharding_indexed", ("codecs", 0)),
        ("uint8", ("data_type",)),
    ]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ZarrV3GroupMetadata(CONSOLIDATED, context=CORE_AND_EXTENSIONS),
        lambda: read_group_metadata_v3(CONSOLIDATED, context=CORE_AND_EXTENSIONS).metadata,
    ],
    ids=["constructor", "reader"],
)
def test_a_models_reading_holds_the_model_and_with_context_moves_the_whole_tree(
    build: Callable[[], ZarrV3GroupMetadata | None],
) -> None:
    """However a group was built, its reading holds it, each nested reading holds the nested model, and `with_context` into a scope that reads every claim identically moves every nested model, the nested ones' too, to the new scope without reading again."""
    group = build()
    assert group is not None
    assert group.reading.metadata is group
    nested = group.consolidated_metadata
    assert isinstance(nested, ZarrV3ConsolidatedMetadata)
    for path, node in nested.metadata.items():
        assert group.reading.consolidated[path].metadata is node
        assert node.reading.metadata is node
    scope = CORE.extended_with(ZSTD_CODEC)
    moved = group.with_context(scope)
    assert moved.reading is not group.reading
    held = moved.consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    assert held.context is scope
    before = nested.metadata["a"]
    after = held.metadata["a"]
    assert isinstance(before, ZarrV3ArrayMetadata)
    assert isinstance(after, ZarrV3ArrayMetadata)
    assert after.reading.pipeline is before.reading.pipeline  # not read again
    for node in held.metadata.values():
        assert node.context is scope
        assert node.reading.metadata is node
    inner = held.metadata["b"]
    assert isinstance(inner, ZarrV3GroupMetadata)
    innermost = inner.consolidated_metadata
    assert isinstance(innermost, ZarrV3ConsolidatedMetadata)
    assert innermost.context is scope
    assert innermost.metadata["c"].context is scope
    assert innermost.metadata["c"].reading.metadata is innermost.metadata["c"]


def test_consolidated_metadata_refines_nothing_of_another_type() -> None:
    """`refines` of consolidated metadata says False of what is not consolidated metadata, as the array's and group's do, rather than raising."""
    held = ZarrV3GroupMetadata(CONSOLIDATED).consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    assert held.refines(cast("Any", ZarrV3GroupMetadata(GROUP))) is False


@pytest.mark.parametrize(
    ("document", "change"),
    [
        (
            {**ARRAY, "attributes": {"a": {"b": 1}}},
            lambda model: operator.setitem(model.attributes["a"], "b", 2),
        ),
        (
            {**ARRAY, "acme": {"x": 1, "must_understand": False}},
            lambda model: operator.setitem(model.extra_fields["acme"], "x", 2),
        ),
        (
            {
                **ARRAY,
                "data_type": {
                    "name": "struct",
                    "configuration": {"fields": [{"name": "a", "data_type": "uint8"}]},
                },
                "fill_value": {"a": 0},
            },
            lambda model: operator.setitem(model.fill_value, "a", 7),
        ),
    ],
    ids=["attributes", "extra-fields", "fill-value"],
)
def test_a_models_views_cannot_be_changed_in_place(
    document: dict[str, Any], change: Callable[[ZarrV3ArrayMetadata], None]
) -> None:
    """What a model shows -- attributes, extra fields, a fill value -- is read-only at every level, so a model cannot be put in a state its document, its key and `refines` disagree about."""
    model = ZarrV3ArrayMetadata(document)
    same = ZarrV3ArrayMetadata(document)
    with pytest.raises(TypeError):
        change(model)
    assert model == same
    assert model.to_json() == same.to_json()
    assert model.refines(same)


@pytest.mark.parametrize(
    "problem",
    [
        {"attributes": {1: "x"}},
        {"attributes": 5},
        {
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": False,
                "metadata": {"a": ARRAY, "b": {**GROUP, "attributes": {"s": {1, 2}}}},
            }
        },
        {
            "consolidated_metadata": {
                "kind": "inline",
                "must_understand": False,
                "metadata": {"a": ARRAY, "b": {**ARRAY, "shape": "x"}},
            }
        },
    ],
    ids=["non-string-key", "attributes-not-an-object", "sibling-not-json", "sibling-invalid"],
)
def test_a_reading_holds_a_model_of_each_nested_document_without_a_problem(
    problem: dict[str, Any],
) -> None:
    """A group document with a problem still holds, in its reading, a model of each document its consolidated metadata holds that has no problem, whatever the problem elsewhere is: one a reader walks past, or one it refuses."""
    member = {"kind": "inline", "must_understand": False, "metadata": {"a": ARRAY}}
    document = {**GROUP, "consolidated_metadata": member, **problem}
    reading = read_group_metadata_v3(document)
    assert len(reading.problems) != 0
    assert reading.metadata is None
    held = reading.consolidated["a"].metadata
    assert isinstance(held, ZarrV3ArrayMetadata)
    assert held == ZarrV3ArrayMetadata(ARRAY)


def test_a_scope_conflict_inside_consolidated_metadata_is_located_there() -> None:
    """A group's `refined_in` locates a conflict in a document its consolidated metadata holds under that document's path, in a listing a listed group holds too."""
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=CORE)
    with pytest.raises(ScopeConflictError) as raised:
        group.refined_in(Context.of())
    locs = {conflict.loc for conflict in raised.value.conflicts}
    assert ("consolidated_metadata", "metadata", "a", "codecs", 0) in locs
    assert (
        "consolidated_metadata",
        "metadata",
        "b",
        "consolidated_metadata",
        "metadata",
        "c",
        "codecs",
        0,
    ) in locs


def test_a_models_reading_cannot_be_changed_in_place() -> None:
    """What a model's reading holds of the documents its consolidated metadata holds is read-only, so nothing planted there is taken up by `with_context`."""
    group = ZarrV3GroupMetadata(CONSOLIDATED)
    planted = ZarrV3ArrayMetadata({**ARRAY, "shape": [9]}).reading
    with pytest.raises(TypeError):
        operator.setitem(cast("Any", group.reading.consolidated), "a", planted)
    moved = group.with_context(CORE_AND_EXTENSIONS)
    assert moved == ZarrV3GroupMetadata(CONSOLIDATED)


@pytest.mark.parametrize(
    "fault",
    [{"attributes": "bad"}, {"attributes": {1: "bad"}}, {"attributes": {"s": {1, 2}}}],
    ids=["refused", "non-string-key", "not-json"],
)
def test_a_reading_holds_a_model_of_each_healthy_document_in_a_listed_groups_own_listing(
    fault: dict[str, Any],
) -> None:
    """A listed group with a problem of its own -- one the reader refuses, or one it walks past -- still holds, in its reading, a model of each document in its own listing that has no problem, as the top group does."""
    inline = {"kind": "inline", "must_understand": False}
    listed = {
        **GROUP,
        **fault,
        "consolidated_metadata": {
            **inline,
            "metadata": {"x": ARRAY, "y": {**ARRAY, "shape": [-1]}},
        },
    }
    document = {
        **GROUP,
        "consolidated_metadata": {
            **inline,
            "metadata": {"a": listed, "a/x": ARRAY, "a/y": {**ARRAY, "shape": [-1]}},
        },
    }
    reading = read_group_metadata_v3(document)
    assert reading.metadata is None
    nested = reading.consolidated["a"]
    assert isinstance(nested, ZarrV3GroupMetadataReading)
    assert nested.metadata is None
    held = nested.consolidated["x"].metadata
    assert isinstance(held, ZarrV3ArrayMetadata)
    assert held == ZarrV3ArrayMetadata(ARRAY)


@pytest.mark.parametrize("document", [GROUP, CONSOLIDATED], ids=["group", "consolidated"])
def test_a_group_reading_pickles_and_copies(document: dict[str, Any]) -> None:
    """A group's reading pickles and deep-copies, with the documents its consolidated metadata holds and the model it built, and compares equal afterwards, as an array's does."""
    reading = read_group_metadata_v3(document)
    for again in (pickle.loads(pickle.dumps(reading)), copy.deepcopy(reading)):
        assert again == reading
        assert again.metadata == reading.metadata
        assert set(again.consolidated) == set(reading.consolidated)
        # One model per document still: the reading comes back through
        # the model it holds, which reads once.
        assert again.metadata is not None
        assert again.metadata.reading is again
        for path, nested in again.consolidated.items():
            held = again.metadata.consolidated_metadata
            assert isinstance(held, ZarrV3ConsolidatedMetadata)
            assert nested.metadata is held.metadata[path]


def test_an_array_reading_pickles_through_its_model() -> None:
    """An array's reading that holds a model pickles and deep-copies as that model does, and comes back as the model's own reading."""
    reading = read_array_metadata_v3(ARRAY)
    for again in (pickle.loads(pickle.dumps(reading)), copy.deepcopy(reading)):
        assert again == reading
        assert again.metadata is not None
        assert again.metadata.reading is again


def test_error_a_models_fields_cannot_be_changed_in_place() -> None:
    """The fields a model hands out -- a codec's configuration, the fields a shard holds -- are read-only, so `codecs ==` and `refines` cannot drift from `==`."""
    model = ZarrV3ArrayMetadata(ARRAY)
    same = ZarrV3ArrayMetadata(ARRAY)
    codec = model.codecs[0]
    assert isinstance(codec, AcceptedField)
    with pytest.raises(TypeError):
        codec.configuration["endian"] = "big"  # pyright: ignore[reportIndexIssue]
    assert model.codecs == same.codecs
    assert model.refines(same)
