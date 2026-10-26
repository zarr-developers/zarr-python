"""Tests that a v3 model is a document plus the scope it was read in, and for the behavior that follows from that."""

from __future__ import annotations

import copy
import dataclasses
import operator
import pickle
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata._json import refine_json
from zarr_metadata.definition import (
    AcceptedField,
    EmptyConfiguration,
    Nested,
    ScopeConflictError,
    UnclaimedField,
)
from zarr_metadata.model import (
    UNSET,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3ConsolidatedMetadataReading,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    read_array_metadata_v3,
    read_consolidated_metadata_v3,
    read_group_metadata_v3,
)
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v3.codec.bytes import BYTES_CODEC
from zarr_metadata.v3.codec.zstd import ZSTD_CODEC
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
    ZarrV3Context,
)

if TYPE_CHECKING:
    from zarr_metadata import ValidationProblem

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
MY_BYTES = CodecDefinition(
    name="bytes", configuration=EmptyConfiguration, kind="array_bytes", size="static"
)
"""A private `bytes` definition: a different meaning under the core name, with no configuration."""
PRIVATE = CORE.update(codecs={**CORE.codecs, "bytes": MY_BYTES})
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
        (SPELLED_OUT, ZarrV3Context(), ZarrV3Context()),
    ],
    ids=["default-scope", "core", "empty-scope"],
)
def test_a_model_is_its_document_read_in_its_scope(
    document: dict[str, Any], context: ZarrV3Context | None, expected_context: ZarrV3Context
) -> None:
    """A model built from a document holds that document, as written and refined, and the scope it was read in. The scope is `CORE_AND_EXTENSIONS` when none is given."""
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
    """`to_json` writes each field as the document wrote it, not in a reader's preferred spelling, because the model is the document."""
    assert ZarrV3ArrayMetadata(document).to_json()[key] == written


def test_properties_are_what_the_reading_holds() -> None:
    """The typed members (fields as the scope read them, shape, fill value, attributes) are read-only views of the reading."""
    model = ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": [1]}, "acme": 1})
    assert isinstance(model.data_type, AcceptedField)
    assert isinstance(model.codecs[0], AcceptedField)
    assert model.shape == (4,)
    assert model.fill_value == 0
    assert model.dimension_names is UNSET
    assert isinstance(model.attributes, Mapping)
    assert model.attributes == {"a": (1,)}
    assert model.extra_fields["acme"].to_json() == {"acme": 1}
    assert (model.zarr_format, model.node_type) == (3, "array")
    with pytest.raises(TypeError):
        model.attributes["b"] = 1  # pyright: ignore[reportIndexIssue]
    with pytest.raises(AttributeError):
        model.shape = (5,)  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(ZarrV3ArrayMetadata(ARRAY, context=ZarrV3Context()).data_type, UnclaimedField)


def test_a_reading_without_problems_builds_the_model_without_reading_again() -> None:
    """`read_array_metadata_v3` passes its reading to the model it builds, so the model's `reading` is that same reading, not a second one."""
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
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=ZarrV3Context()), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata({**ARRAY, "attributes": {}}), True),
    ],
    ids=["spellings", "unused-definitions", "private-bytes", "unclaimed", "empty-attributes"],
)
def test_models_are_equal_by_what_their_documents_mean(
    left: ZarrV3ArrayMetadata, right: ZarrV3ArrayMetadata, equal: bool
) -> None:
    """Two models are equal when every field reads the same with the same definition and the rest is the same JSON. Spelling and unused definitions do not matter. A private definition under a core name does matter, and so does a name that one model reads and the other leaves unclaimed."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


@pytest.mark.parametrize(
    "context", [None, CORE, PRIVATE, ZarrV3Context()], ids=["default", "core", "private", "empty"]
)
def test_a_model_round_trips_through_its_document_in_its_scope(
    context: ZarrV3Context | None,
) -> None:
    """`from_json(m.to_json(), context=m.context) == m`. The document and the scope determine the model, and pickle preserves both."""
    model = ZarrV3ArrayMetadata(ARRAY, context=context)
    assert ZarrV3ArrayMetadata(model.to_json(), context=model.context) == model
    loaded = pickle.loads(pickle.dumps(model))
    assert loaded == model
    assert loaded.context == model.context
    assert loaded.to_json() == model.to_json()


def test_error_a_model_whose_scope_does_not_pickle_says_so() -> None:
    """A scope that holds a definition with a local function cannot be pickled. Pickling the model raises the same error as pickling the `Context`."""
    local = CodecDefinition(
        name="acme.c",
        configuration=EmptyConfiguration,
        kind="bytes_bytes",
        size="dynamic",
        rules=lambda configuration, nested: iter(()),
    )
    model = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ["bytes", "acme.c"]},
        context=CORE.update(codecs=[*CORE.codecs.values(), local]),
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
    """`update` replaces the document's members with the given JSON members, removes a member given as `UNSET`, and reads the result in the model's own scope. So a model read in a private scope stays in that scope."""
    updated = model.update(**members)
    assert updated == expected
    assert updated.context == model.context


def test_error_update_refuses_a_document_with_a_problem() -> None:
    """`update` raises `MetadataValidationError` when the members make an invalid document, as the constructor does. Here the invalid document has two dimension names for one dimension."""
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadata(ARRAY).update(dimension_names=("x", "y"))


@pytest.mark.parametrize(
    ("model", "context", "gained"),
    [
        (ZarrV3ArrayMetadata(ARRAY, context=ZarrV3Context()), CORE, True),
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), CORE_AND_EXTENSIONS, False),
        (ZarrV3ArrayMetadata(FLOAT, context=ZarrV3Context()), CORE, True),
        (ZarrV3ArrayMetadata(SHARDED, context=ZarrV3Context()), CORE, True),
    ],
    ids=["gain", "nothing-to-gain", "gain-data-type-with-fill-value", "gain-of-a-shard"],
)
def test_refined_in_moves_a_model_up_the_order(
    model: ZarrV3ArrayMetadata, context: ZarrV3Context, gained: bool
) -> None:
    """`refined_in` reads the document in a scope that claims what the current scope left unclaimed and contradicts nothing. The result refines the model and keeps its document. When the new scope reads nothing differently, the result equals the original model."""
    refined = model.refined_in(context)
    assert refined.context == context
    assert refined.to_json() == model.to_json()
    assert refined.refines(model)
    assert (refined == model) is not gained
    if not gained:
        # Not read again: the reading's pipeline is the original one.
        assert refined.reading.pipeline is model.reading.pipeline


@pytest.mark.parametrize(
    ("model", "context"),
    [
        (ZarrV3ArrayMetadata(ARRAY), PRIVATE),
        (ZarrV3ArrayMetadata(ARRAY, context=CORE), ZarrV3Context()),
    ],
    ids=["conflict", "loss"],
)
def test_error_refined_in_refuses_a_conflict_or_a_loss(
    model: ZarrV3ArrayMetadata, context: ZarrV3Context
) -> None:
    """`refined_in` raises `ScopeConflictError` that names each name the new scope reads with another definition, or with no definition."""
    with pytest.raises(ScopeConflictError) as raised:
        model.refined_in(context)
    assert "bytes" in [conflict.key[1] for conflict in raised.value.conflicts]


def test_with_context_reads_the_document_in_any_scope() -> None:
    """`with_context` reads the same document in another scope, whatever that changes. A private `bytes` is read with the private definition, and a scope that claims nothing leaves every name unclaimed."""
    model = ZarrV3ArrayMetadata(ARRAY)
    private = model.with_context(PRIVATE)
    assert private.context == PRIVATE
    assert private.codecs[0].definition == MY_BYTES
    assert isinstance(model.with_context(ZarrV3Context()).codecs[0], UnclaimedField)
    assert model.with_context(None).context == CORE_AND_EXTENSIONS


def test_error_with_context_refuses_a_document_the_scope_reads_with_a_problem() -> None:
    """`with_context` raises `MetadataValidationError` when the document has a problem in the new scope. Here the problem is a gzip `level` that the core definition refuses."""
    loose = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 12}}]},
        context=ZarrV3Context(),
    )
    with pytest.raises(MetadataValidationError):
        loose.with_context(CORE)


@pytest.mark.parametrize(
    ("upper", "lower", "expected"),
    [
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(ARRAY, context=ZarrV3Context()), True),
        (ZarrV3ArrayMetadata(ARRAY, context=ZarrV3Context()), ZarrV3ArrayMetadata(ARRAY), False),
        (ZarrV3ArrayMetadata(ARRAY), ZarrV3ArrayMetadata(SPELLED_OUT), True),
        (ZarrV3ArrayMetadata(ARRAY, context=PRIVATE), ZarrV3ArrayMetadata(ARRAY), False),
        (ZarrV3ArrayMetadata({**ARRAY, "attributes": {"a": 1}}), ZarrV3ArrayMetadata(ARRAY), False),
        (
            ZarrV3ArrayMetadata(FLOAT, context=CORE),
            ZarrV3ArrayMetadata(FLOAT, context=ZarrV3Context()),
            True,
        ),
        (
            ZarrV3ArrayMetadata(SHARDED, context=CORE),
            ZarrV3ArrayMetadata(SHARDED, context=ZarrV3Context()),
            True,
        ),
        (
            ZarrV3ArrayMetadata(FLOAT, context=CORE),
            ZarrV3ArrayMetadata({**FLOAT, "fill_value": "banana"}, context=ZarrV3Context()),
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
    """A model refines another when every field refines its counterpart, and every other member is the same. Fields held inside a field count too, so gaining a shard's inner fields is a gain. The fill value is compared as the more informed data type spells it. A fill value that this data type's definition refuses is not a refinement, and does not raise an error."""
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


MEMBER: dict[str, Any] = CONSOLIDATED["consolidated_metadata"]


def test_a_group_holds_its_consolidated_metadata_as_written() -> None:
    """A group model holds its `consolidated_metadata` member as written, refined and read-only, and does not read the documents it lists. `ZarrV3ConsolidatedMetadata` reads them: each becomes a model in its scope, and writes its document as written."""
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=CORE)
    member = group.consolidated_metadata
    assert member == refine_json(MEMBER)[0]
    with pytest.raises(TypeError):
        operator.setitem(cast("Any", member)["metadata"], "a", ARRAY)
    assert group.to_json() == refine_json(CONSOLIDATED)[0]
    assert ZarrV3GroupMetadata(GROUP).consolidated_metadata is UNSET
    held = ZarrV3ConsolidatedMetadata(member, context=group.context)
    assert set(held.metadata) == {"a", "b", "b/c"}
    for node in held.metadata.values():
        assert node.context is held.context
    assert held.metadata["b/c"].to_json()["data_type"] == {"name": "uint8"}


def test_consolidated_metadata_reads_each_nested_field_once() -> None:
    """Reading a group calls no definition's rules for the documents its consolidated metadata lists. Reading the consolidated metadata calls them once per field, and builds the models from that read."""
    calls: list[int] = []

    def counted(configuration: object, nested: Nested) -> Iterator[ValidationProblem]:
        calls.append(1)
        return iter(())

    context = CORE.update(
        codecs={**CORE.codecs, "bytes": dataclasses.replace(BYTES_CODEC, rules=counted)}
    )
    ZarrV3GroupMetadata(CONSOLIDATED, context=context)
    assert calls == []
    ZarrV3ConsolidatedMetadata(MEMBER, context=context)
    assert len(calls) == 3  # `a`, `b/c`, and `c` in `b`'s own listing, each one bytes codec


def _listing(a: dict[str, Any]) -> dict[str, Any]:
    """`MEMBER` with `a` listed at `a`."""
    return {**MEMBER, "metadata": {**MEMBER["metadata"], "a": a}}


@pytest.mark.parametrize(
    ("left", "right", "equal"),
    [
        (ZarrV3GroupMetadata(CONSOLIDATED), ZarrV3GroupMetadata(CONSOLIDATED, context=CORE), True),
        (
            ZarrV3GroupMetadata(CONSOLIDATED),
            ZarrV3GroupMetadata(CONSOLIDATED, context=PRIVATE),
            True,
        ),
        (ZarrV3GroupMetadata(GROUP), ZarrV3GroupMetadata({**GROUP, "attributes": {"g": 2}}), False),
        (
            ZarrV3GroupMetadata({**GROUP, "consolidated_metadata": _listing(ARRAY)}),
            ZarrV3GroupMetadata({**GROUP, "consolidated_metadata": _listing(SPELLED_OUT)}),
            False,
        ),
        (
            ZarrV3ConsolidatedMetadata(MEMBER),
            ZarrV3ConsolidatedMetadata(MEMBER, context=CORE),
            True,
        ),
        (
            ZarrV3ConsolidatedMetadata(MEMBER),
            ZarrV3ConsolidatedMetadata(MEMBER, context=PRIVATE),
            False,
        ),
        (
            ZarrV3ConsolidatedMetadata(_listing(ARRAY)),
            ZarrV3ConsolidatedMetadata(_listing(SPELLED_OUT)),
            True,
        ),
    ],
    ids=[
        "group-unused-definitions",
        "group-private-bytes-inside",
        "group-attributes",
        "group-listing-spelled-otherwise",
        "consolidated-unused-definitions",
        "consolidated-private-bytes-inside",
        "consolidated-listing-spelled-otherwise",
    ],
)
def test_groups_and_consolidated_metadata_are_equal_by_what_they_read(
    left: ZarrV3GroupMetadata | ZarrV3ConsolidatedMetadata,
    right: ZarrV3GroupMetadata | ZarrV3ConsolidatedMetadata,
    equal: bool,
) -> None:
    """A group is compared by its attributes, consolidated metadata and extra fields as JSON text, because it reads no field: no scope changes it, and a document listed in another spelling makes another group. Consolidated metadata is compared by what each document it lists means in its scope. Equal models have equal hashes."""
    assert (left == right) is equal
    if equal:
        assert hash(left) == hash(right)


@pytest.mark.parametrize("document", [GROUP, CONSOLIDATED], ids=["group", "consolidated"])
def test_a_group_round_trips_through_its_document_in_its_scope(document: dict[str, Any]) -> None:
    """`from_json(g.to_json(), context=g.context) == g`, and pickle preserves both the document and the scope."""
    group = ZarrV3GroupMetadata(document, context=CORE)
    assert ZarrV3GroupMetadata(group.to_json(), context=group.context) == group
    assert pickle.loads(pickle.dumps(group)) == group


def test_group_update_with_context_and_refined_in_behave_as_the_arrays_do() -> None:
    """`update` reads in the group's scope and keeps the consolidated metadata unless a new one is given. A group's fields are its extra fields: in a scope that claims nothing, `consolidated_metadata` is unclaimed, and `refined_in` and `with_context` move the group to a scope that reads it as the envelope, which is a gain."""
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=ZarrV3Context())
    assert group.update(attributes={"g": 2}).consolidated_metadata == group.consolidated_metadata
    assert isinstance(group.extra_fields["consolidated_metadata"], UnclaimedField)
    for moved in (group.refined_in(CORE), group.with_context(PRIVATE)):
        assert isinstance(moved.extra_fields["consolidated_metadata"], AcceptedField)
        assert moved.consolidated_metadata == group.consolidated_metadata
        assert moved.refines(group)
        assert not group.refines(moved)
        assert moved != group
    assert group.refined_in(CORE).context == CORE


def test_error_group_refined_in_refuses_a_scope_that_loses_an_extra_field() -> None:
    """A group read in `CORE` reads its `consolidated_metadata` as the envelope. A scope that leaves the member unclaimed would lose that reading, so `refined_in` raises `ScopeConflictError`, located at the member."""
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=CORE)
    with pytest.raises(ScopeConflictError) as raised:
        group.refined_in(ZarrV3Context())
    assert [conflict.loc for conflict in raised.value.conflicts] == [("consolidated_metadata",)]


def test_consolidated_metadata_with_context_and_refined_in_behave_as_the_arrays_do() -> None:
    """`refined_in` moves every document up the order. `with_context` reads every document in another scope."""
    held = ZarrV3ConsolidatedMetadata(MEMBER, context=ZarrV3Context())
    refined = held.refined_in(CORE)
    assert refined.refines(held)
    assert refined.context == CORE
    array = refined.metadata["a"]
    assert isinstance(array, ZarrV3ArrayMetadata)
    assert isinstance(array.codecs[0], AcceptedField)
    private_array = held.with_context(PRIVATE).metadata["a"]
    assert isinstance(private_array, ZarrV3ArrayMetadata)
    assert private_array.codecs[0].definition == MY_BYTES


def test_error_consolidated_metadata_refined_in_refuses_a_conflict() -> None:
    """`refined_in` raises `ScopeConflictError` for a scope that reads a name of a listed document by another definition, and locates each conflict under that document's path."""
    held = ZarrV3ConsolidatedMetadata(MEMBER, context=CORE)
    with pytest.raises(ScopeConflictError) as raised:
        held.refined_in(PRIVATE)
    assert {conflict.loc for conflict in raised.value.conflicts} == {
        ("metadata", "a", "codecs", 0),
        ("metadata", "b/c", "codecs", 0),
    }


def test_consolidated_metadata_reads_on_its_own() -> None:
    """`ZarrV3ConsolidatedMetadata(member, context)` reads a member as written, or as a group holds it, with each document as a model in that scope. `read_consolidated_metadata_v3` returns the reading of the same model."""
    held = ZarrV3ConsolidatedMetadata(MEMBER, context=CORE)
    group = ZarrV3GroupMetadata(CONSOLIDATED, context=CORE)
    assert held == ZarrV3ConsolidatedMetadata(group.consolidated_metadata, context=CORE)
    assert held.to_json() == refine_json(MEMBER)[0]
    assert ZarrV3ConsolidatedMetadata.from_json(MEMBER) == ZarrV3ConsolidatedMetadata(MEMBER)
    reading = read_consolidated_metadata_v3(MEMBER, context=CORE)
    assert reading.problems == ()
    assert reading.metadata == held
    assert held.reading.metadata is held


def test_error_consolidated_metadata_with_a_problem_is_refused_at_the_documents_path() -> None:
    """A problem in a listed document is reported as a problem of the consolidated metadata, located under `metadata.<path>`. `read_consolidated_metadata_v3` returns the same problems and no model."""
    member = {**MEMBER, "metadata": {"a": {**ARRAY, "shape": [-1]}}}
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ConsolidatedMetadata(member)
    assert [problem.loc for problem in raised.value.problems] == [("metadata", "a", "shape")]
    reading = read_consolidated_metadata_v3(member)
    assert reading.problems == raised.value.problems
    assert reading.metadata is None


def test_the_held_field_machinery_is_gone() -> None:
    """A v3 model is built only by reading its document. So nothing in the package accepts fields that were already read, or re-reads a model in an empty scope. `NO_SCOPE`, `overlapping` and `held` are removed."""
    import inspect

    import zarr_metadata.model._array_v3 as validation

    assert not hasattr(validation, "NO_SCOPE")
    assert not hasattr(validation, "overlapping")
    assert "held" not in inspect.signature(validation.read_array_v3).parameters


def test_error_refined_in_refuses_a_gain_that_surfaces_a_problem() -> None:
    """A scope that claims a name the current scope left unclaimed may refuse what was written under that name. Then `refined_in` raises `MetadataValidationError`, because the document has a problem in that scope."""
    loose = ZarrV3ArrayMetadata({**ARRAY, "fill_value": "banana"}, context=ZarrV3Context())
    with pytest.raises(MetadataValidationError) as raised:
        loose.refined_in(CORE)
    assert [problem.loc for problem in raised.value.problems] == [("fill_value",)]


def test_a_scope_conflict_says_where_each_conflict_sits() -> None:
    """`refined_in` reports each conflict with the field's location in the document, the same way a problem is located. For example, a loss of `bytes` is reported at `codecs.0`, including inside the shard."""
    model = ZarrV3ArrayMetadata(SHARDED, context=CORE)
    with pytest.raises(ScopeConflictError) as raised:
        model.refined_in(ZarrV3Context())
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
        lambda: ZarrV3ConsolidatedMetadata(MEMBER, context=CORE_AND_EXTENSIONS),
        lambda: read_consolidated_metadata_v3(MEMBER, context=CORE_AND_EXTENSIONS).metadata,
    ],
    ids=["constructor", "reader"],
)
def test_with_context_moves_every_listed_model_without_reading_again(
    build: Callable[[], ZarrV3ConsolidatedMetadata | None],
) -> None:
    """However consolidated metadata was built, its reading holds it, and each listed model's reading holds that model. `with_context` into a scope that reads every claim identically moves every listed model to the new scope without reading again."""
    held = build()
    assert held is not None
    assert held.reading.metadata is held
    for node in held.metadata.values():
        assert node.reading.metadata is node
    context = CORE.update(codecs=[*CORE.codecs.values(), ZSTD_CODEC])
    moved = held.with_context(context)
    assert moved.context is context
    before = held.metadata["a"]
    after = moved.metadata["a"]
    assert isinstance(before, ZarrV3ArrayMetadata)
    assert isinstance(after, ZarrV3ArrayMetadata)
    assert after.reading.pipeline is before.reading.pipeline  # not read again
    for node in moved.metadata.values():
        assert node.context is context
        assert node.reading.metadata is node


def test_consolidated_metadata_refines_nothing_of_another_type() -> None:
    """`refines` on consolidated metadata returns False for an object that is not consolidated metadata, as it does for arrays and groups, rather than raising."""
    held = ZarrV3ConsolidatedMetadata(MEMBER)
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
    """What a model exposes (attributes, extra fields, a fill value) is read-only at every level. So a model cannot reach a state where its document, its comparison key and `refines` disagree."""
    model = ZarrV3ArrayMetadata(document)
    same = ZarrV3ArrayMetadata(document)
    with pytest.raises(TypeError):
        change(model)
    assert model == same
    assert model.to_json() == same.to_json()
    assert model.refines(same)


@pytest.mark.parametrize(
    "reading",
    [
        read_group_metadata_v3(GROUP),
        read_group_metadata_v3(CONSOLIDATED),
        read_group_metadata_v3({**GROUP, "attributes": 5}),
        read_consolidated_metadata_v3(MEMBER),
        read_consolidated_metadata_v3({**MEMBER, "kind": "sidecar"}),
    ],
    ids=["group", "consolidated", "group-problem", "member", "member-problem"],
)
def test_a_group_or_consolidated_reading_pickles_and_copies(
    reading: ZarrV3GroupMetadataReading | ZarrV3ConsolidatedMetadataReading,
) -> None:
    """A group's reading and a consolidated-metadata reading pickle and deep-copy, with the model they hold, and compare equal afterwards, as an array's reading does. A reading that holds a model is restored through that model, which reads once."""
    for again in (pickle.loads(pickle.dumps(reading)), copy.deepcopy(reading)):
        assert again == reading
        assert again.metadata == reading.metadata
        if again.metadata is not None:
            assert again.metadata.reading == again


def test_an_array_reading_pickles_through_its_model() -> None:
    """An array's reading that holds a model pickles and deep-copies the same way that model does, and is restored as the model's own reading."""
    reading = read_array_metadata_v3(ARRAY)
    for again in (pickle.loads(pickle.dumps(reading)), copy.deepcopy(reading)):
        assert again == reading
        assert again.metadata is not None
        assert again.metadata.reading is again


def test_error_a_models_fields_cannot_be_changed_in_place() -> None:
    """The fields a model returns (a codec's configuration, the fields a shard holds) are read-only, so `codecs ==` and `refines` cannot become inconsistent with `==`."""
    model = ZarrV3ArrayMetadata(ARRAY)
    same = ZarrV3ArrayMetadata(ARRAY)
    codec = model.codecs[0]
    assert isinstance(codec, AcceptedField)
    with pytest.raises(TypeError):
        codec.configuration["endian"] = "big"  # pyright: ignore[reportIndexIssue]
    assert model.codecs == same.codecs
    assert model.refines(same)
