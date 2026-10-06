"""A group's consolidated metadata built from node models: each accepted when it reads the same in the group's scope, or gains there; refused when it conflicts or would lose."""

from __future__ import annotations

import pickle
from typing import Any, cast

import pytest

from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    is_group_metadata_v3,
    parse_group_metadata_v3,
    read_group_metadata_v3,
    validate_group_metadata_v3,
)
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS, CodecDefinition, Context

ARRAY: dict[str, Any] = {
    **ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json(),
    "codecs": ("bytes",),
}
"""A `uint8` array whose `bytes` takes no configuration, so a private `bytes` of none reads it too."""
ZSTD = {"name": "zstd", "configuration": {"level": 3, "checksum": False}}
LOOSE_ZSTD = {"name": "zstd", "configuration": {"level": 30}}
WITH_ZSTD = {**ARRAY, "codecs": (*ARRAY["codecs"], ZSTD)}
WITH_LOOSE_ZSTD = {**ARRAY, "codecs": (*ARRAY["codecs"], LOOSE_ZSTD)}
MY_BYTES = CodecDefinition(name="bytes", configuration=Empty, kind="array_bytes", size="static")
PRIVATE = CORE.extended_with(MY_BYTES)
INLINE: dict[str, Any] = {"kind": "inline", "must_understand": False}


def _group(scope: Context | None = None, **entries: object) -> ZarrV3GroupMetadata:
    member: Any = {**INLINE, "metadata": entries}
    return ZarrV3GroupMetadata.create_default(context=scope, consolidated_metadata=member)


@pytest.mark.parametrize(
    ("scope", "child", "beside", "gained"),
    [
        (CORE_AND_EXTENSIONS, ZarrV3ArrayMetadata(ARRAY), {}, False),
        (CORE_AND_EXTENSIONS, ZarrV3ArrayMetadata(ARRAY, context=CORE), {}, False),
        (CORE_AND_EXTENSIONS, ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE), {}, True),
        (
            CORE,
            ZarrV3GroupMetadata(
                _group(CORE, x=ZarrV3ArrayMetadata(ARRAY, context=CORE)).to_json(),
                context=Context.of(),
            ),
            {"a/x": ZarrV3ArrayMetadata(ARRAY, context=Context.of())},
            True,
        ),
        (
            CORE,
            {
                "zarr_format": 3,
                "node_type": "group",
                "consolidated_metadata": {**INLINE, "metadata": {"x": ZarrV3ArrayMetadata(ARRAY)}},
            },
            {"a/x": ARRAY},
            False,
        ),
    ],
    ids=["same-scope", "unused-definitions", "gain", "nested-models", "model-in-a-document"],
)
def test_a_model_is_accepted_as_a_consolidated_entry_when_it_refines_into_the_scope(
    scope: Context,
    child: ZarrV3ArrayMetadata | ZarrV3GroupMetadata | dict[str, Any],
    beside: dict[str, Any],
    gained: bool,
) -> None:
    """A node model given where consolidated metadata lists a document -- at the top, or inside a document listed there -- is accepted when the group's scope reads every claim of it identically -- its reading is kept, not read again -- or claims what the model's scope left unclaimed, when its document is read again there; the group then holds it in its own scope, equal to the group built of the documents, and writes plain JSON. A listed group's own listing is listed flat beside it, as the convention asks."""
    group = _group(scope, a=child, **beside)
    held = group.consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    node = held.metadata["a"]
    assert node.context is group.context
    if isinstance(child, dict):
        child = ZarrV3GroupMetadata(_documents(child), context=scope)
    assert cast("Any", node).refines(child)
    assert (node == child) is not gained
    if not gained and isinstance(child, ZarrV3ArrayMetadata):
        assert isinstance(node, ZarrV3ArrayMetadata)
        assert node.reading.pipeline is child.reading.pipeline
    written = {path: _documents(entry) for path, entry in beside.items()}
    assert group == _group(scope, a=child.to_json(), **written)
    assert is_group_metadata_v3(group.to_json())


def _documents(value: object) -> object:
    """`value` with every node model in it replaced by its document, however deep."""
    if isinstance(value, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
        return value.to_json()
    if isinstance(value, dict):
        return {key: _documents(item) for key, item in cast("dict[str, object]", value).items()}
    return value


def test_documents_as_written_and_pickle() -> None:
    """A group built of models writes each child's document as the child wrote it, and pickles as any group does."""
    child = ZarrV3ArrayMetadata({**ARRAY, "data_type": {"name": "uint8"}})
    group = _group(None, a=child)
    written = cast("Any", group.to_json()["consolidated_metadata"])
    assert written["metadata"]["a"] == child.to_json()
    assert pickle.loads(pickle.dumps(group)) == group


def test_error_a_model_that_conflicts_with_the_scope_is_refused_at_its_path() -> None:
    """A model read with a private `bytes`, given to a group whose scope reads the core `bytes`, is a conflict: a problem at the entry's path, naming the kind and the name, and telling the two definitions apart, not a silent re-read."""
    child = ZarrV3ArrayMetadata(ARRAY, context=PRIVATE)
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE_AND_EXTENSIONS, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("consolidated_metadata", "metadata", "a", "codecs", 0)
    assert problem.kind == "invalid_value"
    assert problem.message == (
        "expected a document read in the group's scope, got a model that reads the codec "
        "'bytes' by CodecDefinition(name='bytes') of Empty, which the group's scope reads by "
        "CodecDefinition(name='bytes') of BytesCodecConfiguration"
    )


def test_error_a_model_that_would_lose_a_meaning_is_refused_at_its_path() -> None:
    """A model read where `zstd` is claimed, given to a group whose scope leaves it unclaimed, would lose what it reads: refused as a conflict is."""
    child = ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("consolidated_metadata", "metadata", "a", "codecs", 1)
    assert problem.message.endswith("which the group's scope leaves unclaimed")


def test_error_a_gain_that_surfaces_a_problem_is_reported_at_the_problem() -> None:
    """A model whose scope left `zstd` unclaimed, given to a group whose scope claims it, is read again there: a level that definition refuses is a problem where it sits."""
    child = ZarrV3ArrayMetadata(WITH_LOOSE_ZSTD, context=CORE)
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE_AND_EXTENSIONS, a=child)
    assert [problem.loc for problem in raised.value.problems] == [
        ("consolidated_metadata", "metadata", "a", "codecs", 1, "configuration", "level")
    ]


def test_readers_see_models_as_the_constructor_does() -> None:
    """`read_group_metadata_v3` and `validate_group_metadata_v3` given a document holding models read them as the constructor does: the reading holds the adopted model, and the validator reports a conflict."""
    fine = {**INLINE, "metadata": {"a": ZarrV3ArrayMetadata(ARRAY)}}
    document = {"zarr_format": 3, "node_type": "group", "consolidated_metadata": fine}
    reading = read_group_metadata_v3(document)
    assert reading.problems == ()
    assert isinstance(reading.consolidated["a"].metadata, ZarrV3ArrayMetadata)
    assert validate_group_metadata_v3(document) == ()
    clashing = {
        **document,
        "consolidated_metadata": {
            **INLINE,
            "metadata": {"a": ZarrV3ArrayMetadata(ARRAY, context=PRIVATE)},
        },
    }
    assert [problem.loc for problem in validate_group_metadata_v3(clashing)] == [
        ("consolidated_metadata", "metadata", "a", "codecs", 0)
    ]


def test_consolidated_metadata_given_whole_is_taken_as_its_models() -> None:
    """A `ZarrV3ConsolidatedMetadata` given as the member is its models at their paths: `update(consolidated_metadata=other.consolidated_metadata)` carries them over."""
    source = _group(CORE_AND_EXTENSIONS, a=ZarrV3ArrayMetadata(ARRAY))
    target = ZarrV3GroupMetadata.create_default(attributes={"t": 1}).update(
        consolidated_metadata=source.consolidated_metadata
    )
    assert target.consolidated_metadata == source.consolidated_metadata
    assert target.attributes == {"t": 1}


def test_children_of_different_scopes_consolidate_in_their_join() -> None:
    """Children read in different scopes are consolidated in `Context.joined` of them: each refines into the join, and the group reads in it."""
    a = ZarrV3ArrayMetadata(ARRAY, context=CORE)
    b = ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    scope = Context.joined(a.context, b.context)
    group = _group(scope, a=a, b=b)
    assert group.context == CORE_AND_EXTENSIONS
    held = group.consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    assert held.metadata["a"] == a
    assert held.metadata["b"] == b


def test_update_keeps_a_models_scope_apart_from_the_groups() -> None:
    """A model given to `update` keeps nothing of its own scope in the group: the group's `context` is the group's, and the child's is the group's too."""
    group = ZarrV3GroupMetadata.create_default(context=CORE)
    updated = group.update(
        consolidated_metadata={
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": ZarrV3ArrayMetadata(ARRAY)},
        }
    )
    assert updated.context == CORE
    held = updated.consolidated_metadata
    assert isinstance(held, ZarrV3ConsolidatedMetadata)
    assert held.metadata["a"].context == CORE


def test_parse_gives_json_for_a_document_holding_models() -> None:
    """`parse_group_metadata_v3` of a document holding models gives JSON, each model as its document, which the guard then says yes to: the trio agree."""
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {**INLINE, "metadata": {"a": ZarrV3ArrayMetadata(ARRAY)}},
    }
    parsed = parse_group_metadata_v3(document)
    member = cast("Any", parsed["consolidated_metadata"])
    assert member["metadata"]["a"] == ZarrV3ArrayMetadata(ARRAY).to_json()
    assert is_group_metadata_v3(parsed)


def test_a_refused_model_entrys_reading_holds_no_model_of_another_scope() -> None:
    """A group model refused as an entry leaves, in its reading, no model of the scope it was read in: a reading of the group's scope holds models of that scope alone."""
    child = ZarrV3GroupMetadata(
        _group(CORE_AND_EXTENSIONS, x=ZarrV3ArrayMetadata(WITH_ZSTD)).to_json(),
        context=CORE_AND_EXTENSIONS,
    )
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {**INLINE, "metadata": {"a": child, "a/x": WITH_ZSTD}},
    }
    reading = read_group_metadata_v3(document, context=CORE)
    nested = reading.consolidated["a"]
    assert isinstance(nested, ZarrV3GroupMetadataReading)
    assert nested.metadata is None
    assert len(nested.problems) != 0
    assert nested.consolidated["x"].metadata is None
