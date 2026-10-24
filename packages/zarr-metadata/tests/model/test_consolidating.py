"""Tests for building consolidated metadata from node models. A model is accepted when it reads the same in the scope of the consolidated metadata, or gains information there. It is refused when it conflicts or would lose information. A group takes the result through `update`, and writes its document."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any, cast, get_type_hints

import pytest

from zarr_metadata._json import JSON_DEPTH
from zarr_metadata.definition import EmptyConfiguration
from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3ConsolidatedMetadataInput,
    ZarrV3GroupMetadata,
    ZarrV3NodeMetadataInput,
    is_group_metadata_v3,
    read_consolidated_metadata_v3,
    validate_group_metadata_v3,
)
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
    ZarrV3Context,
)

ARRAY: dict[str, Any] = {
    **ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json(),
    "codecs": ("bytes",),
}
"""A `uint8` array whose `bytes` codec has no configuration, so a private `bytes` definition with no configuration also reads it."""
ZSTD = {"name": "zstd", "configuration": {"level": 3, "checksum": False}}
LOOSE_ZSTD = {"name": "zstd", "configuration": {"level": 30}}
WITH_ZSTD = {**ARRAY, "codecs": (*ARRAY["codecs"], ZSTD)}
WITH_LOOSE_ZSTD = {**ARRAY, "codecs": (*ARRAY["codecs"], LOOSE_ZSTD)}
MY_BYTES = CodecDefinition(
    name="bytes", configuration=EmptyConfiguration, kind="array_bytes", size="static"
)
PRIVATE = CORE.extended_with(codecs=[MY_BYTES])
INLINE: dict[str, Any] = {"kind": "inline", "must_understand": False}


def _consolidated(
    context: ZarrV3Context | None = None, **entries: object
) -> ZarrV3ConsolidatedMetadata:
    """The consolidated metadata that lists `entries`, read in `context`."""
    return ZarrV3ConsolidatedMetadata({**INLINE, "metadata": entries}, context=context)


def _group_document(**entries: object) -> dict[str, Any]:
    """A group document whose consolidated metadata lists `entries`."""
    return {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {**INLINE, "metadata": entries},
    }


@pytest.mark.parametrize(
    ("context", "child", "beside", "gained"),
    [
        (CORE_AND_EXTENSIONS, ZarrV3ArrayMetadata(ARRAY), {}, False),
        (CORE_AND_EXTENSIONS, ZarrV3ArrayMetadata(ARRAY, context=CORE), {}, False),
        (CORE_AND_EXTENSIONS, ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE), {}, True),
        (
            CORE,
            ZarrV3GroupMetadata(_group_document(x=ARRAY), context=ZarrV3Context()),
            {"a/x": ZarrV3ArrayMetadata(ARRAY, context=ZarrV3Context())},
            False,
        ),
    ],
    ids=["same-scope", "unused-definitions", "gain", "listed-group-and-its-child"],
)
def test_a_model_is_accepted_as_a_consolidated_entry_when_it_refines_into_the_scope(
    context: ZarrV3Context,
    child: ZarrV3ArrayMetadata | ZarrV3GroupMetadata,
    beside: dict[str, Any],
    gained: bool,
) -> None:
    """A node model can be given wherever consolidated metadata lists a document. The model is accepted in two cases. If the scope of the consolidated metadata reads every claim of the model identically, the model's reading is kept and not read again. If that scope claims something that the model's scope left unclaimed, the model's document is read again in that scope. In both cases the consolidated metadata holds the model in its own scope, equals the one built from the documents, and a group it is given to writes plain JSON. A group model has no fields, so it is always kept. A listed group's own listing is listed flat next to it, as the convention asks."""
    held = _consolidated(context, a=child, **beside)
    node = held.metadata["a"]
    assert node.context is held.context
    assert cast("Any", node).refines(child)
    assert (node == child) is not gained
    if not gained and isinstance(child, ZarrV3ArrayMetadata):
        assert isinstance(node, ZarrV3ArrayMetadata)
        assert node.reading.pipeline is child.reading.pipeline
    written = {path: _documents(entry) for path, entry in beside.items()}
    assert held == _consolidated(context, a=child.to_json(), **written)
    group = ZarrV3GroupMetadata.create_default(context=context, consolidated_metadata=held)
    assert is_group_metadata_v3(group.to_json())


def _documents(value: object) -> object:
    """Return `value` with every node model in it, at any depth, replaced by its document."""
    if isinstance(value, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
        return value.to_json()
    if isinstance(value, dict):
        return {key: _documents(item) for key, item in cast("dict[str, object]", value).items()}
    return value


def test_documents_as_written_and_pickle() -> None:
    """Consolidated metadata built from models writes each child's document as the child wrote it, and so does a group it is given to. Both pickle like any other model."""
    child = ZarrV3ArrayMetadata({**ARRAY, "data_type": {"name": "uint8"}})
    held = _consolidated(None, a=child)
    assert held.to_json()["metadata"]["a"] == child.to_json()
    group = ZarrV3GroupMetadata.create_default(consolidated_metadata=held)
    written = cast("Any", group.to_json())["consolidated_metadata"]
    assert written["metadata"]["a"] == child.to_json()
    assert pickle.loads(pickle.dumps(held)) == held
    assert pickle.loads(pickle.dumps(group)) == group


def test_error_a_model_that_conflicts_with_the_scope_is_refused_at_its_path() -> None:
    """A model read with a private `bytes`, given to consolidated metadata whose scope reads the core `bytes`, is a conflict. The problem is at the entry's path, names the kind and the name, and tells the two definitions apart. The model is not silently read again."""
    child = ZarrV3ArrayMetadata(ARRAY, context=PRIVATE)
    with pytest.raises(MetadataValidationError) as raised:
        _consolidated(CORE_AND_EXTENSIONS, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("metadata", "a", "codecs", 0)
    assert problem.kind == "invalid_value"
    assert problem.message == (
        "expected a document read in the scope of the consolidated metadata, got a model that "
        "reads the codec 'bytes' by CodecDefinition(name='bytes') of EmptyConfiguration, which "
        "that scope reads by CodecDefinition(name='bytes') of BytesCodecConfiguration"
    )


def test_error_a_model_that_would_lose_a_meaning_is_refused_at_its_path() -> None:
    """A model read in a scope that claims `zstd`, given to consolidated metadata whose scope leaves `zstd` unclaimed, would lose information. It is refused, the same way a conflict is."""
    child = ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    with pytest.raises(MetadataValidationError) as raised:
        _consolidated(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("metadata", "a", "codecs", 1)
    assert problem.message.endswith("which that scope leaves unclaimed")


def test_error_a_gain_that_surfaces_a_problem_is_reported_at_the_problem() -> None:
    """A model whose scope left `zstd` unclaimed, given to consolidated metadata whose scope claims it, is read again in that scope. A level that the `zstd` definition refuses is reported as a problem at its location."""
    child = ZarrV3ArrayMetadata(WITH_LOOSE_ZSTD, context=CORE)
    with pytest.raises(MetadataValidationError) as raised:
        _consolidated(CORE_AND_EXTENSIONS, a=child)
    assert [problem.loc for problem in raised.value.problems] == [
        ("metadata", "a", "codecs", 1, "configuration", "level")
    ]


def test_the_reader_sees_models_as_the_constructor_does() -> None:
    """Given a member that lists models, `read_consolidated_metadata_v3` reads them the same way the constructor does, and holds the adopted model."""
    reading = read_consolidated_metadata_v3(
        {**INLINE, "metadata": {"a": ZarrV3ArrayMetadata(ARRAY)}}
    )
    assert reading.problems == ()
    assert reading.metadata is not None
    assert isinstance(reading.metadata.metadata["a"], ZarrV3ArrayMetadata)


def test_error_the_reader_reports_a_conflicting_model_as_the_constructor_does() -> None:
    """Given a member that lists a model which conflicts with the scope, `read_consolidated_metadata_v3` reports the conflict at the entry's path, as the constructor refuses it."""
    member = {**INLINE, "metadata": {"a": ZarrV3ArrayMetadata(ARRAY, context=PRIVATE)}}
    reading = read_consolidated_metadata_v3(member)
    assert [problem.loc for problem in reading.problems] == [("metadata", "a", "codecs", 0)]
    assert reading.metadata is None


def test_a_group_writes_the_consolidated_metadata_it_is_given_in_its_own_scope() -> None:
    """A `ZarrV3ConsolidatedMetadata` given to `update` is written as its document. It does not bring its own scope into the group: the group's `context` stays its own, and reading the written member in that scope gives the models in it."""
    held = _consolidated(CORE_AND_EXTENSIONS, a=ZarrV3ArrayMetadata(ARRAY))
    group = ZarrV3GroupMetadata.create_default(context=CORE, attributes={"t": 1})
    updated = group.update(consolidated_metadata=held)
    assert updated.context == CORE
    assert updated.attributes == {"t": 1}
    assert updated.consolidated_metadata == held.to_json()
    again = ZarrV3ConsolidatedMetadata(updated.consolidated_metadata, context=updated.context)
    assert again.metadata["a"].context == CORE
    assert again == held


def test_children_of_different_scopes_consolidate_in_their_join() -> None:
    """Children read in different scopes are consolidated in the scope `ZarrV3Context.joined` builds from their scopes. Each child refines into the joined scope, and the consolidated metadata reads in it."""
    a = ZarrV3ArrayMetadata(ARRAY, context=CORE)
    b = ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    context = ZarrV3Context.joined(a.context, b.context)
    held = _consolidated(context, a=a, b=b)
    assert held.context == CORE_AND_EXTENSIONS
    assert held.metadata["a"] == a
    assert held.metadata["b"] == b


def test_error_a_model_inside_a_listed_document_is_not_json() -> None:
    """Node models are taken only as the entries of consolidated metadata. A model inside a listed group's own listing is a value that is not JSON, and is refused there."""
    listed = _group_document(x=ZarrV3ArrayMetadata(ARRAY))
    reading = read_consolidated_metadata_v3({**INLINE, "metadata": {"a": listed, "a/x": ARRAY}})
    assert [(problem.loc, problem.kind) for problem in reading.problems] == [
        (("metadata", "a", "consolidated_metadata", "metadata", "x"), "invalid_type")
    ]


def test_error_a_model_listed_too_deep_is_refused_where_it_sits() -> None:
    """A model that is valid on its own, with attributes nested to the last level a reader walks, goes past the depth cap when listed as a document in consolidated metadata, which sits under a group's key. The reader and the constructor refuse it there, as a group refuses its document."""
    deep: list[object] = []
    for _ in range(JSON_DEPTH - 3):
        deep = [deep]
    child = ZarrV3ArrayMetadata({**ARRAY, "attributes": {"x": deep}})
    member = {**INLINE, "metadata": {"a": child}}
    reading = read_consolidated_metadata_v3(member)
    assert [problem.loc[:3] for problem in reading.problems] == [("metadata", "a", "attributes")]
    assert reading.metadata is None
    with pytest.raises(MetadataValidationError):
        ZarrV3ConsolidatedMetadata(member)
    assert [
        problem.loc[:4]
        for problem in validate_group_metadata_v3(_group_document(a=child.to_json()))
    ] == [("consolidated_metadata", "metadata", "a", "attributes")]


def test_error_a_data_type_conflict_names_the_kind_and_the_written_name() -> None:
    """A conflict over a data type names the kind in words and the name as the document writes it (`r16`, which the `r*` definition claims). When the two definitions print the same, the message says that they differ."""
    strict = dataclasses.replace(
        RAW_BYTES_DATA_TYPE, fill_value_rules=lambda configuration, nested, value: iter(())
    )
    child = ZarrV3ArrayMetadata(
        {**ARRAY, "data_type": "r16", "fill_value": [0, 0], "codecs": ("bytes",)},
        context=CORE.extended_with(data_types=[strict]),
    )
    with pytest.raises(MetadataValidationError) as raised:
        _consolidated(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("metadata", "a", "data_type")
    assert problem.message == (
        "expected a document read in the scope of the consolidated metadata, got a model that "
        "reads the data type 'r16' by another definition than the one that scope reads it by, "
        "DataTypeDefinition(name='r*') of RawBytesConfiguration"
    )


def test_error_a_conflict_between_definitions_alike_says_they_differ() -> None:
    """Two definitions of the same name that read the same TypedDict but have different rules are distinguished in the message by stating that they differ, since nothing else shows the difference."""
    strict = dataclasses.replace(GZIP_CODEC, rules=lambda configuration, nested: iter(()))
    child = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ("bytes", {"name": "gzip", "configuration": {"level": 1}})},
        context=CORE.extended_with(codecs=[strict]),
    )
    with pytest.raises(MetadataValidationError) as raised:
        _consolidated(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.message == (
        "expected a document read in the scope of the consolidated metadata, got a model that "
        "reads the codec 'gzip' by another definition than the one that scope reads it by, "
        "CodecDefinition(name='gzip') of GzipCodecConfiguration"
    )


def test_the_input_types_are_types_a_signature_can_hold() -> None:
    """`ZarrV3NodeMetadataInput` and `ZarrV3ConsolidatedMetadataInput` resolve to types, not strings, when a caller's `get_type_hints` reads them as annotations at run time."""

    def take(entry: ZarrV3NodeMetadataInput, member: ZarrV3ConsolidatedMetadataInput) -> None:
        pass

    hints = get_type_hints(take)
    assert {"entry", "member"} <= set(hints)
    assert not isinstance(hints["entry"], str)


def test_error_a_conflict_is_reported_before_what_the_re_read_finds() -> None:
    """A model read with the core `bytes` and an `endian`, given to consolidated metadata whose private `bytes` takes no configuration, is refused first for the conflict and then for the `endian` key that the private `bytes` does not accept. The cause is reported before its symptom."""
    child = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},)}
    )
    with pytest.raises(MetadataValidationError) as raised:
        _consolidated(PRIVATE, a=child)
    assert [problem.loc for problem in raised.value.problems] == [
        ("metadata", "a", "codecs", 0),
        ("metadata", "a", "codecs", 0, "configuration", "endian"),
    ]
