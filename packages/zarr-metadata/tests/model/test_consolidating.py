"""Tests for building a group's consolidated metadata from node models. A model is accepted when it reads the same in the group's scope, or gains information there. It is refused when it conflicts or would lose information."""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any, cast, get_type_hints

import pytest

from zarr_metadata._json import JSON_DEPTH
from zarr_metadata.definition import Context
from zarr_metadata.definition._scope import claims_of
from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3ConsolidatedMetadataInput,
    ZarrV3GroupMetadata,
    ZarrV3GroupMetadataReading,
    ZarrV3NodeMetadataInput,
    is_group_metadata_v3,
    parse_group_metadata_v3,
    read_group_metadata_v3,
    validate_group_metadata_v3,
)
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v3.codec.crc32c import Empty
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
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
    """A node model can be given wherever consolidated metadata lists a document: at the top level, or inside a document listed there. The model is accepted in two cases. If the group's scope reads every claim of the model identically, the model's reading is kept and not read again. If the group's scope claims something that the model's scope left unclaimed, the model's document is read again in the group's scope. In both cases the group holds the model in its own scope, equals the group built from the documents, and writes plain JSON. A listed group's own listing is listed flat next to it, as the convention asks."""
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
    """Return `value` with every node model in it, at any depth, replaced by its document."""
    if isinstance(value, (ZarrV3ArrayMetadata, ZarrV3GroupMetadata)):
        return value.to_json()
    if isinstance(value, dict):
        return {key: _documents(item) for key, item in cast("dict[str, object]", value).items()}
    return value


def test_documents_as_written_and_pickle() -> None:
    """A group built from models writes each child's document as the child wrote it, and pickles like any other group."""
    child = ZarrV3ArrayMetadata({**ARRAY, "data_type": {"name": "uint8"}})
    group = _group(None, a=child)
    written = cast("Any", group.to_json()["consolidated_metadata"])
    assert written["metadata"]["a"] == child.to_json()
    assert pickle.loads(pickle.dumps(group)) == group


def test_error_a_model_that_conflicts_with_the_scope_is_refused_at_its_path() -> None:
    """A model read with a private `bytes`, given to a group whose scope reads the core `bytes`, is a conflict. The problem is at the entry's path, names the kind and the name, and tells the two definitions apart. The model is not silently read again."""
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
    """A model read in a scope that claims `zstd`, given to a group whose scope leaves `zstd` unclaimed, would lose information. It is refused, the same way a conflict is."""
    child = ZarrV3ArrayMetadata(WITH_ZSTD, context=CORE_AND_EXTENSIONS)
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("consolidated_metadata", "metadata", "a", "codecs", 1)
    assert problem.message.endswith("which the group's scope leaves unclaimed")


def test_error_a_gain_that_surfaces_a_problem_is_reported_at_the_problem() -> None:
    """A model whose scope left `zstd` unclaimed, given to a group whose scope claims it, is read again in the group's scope. A level that the `zstd` definition refuses is reported as a problem at its location."""
    child = ZarrV3ArrayMetadata(WITH_LOOSE_ZSTD, context=CORE)
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE_AND_EXTENSIONS, a=child)
    assert [problem.loc for problem in raised.value.problems] == [
        ("consolidated_metadata", "metadata", "a", "codecs", 1, "configuration", "level")
    ]


def test_readers_see_models_as_the_constructor_does() -> None:
    """Given a document that holds models, `read_group_metadata_v3` and `validate_group_metadata_v3` read them the same way the constructor does. The reading holds the adopted model, and the validator reports a conflict."""
    fine = {**INLINE, "metadata": {"a": ZarrV3ArrayMetadata(ARRAY)}}
    document = {"zarr_format": 3, "node_type": "group", "consolidated_metadata": fine}
    reading = read_group_metadata_v3(document)
    assert reading.problems == ()
    assert isinstance(reading.consolidated["a"].metadata, ZarrV3ArrayMetadata)
    assert validate_group_metadata_v3(document) == ()


def test_error_the_validator_reports_a_conflicting_model_as_the_constructor_does() -> None:
    """Given a document that holds a model which conflicts with the scope, `validate_group_metadata_v3` reports the conflict at the entry's path, as the constructor refuses it."""
    clashing = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            **INLINE,
            "metadata": {"a": ZarrV3ArrayMetadata(ARRAY, context=PRIVATE)},
        },
    }
    assert [problem.loc for problem in validate_group_metadata_v3(clashing)] == [
        ("consolidated_metadata", "metadata", "a", "codecs", 0)
    ]


def test_consolidated_metadata_given_whole_is_taken_as_its_models() -> None:
    """A `ZarrV3ConsolidatedMetadata` given as the member is treated as its models at their paths. So `update(consolidated_metadata=other.consolidated_metadata)` copies them over."""
    source = _group(CORE_AND_EXTENSIONS, a=ZarrV3ArrayMetadata(ARRAY))
    target = ZarrV3GroupMetadata.create_default(attributes={"t": 1}).update(
        consolidated_metadata=source.consolidated_metadata
    )
    assert target.consolidated_metadata == source.consolidated_metadata
    assert target.attributes == {"t": 1}


def test_children_of_different_scopes_consolidate_in_their_join() -> None:
    """Children read in different scopes are consolidated in the scope `Context.joined` builds from their scopes. Each child refines into the joined scope, and the group reads in it."""
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
    """A model given to `update` does not bring its own scope into the group. The group's `context` stays the group's own, and the child's `context` becomes the group's too."""
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
    """`parse_group_metadata_v3` of a document that holds models returns JSON, with each model replaced by its document. The guard then accepts that JSON, so the three entry points agree."""
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {**INLINE, "metadata": {"a": ZarrV3ArrayMetadata(ARRAY)}},
    }
    parsed = parse_group_metadata_v3(document)
    member = cast("Any", parsed["consolidated_metadata"])
    assert member["metadata"]["a"] == ZarrV3ArrayMetadata(ARRAY).to_json()
    assert is_group_metadata_v3(parsed)


def test_error_a_refused_model_entrys_reading_is_of_the_groups_scope() -> None:
    """A group model refused as an entry is read again in the group's scope to build its reading. So the group's reading holds no field and no model from another scope: `claims_of` on its fields finds one scope."""
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
    assert claims_of(reading.fields()) == claims_of(
        read_group_metadata_v3(_documents(document), context=CORE).fields()
    )
    for _, field in reading.fields():
        assert field.definition is None or field.definition in CORE.definitions()


def test_error_a_model_listed_too_deep_is_refused_where_it_sits() -> None:
    """A model that is valid on its own, with attributes nested to the last level a reader walks, goes past the depth cap when listed as a document in consolidated metadata. The validator, the constructor and the reader all refuse it there, as they would refuse its document."""
    deep: list[object] = []
    for _ in range(JSON_DEPTH - 3):
        deep = [deep]
    child = ZarrV3ArrayMetadata({**ARRAY, "attributes": {"x": deep}})
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {**INLINE, "metadata": {"a": child}},
    }
    problems = validate_group_metadata_v3(document)
    assert [problem.loc[:4] for problem in problems] == [
        ("consolidated_metadata", "metadata", "a", "attributes")
    ]
    with pytest.raises(MetadataValidationError):
        ZarrV3GroupMetadata(document)
    assert read_group_metadata_v3(document).metadata is None


def test_error_a_data_type_conflict_names_the_kind_and_the_written_name() -> None:
    """A conflict over a data type names the kind in words and the name as the document writes it (`r16`, which the `r*` definition claims). When the two definitions print the same, the message says that they differ."""
    strict = dataclasses.replace(
        RAW_BYTES_DATA_TYPE, fill_value_rules=lambda configuration, nested, value: iter(())
    )
    child = ZarrV3ArrayMetadata(
        {**ARRAY, "data_type": "r16", "fill_value": [0, 0], "codecs": ("bytes",)},
        context=CORE.extended_with(strict),
    )
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.loc == ("consolidated_metadata", "metadata", "a", "data_type")
    assert problem.message == (
        "expected a document read in the group's scope, got a model that reads the data type "
        "'r16' by another definition than the one the group's scope reads it by, "
        "DataTypeDefinition(name='r*') of RawBytesConfiguration"
    )


def test_error_a_conflict_between_definitions_alike_says_they_differ() -> None:
    """Two definitions of the same name that read the same TypedDict but have different rules are distinguished in the message by stating that they differ, since nothing else shows the difference."""
    strict = dataclasses.replace(GZIP_CODEC, rules=lambda configuration, nested: iter(()))
    child = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ("bytes", {"name": "gzip", "configuration": {"level": 1}})},
        context=CORE.extended_with(strict),
    )
    with pytest.raises(MetadataValidationError) as raised:
        _group(CORE, a=child)
    (problem,) = raised.value.problems
    assert problem.message == (
        "expected a document read in the group's scope, got a model that reads the codec "
        "'gzip' by another definition than the one the group's scope reads it by, "
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
    """A model read with the core `bytes` and an `endian`, given to a group whose private `bytes` takes no configuration, is refused first for the conflict and then for the `endian` key that the private `bytes` does not accept. The cause is reported before its symptom."""
    child = ZarrV3ArrayMetadata(
        {**ARRAY, "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},)}
    )
    with pytest.raises(MetadataValidationError) as raised:
        _group(PRIVATE, a=child)
    assert [problem.loc for problem in raised.value.problems] == [
        ("consolidated_metadata", "metadata", "a", "codecs", 0),
        ("consolidated_metadata", "metadata", "a", "codecs", 0, "configuration", "endian"),
    ]
