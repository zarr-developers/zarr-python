"""A model checks itself when it is built, as pydantic's `__init__` does.

Built by hand, or changed as `dataclasses.replace` changes one, a model
whose document has a problem is refused at the change, with every problem,
so none is ever invalid. A model a read builds is not read a second time,
and `to_key_value` writes a model as it is.
"""

from __future__ import annotations

import dataclasses
import math
from collections import UserDict
from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata.model import (
    UNSET,
    MetadataValidationError,
    ValidationProblem,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
)
from zarr_metadata.model._validation import ZarrV3ArrayMetadataReading, construct
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS, Chunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.v3.definition import Nested

ARRAY = ZarrV3ArrayMetadata.create_default(
    shape=(4,),
    attributes={"a": [1, None]},
    dimension_names=("x",),
    acme={"must_understand": False},
)
GROUP = ZarrV3GroupMetadata.from_json(
    {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"a": 1},
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                "x": ARRAY.to_json(),
                "y": {"zarr_format": 3, "node_type": "group"},
                "y/z": ARRAY.to_json(),
            },
        },
    }
)
V2_ARRAY = ZarrV2ArrayMetadata.create_default(shape=(4,), attributes={"a": 1})
V2_GROUP = ZarrV2GroupMetadata.create_default(attributes={})
V2_CONSOLIDATED = ZarrV2ConsolidatedMetadata.from_json(
    {"zarr_consolidated_format": 1, "metadata": {"a/.zattrs": {"x": 1}}}
)


@pytest.mark.parametrize(
    "model",
    [
        ARRAY,
        GROUP,
        GROUP.consolidated_metadata,
        V2_ARRAY,
        V2_GROUP,
        V2_CONSOLIDATED,
    ],
    ids=["array", "group", "consolidated", "v2-array", "v2-group", "v2-consolidated"],
)
def test_a_model_is_built_as_a_read_builds_it(model: object) -> None:
    # The constructor checks what the read checked, and of the same members
    # builds the same model.
    held = cast("Any", model)
    members = {
        member.name: getattr(held, member.name)
        for member in dataclasses.fields(held)
        if member.init
    }
    assert type(held)(**members) == model


@pytest.mark.parametrize(
    ("model", "changes"),
    [
        (ARRAY, {"shape": [4], "dimension_names": ["x"]}),
        (ARRAY, {"attributes": UserDict({"a": [1, None]})}),
        (GROUP, {"attributes": UserDict({"a": 1})}),
        (V2_ARRAY, {"shape": range(4, 5), "chunks": [4], "attributes": {"a": 1}}),
        (V2_CONSOLIDATED, {"metadata": {"a/.zattrs": UserDict({"x": 1})}}),
    ],
    ids=[
        "array-lists",
        "array-mapping",
        "group-mapping",
        "v2-sequences",
        "v2-consolidated-mapping",
    ],
)
def test_a_model_holds_its_members_as_a_read_refines_them(
    model: object, changes: dict[str, object]
) -> None:
    # Arrays as tuples and objects as dicts, as the read holds them, so a
    # model built of other containers is the model a read builds.
    changed = dataclasses.replace(cast("Any", model), **changes)
    assert changed == model
    assert type(changed).from_key_value(changed.to_key_value()) == changed


@pytest.mark.parametrize(
    ("model", "member"),
    [
        (ARRAY, "extra_fields"),
        (GROUP, "attributes"),
        (V2_GROUP, "attributes"),
        (V2_CONSOLIDATED, "metadata"),
    ],
    ids=["array-extra-fields", "group-attributes", "v2-group-attributes", "v2-consolidated"],
)
def test_a_model_shares_no_container_with_what_it_was_built_of(model: object, member: str) -> None:
    held: dict[str, object] = {"acme.x": {"must_understand": False}}
    built = dataclasses.replace(cast("Any", model), **{member: held})
    held["acme.y"] = math.nan
    cast("dict[str, object]", held["acme.x"])["z"] = math.nan
    assert getattr(built, member) == {"acme.x": {"must_understand": False}}
    assert type(built).from_key_value(built.to_key_value()) == built


def test_a_model_is_read_once_and_written_as_it_is() -> None:
    values: list[object] = []

    def counted(
        configuration: object, nested: Nested, value: object
    ) -> Iterator[ValidationProblem]:
        values.append(value)
        yield from ()

    counting = dataclasses.replace(INT8_DATA_TYPE, fill_value_rules=counted)
    scope = CORE_AND_EXTENSIONS.extended_with(counting)
    model = ZarrV3ArrayMetadata.create_default(context=scope, data_type="int8", fill_value=3)
    model.to_key_value()
    # Built by hand, a model is read once, as its own fields read it.
    dataclasses.replace(model, fill_value=4)
    assert values == [3, 4]
    # A group checks its own members: each document it holds checked itself.
    group = ZarrV3GroupMetadata(
        attributes={},
        consolidated_metadata=ZarrV3ConsolidatedMetadata(metadata={"a": model}),
        extra_fields={},
    )
    dataclasses.replace(group, attributes={"b": 1}).to_key_value()
    assert values == [3, 4]


@pytest.mark.parametrize(
    ("model", "changes", "problems"),
    [
        (ARRAY, {"fill_value": math.nan}, [(("fill_value",), "invalid_value")]),
        (ARRAY, {"dimension_names": ("x", "y")}, [(("dimension_names",), "invalid_value")]),
        # Every problem: the grid and the names are each for one dimension.
        (
            ARRAY,
            {"shape": (4, 4)},
            [
                (("chunk_grid", "configuration", "chunk_shape"), "invalid_value"),
                (("dimension_names",), "invalid_value"),
            ],
        ),
        (ARRAY, {"attributes": {1: "a"}}, [(("attributes",), "invalid_type")]),
        # Empty or not, a value that is no object is judged as one.
        (ARRAY, {"attributes": []}, [(("attributes",), "invalid_type")]),
        (ARRAY, {"attributes": None}, [(("attributes",), "invalid_type")]),
        # Every problem: an extra field named as a member, and the rest.
        (
            ARRAY,
            {"extra_fields": {"shape": [1]}, "fill_value": math.nan},
            [(("extra_fields",), "invalid_value"), (("fill_value",), "invalid_value")],
        ),
        (GROUP, {"attributes": {1: "a"}}, [(("attributes",), "invalid_type")]),
        (GROUP, {"attributes": ()}, [(("attributes",), "invalid_type")]),
        (GROUP, {"extra_fields": {"acme": math.nan}}, [(("acme",), "invalid_value")]),
        (
            GROUP.consolidated_metadata,
            {"metadata": {"x": ARRAY, "x/a": ARRAY, "__b": ARRAY, "c/d": ARRAY}},
            [
                (("metadata", "__b"), "invalid_value"),
                (("metadata", "x/a"), "invalid_value"),
                (("metadata", "c"), "missing_key"),
            ],
        ),
        (V2_ARRAY, {"order": "Q"}, [(("order",), "invalid_value")]),
        (V2_ARRAY, {"chunks": (4, 4)}, [(("chunks",), "invalid_value")]),
        (V2_GROUP, {"attributes": {1: "a"}}, [(("attributes",), "invalid_type")]),
        (
            V2_CONSOLIDATED,
            {"metadata": {"a/.zarray": {"x": math.nan}}},
            [(("metadata", "a/.zarray", "x"), "invalid_value")],
        ),
    ],
    ids=[
        "fill-value-not-json",
        "names-for-another-rank",
        "shape-without-its-grid",
        "attribute-key",
        "attributes-empty-and-no-object",
        "attributes-null",
        "extra-field-named-as-a-member-and-more",
        "group-attribute-key",
        "group-attributes-empty-and-no-object",
        "group-extension-not-json",
        "consolidated-paths",
        "v2-order",
        "v2-chunks-for-another-rank",
        "v2-group-attribute-key",
        "v2-consolidated-entry-not-json",
    ],
)
def test_error_a_model_changed_by_hand_into_an_invalid_one_is_refused_at_the_change(
    model: object, changes: dict[str, object], problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """As a read reads its document: no model is invalid, however it came to be."""
    with pytest.raises(MetadataValidationError) as raised:
        dataclasses.replace(cast("Any", model), **changes)
    assert [(found.loc, found.kind) for found in raised.value.problems] == problems


def test_error_v2_create_default_refuses_chunks_its_default_shape_does_not_take() -> None:
    # Overriding chunks without shape keeps the scalar default shape (), as
    # the v3 model keeps its default shape, and refuses a grid it does not fit.
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata.create_default(chunks=(10, 10))
    assert [(found.loc, found.kind) for found in raised.value.problems] == [
        (("chunks",), "invalid_value")
    ]


def test_error_construct_refuses_a_member_the_model_does_not_take() -> None:
    with pytest.raises(TypeError, match="ZarrV2GroupMetadata has no member \\['bogus'\\] to build"):
        construct(ZarrV2GroupMetadata, attributes=UNSET, bogus=1)


def test_error_construct_refuses_a_model_missing_a_member() -> None:
    with pytest.raises(TypeError, match="ZarrV2GroupMetadata is built with 'attributes'"):
        construct(ZarrV2GroupMetadata)


@pytest.mark.parametrize("model", [ARRAY, GROUP], ids=["array", "group"])
def test_error_extra_fields_are_a_mapping(model: object) -> None:
    with pytest.raises(TypeError, match="extra_fields: expected a mapping of names to JSON"):
        dataclasses.replace(cast("Any", model), extra_fields=[("acme", 1)])


def test_error_consolidated_metadata_paths_are_strings() -> None:
    with pytest.raises(TypeError, match="a document's path is a string, got 1"):
        ZarrV3ConsolidatedMetadata(metadata=cast("Any", {1: ARRAY}))


def test_construct_fills_a_member_from_its_default_factory() -> None:
    reading = construct(ZarrV3ArrayMetadataReading, problems=())
    assert reading.chunk == Chunk()
    assert reading.pipeline == ()
