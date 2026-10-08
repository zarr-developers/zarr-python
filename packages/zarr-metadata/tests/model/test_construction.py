"""A model checks itself when it is built, as pydantic's `__init__` does.

A v3 model is built only by reading its document, and changed only by
`update`, which reads the document it makes: a document with a problem is
refused, with every problem, so no model is ever invalid. A v2 model, a
dataclass still, is refused at a `dataclasses.replace` into an invalid
one. `to_key_value` writes a model as it is, reading nothing.
"""

from __future__ import annotations

import dataclasses
import math
from collections import UserDict
from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata.model import (
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


_INLINE: dict[str, Any] = {"kind": "inline", "must_understand": False}


@pytest.mark.parametrize(
    "model",
    [ARRAY, GROUP, GROUP.consolidated_metadata],
    ids=["array", "group", "consolidated"],
)
def test_a_v3_model_is_built_of_its_document_in_its_scope(model: object) -> None:
    """A v3 model is its document read in its scope: built again of the two, it is the same model."""
    held = cast("Any", model)
    assert type(held)(held.to_json(), context=held.context) == model


@pytest.mark.parametrize(
    "model",
    [
        V2_ARRAY,
        V2_GROUP,
        pytest.param(V2_CONSOLIDATED, marks=pytest.mark.xfail(strict=True, reason="Task 4")),
    ],
    ids=["v2-array", "v2-group", "v2-consolidated"],
)
def test_a_v2_model_is_built_as_a_read_builds_it(model: object) -> None:
    """A v2 model is built of its document in its scope, as a read builds it: the constructor given the model's own document and scope builds an equal model."""
    held = cast("Any", model)
    assert type(held)(held.to_json(), context=held.context) == model


@pytest.mark.parametrize(
    ("model", "changes"),
    [
        (ARRAY, {"shape": [4], "dimension_names": ["x"]}),
        (ARRAY, {"attributes": UserDict({"a": [1, None]})}),
        (GROUP, {"attributes": UserDict({"a": 1})}),
    ],
    ids=["array-lists", "array-mapping", "group-mapping"],
)
def test_a_v3_model_holds_its_members_as_a_read_refines_them(
    model: object, changes: dict[str, object]
) -> None:
    """Arrays as tuples and objects as dicts, as the read holds them, so a model updated with other containers is the model a read builds, and round-trips through its store."""
    changed = cast("Any", model).update(**changes)
    assert changed == model
    assert type(changed).from_key_value(changed.to_key_value()) == changed


@pytest.mark.parametrize(
    ("model", "changes"),
    [
        (V2_ARRAY, {"shape": range(4, 5), "chunks": [4], "attributes": {"a": 1}}),
        pytest.param(
            V2_CONSOLIDATED,
            {"metadata": {"a/.zattrs": UserDict({"x": 1})}},
            marks=pytest.mark.xfail(strict=True, reason="Task 4"),
        ),
    ],
    ids=["v2-sequences", "v2-consolidated-mapping"],
)
def test_a_v2_model_holds_its_members_as_a_read_refines_them(
    model: object, changes: dict[str, object]
) -> None:
    """Arrays as tuples and objects as dicts, as the read holds them, so a v2 model updated with other containers is the model a read builds."""
    changed = cast("Any", model).update(**changes)
    assert changed == model
    assert type(changed).from_key_value(changed.to_key_value()) == changed


@pytest.mark.parametrize(
    ("model", "member"),
    [(ARRAY, "acme.x"), (GROUP, "attributes")],
    ids=["array-extra-field", "group-attributes"],
)
def test_a_v3_model_shares_no_container_with_what_it_was_built_of(
    model: object, member: str
) -> None:
    """A model holds copies of the containers `update` is given: changing them afterwards changes nothing it holds."""
    held: dict[str, object] = {"must_understand": False, "z": {"y": 1}}
    built = cast("Any", model).update(**{member: held})
    held["w"] = math.nan
    cast("dict[str, object]", held["z"])["y"] = math.nan
    expected = {"must_understand": False, "z": {"y": 1}}
    if member == "attributes":
        assert built.attributes == expected
    else:
        assert built.extra_fields[member] == expected
    assert type(built).from_key_value(built.to_key_value()) == built


@pytest.mark.parametrize(
    ("model", "member"),
    [
        (V2_ARRAY, "attributes"),
        (V2_GROUP, "attributes"),
        pytest.param(
            V2_CONSOLIDATED, "metadata", marks=pytest.mark.xfail(strict=True, reason="Task 4")
        ),
    ],
    ids=["v2-array-attributes", "v2-group-attributes", "v2-consolidated"],
)
def test_a_v2_model_shares_no_container_with_what_it_was_built_of(
    model: object, member: str
) -> None:
    """A v2 model holds copies of the containers it is built of."""
    held: dict[str, object] = {"acme.x": {"must_understand": False}}
    built = cast("Any", model).update(**{member: held})
    held["acme.y"] = math.nan
    cast("dict[str, object]", held["acme.x"])["z"] = math.nan
    assert getattr(built, member) == {"acme.x": {"must_understand": False}}
    assert type(built).from_key_value(built.to_key_value()) == built


def test_a_model_is_read_when_built_and_written_as_it_is() -> None:
    """A model is read once, when it is built; `to_key_value` reads nothing; `update` reads the document it makes; a group reads the documents it holds once, as part of its own read."""
    values: list[object] = []

    def counted(
        configuration: object, nested: Nested, value: object
    ) -> Iterator[ValidationProblem]:
        values.append(value)
        yield from ()

    counting = dataclasses.replace(INT8_DATA_TYPE, fill_value_rules=counted)
    scope = CORE_AND_EXTENSIONS.extended_with(counting)
    model = ZarrV3ArrayMetadata.create_default(context=scope, data_type="int8", fill_value=3)
    assert values == [3]
    model.to_key_value()
    assert values == [3]
    changed = model.update(fill_value=4)
    assert values == [3, 4]
    group = ZarrV3GroupMetadata.create_default(
        context=scope, consolidated_metadata={**_INLINE, "metadata": {"a": changed.to_json()}}
    )
    assert values == [3, 4, 4]
    group.to_key_value()
    assert values == [3, 4, 4]


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
        (GROUP, {"attributes": {1: "a"}}, [(("attributes",), "invalid_type")]),
        (GROUP, {"attributes": ()}, [(("attributes",), "invalid_type")]),
        (GROUP, {"acme": math.nan}, [(("acme",), "invalid_value")]),
    ],
    ids=[
        "fill-value-not-json",
        "names-for-another-rank",
        "shape-without-its-grid",
        "attribute-key",
        "attributes-empty-and-no-object",
        "attributes-null",
        "group-attribute-key",
        "group-attributes-empty-and-no-object",
        "group-extension-not-json",
    ],
)
def test_error_a_v3_model_updated_into_an_invalid_one_is_refused_at_the_change(
    model: object, changes: dict[str, object], problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """`update` reads the document it makes, so a change that makes an invalid one is refused with every problem: no model is invalid, however it came to be."""
    with pytest.raises(MetadataValidationError) as raised:
        cast("Any", model).update(**changes)
    assert [(found.loc, found.kind) for found in raised.value.problems] == problems


def test_error_consolidated_metadata_of_documents_at_bad_paths_is_refused() -> None:
    """The consolidated member read on its own refuses documents at paths no node has, and reports a group missing above one, as the group's read does."""
    documents = {
        "x": ARRAY.to_json(),
        "x/a": ARRAY.to_json(),
        "__b": ARRAY.to_json(),
        "c/d": ARRAY.to_json(),
    }
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ConsolidatedMetadata({**_INLINE, "metadata": documents})
    assert [(found.loc, found.kind) for found in raised.value.problems] == [
        (("metadata", "__b"), "invalid_value"),
        (("metadata", "x/a"), "invalid_value"),
        (("metadata", "c"), "missing_key"),
    ]


@pytest.mark.parametrize(
    ("model", "changes", "problems"),
    [
        (V2_ARRAY, {"order": "Q"}, [(("order",), "invalid_value")]),
        (V2_ARRAY, {"chunks": (4, 4)}, [(("chunks",), "invalid_value")]),
        (V2_GROUP, {"attributes": {1: "a"}}, [(("attributes",), "invalid_type")]),
        pytest.param(
            V2_CONSOLIDATED,
            {"metadata": {"a/.zarray": {"x": math.nan}}},
            [(("metadata", "a/.zarray", "x"), "invalid_value")],
            marks=pytest.mark.xfail(strict=True, reason="Task 4"),
        ),
    ],
    ids=[
        "v2-order",
        "v2-chunks-for-another-rank",
        "v2-group-attribute-key",
        "v2-consolidated-entry-not-json",
    ],
)
def test_error_a_v2_model_changed_by_hand_into_an_invalid_one_is_refused_at_the_change(
    model: object, changes: dict[str, object], problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    """As a read reads its document: no v2 model is invalid, however it came to be."""
    with pytest.raises(MetadataValidationError) as raised:
        cast("Any", model).update(**changes)
    assert [(found.loc, found.kind) for found in raised.value.problems] == problems


def test_error_v2_create_default_refuses_chunks_its_default_shape_does_not_take() -> None:
    # Overriding chunks without shape keeps the scalar default shape (), as
    # the v3 model keeps its default shape, and refuses a grid it does not fit.
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata.create_default(chunks=(10, 10))
    assert [(found.loc, found.kind) for found in raised.value.problems] == [
        (("chunks",), "invalid_value")
    ]


def test_error_consolidated_metadata_paths_are_strings() -> None:
    """A `metadata` member keyed by what is no string is a problem of the member, as the read reports it, not a `TypeError`."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ConsolidatedMetadata({**_INLINE, "metadata": {1: ARRAY.to_json()}})
    assert [(found.loc, found.kind) for found in raised.value.problems] == [
        (("metadata",), "invalid_type")
    ]


def test_construct_fills_a_member_from_its_default_factory() -> None:
    reading = construct(ZarrV3ArrayMetadataReading, problems=())
    assert reading.chunk == Chunk()
    assert reading.pipeline == ()
