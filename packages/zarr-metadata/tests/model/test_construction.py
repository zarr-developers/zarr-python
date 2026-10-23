"""A model checks itself when it is built, as pydantic's `__init__` does.

A v3 model is built only by reading its document. It is changed only by
`update`, which reads the new document. A document with a problem is
refused, with every problem reported, so no model is ever invalid. A v2
model is still a dataclass, and a `dataclasses.replace` that would make it
invalid is refused. `to_key_value` writes a model as it is, without
reading anything.
"""

from __future__ import annotations

import dataclasses
import math
from collections import UserDict
from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
)
from zarr_metadata.problems import MetadataValidationError, ValidationProblem
from zarr_metadata.v3.codec.gzip import GZIP_CODEC, GzipCodecConfiguration
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from zarr_metadata.definition import Nested

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


def _rebuilt(model: object, changes: dict[str, object]) -> Any:  # noqa: ANN401 - the tests give the model as object
    """Rebuild the v2 model `model` from its document, with `changes` replacing members, in the model's own scope. This is what `update` does, and it also works for the consolidated model."""
    held = cast("Any", model)
    return type(held)({**held.to_json(), **changes}, context=held.context)


_INLINE: dict[str, Any] = {"kind": "inline", "must_understand": False}


@pytest.mark.parametrize(
    "model",
    [ARRAY, GROUP, GROUP.consolidated_metadata],
    ids=["array", "group", "consolidated"],
)
def test_a_v3_model_is_built_of_its_document_in_its_scope(model: object) -> None:
    """A v3 model is its document read in its scope. Rebuilding it from its document and scope gives the same model."""
    held = cast("Any", model)
    assert type(held)(held.to_json(), context=held.context) == model


@pytest.mark.parametrize(
    "model",
    [
        V2_ARRAY,
        V2_GROUP,
        V2_CONSOLIDATED,
    ],
    ids=["v2-array", "v2-group", "v2-consolidated"],
)
def test_a_v2_model_is_built_as_a_read_builds_it(model: object) -> None:
    """A v2 model is built from its document in its scope, the same way a read builds it. The constructor, given the model's own document and scope, builds an equal model."""
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
    """A model holds arrays as tuples and objects as dicts, as a read holds them. So a model updated with other container types equals the model a read builds, and it round-trips through its store."""
    changed = cast("Any", model).update(**changes)
    assert changed == model
    assert type(changed).from_key_value(changed.to_key_value()) == changed


@pytest.mark.parametrize(
    ("model", "changes"),
    [
        (V2_ARRAY, {"shape": range(4, 5), "chunks": [4], "attributes": {"a": 1}}),
        (V2_CONSOLIDATED, {"metadata": {"a/.zattrs": UserDict({"x": 1})}}),
    ],
    ids=["v2-sequences", "v2-consolidated-mapping"],
)
def test_a_v2_model_holds_its_members_as_a_read_refines_them(
    model: object, changes: dict[str, object]
) -> None:
    """A v2 model holds arrays as tuples and objects as dicts, as a read holds them. So a v2 model rebuilt with other container types equals the model a read builds."""
    changed = _rebuilt(model, changes)
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
    """A model holds copies of the containers passed to `update`. Changing those containers afterwards does not change the model."""
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
        (V2_CONSOLIDATED, "metadata"),
    ],
    ids=["v2-array-attributes", "v2-group-attributes", "v2-consolidated"],
)
def test_a_v2_model_shares_no_container_with_what_it_was_built_of(
    model: object, member: str
) -> None:
    """A v2 model holds copies of the containers it is built from."""
    held: dict[str, object] = {"acme.x": {"must_understand": False}}
    built = _rebuilt(model, {member: held})
    held["acme.y"] = math.nan
    cast("dict[str, object]", held["acme.x"])["z"] = math.nan
    assert getattr(built, member) == {"acme.x": {"must_understand": False}}
    assert type(built).from_key_value(built.to_key_value()) == built


def test_a_model_is_read_when_built_and_written_as_it_is() -> None:
    """A model is read once, when it is built. `to_key_value` reads nothing. `update` reads the new document. A group reads the documents it holds once, as part of its own read."""
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
        # Both problems are reported: the grid and the names each cover one dimension.
        (
            ARRAY,
            {"shape": (4, 4)},
            [
                (("chunk_grid", "configuration", "chunk_shape"), "invalid_value"),
                (("dimension_names",), "invalid_value"),
            ],
        ),
        (ARRAY, {"attributes": {1: "a"}}, [(("attributes",), "invalid_type")]),
        # A value that is not an object is reported as the wrong type, even when empty.
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
    """`update` reads the new document, so a change that makes the document invalid is refused, with every problem reported. No model is invalid, however it was made."""
    with pytest.raises(MetadataValidationError) as raised:
        cast("Any", model).update(**changes)
    assert [(found.loc, found.kind) for found in raised.value.problems] == problems


def test_error_consolidated_metadata_of_documents_at_bad_paths_is_refused() -> None:
    """The consolidated member, read on its own, refuses documents at paths that no node can have. It also reports a missing parent group, as the group's read does."""
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
        (
            V2_CONSOLIDATED,
            {"metadata": {"a/.zarray": {"x": math.nan}}},
            [(("metadata", "a/.zarray", "x"), "invalid_value")],
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
    """The constructor reads the document the same way a read does, so no v2 model is invalid, however it was made."""
    with pytest.raises(MetadataValidationError) as raised:
        _rebuilt(model, changes)
    assert [(found.loc, found.kind) for found in raised.value.problems] == problems


def test_error_v2_create_default_refuses_chunks_its_default_shape_does_not_take() -> None:
    # Overriding chunks without shape keeps the scalar default shape (), as
    # the v3 model keeps its default shape. A chunk grid that does not fit
    # that shape is refused.
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata.create_default(chunks=(10, 10))
    assert [(found.loc, found.kind) for found in raised.value.problems] == [
        (("chunks",), "invalid_value")
    ]


def test_error_consolidated_metadata_paths_are_strings() -> None:
    """A `metadata` member with a key that is not a string is reported as a problem of the member, as the read reports it, and does not raise `TypeError`."""
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ConsolidatedMetadata({**_INLINE, "metadata": {1: ARRAY.to_json()}})
    assert [(found.loc, found.kind) for found in raised.value.problems] == [
        (("metadata",), "invalid_type")
    ]


def test_a_model_s_key_is_computed_when_it_is_first_compared() -> None:
    """`==` and `hash` use a key that represents what the document means. The key is computed the first time either is called, then cached. A model that is only read never calls a definition's `canonical` spelling."""
    asked: list[int] = []

    def counting(configuration: GzipCodecConfiguration) -> GzipCodecConfiguration:
        asked.append(1)
        return configuration

    scope = CORE_AND_EXTENSIONS.extended_with(dataclasses.replace(GZIP_CODEC, canonical=counting))
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "int8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes"}, {"name": "gzip", "configuration": {"level": 1}}],
    }
    model = ZarrV3ArrayMetadata(document, scope)
    assert model.to_json()["codecs"][1] == {"name": "gzip", "configuration": {"level": 1}}
    assert asked == []
    other = ZarrV3ArrayMetadata(document, scope)
    assert model == other
    assert len(asked) == 2
    assert hash(model) == hash(other)
    assert model != ZarrV3ArrayMetadata({**document, "fill_value": 1}, scope)
    assert len(asked) == 3


def test_error_a_canonical_that_raises_is_raised_when_the_model_is_first_compared() -> None:
    """A definition's `canonical` is code written by the extension author, and it is first called when a model is first compared. The model builds, and `hash` raises the error that `canonical` raised, with a note naming the definition."""

    def refuses(configuration: GzipCodecConfiguration) -> GzipCodecConfiguration:
        raise ValueError("no")

    scope = CORE_AND_EXTENSIONS.extended_with(dataclasses.replace(GZIP_CODEC, canonical=refuses))
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "int8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{"name": "bytes"}, {"name": "gzip", "configuration": {"level": 1}}],
    }
    model = ZarrV3ArrayMetadata(document, scope)
    assert model.to_json()["codecs"][1] == {"name": "gzip", "configuration": {"level": 1}}
    with pytest.raises(ValueError, match="no") as raised:
        hash(model)
    assert raised.value.__notes__ == ["raised by the canonical of 'gzip'"]
