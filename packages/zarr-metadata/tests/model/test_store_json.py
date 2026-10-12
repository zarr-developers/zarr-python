"""Where a non-finite number may be: in a node's attributes, as user data.

zarr-python writes attributes with the defaults of Python's `json` module,
so an attribute may hold `NaN`, `Infinity` or `-Infinity`. The model reads
such a store, validates it, and writes it back the same way. Wherever the
spec interprets a value, a non-finite number is refused when read, and a
document the reader would refuse is not written.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

import pytest

from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    is_group_metadata_v3,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class _Written(Protocol):
    """What a model writes: store keys, each with the bytes it holds."""

    def items(self) -> Iterable[tuple[str, bytes]]: ...


class _Stored(Protocol):
    """A model that writes itself to a store."""

    def to_json(self) -> Mapping[str, object]: ...

    def to_key_value(self, *, indent: int | str | None = None) -> _Written: ...


ATTRIBUTES: dict[str, object] = {
    "_FillValue": math.nan,
    "valid_range": [-math.inf, math.inf],
    "cf": {"missing_value": -math.inf},
}


def _held(attributes: object) -> tuple[bool, bool, bool]:
    """Whether each non-finite number in `ATTRIBUTES` is still there, arrays as tuples."""
    assert isinstance(attributes, Mapping)
    fill = attributes["_FillValue"]
    cf = attributes["cf"]
    assert isinstance(cf, Mapping)
    return (
        isinstance(fill, float) and math.isnan(fill),
        attributes["valid_range"] == (-math.inf, math.inf),
        cf["missing_value"] == -math.inf,
    )


_ARRAY: dict[str, object] = dict(ZarrV3ArrayMetadata.create_default().to_json())
_INLINE: dict[str, object] = {"kind": "inline", "must_understand": False}


def _stored(key: str, document: object) -> Mapping[str, bytes]:
    return {key: json.dumps(document).encode()}


def _v3_array() -> Mapping[str, bytes]:
    return _stored("zarr.json", {**_ARRAY, "attributes": ATTRIBUTES})


def _v3_group() -> Mapping[str, bytes]:
    return _stored("zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": ATTRIBUTES})


def _v3_consolidated() -> Mapping[str, bytes]:
    node = {**_ARRAY, "attributes": ATTRIBUTES}
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {**_INLINE, "metadata": {"a": node}},
    }
    return _stored("zarr.json", document)


def _v2_array() -> Mapping[str, bytes]:
    zarray = dict(ZarrV2ArrayMetadata.create_default().to_json())
    return {".zarray": json.dumps(zarray).encode(), ".zattrs": json.dumps(ATTRIBUTES).encode()}


def _v2_group() -> Mapping[str, bytes]:
    return {
        ".zgroup": json.dumps({"zarr_format": 2}).encode(),
        ".zattrs": json.dumps(ATTRIBUTES).encode(),
    }


def _v2_consolidated() -> Mapping[str, bytes]:
    document = {
        "zarr_consolidated_format": 1,
        "metadata": {".zgroup": {"zarr_format": 2}, ".zattrs": ATTRIBUTES},
    }
    return {".zmetadata": json.dumps(document).encode()}


def _consolidated_node_attributes(store: Mapping[str, bytes]) -> object:
    consolidated = ZarrV3GroupMetadata.from_key_value(store).consolidated_metadata
    assert isinstance(consolidated, ZarrV3ConsolidatedMetadata)
    return consolidated.metadata["a"].attributes


# Each case: the store, its reader, the attributes read from a store, and
# the guard the model's document passes. Declared, so that each reader and
# lambda is checked against its model.
_Case = tuple[
    Callable[[], Mapping[str, bytes]],
    Callable[[Mapping[str, bytes]], _Stored],
    Callable[[Mapping[str, bytes]], object],
    Callable[[object], bool] | None,
]
_CASES: dict[str, _Case] = {
    "v3-array": (
        _v3_array,
        ZarrV3ArrayMetadata.from_key_value,
        lambda store: ZarrV3ArrayMetadata.from_key_value(store).attributes,
        is_array_metadata_v3,
    ),
    "v3-group": (
        _v3_group,
        ZarrV3GroupMetadata.from_key_value,
        lambda store: ZarrV3GroupMetadata.from_key_value(store).attributes,
        is_group_metadata_v3,
    ),
    "v3-consolidated-node": (
        _v3_consolidated,
        ZarrV3GroupMetadata.from_key_value,
        _consolidated_node_attributes,
        is_group_metadata_v3,
    ),
    "v2-array": (
        _v2_array,
        ZarrV2ArrayMetadata.from_key_value,
        lambda store: ZarrV2ArrayMetadata.from_key_value(store).attributes,
        is_array_metadata_v2,
    ),
    "v2-group": (
        _v2_group,
        ZarrV2GroupMetadata.from_key_value,
        lambda store: ZarrV2GroupMetadata.from_key_value(store).attributes,
        is_group_metadata_v2,
    ),
    "v2-consolidated": (
        _v2_consolidated,
        ZarrV2ConsolidatedMetadata.from_key_value,
        lambda store: ZarrV2ConsolidatedMetadata.from_key_value(store).metadata[".zattrs"],
        None,
    ),
}


@pytest.mark.parametrize(
    ("store", "read", "attributes_in", "guard"), list(_CASES.values()), ids=list(_CASES)
)
def test_attributes_holding_non_finite_numbers_round_trip_through_the_store(
    store: Callable[[], Mapping[str, bytes]],
    read: Callable[[Mapping[str, bytes]], _Stored],
    attributes_in: Callable[[Mapping[str, bytes]], object],
    guard: Callable[[object], bool] | None,
) -> None:
    stored = store()
    assert _held(attributes_in(stored)) == (True, True, True)
    model = read(stored)
    if guard is not None:
        assert guard(model.to_json())
    written = dict(model.to_key_value().items())
    assert b"NaN" in b"".join(written.values())
    assert _held(attributes_in(written)) == (True, True, True)


def test_a_sequence_is_written_as_an_array() -> None:
    # The writer writes the document the reader's validation passed, arrays
    # normalized as the reader normalizes them.
    model = ZarrV3GroupMetadata.create_default(attributes={"r": range(3)})  # pyright: ignore[reportArgumentType]
    written = json.loads(model.to_key_value()["zarr.json"])
    assert written["attributes"] == {"r": [0, 1, 2]}


_V2_ARRAY: dict[str, object] = dict(ZarrV2ArrayMetadata.create_default().to_json())

# Each case: a reader, a store it refuses, and where the non-finite number is.
_Refused = tuple[
    Callable[[Mapping[str, bytes]], object], Mapping[str, bytes], tuple[str | int, ...]
]
_REFUSED: dict[str, _Refused] = {
    "fill-value": (
        ZarrV3ArrayMetadata.from_key_value,
        _stored("zarr.json", {**_ARRAY, "fill_value": math.nan}),
        ("fill_value",),
    ),
    "codec-configuration": (
        ZarrV3ArrayMetadata.from_key_value,
        _stored(
            "zarr.json",
            {
                **_ARRAY,
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little", "scale": math.inf}}
                ],
            },
        ),
        ("codecs", 0, "configuration", "scale"),
    ),
    "extension-field": (
        ZarrV3ArrayMetadata.from_key_value,
        _stored("zarr.json", {**_ARRAY, "extension": {"must_understand": False, "x": math.nan}}),
        ("extension", "x"),
    ),
    "consolidated-node": (
        ZarrV3GroupMetadata.from_key_value,
        _stored(
            "zarr.json",
            {
                "zarr_format": 3,
                "node_type": "group",
                "consolidated_metadata": {
                    **_INLINE,
                    "metadata": {"a": {**_ARRAY, "fill_value": math.nan}},
                },
            },
        ),
        ("consolidated_metadata", "metadata", "a", "fill_value"),
    ),
    # Only a v3 group holds node documents inline; in an array, a member of
    # that name is an extension the spec does not declare.
    "consolidated-in-an-array": (
        ZarrV3ArrayMetadata.from_key_value,
        _stored(
            "zarr.json",
            {
                **_ARRAY,
                "consolidated_metadata": {
                    "must_understand": False,
                    "metadata": {"a": {"attributes": {"x": math.nan}}},
                },
            },
        ),
        ("consolidated_metadata", "metadata", "a", "attributes", "x"),
    ),
    # A v2 array's attributes live in `.zattrs`; its `.zarray` is RFC 8259
    # throughout -- which is where zarr-python 3.0 once wrote a bare `NaN`.
    "zarray": (
        ZarrV2ArrayMetadata.from_key_value,
        _stored(".zarray", {**_V2_ARRAY, "fill_value": math.nan}),
        ("fill_value",),
    ),
    # Each `.zmetadata` entry is the document its key names.
    "zmetadata-entry": (
        ZarrV2ConsolidatedMetadata.from_key_value,
        _stored(
            ".zmetadata",
            {
                "zarr_consolidated_format": 1,
                "metadata": {"a/.zattrs": {"x": math.nan}, "a/.zarray": {"fill_value": math.nan}},
            },
        ),
        ("metadata", "a/.zarray", "fill_value"),
    ),
}


@pytest.mark.parametrize(("read", "store", "loc"), list(_REFUSED.values()), ids=list(_REFUSED))
def test_error_a_non_finite_number_outside_attributes_is_located_when_read(
    read: Callable[[Mapping[str, bytes]], object],
    store: Mapping[str, bytes],
    loc: tuple[str | int, ...],
) -> None:
    # Python's decoder reads a bare `NaN` as a float; where the spec
    # interprets a value, the document's validator says where it is.
    with pytest.raises(MetadataValidationError) as raised:
        read(store)
    assert [(problem.loc, problem.kind) for problem in raised.value.problems] == [
        (loc, "invalid_value")
    ]


@pytest.mark.parametrize(
    ("model", "problems"),
    [
        (
            ZarrV3ArrayMetadata.create_default(fill_value=math.nan, attributes={"x": math.nan}),
            [(("fill_value",), "invalid_value")],
        ),
        (
            ZarrV3GroupMetadata.create_default(attributes={"s": {1, 2}}),  # pyright: ignore[reportArgumentType]
            [(("attributes", "s"), "invalid_type")],
        ),
        (
            ZarrV3GroupMetadata.create_default(attributes={1: "a"}),  # pyright: ignore[reportArgumentType]
            [(("attributes",), "invalid_type")],
        ),
        (
            ZarrV3ArrayMetadata.create_default(shape=(2,), dimension_names=("x", "y")),
            [(("dimension_names",), "invalid_value")],
        ),
    ],
    ids=["non-finite-fill-value", "not-json", "non-string-key", "dimension-names-past-shape"],
)
def test_error_a_document_the_reader_refuses_is_not_written(
    model: _Stored, problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    # A model built by hand is not validated; the writer validates what it
    # writes as the reader does. A non-string key was written as a string,
    # a value that is not JSON raised `TypeError`, and the last was written
    # and then refused on read.
    with pytest.raises(MetadataValidationError) as raised:
        model.to_key_value()
    assert [(problem.loc, problem.kind) for problem in raised.value.problems] == problems
