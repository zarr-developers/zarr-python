"""Tests for where a non-finite number is allowed: in a node's attributes, as user data.

zarr-python writes attributes with the defaults of Python's `json` module,
so an attribute may hold `NaN`, `Infinity` or `-Infinity`. The model reads
such a store, validates it, and writes it back the same way. Wherever the
spec interprets a value, a non-finite number is refused when read. A
document that the reader would refuse produces no model, so it is not written.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

import pytest

from zarr_metadata.model import (
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    ZarrV3GroupMetadata,
    read_array_metadata_v2,
    read_array_metadata_v3,
    read_group_metadata_v2,
    read_group_metadata_v3,
)
from zarr_metadata.problems import MetadataValidationError, ValidationProblem

if TYPE_CHECKING:
    from collections.abc import Iterable


class _Written(Protocol):
    """The output of a model write: store keys, each with its bytes."""

    def items(self) -> Iterable[tuple[str, bytes]]: ...


class _Stored(Protocol):
    """A model that writes itself to a store."""

    def to_json(self) -> Mapping[str, object]: ...

    def to_key_value(self, *, indent: int | str | None = None) -> _Written: ...


class _Reading(Protocol):
    """What a reader returns: every problem it found."""

    @property
    def problems(self) -> tuple[ValidationProblem, ...]: ...


ATTRIBUTES: dict[str, object] = {
    "_FillValue": math.nan,
    "valid_range": [-math.inf, math.inf],
    "cf": {"missing_value": -math.inf},
}


def _held(attributes: object) -> tuple[bool, bool, bool]:
    """Report whether each non-finite number in `ATTRIBUTES` is still present, with arrays as tuples."""
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


def _consolidated_from_key_value(store: Mapping[str, bytes]) -> ZarrV3ConsolidatedMetadata:
    """The consolidated metadata of the group at `zarr.json` in `store`, with every document it lists read."""
    return ZarrV3ConsolidatedMetadata(
        ZarrV3GroupMetadata.from_key_value(store).consolidated_metadata
    )


def _consolidated_node_attributes(store: Mapping[str, bytes]) -> object:
    return _consolidated_from_key_value(store).metadata["a"].attributes


# Each case: the store, its reader, a function that reads the attributes
# from a store, and the reader of the model's document, which finds no
# problem with it. The type is declared so that each reader and lambda is
# type-checked against its model.
_Case = tuple[
    Callable[[], Mapping[str, bytes]],
    Callable[[Mapping[str, bytes]], _Stored],
    Callable[[Mapping[str, bytes]], object],
    Callable[[object], _Reading] | None,
]
_CASES: dict[str, _Case] = {
    "v3-array": (
        _v3_array,
        ZarrV3ArrayMetadata.from_key_value,
        lambda store: ZarrV3ArrayMetadata.from_key_value(store).attributes,
        read_array_metadata_v3,
    ),
    "v3-group": (
        _v3_group,
        ZarrV3GroupMetadata.from_key_value,
        lambda store: ZarrV3GroupMetadata.from_key_value(store).attributes,
        read_group_metadata_v3,
    ),
    "v3-consolidated-node": (
        _v3_consolidated,
        ZarrV3GroupMetadata.from_key_value,
        _consolidated_node_attributes,
        read_group_metadata_v3,
    ),
    "v2-array": (
        _v2_array,
        ZarrV2ArrayMetadata.from_key_value,
        lambda store: ZarrV2ArrayMetadata.from_key_value(store).attributes,
        read_array_metadata_v2,
    ),
    "v2-group": (
        _v2_group,
        ZarrV2GroupMetadata.from_key_value,
        lambda store: ZarrV2GroupMetadata.from_key_value(store).attributes,
        read_group_metadata_v2,
    ),
    "v2-consolidated": (
        _v2_consolidated,
        ZarrV2ConsolidatedMetadata.from_key_value,
        lambda store: ZarrV2ConsolidatedMetadata.from_key_value(store).metadata[""].attributes,
        None,
    ),
}


@pytest.mark.parametrize(
    ("store", "read", "attributes_in", "reader"), list(_CASES.values()), ids=list(_CASES)
)
def test_attributes_holding_non_finite_numbers_round_trip_through_the_store(
    store: Callable[[], Mapping[str, bytes]],
    read: Callable[[Mapping[str, bytes]], _Stored],
    attributes_in: Callable[[Mapping[str, bytes]], object],
    reader: Callable[[object], _Reading] | None,
) -> None:
    stored = store()
    assert _held(attributes_in(stored)) == (True, True, True)
    model = read(stored)
    if reader is not None:
        assert reader(model.to_json()).problems == ()
    written = dict(model.to_key_value().items())
    assert b"NaN" in b"".join(written.values())
    assert _held(attributes_in(written)) == (True, True, True)


@pytest.mark.parametrize(
    ("key", "member"),
    [
        ("extension", {"must_understand": False, "x": math.nan}),
        # Only a v3 group holds node documents inline. In an array, a member
        # with that name is an extra field that nothing in scope claims.
        (
            "consolidated_metadata",
            {"must_understand": False, "metadata": {"a": {"attributes": {"x": math.nan}}}},
        ),
    ],
    ids=["extension-field", "consolidated-in-an-array"],
)
def test_an_unclaimed_member_holding_non_finite_numbers_round_trips_through_the_store(
    key: str, member: dict[str, object]
) -> None:
    """Nothing reads a member that no definition in scope claims, so like attributes it may hold `NaN` and the infinities: the model reads it, and writes it back as bare tokens."""
    model = ZarrV3ArrayMetadata.from_key_value(_stored("zarr.json", {**_ARRAY, key: member}))
    written = model.to_key_value()
    assert b"NaN" in written["zarr.json"]
    assert ZarrV3ArrayMetadata.from_key_value(written) == model


def test_a_sequence_is_written_as_an_array() -> None:
    # The writer writes the document that passed the reader's validation,
    # with arrays normalized the same way the reader normalizes them.
    model = ZarrV3GroupMetadata.create_default(attributes={"r": range(3)})  # pyright: ignore[reportArgumentType]
    written = json.loads(model.to_key_value()["zarr.json"])
    assert written["attributes"] == {"r": [0, 1, 2]}


_V2_ARRAY: dict[str, object] = dict(ZarrV2ArrayMetadata.create_default().to_json())

# Each case: a reader, a store it refuses, and the location of the non-finite number.
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
    # A group does not read the documents its consolidated metadata lists;
    # reading them finds the number.
    "consolidated-node": (
        _consolidated_from_key_value,
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
        ("metadata", "a", "fill_value"),
    ),
    # A v2 array's attributes live in `.zattrs`. Its `.zarray` must be RFC 8259
    # JSON throughout, and zarr-python 3.0 once wrote a bare `NaN` there.
    "zarray": (
        ZarrV2ArrayMetadata.from_key_value,
        _stored(".zarray", {**_V2_ARRAY, "fill_value": math.nan}),
        (".zarray", "fill_value"),
    ),
    # Each `.zmetadata` entry is validated as the document type its key names.
    "zmetadata-entry": (
        ZarrV2ConsolidatedMetadata.from_key_value,
        _stored(
            ".zmetadata",
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    "a/.zattrs": {"x": math.nan},
                    "a/.zarray": {**_V2_ARRAY, "fill_value": math.nan},
                },
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
    # Python's decoder reads a bare `NaN` as a float. Where the spec
    # interprets a value, the document's validator reports its location.
    with pytest.raises(MetadataValidationError) as raised:
        read(store)
    assert [(problem.loc, problem.kind) for problem in raised.value.problems] == [
        (loc, "invalid_value")
    ]


@pytest.mark.parametrize(
    ("build", "problems"),
    [
        (
            lambda: ZarrV3ArrayMetadata.create_default(
                fill_value=math.nan, attributes={"x": math.nan}
            ),
            [(("fill_value",), "invalid_value")],
        ),
        (
            lambda: ZarrV3GroupMetadata.create_default(attributes={"s": {1, 2}}),  # pyright: ignore[reportArgumentType]
            [(("attributes", "s"), "invalid_type")],
        ),
        (
            lambda: ZarrV3GroupMetadata.create_default(attributes={1: "a"}),  # pyright: ignore[reportArgumentType]
            [(("attributes",), "invalid_type")],
        ),
        (
            lambda: ZarrV3ArrayMetadata.create_default(shape=(2,), dimension_names=("x", "y")),
            [(("dimension_names",), "invalid_value")],
        ),
    ],
    ids=["non-finite-fill-value", "not-json", "non-string-key", "dimension-names-past-shape"],
)
def test_error_a_document_the_reader_refuses_is_not_written(
    build: Callable[[], _Stored], problems: list[tuple[tuple[str | int, ...], str]]
) -> None:
    # A model is built by reading its document, so a document that the reader
    # refuses produces no model to write. When such documents were written, a
    # non-string key became a string, a value that is not JSON raised
    # `TypeError`, and the last case was refused only when read again.
    with pytest.raises(MetadataValidationError) as raised:
        build()
    assert [(problem.loc, problem.kind) for problem in raised.value.problems] == problems


def test_error_store_bytes_nested_deeper_than_python_reads_are_invalid_json() -> None:
    # `json.loads` raises `RecursionError` on a hundred thousand `[`. This is
    # reported as an ingestion failure like any other.
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV3ArrayMetadata.from_key_value({"zarr.json": b"[" * 100_000})
    assert [(p.loc, p.kind) for p in raised.value.problems] == [(("zarr.json",), "invalid_json")]
