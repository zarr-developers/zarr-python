"""What the model reads from and writes to a store, key by key.

A node's attributes are user data, and zarr-python writes them with the
defaults of Python's `json` module, so an attribute may hold `NaN`,
`Infinity` or `-Infinity`. The model reads such a store, validates it, and
writes it back the same way; anywhere else a non-finite number is refused.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Protocol, cast

import pytest

from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV2ArrayMetadata,
    ZarrV2ConsolidatedMetadata,
    ZarrV2GroupMetadata,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    is_array_metadata_v2,
    is_array_metadata_v3,
    is_group_metadata_v2,
    is_group_metadata_v3,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class _Stored(Protocol):
    """A model that writes itself to a store."""

    def to_json(self) -> Mapping[str, object]: ...

    def to_key_value(self, *, indent: int | str | None = None) -> Mapping[str, bytes]: ...


ATTRIBUTES: dict[str, object] = {"_FillValue": math.nan, "valid_range": [-math.inf, math.inf]}


def _held(attributes: object) -> tuple[bool, bool, bool]:
    held = cast("Mapping[str, object]", attributes)
    fill = held["_FillValue"]
    low, high = cast("tuple[float, float]", held["valid_range"])
    return (isinstance(fill, float) and math.isnan(fill), low == -math.inf, high == math.inf)


def _v3_array() -> Mapping[str, bytes]:
    document = {**ZarrV3ArrayMetadata.create_default().to_json(), "attributes": ATTRIBUTES}
    return {"zarr.json": json.dumps(document).encode()}


def _v3_group() -> Mapping[str, bytes]:
    document = {"zarr_format": 3, "node_type": "group", "attributes": ATTRIBUTES}
    return {"zarr.json": json.dumps(document).encode()}


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


@pytest.mark.parametrize(
    ("store", "read", "attributes_of", "guard"),
    [
        (
            _v3_array,
            ZarrV3ArrayMetadata.from_key_value,
            lambda model: model.attributes,
            is_array_metadata_v3,
        ),
        (
            _v3_group,
            ZarrV3GroupMetadata.from_key_value,
            lambda model: model.attributes,
            is_group_metadata_v3,
        ),
        (
            _v2_array,
            ZarrV2ArrayMetadata.from_key_value,
            lambda model: model.attributes,
            is_array_metadata_v2,
        ),
        (
            _v2_group,
            ZarrV2GroupMetadata.from_key_value,
            lambda model: model.attributes,
            is_group_metadata_v2,
        ),
        (
            _v2_consolidated,
            ZarrV2ConsolidatedMetadata.from_key_value,
            lambda model: model.metadata[".zattrs"],
            None,
        ),
    ],
    ids=["v3-array", "v3-group", "v2-array", "v2-group", "v2-consolidated"],
)
def test_attributes_holding_non_finite_numbers_round_trip_through_the_store(
    store: Callable[[], Mapping[str, bytes]],
    read: Callable[[Mapping[str, bytes]], _Stored],
    attributes_of: Callable[[Any], object],
    guard: Callable[[object], bool] | None,
) -> None:
    model = read(store())
    assert _held(attributes_of(model)) == (True, True, True)
    if guard is not None:
        assert guard(model.to_json())
    written = model.to_key_value()
    assert b"NaN" in b"".join(written.values())
    assert _held(attributes_of(read(written))) == (True, True, True)


def test_error_a_non_finite_number_outside_attributes_is_located_when_read() -> None:
    # Python's decoder reads a bare `NaN` as a float; where it may not be,
    # the store says where it is.
    zarray = dict(ZarrV2ArrayMetadata.create_default().to_json())
    zarray["fill_value"] = math.nan
    with pytest.raises(MetadataValidationError) as raised:
        ZarrV2ArrayMetadata.from_key_value({".zarray": json.dumps(zarray).encode()})
    assert [(problem.loc, problem.kind) for problem in raised.value.problems] == [
        (("fill_value",), "invalid_value")
    ]


def test_error_a_non_finite_number_outside_attributes_is_not_written() -> None:
    # A model built by hand is not validated; the writer is what stands
    # between a non-finite fill value and a document no reader accepts.
    model = ZarrV3ArrayMetadata.create_default(fill_value=math.nan, attributes={"x": math.nan})
    with pytest.raises(MetadataValidationError) as raised:
        model.to_key_value()
    assert [problem.loc for problem in raised.value.problems] == [("fill_value",)]
