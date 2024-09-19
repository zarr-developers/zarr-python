from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zarr.core.metadata.v2 import ArrayV2Metadata

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec

import numcodecs
import pytest

from zarr.core.metadata.v2 import parse_zarr_format


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(2) == 2


@pytest.mark.parametrize("data", [None, 1, 3, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 2. Got {data}"):
        parse_zarr_format(data)


@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("filters", [None, (), (numcodecs.GZip(),)])
@pytest.mark.parametrize("compressor", [None, numcodecs.GZip()])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
def test_metadata_to_dict(
    compressor: Codec | None,
    filters: tuple[Codec] | None,
    fill_value: Any,
    order: Literal["C", "F"],
    dimension_separator: Literal[".", "/"] | None,
    attributes: None | dict[str, Any],
) -> None:
    shape = (1, 2, 3)
    chunks = (1,) * len(shape)
    data_type = "|u1"
    metadata_dict = {
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": data_type,
        "order": order,
        "compressor": compressor,
        "filters": filters,
        "fill_value": fill_value,
    }

    if attributes is not None:
        metadata_dict["attributes"] = attributes
    if dimension_separator is not None:
        metadata_dict["dimension_separator"] = dimension_separator

    metadata = ArrayV2Metadata.from_dict(metadata_dict)
    observed = metadata.to_dict()
    expected = metadata_dict.copy()

    if attributes is None:
        assert observed["attributes"] == {}
        observed.pop("attributes")

    if dimension_separator is None:
        expected_dimension_sep = "."
        assert observed["dimension_separator"] == expected_dimension_sep
        observed.pop("dimension_separator")

    assert observed == expected
