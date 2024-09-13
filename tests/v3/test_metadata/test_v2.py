from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

import zarr.api.asynchronous
import zarr.store
from zarr.codecs import GzipCodec
from zarr.core.buffer import cpu
from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayV2Metadata, parse_zarr_format_v2

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format_v2(2) == 2


@pytest.mark.parametrize("data", [None, 1, 3, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 2. Got {data}"):
        parse_zarr_format_v2(data)


@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("filters", [(), (GzipCodec().to_dict())])
@pytest.mark.parametrize("compressor", [None, GzipCodec().to_dict()])
@pytest.mark.parametrize("fill_value", [0, 1])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
def test_metadata_to_dict(
    compressor: Codec | None,
    filters: list[Codec] | None,
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


class TestConsolidated:
    @pytest.fixture
    async def v2_consolidated_metadata(
        self, memory_store: zarr.store.MemoryStore
    ) -> zarr.store.MemoryStore:
        zmetadata = {
            "metadata": {
                ".zattrs": {
                    "Conventions": "COARDS",
                },
                ".zgroup": {"zarr_format": 2},
                "air/.zarray": {
                    "chunks": [730],
                    "compressor": {},
                    "dtype": "<i2",
                    "fill_value": 0,
                    "filters": None,
                    "order": "C",
                    "shape": [730],
                    "zarr_format": 2,
                },
                "air/.zattrs": {
                    "_ARRAY_DIMENSIONS": ["time"],
                    "dataset": "NMC Reanalysis",
                },
                "time/.zarray": {
                    "chunks": [730],
                    "compressor": {},
                    "dtype": "<f4",
                    "fill_value": "0.0",
                    "filters": None,
                    "order": "C",
                    "shape": [730],
                    "zarr_format": 2,
                },
                "time/.zattrs": {
                    "_ARRAY_DIMENSIONS": ["time"],
                    "calendar": "standard",
                    "long_name": "Time",
                    "standard_name": "time",
                    "units": "hours since 1800-01-01",
                },
                "nested/.zattrs": {"key": "value"},
                "nested/.zgroup": {"zarr_format": 2},
                "nested/array/.zarray": {
                    "chunks": [730],
                    "compressor": {},
                    "dtype": "<f4",
                    "fill_value": "0.0",
                    "filters": None,
                    "order": "C",
                    "shape": [730],
                    "zarr_format": 2,
                },
                "nested/array/.zattrs": {
                    "calendar": "standard",
                },
            },
            "zarr_consolidated_format": 1,
        }
        store_dict = {}
        store = zarr.store.MemoryStore(store_dict=store_dict, mode="a")
        await store.set(
            ".zattrs", cpu.Buffer.from_bytes(json.dumps({"Conventions": "COARDS"}).encode())
        )
        await store.set(".zgroup", cpu.Buffer.from_bytes(json.dumps({"zarr_format": 2}).encode()))
        await store.set(".zmetadata", cpu.Buffer.from_bytes(json.dumps(zmetadata).encode()))
        await store.set(
            "air/.zarray",
            cpu.Buffer.from_bytes(json.dumps(zmetadata["metadata"]["air/.zarray"]).encode()),
        )
        await store.set(
            "air/.zattrs",
            cpu.Buffer.from_bytes(json.dumps(zmetadata["metadata"]["air/.zattrs"]).encode()),
        )
        await store.set(
            "time/.zarray",
            cpu.Buffer.from_bytes(json.dumps(zmetadata["metadata"]["time/.zarray"]).encode()),
        )
        await store.set(
            "time/.zattrs",
            cpu.Buffer.from_bytes(json.dumps(zmetadata["metadata"]["time/.zattrs"]).encode()),
        )

        # and a nested group for fun
        await store.set(
            "nested/.zattrs", cpu.Buffer.from_bytes(json.dumps({"key": "value"}).encode())
        )
        await store.set(
            "nested/.zgroup", cpu.Buffer.from_bytes(json.dumps({"zarr_format": 2}).encode())
        )
        await store.set(
            "nested/array/.zarray",
            cpu.Buffer.from_bytes(
                json.dumps(zmetadata["metadata"]["nested/array/.zarray"]).encode()
            ),
        )
        await store.set(
            "nested/array/.zattrs",
            cpu.Buffer.from_bytes(
                json.dumps(zmetadata["metadata"]["nested/array/.zattrs"]).encode()
            ),
        )

        return store

    async def test_read_consolidated_metadata(
        self, v2_consolidated_metadata: zarr.store.MemoryStore
    ):
        # .zgroup, .zattrs, .metadata
        store = v2_consolidated_metadata
        group = zarr.open_consolidated(store=store, zarr_format=2)
        assert group.metadata.consolidated_metadata is not None
        expected = ConsolidatedMetadata(
            metadata={
                "air": ArrayV2Metadata(
                    shape=(730,),
                    fill_value=0,
                    chunks=(730,),
                    attributes={"_ARRAY_DIMENSIONS": ["time"], "dataset": "NMC Reanalysis"},
                    dtype=np.dtype("int16"),
                    order="C",
                    filters=None,
                    dimension_separator=".",
                    compressor={},
                ),
                "time": ArrayV2Metadata(
                    shape=(730,),
                    fill_value=0.0,
                    chunks=(730,),
                    attributes={
                        "_ARRAY_DIMENSIONS": ["time"],
                        "calendar": "standard",
                        "long_name": "Time",
                        "standard_name": "time",
                        "units": "hours since 1800-01-01",
                    },
                    dtype=np.dtype("float32"),
                    order="C",
                    filters=None,
                    dimension_separator=".",
                    compressor={},
                ),
                "nested": GroupMetadata(
                    attributes={"key": "value"},
                    zarr_format=2,
                    consolidated_metadata=ConsolidatedMetadata(
                        metadata={
                            "array": ArrayV2Metadata(
                                shape=(730,),
                                fill_value=0.0,
                                chunks=(730,),
                                attributes={
                                    "calendar": "standard",
                                },
                                dtype=np.dtype("float32"),
                                order="C",
                                filters=None,
                                dimension_separator=".",
                                compressor={},
                            )
                        }
                    ),
                ),
            },
            kind="inline",
            must_understand=False,
        )
        result = group.metadata.consolidated_metadata
        assert result == expected

    async def test_getitem_consolidated(self, v2_consolidated_metadata):
        store = v2_consolidated_metadata
        group = await zarr.api.asynchronous.open_consolidated(store=store, zarr_format=2)
        air = await group.getitem("air")
        assert air.metadata.shape == (730,)
