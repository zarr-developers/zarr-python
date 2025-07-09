from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

import zarr.api.asynchronous
import zarr.storage
from zarr.core.buffer import cpu
from zarr.core.buffer.core import default_buffer_prototype
from zarr.core.dtype.npy.float import Float32, Float64
from zarr.core.dtype.npy.int import Int16
from zarr.core.group import ConsolidatedMetadata, GroupMetadata
from zarr.core.metadata import ArrayV2Metadata
from zarr.core.metadata.v2 import parse_zarr_format

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec


def test_parse_zarr_format_valid() -> None:
    assert parse_zarr_format(2) == 2


@pytest.mark.parametrize("data", [None, 1, 3, 4, 5, "3"])
def test_parse_zarr_format_invalid(data: Any) -> None:
    with pytest.raises(ValueError, match=f"Invalid value. Expected 2. Got {data}"):
        parse_zarr_format(data)


@pytest.mark.parametrize("attributes", [None, {"foo": "bar"}])
@pytest.mark.parametrize("filters", [None, [{"id": "gzip", "level": 1}]])
@pytest.mark.parametrize("compressor", [None, {"id": "gzip", "level": 1}])
@pytest.mark.parametrize("fill_value", [None, 0, 1])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dimension_separator", [".", "/", None])
def test_metadata_to_dict(
    compressor: Codec | None,
    filters: tuple[Codec] | None,
    fill_value: Any,
    order: Literal["C", "F"],
    dimension_separator: Literal[".", "/"] | None,
    attributes: dict[str, Any] | None,
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


def test_filters_empty_tuple_warns() -> None:
    metadata_dict = {
        "zarr_format": 2,
        "shape": (1,),
        "chunks": (1,),
        "dtype": "|u1",
        "order": "C",
        "compressor": None,
        "filters": (),
        "fill_value": 0,
    }
    with pytest.warns(
        UserWarning, match="Found an empty list of filters in the array metadata document."
    ):
        meta = ArrayV2Metadata.from_dict(metadata_dict)
    assert meta.filters is None


class TestConsolidated:
    @pytest.fixture
    async def v2_consolidated_metadata(
        self, memory_store: zarr.storage.MemoryStore
    ) -> zarr.storage.MemoryStore:
        zmetadata = {
            "metadata": {
                ".zattrs": {
                    "Conventions": "COARDS",
                },
                ".zgroup": {"zarr_format": 2},
                "air/.zarray": {
                    "chunks": [730],
                    "compressor": None,
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
                    "compressor": None,
                    "dtype": "<f4",
                    "fill_value": 0.0,
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
                    "compressor": None,
                    "dtype": "<f4",
                    "fill_value": 0.0,
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
        store = zarr.storage.MemoryStore(store_dict=store_dict)
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
        self, v2_consolidated_metadata: zarr.storage.MemoryStore
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
                    dtype=Int16(),
                    order="C",
                    filters=None,
                    dimension_separator=".",
                    compressor=None,
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
                    dtype=Float32(),
                    order="C",
                    filters=None,
                    dimension_separator=".",
                    compressor=None,
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
                                dtype=Float32(),
                                order="C",
                                filters=None,
                                dimension_separator=".",
                                compressor=None,
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


def test_from_dict_extra_fields() -> None:
    data = {
        "_nczarr_array": {"dimrefs": ["/dim1", "/dim2"], "storage": "chunked"},
        "attributes": {"key": "value"},
        "chunks": [8],
        "compressor": None,
        "dtype": "<f8",
        "fill_value": 0.0,
        "filters": None,
        "order": "C",
        "shape": [8],
        "zarr_format": 2,
    }

    result = ArrayV2Metadata.from_dict(data)
    expected = ArrayV2Metadata(
        attributes={"key": "value"},
        shape=(8,),
        dtype=Float64(),
        chunks=(8,),
        fill_value=0.0,
        order="C",
    )
    assert result == expected


def test_zstd_checksum() -> None:
    arr = zarr.create_array(
        {},
        shape=(10,),
        chunks=(10,),
        dtype="int32",
        compressors={"id": "zstd", "level": 5, "checksum": False},
        zarr_format=2,
    )
    metadata = json.loads(
        arr.metadata.to_buffer_dict(default_buffer_prototype())[".zarray"].to_bytes()
    )
    assert "checksum" not in metadata["compressor"]


@pytest.mark.parametrize("fill_value", [np.void((0, 0), np.dtype([("foo", "i4"), ("bar", "i4")]))])
def test_structured_dtype_fill_value_serialization(tmp_path, fill_value):
    zarr_format = 2
    group_path = tmp_path / "test.zarr"
    root_group = zarr.open_group(group_path, mode="w", zarr_format=zarr_format)
    dtype = np.dtype([("foo", "i4"), ("bar", "i4")])
    root_group.create_array(
        name="structured_dtype",
        shape=(100, 100),
        chunks=(100, 100),
        dtype=dtype,
        fill_value=fill_value,
    )

    zarr.consolidate_metadata(root_group.store, zarr_format=zarr_format)
    root_group = zarr.open_group(group_path, mode="r")
    observed = root_group.metadata.consolidated_metadata.metadata["structured_dtype"].fill_value
    assert observed == fill_value
