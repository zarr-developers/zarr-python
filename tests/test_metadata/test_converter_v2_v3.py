import numcodecs
import pytest
from typer.testing import CliRunner

import zarr
from zarr.abc.store import Store
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.dtype.npy.int import UInt16
from zarr.core.metadata.converter.cli import app

runner = CliRunner()


def test_convert_array(local_store: Store) -> None:
    shape = (10, 10)
    chunks = (10, 10)
    dtype = "uint16"
    compressors = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)
    fill_value = 2
    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}

    zarr.create_array(
        store=local_store,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressors=compressors,
        zarr_format=2,
        fill_value=fill_value,
        attributes=attributes,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.node_type == "array"
    assert metadata.shape == shape
    assert metadata.chunk_grid == RegularChunkGrid(chunk_shape=chunks)
    assert metadata.chunk_key_encoding == DefaultChunkKeyEncoding(separator=".")
    assert metadata.data_type == UInt16("little")
    assert metadata.codecs == (
        BytesCodec(endian="little"),
        BloscCodec(typesize=2, cname="zstd", clevel=3, shuffle="shuffle", blocksize=0),
    )
    assert metadata.fill_value == fill_value
    assert metadata.attributes == attributes
    assert metadata.dimension_names is None
    assert metadata.storage_transformers == ()


def test_convert_group(local_store: Store) -> None:
    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    zarr.create_group(store=local_store, zarr_format=2, attributes=attributes)

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.node_type == "group"
    assert metadata.attributes == attributes
    assert metadata.consolidated_metadata is None


def test_convert_nested_groups_and_arrays(local_store: Store) -> None:
    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}

    # 3 levels of nested groups
    group_1 = zarr.create_group(store=local_store, zarr_format=2, attributes=attributes)
    group_2 = group_1.create_group(name="group_2", attributes=attributes)
    group_3 = group_2.create_group(name="group_3", attributes=attributes)

    # 1 array per group
    array_1 = group_1.create_array(
        name="array_1", shape=(10, 10), chunks=(10, 10), dtype="uint16", attributes=attributes
    )
    array_2 = group_2.create_array(
        name="array_2", shape=(10, 10), chunks=(10, 10), dtype="uint16", attributes=attributes
    )
    array_3 = group_3.create_array(
        name="array_3", shape=(10, 10), chunks=(10, 10), dtype="uint16", attributes=attributes
    )

    paths = [group_1.path, group_2.path, group_3.path, array_1.path, array_2.path, array_3.path]

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0

    # check zarr.json were created for every group and array
    total_zarr_jsons = 0
    for _, _, filenames in local_store.root.walk():
        assert "zarr.json" in filenames
        total_zarr_jsons += 1
    assert total_zarr_jsons == 6

    # Check converted zarr can be opened + metadata accessed at all levels
    zarr_array = zarr.open(local_store.root, zarr_format=3)
    for path in paths:
        zarr_v3 = zarr_array[path]
        metadata = zarr_v3.metadata
        assert metadata.zarr_format == 3
        assert metadata.attributes == attributes


@pytest.mark.parametrize("node_type", ["array", "group"])
def test_convert_v3(local_store: Store, node_type: str) -> None:
    """Attempting to convert a v3 array/group should always fail"""

    if node_type == "array":
        zarr.create_array(
            store=local_store, shape=(10, 10), chunks=(10, 10), zarr_format=3, dtype="uint16"
        )
    else:
        zarr.create_group(store=local_store, zarr_format=3)

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert str(result.exception) == "Only arrays / groups with zarr v2 metadata can be converted"
