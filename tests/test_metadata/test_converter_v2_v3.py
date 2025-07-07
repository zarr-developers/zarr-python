import lzma
from pathlib import Path
from typing import Any

import numcodecs
import numcodecs.abc
import pytest
from numcodecs.zarr3 import LZMA, Delta
from typer.testing import CliRunner

import zarr
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.dtype.npy.int import BaseInt, UInt8, UInt16
from zarr.core.metadata.converter.cli import app

runner = CliRunner()


def create_nested_zarr(store: Store, attributes: dict[str, Any], separator: str) -> list[str]:
    """Create a zarr with nested groups / arrays, returning the paths to all."""

    # 3 levels of nested groups
    group_0 = zarr.create_group(store=store, zarr_format=2, attributes=attributes)
    group_1 = group_0.create_group(name="group_1", attributes=attributes)
    group_2 = group_1.create_group(name="group_2", attributes=attributes)
    paths = [group_0.path, group_1.path, group_2.path]

    # 1 array per group
    for i, group in enumerate([group_0, group_1, group_2]):
        array = group.create_array(
            name=f"array_{i}",
            shape=(10, 10),
            chunks=(5, 5),
            dtype="uint16",
            attributes=attributes,
            chunk_key_encoding={"name": "v2", "separator": separator},
        )
        array[:] = 1
        paths.append(array.path)

    return paths


@pytest.fixture
def expected_paths_no_metadata() -> list[Path]:
    """Expected paths from create_nested_zarr, with no metadata files"""
    return [
        Path("array_0"),
        Path("array_0/0.0"),
        Path("array_0/0.1"),
        Path("array_0/1.0"),
        Path("array_0/1.1"),
        Path("group_1"),
        Path("group_1/array_1"),
        Path("group_1/array_1/0.0"),
        Path("group_1/array_1/0.1"),
        Path("group_1/array_1/1.0"),
        Path("group_1/array_1/1.1"),
        Path("group_1/group_2"),
        Path("group_1/group_2/array_2"),
        Path("group_1/group_2/array_2/0.0"),
        Path("group_1/group_2/array_2/0.1"),
        Path("group_1/group_2/array_2/1.0"),
        Path("group_1/group_2/array_2/1.1"),
    ]


@pytest.fixture
def expected_paths_v3_metadata(expected_paths_no_metadata: list[Path]) -> list[Path]:
    """Expected paths from create_nested_zarr, with v3 metadata files"""
    v3_paths = [
        Path("array_0/zarr.json"),
        Path("group_1/array_1/zarr.json"),
        Path("group_1/group_2/array_2/zarr.json"),
        Path("zarr.json"),
        Path("group_1/zarr.json"),
        Path("group_1/group_2/zarr.json"),
    ]
    expected_paths_no_metadata.extend(v3_paths)

    return sorted(expected_paths_no_metadata)


@pytest.fixture
def expected_paths_v2_metadata(expected_paths_no_metadata: list[Path]) -> list[Path]:
    """Expected paths from create_nested_zarr, with v2 metadata files"""
    v2_paths = [
        Path("array_0/.zarray"),
        Path("array_0/.zattrs"),
        Path("group_1/array_1/.zarray"),
        Path("group_1/array_1/.zattrs"),
        Path("group_1/group_2/array_2/.zarray"),
        Path("group_1/group_2/array_2/.zattrs"),
        Path(".zgroup"),
        Path(".zattrs"),
        Path("group_1/.zgroup"),
        Path("group_1/.zattrs"),
        Path("group_1/group_2/.zgroup"),
        Path("group_1/group_2/.zattrs"),
    ]
    expected_paths_no_metadata.extend(v2_paths)

    return sorted(expected_paths_no_metadata)


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


@pytest.mark.parametrize("separator", [".", "/"])
def test_convert_nested_groups_and_arrays(local_store: Store, separator: str) -> None:
    """Test that zarr.json are made at the correct points in a hierarchy of groups and arrays
    (including when there are additional dirs due to using a / separator)"""

    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    paths = create_nested_zarr(local_store, attributes, separator)

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0

    # check zarr.json were created for every group and array
    total_zarr_jsons = 0
    for _, _, filenames in local_store.root.walk():
        # group / array directories
        if ".zattrs" in filenames:
            assert "zarr.json" in filenames
            total_zarr_jsons += 1
        # other directories e.g. for chunks when separator is /
        else:
            assert "zarr.json" not in filenames
    assert total_zarr_jsons == 6

    # Check converted zarr can be opened + metadata accessed at all levels
    zarr_array = zarr.open(local_store.root, zarr_format=3)
    for path in paths:
        zarr_v3 = zarr_array[path]
        metadata = zarr_v3.metadata
        assert metadata.zarr_format == 3
        assert metadata.attributes == attributes


@pytest.mark.parametrize("separator", [".", "/"])
def test_convert_nested_with_path(local_store: Store, separator: str) -> None:
    """Test that only arrays/groups within group_1 are converted (+ no other files in store)"""

    create_nested_zarr(local_store, {}, separator)

    result = runner.invoke(app, ["convert", str(local_store.root), "--path", "group_1"])
    assert result.exit_code == 0

    group_path = local_store.root / "group_1"

    total_zarr_jsons = 0
    for dirpath, _, filenames in local_store.root.walk():
        inside_group = (dirpath == group_path) or (group_path in dirpath.parents)
        if (".zattrs" in filenames) and inside_group:
            # group / array directories inside the group
            assert "zarr.json" in filenames
            total_zarr_jsons += 1
        else:
            assert "zarr.json" not in filenames

    assert total_zarr_jsons == 4


@pytest.mark.parametrize(
    ("compressor_v2", "compressor_v3"),
    [
        (
            numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1),
            BloscCodec(typesize=2, cname="zstd", clevel=3, shuffle="shuffle", blocksize=0),
        ),
        (numcodecs.Zstd(level=3), ZstdCodec(level=3)),
        (numcodecs.GZip(level=3), GzipCodec(level=3)),
        (
            numcodecs.LZMA(
                format=1, check=-1, preset=None, filters=[{"id": lzma.FILTER_DELTA, "dist": 4}]
            ),
            LZMA(format=1, check=-1, preset=None, filters=[{"id": lzma.FILTER_DELTA, "dist": 4}]),
        ),
    ],
    ids=["blosc", "zstd", "gzip", "numcodecs-compressor"],
)
def test_convert_compressor(
    local_store: Store, compressor_v2: numcodecs.abc.Codec, compressor_v3: Codec
) -> None:
    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        compressors=compressor_v2,
        zarr_format=2,
        fill_value=0,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.codecs == (
        BytesCodec(endian="little"),
        compressor_v3,
    )


def test_convert_filter(local_store: Store) -> None:
    filter_v2 = numcodecs.Delta(dtype="<u2", astype="<u2")
    filter_v3 = Delta(dtype="<u2", astype="<u2")

    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        compressors=None,
        filters=filter_v2,
        zarr_format=2,
        fill_value=0,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.codecs == (
        filter_v3,
        BytesCodec(endian="little"),
    )


@pytest.mark.parametrize(
    ("order", "expected_codecs"),
    [
        ("C", (BytesCodec(endian="little"),)),
        ("F", (TransposeCodec(order=(1, 0)), BytesCodec(endian="little"))),
    ],
)
def test_convert_C_vs_F_order(
    local_store: Store, order: str, expected_codecs: tuple[Codec]
) -> None:
    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        compressors=None,
        zarr_format=2,
        fill_value=0,
        order=order,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3

    assert metadata.codecs == expected_codecs


@pytest.mark.parametrize(
    ("dtype", "expected_data_type", "expected_codecs"),
    [
        ("uint8", UInt8(), (BytesCodec(endian=None),)),
        ("uint16", UInt16(), (BytesCodec(endian="little"),)),
    ],
    ids=["single_byte", "multi_byte"],
)
def test_convert_endian(
    local_store: Store, dtype: str, expected_data_type: BaseInt, expected_codecs: tuple[Codec]
) -> None:
    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype=dtype,
        compressors=None,
        zarr_format=2,
        fill_value=0,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.data_type == expected_data_type
    assert metadata.codecs == expected_codecs


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


def test_convert_unknown_codec(local_store: Store) -> None:
    """Attempting to convert a codec without a v3 equivalent should always fail"""

    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        filters=[numcodecs.Categorize(labels=["a", "b"], dtype=object)],
        zarr_format=2,
        fill_value=0,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert (
        str(result.exception) == "Couldn't find corresponding numcodecs.zarr3 codec for categorize"
    )


def test_convert_incorrect_filter(local_store: Store) -> None:
    """Attempting to convert a filter (which is the wrong type of codec) should always fail"""

    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        filters=[numcodecs.Zstd(level=3)],
        zarr_format=2,
        fill_value=0,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert (
        str(result.exception) == "Filter <class 'numcodecs.zarr3.Zstd'> is not an ArrayArrayCodec"
    )


def test_convert_incorrect_compressor(local_store: Store) -> None:
    """Attempting to convert a compressor (which is the wrong type of codec) should always fail"""

    zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        compressors=numcodecs.Delta(dtype="<u2", astype="<u2"),
        zarr_format=2,
        fill_value=0,
    )

    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert (
        str(result.exception)
        == "Compressor <class 'numcodecs.zarr3.Delta'> is not a BytesBytesCodec"
    )


def test_remove_metadata_v2(local_store: Store, expected_paths_no_metadata: list[Path]) -> None:
    """Test all v2 metadata can be removed (leaving all groups / arrays as-is)"""

    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    create_nested_zarr(local_store, attributes, ".")

    result = runner.invoke(app, ["clear", str(local_store.root), "2"])
    assert result.exit_code == 0

    # check metadata files removed, but all groups / arrays still remain
    paths = sorted(local_store.root.rglob("*"))

    expected_paths = [local_store.root / p for p in expected_paths_no_metadata]
    assert paths == expected_paths


def test_remove_metadata_v2_with_path(
    local_store: Store, expected_paths_no_metadata: list[Path]
) -> None:
    """Test only v2 metadata within the given path (group_1) is removed"""

    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    create_nested_zarr(local_store, attributes, ".")

    result = runner.invoke(app, ["clear", str(local_store.root), "2", "--path", "group_1"])
    assert result.exit_code == 0

    # check all metadata files inside group_1 are removed (.zattrs / .zgroup / .zarray should remain only inside the top
    # group)
    paths = sorted(local_store.root.rglob("*"))

    expected_paths = [local_store.root / p for p in expected_paths_no_metadata]
    expected_paths.append(local_store.root / ".zattrs")
    expected_paths.append(local_store.root / ".zgroup")
    expected_paths.append(local_store.root / "array_0" / ".zarray")
    expected_paths.append(local_store.root / "array_0" / ".zattrs")
    assert paths == sorted(expected_paths)


@pytest.mark.parametrize(
    ("zarr_format", "expected_paths"),
    [("2", "expected_paths_v3_metadata"), ("3", "expected_paths_v2_metadata")],
)
def test_remove_metadata_after_conversion(
    local_store: Store, request: pytest.FixtureRequest, zarr_format: str, expected_paths: list[Path]
) -> None:
    """Test all v2/v3 metadata can be removed after metadata conversion (all groups / arrays /
    metadata of other versions should remain as-is)"""

    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    create_nested_zarr(local_store, attributes, ".")

    # convert v2 metadata to v3 (so now both v2 and v3 metadata present!), then remove either the v2 or v3 metadata
    result = runner.invoke(app, ["convert", str(local_store.root)])
    assert result.exit_code == 0
    result = runner.invoke(app, ["clear", str(local_store.root), zarr_format])
    assert result.exit_code == 0

    paths = sorted(local_store.root.rglob("*"))
    expected_paths = [local_store.root / p for p in request.getfixturevalue(expected_paths)]
    assert paths == expected_paths
