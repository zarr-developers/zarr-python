import lzma
from pathlib import Path

import numcodecs
import numcodecs.abc
import pytest
from numcodecs.zarr3 import LZMA, Delta

import zarr
from tests.test_cli.conftest import create_nested_zarr
from zarr.abc.codec import Codec
from zarr.abc.store import Store
from zarr.codecs.blosc import BloscCodec
from zarr.codecs.bytes import BytesCodec
from zarr.codecs.gzip import GzipCodec
from zarr.codecs.transpose import TransposeCodec
from zarr.codecs.zstd import ZstdCodec
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.chunk_key_encodings import V2ChunkKeyEncoding
from zarr.core.common import ZarrFormat
from zarr.core.dtype.npy.int import BaseInt, UInt8, UInt16
from zarr.storage._local import LocalStore

typer_testing = pytest.importorskip(
    "typer.testing", reason="optional cli dependencies aren't installed"
)
cli = pytest.importorskip(
    "zarr.core.metadata.converter.cli", reason="optional cli dependencies aren't installed"
)

runner = typer_testing.CliRunner()


def test_migrate_array(local_store: Store) -> None:
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

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.node_type == "array"
    assert metadata.shape == shape
    assert metadata.chunk_grid == RegularChunkGrid(chunk_shape=chunks)
    assert metadata.chunk_key_encoding == V2ChunkKeyEncoding(separator=".")
    assert metadata.data_type == UInt16(endianness="little")
    assert metadata.codecs == (
        BytesCodec(endian="little"),
        BloscCodec(typesize=2, cname="zstd", clevel=3, shuffle="shuffle", blocksize=0),
    )
    assert metadata.fill_value == fill_value
    assert metadata.attributes == attributes
    assert metadata.dimension_names is None
    assert metadata.storage_transformers == ()


def test_migrate_group(local_store: Store) -> None:
    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    zarr.create_group(store=local_store, zarr_format=2, attributes=attributes)

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.node_type == "group"
    assert metadata.attributes == attributes
    assert metadata.consolidated_metadata is None


@pytest.mark.parametrize("separator", [".", "/"])
def test_migrate_nested_groups_and_arrays_in_place(
    local_store: Store, separator: str, expected_v3_metadata: list[Path]
) -> None:
    """Test that zarr.json are made at the correct points in a hierarchy of groups and arrays
    (including when there are additional dirs due to using a / separator)"""

    attributes = {"baz": 42, "qux": [1, 4, 7, 12]}
    paths = create_nested_zarr(local_store, attributes=attributes, separator=separator)

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0

    zarr_json_paths = sorted(local_store.root.rglob("zarr.json"))
    expected_zarr_json_paths = [local_store.root / p for p in expected_v3_metadata]
    assert zarr_json_paths == expected_zarr_json_paths

    # Check converted zarr can be opened + metadata accessed at all levels
    zarr_array = zarr.open(local_store.root, zarr_format=3)
    for path in paths:
        zarr_v3 = zarr_array[path]
        metadata = zarr_v3.metadata
        assert metadata.zarr_format == 3
        assert metadata.attributes == attributes


@pytest.mark.parametrize("separator", [".", "/"])
async def test_migrate_nested_groups_and_arrays_separate_location(
    tmp_path: Path,
    separator: str,
    expected_v2_metadata: list[Path],
    expected_v3_metadata: list[Path],
) -> None:
    """Test that zarr.json are made at the correct paths, when saving to a separate output location."""

    input_zarr_path = tmp_path / "input.zarr"
    output_zarr_path = tmp_path / "output.zarr"

    local_store = await LocalStore.open(str(input_zarr_path))
    create_nested_zarr(local_store, separator=separator)

    result = runner.invoke(cli.app, ["migrate", "v3", str(input_zarr_path), str(output_zarr_path)])
    assert result.exit_code == 0

    # Files in input zarr should be unchanged i.e. still v2 only
    zarr_json_paths = sorted(input_zarr_path.rglob("zarr.json"))
    assert len(zarr_json_paths) == 0

    paths = [
        path
        for path in input_zarr_path.rglob("*")
        if path.stem in [".zarray", ".zgroup", ".zattrs"]
    ]
    expected_paths = [input_zarr_path / p for p in expected_v2_metadata]
    assert sorted(paths) == expected_paths

    # Files in output zarr should only contain v3 metadata
    zarr_json_paths = sorted(output_zarr_path.rglob("zarr.json"))
    expected_zarr_json_paths = [output_zarr_path / p for p in expected_v3_metadata]
    assert zarr_json_paths == expected_zarr_json_paths


def test_remove_v2_metadata_option_in_place(
    local_store: Store, expected_paths_v3_metadata: list[Path]
) -> None:
    create_nested_zarr(local_store)

    # convert v2 metadata to v3, then remove v2 metadata
    result = runner.invoke(
        cli.app, ["migrate", "v3", str(local_store.root), "--remove-v2-metadata"]
    )
    assert result.exit_code == 0

    paths = sorted(local_store.root.rglob("*"))
    expected_paths = [local_store.root / p for p in expected_paths_v3_metadata]
    assert paths == expected_paths


async def test_remove_v2_metadata_option_separate_location(
    tmp_path: Path, expected_paths_v2_metadata: list[Path]
) -> None:
    """Check that when using --remove-v2-metadata with a separate output location, no v2 metadata is removed from
    the input location."""

    input_zarr_path = tmp_path / "input.zarr"
    output_zarr_path = tmp_path / "output.zarr"

    local_store = await LocalStore.open(str(input_zarr_path))
    create_nested_zarr(local_store)

    result = runner.invoke(
        cli.app,
        ["migrate", "v3", str(input_zarr_path), str(output_zarr_path), "--remove-v2-metadata"],
    )
    assert result.exit_code == 0

    # input image should be unchanged
    paths = sorted(input_zarr_path.rglob("*"))
    expected_paths = [input_zarr_path / p for p in expected_paths_v2_metadata]
    assert paths == expected_paths


def test_overwrite_option_in_place(
    local_store: Store, expected_paths_v2_v3_metadata: list[Path]
) -> None:
    create_nested_zarr(local_store)

    # add v3 metadata in place
    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0

    # check that v3 metadata can be overwritten with --overwrite
    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root), "--overwrite"])
    assert result.exit_code == 0

    paths = sorted(local_store.root.rglob("*"))
    expected_paths = [local_store.root / p for p in expected_paths_v2_v3_metadata]
    assert paths == expected_paths


@pytest.mark.parametrize("separator", [".", "/"])
def test_migrate_sub_group(
    local_store: Store, separator: str, expected_v3_metadata: list[Path]
) -> None:
    """Test that only arrays/groups within group_1 are converted (+ no other files in store)"""

    create_nested_zarr(local_store, separator=separator)
    group_path = local_store.root / "group_1"

    result = runner.invoke(cli.app, ["migrate", "v3", str(group_path)])
    assert result.exit_code == 0

    zarr_json_paths = sorted(local_store.root.rglob("zarr.json"))
    expected_zarr_json_paths = [
        local_store.root / p
        for p in expected_v3_metadata
        if group_path in (local_store.root / p).parents
    ]
    assert zarr_json_paths == expected_zarr_json_paths


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
                format=lzma.FORMAT_RAW,
                check=-1,
                preset=None,
                filters=[
                    {"id": lzma.FILTER_DELTA, "dist": 4},
                    {"id": lzma.FILTER_LZMA2, "preset": 1},
                ],
            ),
            LZMA(
                format=lzma.FORMAT_RAW,
                check=-1,
                preset=None,
                filters=[
                    {"id": lzma.FILTER_DELTA, "dist": 4},
                    {"id": lzma.FILTER_LZMA2, "preset": 1},
                ],
            ),
        ),
    ],
    ids=["blosc", "zstd", "gzip", "numcodecs-compressor"],
)
def test_migrate_compressor(
    local_store: Store, compressor_v2: numcodecs.abc.Codec, compressor_v3: Codec
) -> None:
    zarr_array = zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        compressors=compressor_v2,
        zarr_format=2,
        fill_value=0,
    )
    zarr_array[:] = 1

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.codecs == (
        BytesCodec(endian="little"),
        compressor_v3,
    )
    assert (zarr_array[:] == 1).all()


def test_migrate_filter(local_store: Store) -> None:
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

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
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
def test_migrate_C_vs_F_order(
    local_store: Store, order: str, expected_codecs: tuple[Codec]
) -> None:
    zarr_array = zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype="uint16",
        compressors=None,
        zarr_format=2,
        fill_value=0,
        order=order,
    )
    zarr_array[:] = 1

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.codecs == expected_codecs
    assert (zarr_array[:] == 1).all()


@pytest.mark.parametrize(
    ("dtype", "expected_data_type", "expected_codecs"),
    [
        ("uint8", UInt8(), (BytesCodec(endian=None),)),
        ("uint16", UInt16(), (BytesCodec(endian="little"),)),
    ],
    ids=["single_byte", "multi_byte"],
)
def test_migrate_endian(
    local_store: Store, dtype: str, expected_data_type: BaseInt, expected_codecs: tuple[Codec]
) -> None:
    zarr_array = zarr.create_array(
        store=local_store,
        shape=(10, 10),
        chunks=(10, 10),
        dtype=dtype,
        compressors=None,
        zarr_format=2,
        fill_value=0,
    )
    zarr_array[:] = 1

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0
    assert (local_store.root / "zarr.json").exists()

    zarr_array = zarr.open(local_store.root, zarr_format=3)
    metadata = zarr_array.metadata
    assert metadata.zarr_format == 3
    assert metadata.data_type == expected_data_type
    assert metadata.codecs == expected_codecs
    assert (zarr_array[:] == 1).all()


@pytest.mark.parametrize("node_type", ["array", "group"])
def test_migrate_v3(local_store: Store, node_type: str) -> None:
    """Attempting to convert a v3 array/group should always fail"""

    if node_type == "array":
        zarr.create_array(
            store=local_store, shape=(10, 10), chunks=(10, 10), zarr_format=3, dtype="uint16"
        )
    else:
        zarr.create_group(store=local_store, zarr_format=3)

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert str(result.exception) == "Only arrays / groups with zarr v2 metadata can be converted"


def test_migrate_unknown_codec(local_store: Store) -> None:
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

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert (
        str(result.exception) == "Couldn't find corresponding numcodecs.zarr3 codec for categorize"
    )


def test_migrate_incorrect_filter(local_store: Store) -> None:
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

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert (
        str(result.exception) == "Filter <class 'numcodecs.zarr3.Zstd'> is not an ArrayArrayCodec"
    )


def test_migrate_incorrect_compressor(local_store: Store) -> None:
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

    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert (
        str(result.exception)
        == "Compressor <class 'numcodecs.zarr3.Delta'> is not a BytesBytesCodec"
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_remove_metadata_fails_without_force(local_store: Store, zarr_format: ZarrFormat) -> None:
    """Test removing metadata (when no alternate metadata is present) fails without --force."""

    create_nested_zarr(local_store, zarr_format=zarr_format)

    result = runner.invoke(cli.app, ["remove-metadata", f"v{zarr_format}", str(local_store.root)])
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert str(result.exception).startswith(f"Cannot remove v{zarr_format} metadata at file")


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_remove_metadata_succeeds_with_force(
    local_store: Store, zarr_format: ZarrFormat, expected_paths_no_metadata: list[Path]
) -> None:
    """Test removing metadata (when no alternate metadata is present) succeeds with --force."""

    create_nested_zarr(local_store, zarr_format=zarr_format)

    result = runner.invoke(
        cli.app, ["remove-metadata", f"v{zarr_format}", str(local_store.root), "--force"]
    )
    assert result.exit_code == 0

    paths = sorted(local_store.root.rglob("*"))
    expected_paths = [local_store.root / p for p in expected_paths_no_metadata]
    assert paths == expected_paths


def test_remove_metadata_sub_group(
    local_store: Store, expected_paths_no_metadata: list[Path]
) -> None:
    """Test only v2 metadata within group_1 is removed and rest remains un-changed."""

    create_nested_zarr(local_store)

    result = runner.invoke(
        cli.app, ["remove-metadata", "v2", str(local_store.root / "group_1"), "--force"]
    )
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
    ("zarr_format", "expected_output_paths"),
    [("v2", "expected_paths_v3_metadata"), ("v3", "expected_paths_v2_metadata")],
)
def test_remove_metadata_after_conversion(
    local_store: Store,
    request: pytest.FixtureRequest,
    zarr_format: str,
    expected_output_paths: str,
) -> None:
    """Test all v2/v3 metadata can be removed after metadata conversion (all groups / arrays /
    metadata of other versions should remain as-is)"""

    create_nested_zarr(local_store)

    # convert v2 metadata to v3 (so now both v2 and v3 metadata present!), then remove either the v2 or v3 metadata
    result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root)])
    assert result.exit_code == 0
    result = runner.invoke(cli.app, ["remove-metadata", zarr_format, str(local_store.root)])
    assert result.exit_code == 0

    paths = sorted(local_store.root.rglob("*"))
    expected_paths = request.getfixturevalue(expected_output_paths)
    expected_paths = [local_store.root / p for p in expected_paths]
    assert paths == expected_paths


@pytest.mark.parametrize("cli_command", ["migrate", "remove-metadata"])
def test_dry_run(
    local_store: Store, cli_command: str, expected_paths_v2_metadata: list[Path]
) -> None:
    """Test that all files are un-changed after a dry run"""

    create_nested_zarr(local_store)

    if cli_command == "migrate":
        result = runner.invoke(cli.app, ["migrate", "v3", str(local_store.root), "--dry-run"])
    else:
        result = runner.invoke(
            cli.app, ["remove-metadata", "v2", str(local_store.root), "--force", "--dry-run"]
        )

    assert result.exit_code == 0

    paths = sorted(local_store.root.rglob("*"))
    expected_paths = [local_store.root / p for p in expected_paths_v2_metadata]
    assert paths == expected_paths
