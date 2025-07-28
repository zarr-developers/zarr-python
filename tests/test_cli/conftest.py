from pathlib import Path
from typing import Any, Literal

import pytest

import zarr
from zarr.abc.store import Store
from zarr.core.common import ZarrFormat


def create_nested_zarr(
    store: Store,
    attributes: dict[str, Any] | None = None,
    separator: Literal[".", "/"] = ".",
    zarr_format: ZarrFormat = 2,
) -> list[str]:
    """Create a zarr with nested groups / arrays for testing, returning the paths to all."""

    if attributes is None:
        attributes = {"baz": 42, "qux": [1, 4, 7, 12]}

    # 3 levels of nested groups
    group_0 = zarr.create_group(store=store, zarr_format=zarr_format, attributes=attributes)
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
def expected_paths() -> list[Path]:
    """Expected paths for create_nested_zarr, with no metadata files or chunks"""
    return [
        Path("array_0"),
        Path("group_1"),
        Path("group_1/array_1"),
        Path("group_1/group_2"),
        Path("group_1/group_2/array_2"),
    ]


@pytest.fixture
def expected_chunks() -> list[Path]:
    """Expected chunks for create_nested_zarr"""
    return [
        Path("array_0/0.0"),
        Path("array_0/0.1"),
        Path("array_0/1.0"),
        Path("array_0/1.1"),
        Path("group_1/array_1/0.0"),
        Path("group_1/array_1/0.1"),
        Path("group_1/array_1/1.0"),
        Path("group_1/array_1/1.1"),
        Path("group_1/group_2/array_2/0.0"),
        Path("group_1/group_2/array_2/0.1"),
        Path("group_1/group_2/array_2/1.0"),
        Path("group_1/group_2/array_2/1.1"),
    ]


@pytest.fixture
def expected_v3_metadata() -> list[Path]:
    """Expected v3 metadata for create_nested_zarr"""
    return sorted(
        [
            Path("zarr.json"),
            Path("array_0/zarr.json"),
            Path("group_1/zarr.json"),
            Path("group_1/array_1/zarr.json"),
            Path("group_1/group_2/zarr.json"),
            Path("group_1/group_2/array_2/zarr.json"),
        ]
    )


@pytest.fixture
def expected_v2_metadata() -> list[Path]:
    """Expected v2 metadata for create_nested_zarr"""
    return sorted(
        [
            Path(".zgroup"),
            Path(".zattrs"),
            Path("array_0/.zarray"),
            Path("array_0/.zattrs"),
            Path("group_1/.zgroup"),
            Path("group_1/.zattrs"),
            Path("group_1/array_1/.zarray"),
            Path("group_1/array_1/.zattrs"),
            Path("group_1/group_2/.zgroup"),
            Path("group_1/group_2/.zattrs"),
            Path("group_1/group_2/array_2/.zarray"),
            Path("group_1/group_2/array_2/.zattrs"),
        ]
    )


@pytest.fixture
def expected_paths_no_metadata(
    expected_paths: list[Path], expected_chunks: list[Path]
) -> list[Path]:
    return sorted(expected_paths + expected_chunks)


@pytest.fixture
def expected_paths_v3_metadata(
    expected_paths: list[Path], expected_chunks: list[Path], expected_v3_metadata: list[Path]
) -> list[Path]:
    return sorted(expected_paths + expected_chunks + expected_v3_metadata)


@pytest.fixture
def expected_paths_v3_metadata_no_chunks(
    expected_paths: list[Path], expected_v3_metadata: list[Path]
) -> list[Path]:
    return sorted(expected_paths + expected_v3_metadata)


@pytest.fixture
def expected_paths_v2_metadata(
    expected_paths: list[Path], expected_chunks: list[Path], expected_v2_metadata: list[Path]
) -> list[Path]:
    return sorted(expected_paths + expected_chunks + expected_v2_metadata)


@pytest.fixture
def expected_paths_v2_v3_metadata(
    expected_paths: list[Path],
    expected_chunks: list[Path],
    expected_v2_metadata: list[Path],
    expected_v3_metadata: list[Path],
) -> list[Path]:
    return sorted(expected_paths + expected_chunks + expected_v2_metadata + expected_v3_metadata)
