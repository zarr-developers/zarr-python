"""
Benchmarks for end-to-end read/write performance of Zarr
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.benchmarks.common import Layout

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

    from zarr.abc.store import Store
    from zarr.core.common import NamedConfig
from operator import getitem, setitem
from typing import Any, Literal

import pytest

from zarr import create_array

CompressorName = Literal["gzip"] | None

compressors: dict[CompressorName, NamedConfig[Any, Any] | None] = {
    None: None,
    "gzip": {"name": "gzip", "configuration": {"level": 1}},
}


layouts: tuple[Layout, ...] = (
    # No shards, just 1000 chunks
    Layout(shape=(1_000_000,), chunks=(1000,), shards=None),
    # 1:1 chunk:shard shape, should measure overhead of sharding
    Layout(shape=(1_000_000,), chunks=(1000,), shards=(1000,)),
    # One shard with all the chunks, should measure overhead of handling inner shard chunks
    Layout(shape=(1_000_000,), chunks=(100,), shards=(10000 * 100,)),
)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_write_array(
    store: Store, layout: Layout, compression_name: CompressorName, benchmark: BenchmarkFixture
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )

    benchmark(setitem, arr, Ellipsis, 1)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_read_array(
    store: Store, layout: Layout, compression_name: CompressorName, benchmark: BenchmarkFixture
) -> None:
    """
    Test the time required to fill an array with a single value
    """
    arr = create_array(
        store,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        shards=layout.shards,
        compressors=compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )
    arr[:] = 1
    benchmark(getitem, arr, Ellipsis)


# Zarr v2 benchmark tests

v2_compressors: dict[CompressorName, NamedConfig[Any, Any] | None] = {
    None: None,
    "gzip": {"name": "gzip", "configuration": {"level": 1}},
}

v2_layouts: tuple[Layout, ...] = (
    # No shards, just 1000 chunks
    Layout(shape=(1_000_000,), chunks=(1000,), shards=None),
    # Larger chunks (v2 doesn't support shards, so we skip the other layouts)
    Layout(shape=(1_000_000,), chunks=(10000,), shards=None),
)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", v2_layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_read_array_v2(
    store: Store, layout: Layout, compression_name: CompressorName, benchmark: BenchmarkFixture
) -> None:
    """
    Test the time required to read a Zarr v2 array

    This benchmark tests reading performance of Zarr v2 format arrays,
    which only supports traditional chunking without shards.
    """
    arr = create_array(
        store,
        zarr_format=2,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        compressors=v2_compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )
    arr[:] = 1
    benchmark(getitem, arr, Ellipsis)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", v2_layouts, ids=str)
@pytest.mark.parametrize("store", ["memory", "local"], indirect=["store"])
def test_write_array_v2(
    store: Store, layout: Layout, compression_name: CompressorName, benchmark: BenchmarkFixture
) -> None:
    """
    Test the time required to write a Zarr v2 array

    This benchmark tests writing performance of Zarr v2 format arrays,
    which only supports traditional chunking without shards.
    """
    arr = create_array(
        store,
        zarr_format=2,
        dtype="uint8",
        shape=layout.shape,
        chunks=layout.chunks,
        compressors=v2_compressors[compression_name],  # type: ignore[arg-type]
        fill_value=0,
    )

    benchmark(setitem, arr, Ellipsis, 1)
