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
    Layout(shape=(1024**2,), chunks=(1024,), shards=None),
    Layout(shape=(1024**2,), chunks=(1024,), shards=(1024,)),
    Layout(shape=(1024**2,), chunks=(1024,), shards=(1024 * 64,)),
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
@pytest.mark.parametrize("layout", layouts)
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
