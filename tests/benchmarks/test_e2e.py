"""
Test the basic end-to-end read/write performance of Zarr
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


@dataclass(kw_only=True, frozen=True)
class Layout:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    shards: tuple[int, ...] | None


layouts: tuple[Layout, ...] = (
    Layout(shape=(16,), chunks=(1,), shards=None),
    Layout(shape=(16,), chunks=(16,), shards=None),
    Layout(shape=(16,), chunks=(1,), shards=(1,)),
    Layout(shape=(16,), chunks=(1,), shards=(16,)),
    Layout(shape=(16,) * 2, chunks=(1,) * 2, shards=None),
    Layout(shape=(16,) * 2, chunks=(16,) * 2, shards=None),
    Layout(shape=(16,) * 2, chunks=(1,) * 2, shards=(1,) * 2),
    Layout(shape=(16,) * 2, chunks=(1,) * 2, shards=(16,) * 2),
)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("store", ["memory", "local", "zip"], indirect=["store"])
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
        compressors=compressors[compression_name],
        fill_value=0,
    )

    benchmark.pedantic(setitem, args=(arr, Ellipsis, 1), rounds=16)


@pytest.mark.parametrize("compression_name", [None, "gzip"])
@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("store", ["memory", "local", "zip"], indirect=["store"])
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
        compressors=compressors[compression_name],
        fill_value=0,
    )
    arr[:] = 1
    benchmark.pedantic(getitem, args=(arr, Ellipsis), rounds=16)
