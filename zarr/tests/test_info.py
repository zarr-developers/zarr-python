import numcodecs
import pytest

import zarr
from zarr.util import InfoReporter


@pytest.mark.parametrize("array_size", [10, 15000])
def test_info(array_size):
    # setup
    g = zarr.group(store=dict(), chunk_store=dict(), synchronizer=zarr.ThreadSynchronizer())
    g.create_group("foo")
    z = g.zeros("bar", shape=array_size, filters=[numcodecs.Adler32()])

    # test group info
    items = g.info_items()
    keys = sorted([k for k, _ in items])
    expected_keys = sorted(
        [
            "Type",
            "Read-only",
            "Synchronizer type",
            "Store type",
            "Chunk store type",
            "No. members",
            "No. arrays",
            "No. groups",
            "Arrays",
            "Groups",
            "Name",
        ]
    )
    assert expected_keys == keys

    # can also get a string representation of info via the info attribute
    assert isinstance(g.info, InfoReporter)
    assert "Type" in repr(g.info)

    # test array info
    items = z.info_items()
    keys = sorted([k for k, _ in items])
    expected_keys = sorted(
        [
            "Type",
            "Data type",
            "Shape",
            "Chunk shape",
            "Order",
            "Read-only",
            "Filter [0]",
            "Compressor",
            "Synchronizer type",
            "Store type",
            "Chunk store type",
            "No. bytes",
            "No. bytes stored",
            "Storage ratio",
            "Chunks initialized",
            "Name",
        ]
    )
    assert expected_keys == keys

    # can also get a string representation of info via the info attribute
    assert isinstance(z.info, InfoReporter)
    assert "Type" in repr(z.info)
