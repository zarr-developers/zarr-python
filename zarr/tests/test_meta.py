from zarr.meta import dumps, loads
import zarr

from nose.tools import eq_ as eq


def test_simple():
    for dt in ['f8', [('a', 'f8')], [('a', 'f8'), ('b', 'i1')]]:
        x = zarr.empty(shape=(1000, 1000), chunks=(100, 100), dtype=dt)
        meta = x.store.meta
        eq(loads(dumps(meta)), meta)
