from zarr.v3 import Array
from zarr.v3.store import MemoryStore


def test_create_array_v3():

    store = MemoryStore()
    _ = Array.create(
        store,
        shape=(100,),
        chunk_shape=(100,),
        dtype="int32",
    )
