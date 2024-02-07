from zarr.v3 import Array
from zarr.v3.group import Group
from zarr.v3.store import MemoryStore


def test_create_array():

    store = MemoryStore()
    _ = Array.create(
        store,
        shape=(100,),
        chunk_shape=(100,),
        dtype="int32",
    )


def test_create_array_from_group():
    store = MemoryStore()
    group = Group.create(store)
    _a = group.create_array(
        "a",
        shape=(100,),
        chunk_shape=(100,),
        dtype="int32",
    )
    _b = group.create_array(
        "b",
        shape=(200,),
        chunk_shape=(100,),
        dtype="int32",
    )
