import json

from zarr.v3 import Array
from zarr.v3.group import Group
from zarr.v3.store import MemoryStore
from zarr.v3.sync import sync


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
    a = group.create_array(
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
    print(sync(store.get('a/zarr.json')))

    meta = {
        "shape": [
            100
        ],
        "data_type": "int32",
        "chunk_grid": {
            "configuration": {
                "axis": 0,
                "arrays": [
                    "a",
                    "b"
                ]

            },
            "name": "concatenate"
        },
        # "chunk_grid": {
        #     "configuration": {
        #     "chunk_shape": [
        #         100
        #     ]
        #     },
        #     "name": "regular"
        # },
        "chunk_key_encoding": {},
        # "chunk_key_encoding": {
        #     "configuration": {
        #     "separator": "/"
        #     },
        #     "name": "default"
        # },
        "fill_value": 0,
        "codecs": [
            {
            "configuration": {
                "endian": "little"
            },
            "name": "bytes"
            }
        ],
        "attributes": {},
        "zarr_format": 3,
        "node_type": "array"
    }