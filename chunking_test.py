import json
import os

import zarr

store = zarr.DirectoryStore("data/chunking_test.zarr")
z = zarr.zeros((20, 3), chunks=(3, 3), shards=(2, 2), store=store, overwrite=True, compressor=None)
z[...] = 42
z[15, 1] = 389
z[19, 2] = 1
z[0, 1] = -4.2

print("ONDISK", sorted(os.listdir("data/chunking_test.zarr")))
assert json.loads(store[".zarray"].decode()) ["shards"] == [2, 2]

print("STORE", list(store))
print("CHUNKSTORE (SHARDED)", list(z.chunk_store))

z_reopened = zarr.open("data/chunking_test.zarr")
assert z_reopened.shards == (2, 2)
assert z_reopened[15, 1] == 389
assert z_reopened[19, 2] == 1
assert z_reopened[0, 1] == -4.2
assert z_reopened[0, 0] == 42
