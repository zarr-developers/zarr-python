import json
import os

import zarr

store = zarr.DirectoryStore("data/chunking_test.zarr")
z = zarr.zeros((20, 3), chunks=(3, 2), shards=(2, 2), store=store, overwrite=True, compressor=None)
z[:10, :] = 42
z[15, 1] = 389
z[19, 2] = 1
z[0, 1] = -4.2

print(store[".zarray"].decode())
# {
#     "chunks": [
#         3,
#         2
#     ],
#     "compressor": null,
#     "dtype": "<f8",
#     "fill_value": 0.0,
#     "filters": null,
#     "order": "C",
#     "shape": [
#         20,
#         3
#     ],
#     "shard_format": "morton_order",
#     "shards": [
#         2,
#         2
#     ],
#     "zarr_format": 2
# }

assert json.loads(store[".zarray"].decode()) ["shards"] == [2, 2]

print("ONDISK", sorted(os.listdir("data/chunking_test.zarr")))
print("STORE", sorted(store))
print("CHUNKSTORE (SHARDED)", sorted(z.chunk_store))

# ONDISK ['.zarray', '0.0', '1.0', '2.0', '3.0']
# STORE ['.zarray', '0.0', '1.0', '2.0', '3.0']
# CHUNKSTORE (SHARDED) ['.zarray', '0.0', '0.1', '1.0', '1.1', '2.0', '2.1', '3.0', '3.1', '5.0', '6.1']

z_reopened = zarr.open("data/chunking_test.zarr")
assert z_reopened.shards == (2, 2)
assert z_reopened[15, 1] == 389
assert z_reopened[19, 2] == 1
assert z_reopened[0, 1] == -4.2
assert z_reopened[0, 0] == 42
