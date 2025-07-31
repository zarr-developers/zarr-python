import zarr
import zarr.storage

local_store = zarr.storage.LocalStore('test.zarr')  # doctest: +SKIP
cache = zarr.storage.LRUStoreCache(local_store, max_size=2**28)
zarr_array = zarr.ones((100, 100), chunks=(10, 10), dtype='f8', store=cache, mode='w')  # doctest: +REMOTE_DATA
