import zarr

store = zarr.DirectoryStore("data")
r = zarr.open_group(store=store)
z = r.full("myArray", 42, shape=(), dtype="i4", compressor=None)

print(z.oindex[...])
