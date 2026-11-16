The following have been removed, having been deprecated since Zarr-Python 3.0.0:

- `zarr.tree`, `zarr.api.synchronous.tree` and `zarr.api.asynchronous.tree`. Use `Group.tree` / `AsyncGroup.tree` instead.
- The `Array.compressor` and `AsyncArray.compressor` properties. Use `Array.compressors` instead (for Zarr format 2 arrays, `compressors[0]` is the old `compressor`, and an empty tuple means no compressor).
