When creating arrays without explicitly specifying a chunk size using `zarr.create` and other
array creation routines, the chunk size will now set automatically instead of defaulting to the data shape.
For large arrays this will result in smaller default chunk sizes.
To retain previous behaviour, explicitly set the chunk shape to the data shape.

This fix matches the existing chunking behaviour of
`zarr.save_array` and `zarr.api.asynchronous.AsyncArray.create`.
