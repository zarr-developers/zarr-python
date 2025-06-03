import zarr

# Write fortran style array
group = zarr.group(store={})
array = group.create_array(
    name="example",
    shape=(128, 128),
    dtype="float32",
    order="F",
    dimension_names=("row", "col"),
)
