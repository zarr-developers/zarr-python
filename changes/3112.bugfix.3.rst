When using ``from_array`` to copy a Zarr format 2 array to a Zarr format 3 array, if the memory order of the input array is ``"F"`` a warning is raised and the order ignored.
This is because Zarr format 3 arrays are always stored in "C" order.
