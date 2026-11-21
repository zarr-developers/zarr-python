def test_shards_initialized_with_sibling_prefix():
    import zarr

    g = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    lat = g.create_array("lat", shape=(4,), chunks=(2,), dtype="f8")
    g.create_array("lat_bnds", shape=(4, 2), chunks=(2, 2), dtype="f8")
    lat[:2] = 1.0
    assert lat.nchunks_initialized == 1
