import zarr


class OneDIndexingSuite:
    def time_1d_fill_no_sharding_no_compression(self) -> None:
        array = zarr.create_array(
            store={},
            shape=(1000000,),
            dtype="i4",
            compressors=None,
            filters=None,
            chunks=(10000,),
            fill_value=0,
        )
        array[:] = 1

    def time_1d_fill_sharding_no_compression(self) -> None:
        array = zarr.create_array(
            store={},
            shape=(1000000,),
            dtype="i4",
            compressors=None,
            filters=None,
            chunks=(10000,),
            shards=(50000,),
            fill_value=0,
        )
        array[:] = 1
