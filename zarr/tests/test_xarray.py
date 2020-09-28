import sys


if sys.version_info >= (3, 8):
    import requests
    import zarr
    import xarray as xr
    from zarr.v3 import (
        V2from3Adapter,
        SyncV3MemoryStore,
        SyncV3DirectoryStore,
        StoreComparer,
    )
    from zarr import DirectoryStore

    from pathlib import Path

    def test_xarray():

        p = Path("rasm.nc")
        if not p.exists():
            r = requests.get("https://github.com/pydata/xarray-data/raw/master/rasm.nc")
            with open("rasm.nc", "wb") as f:
                f.write(r.content)

        ds = xr.open_dataset("rasm.nc")

        compressor = zarr.Blosc(cname="zstd", clevel=3)
        encoding = {vname: {"compressor": compressor} for vname in ds.data_vars}

        v33 = SyncV3DirectoryStore("v3.zarr")
        v23 = V2from3Adapter(v33)

        # use xarray to write to a v3 store via the adapter, so this will create a v3-zarr file
        ds.to_zarr(v23, encoding=encoding)

        # now we open directly the v3 store and check we get the right things
        zarr_ds = xr.open_zarr(store=v33)

        assert len(zarr_ds.attrs) == 11
        assert zarr_ds.Tair.shape == (36, 205, 275)
