# Rectilinear Chunk Grids

This example demonstrates rectilinear (variable-sized) chunk grids, introduced in
[#3802](https://github.com/zarr-developers/zarr-python/pull/3802). Rectilinear grids
allow different chunk sizes along each dimension, which is useful for data that
doesn't partition evenly — for example, sparse HEALPix cells grouped by parent tile,
boundary-padded HPC arrays, or ingesting existing variable-chunked datasets via
VirtualiZarr. See [Rectilinear (variable) chunk grids](../arrays.md#rectilinear-variable-chunk-grids)
in the arrays guide for an introduction to the feature.

The example chunks a HEALPix dataset by parent tile, writes it as a Zarr v3 array
with a rectilinear chunk grid, and verifies the round trip through
[Xarray](https://xarray.dev).

!!! warning "Experimental"
    Rectilinear chunk grids are an experimental feature and may change in future
    releases. In addition, this example currently requires
    [a fork of Xarray](https://github.com/maxrjones/xarray/tree/poc/unified-zarr-chunk-grid)
    with rectilinear chunk grid support (this will ideally be incorporated into a
    future Xarray release), as well as the `dask`, `healpix-geo`, and `obstore`
    packages, and it reads an example dataset from a remote server. For these
    reasons the code on this page is not executed when the documentation is built;
    the outputs shown were captured from a live run.

## Setup

Rectilinear chunk grids are disabled by default and must be explicitly enabled via
the `array.rectilinear_chunks` configuration option:

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
from healpix_geo import nested
from obstore.store import HTTPStore

import zarr
from zarr.storage import ObjectStore

# Increase concurrency for better performance with obstore
zarr.config.set({'async.concurrency': 128})
# Opt in to rectilinear chunks
zarr.config.set({'array.rectilinear_chunks': True})
```

## Inspect the HEALPix dataset

Load the remote Zarr store to understand the data structure before chunking it:

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
ob_store = HTTPStore.from_url("https://data-taos.ifremer.fr/GRID4EARTH/no_chunk_healpix.zarr")
store = ObjectStore(ob_store)
g = zarr.open_group(store, mode="r", zarr_format=2, use_consolidated=True)
arr = g['da']

print("Members:", list(g.members()))
print("Attrs:", dict(g.attrs))
print("Write chunk sizes:", arr.write_chunk_sizes)
```

```text
Members: [('cell_ids', <Array object_store://HTTPStore("https://data-taos.ifremer.fr/GRID4EARTH/no_chunk_healpix.zarr")/cell_ids shape=(222442,) dtype=int64>), ('da', <Array object_store://HTTPStore("https://data-taos.ifremer.fr/GRID4EARTH/no_chunk_healpix.zarr")/da shape=(222442,) dtype=float32>)]
Attrs: {}
Write chunk sizes: ((55611, 55611, 55611, 55609),)
```

## HEALPix-style variable chunking

Inspired by [this use case](https://github.com/zarr-developers/zarr-python/pull/3534#issuecomment-3848669859):
HEALPix grids where cells are grouped by parent tile at a coarser resolution level,
producing variable-sized chunks along the cell dimension when accounting for sparsity.

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
da = xr.open_zarr(
    store,
    zarr_format=2,
    consolidated=True,
)

depth = da.cell_ids.attrs['level']
new_depth = depth - 6
parents = nested.zoom_to(da.cell_ids, depth=depth, new_depth=new_depth)
_, chunk_sizes = np.unique(parents, return_counts=True)
print(chunk_sizes)
```

```text
[  25  645 1510 2363 3203   74  769 3963 4096  233 1603 2450 4096 4096
 3327 4047 4096 4096 1278 2113 4096 3879 4096 3842 2173  983 4046 2187
 4095 1369 4096 4096 4096 4096 3515 1395 4096 3622 4096 4096 3875 4096
 4096 4096 4096 4096 2034 4096  358 3991 4096 4096 4096 4096 2714 1210
 4096 4096 4096 4096   92 3826 4096 2629 4096 1438 4096  353 4078 3410
 2407  226  132 2738 1223   23]
```

Rechunk the dataset with these variable-sized chunks:

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
da = da.chunk({"cell_ids": tuple(chunk_sizes.tolist())})
print(da.chunks)
```

```text
Frozen({'cell_ids': (25, 645, 1510, 2363, 3203, 74, 769, 3963, 4096, 233, 1603, 2450, 4096, 4096, 3327, 4047, 4096, 4096, 1278, 2113, 4096, 3879, 4096, 3842, 2173, 983, 4046, 2187, 4095, 1369, 4096, 4096, 4096, 4096, 3515, 1395, 4096, 3622, 4096, 4096, 3875, 4096, 4096, 4096, 4096, 4096, 2034, 4096, 358, 3991, 4096, 4096, 4096, 4096, 2714, 1210, 4096, 4096, 4096, 4096, 92, 3826, 4096, 2629, 4096, 1438, 4096, 353, 4078, 3410, 2407, 226, 132, 2738, 1223, 23)})
```

## Write as rectilinear Zarr v3

Write the variable-chunked dataset to a local Zarr v3 store with rectilinear chunk
grids enabled:

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
output_path = Path(tempfile.mkdtemp()) / "healpix_rectilinear.zarr"

encoding = {
    "da": {"chunks": [chunk_sizes.tolist()]},
    "cell_ids": {"chunks": [chunk_sizes.tolist()]},
}

da.to_zarr(output_path, zarr_format=3, mode="w", encoding=encoding, consolidated=False)

print(f"Written to: {output_path}")
```

```text
Written to: /var/folders/.../T/tmp6dibcrho/healpix_rectilinear.zarr
```

## Verify the rectilinear metadata

Inspect the output store to confirm the chunk grid is serialized as `"rectilinear"`
in `zarr.json`, following the
[rectilinear chunk grid extension spec](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-grids/rectilinear).

Key things to look for in `chunk_grid`:

- **`name`**: `"rectilinear"` (the extension identifier)
- **`configuration.kind`**: `"inline"` (edge lengths stored directly in metadata)
- **`configuration.chunk_shapes`**: one entry per dimension — here a single list for
  the 1D `cell_ids` axis. Each element is either:
    - a **bare integer** for a unique edge length (e.g., `25`, `645`)
    - a **`[value, count]` array** using
      [run-length encoding](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-grids/rectilinear#run-length-encoding)
      for consecutive repeated sizes (e.g., `[4096, 4]` means four consecutive chunks
      of size 4096)

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
# Read the zarr.json for the 'da' array
da_meta_path = output_path / "da" / "zarr.json"
meta = json.loads(da_meta_path.read_text())
print(meta['chunk_grid'])
```

```text
{'name': 'rectilinear', 'configuration': {'kind': 'inline', 'chunk_shapes': [[25, 645, 1510, 2363, 3203, 74, 769, 3963, 4096, 233, 1603, 2450, [4096, 2], 3327, 4047, [4096, 2], 1278, 2113, 4096, 3879, 4096, 3842, 2173, 983, 4046, 2187, 4095, 1369, [4096, 4], 3515, 1395, 4096, 3622, [4096, 2], 3875, [4096, 5], 2034, 4096, 358, 3991, [4096, 4], 2714, 1210, [4096, 4], 92, 3826, 4096, 2629, 4096, 1438, 4096, 353, 4078, 3410, 2407, 226, 132, 2738, 1223, 23]]}}
```

## Round-trip verification

Read the rectilinear store back and confirm the chunk sizes are preserved:

```python exec="false" reason="requires an xarray fork with rectilinear chunk grid support and remote example data"
roundtrip = xr.open_zarr(output_path, zarr_format=3, consolidated=False)

print("Round-trip chunk sizes:", roundtrip.chunks)
```

```text
Round-trip chunk sizes: Frozen({'cell_ids': (25, 645, 1510, 2363, 3203, 74, 769, 3963, 4096, 233, 1603, 2450, 4096, 4096, 3327, 4047, 4096, 4096, 1278, 2113, 4096, 3879, 4096, 3842, 2173, 983, 4046, 2187, 4095, 1369, 4096, 4096, 4096, 4096, 3515, 1395, 4096, 3622, 4096, 4096, 3875, 4096, 4096, 4096, 4096, 4096, 2034, 4096, 358, 3991, 4096, 4096, 4096, 4096, 2714, 1210, 4096, 4096, 4096, 4096, 92, 3826, 4096, 2629, 4096, 1438, 4096, 353, 4078, 3410, 2407, 226, 132, 2738, 1223, 23)})
```
