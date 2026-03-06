# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr[server] @ git+https://github.com/zarr-developers/zarr-python.git@main",
#   "httpx",
# ]
# ///
"""
Serve a Zarr array over HTTP and fetch its metadata and chunks.

This example creates an in-memory array, serves it in a background thread,
then uses ``httpx`` to request the ``zarr.json`` metadata document and a raw
chunk.
"""

import json

import httpx
import numpy as np

import zarr
from zarr.experimental.serve import serve_node
from zarr.storage import MemoryStore

# -- create an array --------------------------------------------------------
store = MemoryStore()
data = np.arange(1000, dtype="uint8").reshape(10, 10, 10)
# no compression
arr = zarr.create_array(store, data=data, chunks=(5, 5, 5), write_data=True, compressors=None)

# -- serve it in the background ---------------------------------------------
with serve_node(arr, host="127.0.0.1", port=8000, background=True) as server:
    # -- fetch metadata ------------------------------------------------------
    resp = httpx.get(f"{server.url}/zarr.json")
    assert resp.status_code == 200
    meta = resp.json()
    print("zarr.json:")
    print(json.dumps(meta, indent=2))

    # -- fetch a raw chunk ---------------------------------------------------
    resp = httpx.get(f"{server.url}/c/0/0/0")
    assert resp.status_code == 200
    print(f"\nchunk c/0/0/0: {len(resp.content)} bytes")
