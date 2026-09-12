# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "zarr-http-server @ git+https://github.com/zarr-developers/zarr-python.git@main#subdirectory=packages/zarr-http-server",
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
from zarr.storage import MemoryStore

from zarr_http_server import node_app, serve_background

# -- create an array --------------------------------------------------------
store = MemoryStore()
data = np.arange(1000, dtype="uint8").reshape(10, 10, 10)
# no compression
arr = zarr.create_array(store, data=data, chunks=(5, 5, 5), write_data=True, compressors=None)

# -- serve it in the background ---------------------------------------------
# port=0 asks the OS for a free port, so running this twice -- or running it
# while something else holds 8000 -- works. `server.url` reports what it bound.
with serve_background(node_app(arr), host="127.0.0.1") as server:
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
