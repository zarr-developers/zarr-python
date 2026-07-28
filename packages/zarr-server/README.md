# zarr-server

HTTP server for Zarr stores, arrays, and groups.

`zarr-server` exposes a Zarr `Store`, `Array`, or `Group` over HTTP via an
ASGI app, so any HTTP-capable client (including zarr-python itself, via
`FsspecStore` or `ObjectStore`) can read the data. The app is built on
[Starlette](https://www.starlette.io/) and can be run with any ASGI server;
`serve_store`/`serve_node` conveniences run it with
[Uvicorn](https://www.uvicorn.org/).

## Installation

```bash
pip install zarr-server
```

### Building an ASGI App

`store_app` creates an ASGI app that exposes every key
in a store:

```python
import zarr
from zarr_server import store_app

store = zarr.storage.MemoryStore()
zarr.create_array(store, shape=(100, 100), chunks=(10, 10), dtype="float64")

app = store_app(store)

# Run with any ASGI server, e.g. Uvicorn:
# uvicorn my_module:app --host 0.0.0.0 --port 8000
```

`node_app` creates an ASGI app that only serves keys
belonging to a specific `Array` or `Group`.  Requests for keys outside the node
receive a 404, even if those keys exist in the underlying store:

```python
import zarr
from zarr_server import node_app

store = zarr.storage.MemoryStore()
root = zarr.open_group(store)
root.create_array("a", shape=(10,), dtype="int32")
root.create_array("b", shape=(20,), dtype="float64")

# Only serve the array at "a" — requests for "b" will return 404.
arr = root["a"]
app = node_app(arr)
```

### Running the Server

`serve_store` and `serve_node`
build an ASGI app *and* start a [Uvicorn](https://www.uvicorn.org/) server.
By default they block until the server is shut down:

```python
from zarr_server import serve_store

serve_store(store, host="127.0.0.1", port=8000)
```

Pass `background=True` to start the server in a daemon thread and return
immediately.  The returned `BackgroundServer`
can be used as a context manager for automatic shutdown:

```python
import numpy as np

import zarr
from zarr_server import serve_node
from zarr.storage import MemoryStore

store = MemoryStore()
arr = zarr.create_array(store, shape=(100,), chunks=(10,), dtype="float64")
arr[:] = np.arange(100, dtype="float64")

with serve_node(arr, host="127.0.0.1", port=8000, background=True) as server:
    # Now open the served array from another zarr client.
    remote = zarr.open_array(server.url, mode="r")
    np.testing.assert_array_equal(remote[:], arr[:])
# Server is shut down automatically when the block exits.
```

### CORS Support

Both `store_app` and `node_app` (and their `serve_*` counterparts) accept a
`CorsOptions` parameter to enable
[CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) middleware for
browser-based clients:

```python
from zarr_server import CorsOptions, store_app

app = store_app(
    store,
    cors_options=CorsOptions(
        allow_origins=["*"],
        allow_methods=["GET"],
    ),
)
```

### HTTP Range Requests

The server supports the standard `Range` header for partial reads.  The three
forms defined by [RFC 7233](https://httpwg.org/specs/rfc7233.html) are supported:

| Header               | Meaning                        |
| -------------------- | ------------------------------ |
| `bytes=0-99`         | First 100 bytes                |
| `bytes=100-`         | Everything from byte 100       |
| `bytes=-50`          | Last 50 bytes                  |

A successful range request returns HTTP 206 (Partial Content).

### Write Support

By default only `GET` requests are accepted.  To enable writes, pass
`methods={"GET", "PUT"}`:

```python
app = store_app(store, methods={"GET", "PUT"})
```

A `PUT` request stores the request body at the given path and returns 204 (No Content).

## Example

`examples/serve.py` creates an in-memory Zarr array, serves it over HTTP with
`serve_node`, and fetches the `zarr.json` metadata document and a raw chunk
using `httpx`.

```bash
python examples/serve.py
```

Or run with uv:

```bash
uv run examples/serve.py
```
