# zarr-http-server

HTTP server for Zarr stores, arrays, and groups.

`zarr-http-server` is developed in the
[zarr-python repository](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-http-server)
and released independently of `zarr` itself. Install it with:

```
pip install zarr-http-server
```

!!! warning "Experimental"

    This package is experimental. Its API may change or be removed at any
    point.

## What this is

`zarr-http-server` exposes a Zarr `Store`, `Array`, or `Group` over HTTP via an
ASGI app, so any HTTP-capable client — including zarr-python itself, via
`FsspecStore` or `ObjectStore` — can read the data. The app is built on
[Starlette](https://www.starlette.io/) and can be run with any ASGI server;
the `serve` / `serve_background` helpers run it with
[Uvicorn](https://uvicorn.dev/).

Building an app and running it are separate steps, and either app works with
either runner:

- **Build** with [`store_app`][zarr_http_server.store_app] to serve every key
  in a store, or [`node_app`][zarr_http_server.node_app] to serve only the keys
  belonging to one `Array` or `Group` — requests for keys outside that node
  return 404 even when those keys exist in the underlying store.
- **Run** with [`serve`][zarr_http_server.serve], which blocks, or
  [`serve_background`][zarr_http_server.serve_background], which returns a
  handle you can shut down later.

Byte-range reads, configurable CORS headers, and a configurable set of allowed
HTTP methods are handled by the app.

## Quick start

```python
import zarr
from zarr_http_server import node_app, serve_background

store = zarr.storage.MemoryStore()
array = zarr.create_array(store, shape=(100,), chunks=(10,), dtype="float64")

with serve_background(node_app(array)) as server:
    print(server.url)   # e.g. http://127.0.0.1:8000
```

Reads are all that is enabled by default: `GET` and `HEAD` are served, and
`PUT`, `POST`, `DELETE` and `PATCH` are answered with 405.

!!! danger "Serving a whole store grants access to all of it"

    `store_app` applies no per-key filtering. Only point it at a store whose
    full contents are safe to serve, and note that enabling `PUT` grants write
    access to everything the store contains. See
    [read-only serving](guide.md#read-only-serving) for the guarantees
    available.

## Next steps

- [User guide](guide.md) — building apps, running them, byte ranges, CORS,
  writes, notebooks, and Uvicorn configuration
- [API reference](api/index.md)
- [Release notes](release-notes.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-http-server/LICENSE.txt)
