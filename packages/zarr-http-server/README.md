# zarr-http-server

HTTP server for Zarr stores, arrays, and groups.

Documentation: <https://zarr-http-server.readthedocs.io/>

`zarr-http-server` exposes a Zarr `Store`, `Array`, or `Group` over HTTP via an
ASGI app, so any HTTP-capable client — including zarr-python itself, via
`FsspecStore` or `ObjectStore` — can read the data. The app is built on
[Starlette](https://www.starlette.io/) and can be run with any ASGI server;
the `serve` / `serve_background` helpers run it with
[Uvicorn](https://www.uvicorn.org/).

> [!WARNING]
> This package is experimental. Its API may change or be removed at any point.

## Installation

```bash
pip install zarr-http-server
```

## Quick start

```python
import zarr
from zarr_http_server import node_app, serve_background

store = zarr.storage.MemoryStore()
array = zarr.create_array(store, shape=(100,), chunks=(10,), dtype="float64")

with serve_background(node_app(array)) as server:
    print(server.url)   # e.g. http://127.0.0.1:8000
```

Building an app and running it are separate steps, and either app works with
either runner:

- **Build** with `store_app` to serve every key in a store, or `node_app` to
  serve only the keys belonging to one `Array` or `Group` — requests for keys
  outside that node return 404 even when those keys exist in the underlying
  store.
- **Run** with `serve`, which blocks, or `serve_background`, which returns a
  handle you can shut down later.

Reads are all that is enabled by default: `GET` and `HEAD` are served, and
`PUT`, `POST`, `DELETE` and `PATCH` are answered with 405.

> [!CAUTION]
> `store_app` applies no per-key filtering. Only point it at a store whose full
> contents are safe to serve, and note that enabling `PUT` grants write access
> to everything the store contains. The
> [user guide](https://zarr-http-server.readthedocs.io/en/latest/guide/#read-only-serving)
> covers the read-only guarantees available.

The [user guide](https://zarr-http-server.readthedocs.io/en/latest/guide/)
covers byte ranges, CORS, writes, serving several nodes, running from a
notebook, and Uvicorn configuration.

## Examples

[`examples/`](examples/) holds a runnable script and a notebook, both executed
by the test suite so neither can drift from the code:

```bash
uv run examples/serve.py
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
