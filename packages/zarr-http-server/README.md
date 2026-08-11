# zarr-http-server

HTTP server for Zarr stores, arrays, and groups.

`zarr-http-server` exposes a Zarr `Store`, `Array`, or `Group` over HTTP via an
ASGI app, so any HTTP-capable client (including zarr-python itself, via
`FsspecStore` or `ObjectStore`) can read the data. The app is built on
[Starlette](https://www.starlette.io/) and can be run with any ASGI server;
`serve_store`/`serve_node` conveniences run it with
[Uvicorn](https://www.uvicorn.org/).

## Installation

```bash
pip install zarr-http-server
```

### Building an ASGI App

`store_app` creates an ASGI app that exposes every key
in a store. Only point it at a store whose full contents are safe to serve
publicly — it grants read (and, if `PUT` is enabled, write) access to
everything the store contains, with no per-key filtering:

```python
import zarr
from zarr_http_server import store_app

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
from zarr_http_server import node_app

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
from zarr_http_server import serve_store

serve_store(store, host="127.0.0.1", port=8000)
```

Pass `background=True` to start the server in a daemon thread and return
immediately.  The returned `BackgroundServer`
can be used as a context manager for automatic shutdown:

The example below also *reads back* over HTTP, which is a client-side
concern: `zarr.open_array(server.url)` goes through `FsspecStore`, which needs
an HTTP-capable fsspec that `zarr-http-server` does not pull in.

```bash
pip install "fsspec[http]"
```


```python
import numpy as np

import zarr
from zarr_http_server import serve_node
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

### Uvicorn Configuration

`serve_store` and `serve_node` name the options most callers need — `host`,
`port`, `background`, `shutdown_timeout` — and forward anything else to
`uvicorn.Config` through `uvicorn_options`, so nothing uvicorn can do is out
of reach:

```python
server = serve_store(
    store,
    host="0.0.0.0",
    port=8443,
    background=True,
    uvicorn_options={
        "ssl_keyfile": "key.pem",
        "ssl_certfile": "cert.pem",
        "proxy_headers": True,
        "forwarded_allow_ips": "10.0.0.0/8",
        "log_level": "warning",
    },
)
```

Keys you pass are merged over the ones set for you, so they win. `server.url`
reflects the scheme actually in use (`https` when TLS is configured) and is
`None` when the server is not bound to a TCP host and port — a `uds` or `fd`
bind has no URL to report.

### CORS Support

Both `store_app` and `node_app` (and their `serve_*` counterparts) accept a
`CorsOptions` parameter to enable
[CORS](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) middleware for
browser-based clients:

```python
from zarr_http_server import CorsOptions, store_app

app = store_app(
    store,
    cors_options=CorsOptions(
        allow_origins=["*"],
        allow_methods=["GET"],
    ),
)
```

`CorsOptions` carries every parameter Starlette's `CORSMiddleware` accepts —
`allow_headers`, `allow_credentials`, `allow_origin_regex`,
`allow_private_network`, `expose_headers` and `max_age` as well as the two
above — so configuring CORS never means reaching around this package. All keys
are optional.

Two defaults differ from Starlette's, because the server knows things the
caller should not have to. It emits `Content-Range` on every ranged response,
which is *not* a CORS-safelisted response header, so `expose_headers` defaults
to `["Content-Range"]` — otherwise a browser client can read the bytes but not
learn which bytes it got. And it accepts a `Range` request header, so
`allow_headers` defaults to `["Range"]` — otherwise a preflight naming `Range`
is rejected. A key you supply replaces the default outright, so
`expose_headers=[]` means "expose nothing".

### HTTP Range Requests

The server supports the standard `Range` header for partial reads.  The three
forms defined by [RFC 7233](https://httpwg.org/specs/rfc7233.html) are supported:

| Header               | Meaning                        |
| -------------------- | ------------------------------ |
| `bytes=0-99`         | First 100 bytes                |
| `bytes=100-`         | Everything from byte 100       |
| `bytes=-50`          | Last 50 bytes                  |

A successful range request returns HTTP 206 (Partial Content) with a
`Content-Range` header, including for suffix ranges — the server resolves
`bytes=-50` against the object's size so the response says which bytes it
carries.

A range that is well-formed but names nothing readable — one lying wholly
beyond the end of the object, an inverted one such as `bytes=5-2`, or
`bytes=-0` — returns 416 (Range Not Satisfiable). A last-byte-position past
the end of the object is *not* in that category: per RFC 9110 §14.1.2 it is
clamped, so `bytes=0-999999` on a short object returns the whole thing.

A `Range` header the server cannot use is **ignored** rather than refused, per
RFC 9110 §14.2: an unrecognized unit (`chars=0-7`), a multi-range request
(`bytes=0-7, 10-20`, which this server does not build multipart responses
for), or malformed syntax all return 200 with the full representation.

### Write Support

By default only reads are accepted: `GET`, and `HEAD` alongside it. Starlette
routes `HEAD` wherever `GET` goes, as RFC 9110 §9.3.2 asks of every origin
server, so naming `GET` gets you both — a `HEAD` is answered from the value's
size without transferring it. To enable writes, pass `methods={"GET", "PUT"}`:

```python
app = store_app(store, methods={"GET", "PUT"})
```

Accepted methods are `GET`, `HEAD`, and `PUT`; anything else raises
`ValueError` when the app is built, since the handler has no behavior for it.

A `PUT` request stores the request body at the given path and returns 204 (No
Content). Bodies are capped at `DEFAULT_MAX_BODY_SIZE` (256 MiB) and a larger
one returns 413 (Content Too Large) -- `Store.set` takes a whole buffer, so an
accepted body is held in memory in full. Raise or remove the cap with
`max_body_size`:

```python
from zarr_http_server import DEFAULT_MAX_BODY_SIZE, store_app

app = store_app(store, methods={"GET", "PUT"}, max_body_size=None)
```

Note that `store_app` exposes every key in the store, so `PUT` grants
unrestricted write access to all of it. `node_app` confines writes to keys
belonging to the node -- though a client that can write a node's metadata can
change what that node contains, and so what it will serve.

`store_app` also does not *validate* keys, because it proxies the store's raw
key space and has no array semantics to check against. A client that misspells
a chunk key — `c/00/00` where zarr writes `c/0/0` — gets a successful write to
a key no reader will ever consult, so the data is stored but invisible to
anyone opening the array. `node_app` rejects such a key with 404, since it
knows which node it is serving and therefore which keys are real. If you are
serving a zarr hierarchy to clients you do not control and writes are enabled,
prefer `node_app`.

### Shutting Down

`BackgroundServer.shutdown()` (and leaving the `with` block) waits up to
`shutdown_timeout` seconds -- 5 by default -- for in-flight requests to finish
before forcing the server closed:

```python
with serve_node(arr, background=True, shutdown_timeout=30) as server:
    ...
```

## Example

`examples/serve.py` creates an in-memory Zarr array, serves it over HTTP with
`serve_node`, and fetches the `zarr.json` metadata document and a raw chunk
using `httpx`.

Running it with uv is the simplest route — the script declares its own
dependencies inline, so uv installs them for you:

```bash
uv run examples/serve.py
```

To run it with a plain interpreter, install its `httpx` dependency first:

```bash
pip install httpx
python examples/serve.py
```
