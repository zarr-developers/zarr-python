# zarr-http-server

HTTP server for Zarr stores, arrays, and groups.

`zarr-http-server` exposes a Zarr `Store`, `Array`, or `Group` over HTTP via an
ASGI app, so any HTTP-capable client (including zarr-python itself, via
`FsspecStore` or `ObjectStore`) can read the data. The app is built on
[Starlette](https://www.starlette.io/) and can be run with any ASGI server;
the `serve` / `serve_background` helpers run it with
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

Build an app with `store_app` or `node_app`, then run it. `serve` blocks
until the server is stopped, which is the shape for a script or a container
entrypoint:

```python
from zarr_http_server import serve, store_app

serve(store_app(store), host="127.0.0.1", port=8000)
```

`serve_background` instead starts the server in a daemon thread and returns a
`BackgroundServer` as soon as the socket is listening, so the caller can carry
on. These are two functions rather than one with a flag, because they differ
in the only thing that matters at a call site: whether control comes back. The
handle is also a context manager:

Both default to `port="auto"`, which prefers port 8000 but falls back to any
free port if it is taken, reporting the result through `server.url` and
uvicorn's startup line. An **explicit** port means the opposite — bind exactly
that or fail — because a caller who names one usually has a proxy or a
container port mapping expecting the server there, and silently moving would
break it while looking healthy. `port=0` keeps its usual meaning of "any free
port, no preference".

The example below also *reads back* over HTTP, which is a client-side
concern: `zarr.open_array(server.url)` goes through `FsspecStore`, which needs
an HTTP-capable fsspec that `zarr-http-server` does not pull in.

```bash
pip install "fsspec[http]"
```


```python
import numpy as np

import zarr
from zarr_http_server import node_app, serve_background
from zarr.storage import MemoryStore

store = MemoryStore()
arr = zarr.create_array(store, shape=(100,), chunks=(10,), dtype="float64")
arr[:] = np.arange(100, dtype="float64")

with serve_background(node_app(arr), host="127.0.0.1") as server:
    # Now open the served array from another zarr client.
    remote = zarr.open_array(server.url, mode="r")
    np.testing.assert_array_equal(remote[:], arr[:])
# Server is shut down automatically when the block exits.
```

### Serving Several Nodes

Serving two arrays does not mean running two servers. Which approach fits
depends on where the arrays live.

If they share a parent group, serve the parent — `node_app` recurses through
its members, so both are reachable under one port and node scoping still
applies to everything outside it:

```python
server = serve_background(node_app(root))
# -> /a/zarr.json, /a/c/0, /b/zarr.json, ...
```

If everything in the store is safe to expose, `store_app(store)` does the
same for the whole key space.

Otherwise — arrays in *different* stores, or nodes that are not siblings —
`store_app` and `node_app` return plain Starlette apps, so mount them and run
the result:

```python
from starlette.applications import Starlette
from starlette.routing import Mount

from zarr_http_server import node_app, serve_background

app = Starlette(routes=[
    Mount("/first", app=node_app(one)),
    Mount("/second", app=node_app(other)),
])
server = serve_background(app)
```

Each mount keeps its own validation, so a request under one cannot reach
another's data — `/first/../second/zarr.json` and its percent-encoded
spellings all return 404.

Both runners take any ASGI app, so the split is clean: what an app *serves*
(`methods`, `cors_options`, `max_body_size`) is settled when the app is built,
while `serve` / `serve_background` only decide how it runs (`host`, `port`,
`shutdown_timeout`, `uvicorn_options`).

### Serving from a Notebook

A notebook needs a server that outlives the cell that started it, so the
`with serve_background(...)` form above is the wrong shape — it shuts the server down
as soon as the block ends. Start it, keep the handle, and stop it later:

```python
# cell 1 — start
server = serve_background(node_app(array), host="127.0.0.1")
print(server.url)          # e.g. http://127.0.0.1:54635

# cell 2..n — use it, across as many cells as you like
httpx.get(f"{server.url}/zarr.json")

# last cell — stop
server.shutdown()
```

Two things make this comfortable in a kernel you re-run. `serve_background`
runs Uvicorn in a daemon thread with its own event loop, so it never touches
the kernel's loop and cannot block it. And its default `port="auto"` falls
back to a free port when 8000 is taken — re-running a start cell without
stopping the previous server is the classic notebook mistake, and a fixed port
fails there with *address already in use*. `server.url` reports the port
actually bound.

If you forget to stop one, the thread is a daemon, so restarting the kernel
always clears it — and since each start takes a fresh port, a forgotten server
does not block the next one.

[`examples/serve_notebook.ipynb`](examples/serve_notebook.ipynb) is a runnable
version of this, covering metadata and chunk reads, byte ranges, and that
writes are refused by default. It is executed by the test suite, so it cannot
drift from the code.

### Uvicorn Configuration

`serve` and `serve_background` name the options most callers need — `host`,
`port`, `shutdown_timeout` — and forward anything else to
`uvicorn.Config` through `uvicorn_options`, so nothing uvicorn can do is out
of reach:

```python
server = serve_background(
    store_app(store),
    host="0.0.0.0",
    port=8443,
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

Both `store_app` and `node_app` accept a `CorsOptions` parameter to enable
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

### Read-only Serving

Read-only is the default. `store_app(store)` and `node_app(node)` accept
`GET` and `HEAD` and answer **405** to `PUT`, `POST`, `DELETE` and `PATCH` —
no argument is needed to get there.

`POST` is not merely unrouted, it is unconfigurable: the accepted methods are
`GET`, `HEAD` and `PUT`, and asking for anything else raises `ValueError` when
the app is built. There is no handler behavior for `POST`, so no configuration
can produce one.

Two named sets let a call site state which it is, instead of leaving it to the
presence or absence of an argument:

```python
from zarr_http_server import READ_ONLY_HTTP_METHODS, READ_WRITE_HTTP_METHODS, store_app

app = store_app(store, methods=READ_ONLY_HTTP_METHODS)   # GET, HEAD
app = store_app(store, methods=READ_WRITE_HTTP_METHODS)  # GET, HEAD, PUT
```

The distinction also exists in the type domain. `ReadOnlyHTTPMethod` is a
`Literal["GET", "HEAD"]`, so a read-only set can be *declared* rather than
merely configured — a `frozenset[ReadOnlyHTTPMethod]` containing `"PUT"` is a
type error, not a runtime surprise:

```python
from zarr_http_server import ReadOnlyHTTPMethod

reads: frozenset[ReadOnlyHTTPMethod] = frozenset({"GET", "HEAD"})  # ok
reads = frozenset({"GET", "PUT"})                                  # type error
```

Both constants are derived from these Literals, so the runtime sets and the
static types cannot disagree about what the server serves.

`READ_ONLY_HTTP_METHODS` is exactly the default, so passing it changes nothing
except that the intent is now written down. The practical value is the other
direction: a writable app *must* name a method set, so
`grep -r 'methods=' ` finds every place that opts into writes.

For a guarantee that does not depend on getting `methods` right, make the
*store* read-only. The store refuses writes itself, so no routing mistake —
now or in a later edit — can produce one:

```python
app = store_app(store.with_read_only(True))

# For a node, open it read-only and node_app inherits that store:
app = node_app(zarr.open_array(store, mode="r"))
```

These two layers are independent, and the store is the stronger one: it holds
even if the HTTP layer is misconfigured. Asking for both at once —
`methods={"GET", "PUT"}` on a read-only store — is a contradiction that can
never succeed, so it raises `ValueError` at construction rather than turning
into a 403 for whichever client tries to write first.

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
with serve_background(node_app(arr), shutdown_timeout=30) as server:
    ...
```

## Example

`examples/serve.py` creates an in-memory Zarr array, serves it over HTTP with
`serve_background`, and fetches the `zarr.json` metadata document and a raw chunk
using `httpx`.

`examples/serve_notebook.ipynb` is the notebook equivalent, showing how to
start a server in one cell and stop it in another. Both are executed by the
test suite.

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
