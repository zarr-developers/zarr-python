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
[Uvicorn](https://www.uvicorn.org/).

Two levels of exposure are available:

- **Whole store** ([`store_app`][zarr_http_server.store_app],
  run with [`serve`][zarr_http_server.serve]) — serves every key in a
  store, exposing its entire key/value space.
- **Single node** ([`node_app`][zarr_http_server.node_app],
  run with [`serve_background`][zarr_http_server.serve_background]) — serves only the keys
  belonging to one `Array` or `Group`. Requests for keys outside that node
  return 404 even when those keys exist in the underlying store.

Byte-range reads, configurable CORS headers, and a configurable set of allowed
HTTP methods are handled by the app.

!!! danger "Serving a whole store grants access to all of it"

    `store_app` applies no per-key filtering. Only point it
    at a store whose full contents are safe to serve, and note that enabling
    `PUT` grants write access to everything the store contains.

## Getting started

The [README](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-http-server/README.md)
carries worked examples for building an ASGI app, running a blocking or
background server, and configuring CORS and allowed methods. Runnable
versions live in
[`examples/`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-http-server/examples).

## Reference

- [API reference](api/index.md)
- [Changelog](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-http-server/CHANGELOG.md)
- [License (MIT)](https://github.com/zarr-developers/zarr-python/blob/main/packages/zarr-http-server/LICENSE.txt)
