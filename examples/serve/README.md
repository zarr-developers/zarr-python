# Serve a Zarr Array over HTTP

This example creates an in-memory Zarr array, serves it over HTTP with
`zarr.experimental.serve.serve_node`, and fetches the `zarr.json` metadata
document and a raw chunk using `httpx`.

## Running the Example

```bash
python examples/serve/serve.py
```

Or run with uv:

```bash
uv run examples/serve/serve.py
```
