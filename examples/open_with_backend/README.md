# Open With a Different Backend (Engine) Example

This example demonstrates how to open the **same array data with a different
backend** -- what zarr-python calls an *engine*.

An engine is purely an *execution* setting: it selects which compute backend
reads and writes an array's chunks. The bytes on disk are identical regardless
of the engine, so the same Zarr array can be driven by a different backend
without rewriting any data.

The example shows how to:

- Discover the available engines with `zarr.list_engines()`.
- Inspect the engine an array is using via its public `array.engine` property.
- Write one array once and read it back through several engines, asserting the
  results are byte-for-byte identical (`numpy.testing.assert_array_equal`).
- Select an engine **globally** via `zarr.config.set({"array.engine": ...})`.
- Select an engine **per call** via the `engine=` kwarg on `zarr.open_array`
  and `zarr.create_array`.
- Use the `"reference"` (pure-Python) engine, which works on any store.
- Use the `"zarrista"` (Rust-backed) engine, which ingests a `LocalStore` or an
  obstore-backed `ObjectStore`
  (the example guards this so it still runs if the package is absent).
- See the strict policy in action: advanced indexing (`.oindex`/`.vindex`) under
  a non-native engine raises `NotImplementedError` rather than silently falling
  back to native.

## Available engines

| Engine        | Backend            | Stores                | Notes                          |
| ------------- | ------------------ | --------------------- | ------------------------------ |
| `"zarr"`      | native Python      | any                   | the default                    |
| `"reference"` | pure Python (crud) | any                   | always available               |
| `"zarrista"`  | Rust (`zarrista`)  | `LocalStore`, obstore `ObjectStore` | requires the `zarrista` package |
| `"zarrs"`     | Rust (in-repo)     | any                   | requires building the extension |

## Running the Example

This example demonstrates an **unreleased** feature that currently lives only on
the `zarrs-bindings` development branch of zarr-python (it is not in any
published release or in `main`). It therefore does **not** carry a PEP 723
inline-dependency header like the other examples: such a header would install a
zarr without this feature and fail at runtime.

Run it from a checkout of the `zarrs-bindings` branch, with the `zarrista`
dependency group synced:

```bash
uv sync --group zarrista
uv run examples/open_with_backend/open_with_backend.py
```

If `zarrista` is not installed, the example still runs and demonstrates the
native and `reference` engines; the `zarrista` portion is skipped.
