# Open With a Different Engine Example

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
- Select an engine **per call** via the `engine=` kwarg on `zarr.open_array`
  and `zarr.create_array`.
- Use the `"default"` engine, which works on any store and format.
- Use the `"zarrista"` (Rust-backed) engine, which serves Zarr v3 arrays on a
  `LocalStore` or an obstore-backed `ObjectStore` (the example guards this so it
  still runs if the package is absent).

## Available engines

| Engine       | Backend           | Stores                              | Notes                           |
| ------------ | ----------------- | ----------------------------------- | ------------------------------- |
| `"default"`  | built-in Python   | any                                 | used when `engine=` is omitted  |
| `"zarrista"` | Rust (`zarrista`) | `LocalStore`, obstore `ObjectStore` | requires the `zarrista` package |

## Running the Example

This example demonstrates an **unreleased** feature and the `"zarrista"` engine
depends on the optional `zarrista` package (pulled in through the `zarrista`
dependency group). It therefore does **not** carry a PEP 723 inline-dependency
header like some other examples: such a header would install a zarr without this
feature and fail at runtime.

Run it from a checkout of this branch:

```bash
uv run python examples/open_with_engine/open_with_engine.py
```

To exercise the Rust-backed engine, sync the `zarrista` dependency group:

```bash
uv run --group zarrista python examples/open_with_engine/open_with_engine.py
```

If `zarrista` is not installed, the example still runs and demonstrates the
default engine; the `zarrista` portion is skipped.
