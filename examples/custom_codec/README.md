# Custom Codec Example

This example demonstrates how to extend Zarr Python by defining a new codec.

The example shows how to:

- Define custom `Codec` classes
- Implement all required methods for serialization and deserialization
- Register the custom codecs with Zarr's registry
- Create and use arrays with the custom codecs in both Zarr v2 and v3 formats

## Running the Example

```bash
python examples/custom_dtype/custom_codec.py
```

Or run with uv:

```bash
uv run examples/custom_dtype/custom_codec.py
```
