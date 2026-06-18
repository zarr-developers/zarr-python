# Custom Data Type Example

This example demonstrates how to extend Zarr Python by defining a new data type.

The example shows how to:

- Define a custom `ZDType` class for the `int2` data type from [`ml_dtypes`](https://pypi.org/project/ml-dtypes/)
- Implement all required methods for serialization and deserialization
- Register the custom data type with Zarr's registry
- Create and use arrays with the custom data type in both Zarr v2 and v3 formats

## Running the Example

```bash
python examples/custom_dtype/custom_dtype.py
```

Or run with uv:

```bash
uv run examples/custom_dtype/custom_dtype.py
```
