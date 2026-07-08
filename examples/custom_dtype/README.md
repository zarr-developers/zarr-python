# Custom Data Type Example

This example demonstrates how to extend Zarr Python by defining a new data type.

The example shows how to:

- Define a custom `ZDType` class for the `int2` data type from [`ml_dtypes`](https://pypi.org/project/ml-dtypes/)
- Implement all required methods for serialization and deserialization
- Register the custom data type with Zarr's registry
- Create and use arrays with the custom data type in both Zarr v2 and v3 formats

## Running the Example

The script declares its dependencies inline
([PEP 723](https://peps.python.org/pep-0723/)), so the easiest way to run it is
with [uv](https://docs.astral.sh/uv/), which installs them automatically:

```bash
uv run examples/custom_dtype/custom_dtype.py
```

Alternatively, run it with plain Python, in which case you must first install
`zarr`, `ml_dtypes`, and `pytest` yourself:

```bash
python examples/custom_dtype/custom_dtype.py
```
