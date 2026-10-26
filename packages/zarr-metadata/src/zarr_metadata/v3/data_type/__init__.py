"""Zarr v3 data type spec types.

Each v3 data type has its own submodule:

- Core primitives: `bool`, `int8`/`16`/`32`/`64`, `uint8`/`16`/`32`/`64`,
  `float16`/`32`/`64`, `complex64`/`128`, `raw` (for `r<N>`)
- zarr-extensions: `bytes`, `string`, `numpy_datetime64`, `numpy_timedelta64`,
  `struct`

Import a data type's types and its definition from its submodule:

- `<X>DataTypeName` -- the literal type of the dtype's `data_type` string
  (or, for named-config dtypes, the literal value of their `name` field)
- `<X>FillValue` -- the permitted JSON shape of the `fill_value` field
- the envelope and configuration TypedDicts of named-config dtypes
  (`numpy_datetime64`, `numpy_timedelta64`, `struct`), the branded
  `HexFloat<N>` / `Base64Bytes` types, and their validator functions
- `<X>_DATA_TYPE`, the definition

See https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html
"""

__all__: list[str] = []
