# Repairing known writer bugs

The readers are strict, so a document that a known writer bug made invalid
has problems. The repair readers in `zarr_metadata.model`
(`read_repaired_node_metadata_v3`, `read_repaired_node_metadata_v2` and
`read_repaired_consolidated_metadata_v2`) undo these bugs first, then read
the result strictly, and return each change they made as a `Repair`.

## Bugs the repair readers undo

- A chunk length of 0 in a v3 [`regular` chunk grid](https://zarr-specs.readthedocs.io/en/latest/v3/chunk-grids/regular-grid/index.html#description), along an empty
  dimension, which zarr-python 3.0 and 3.1 wrote, is written as 1.
- `consolidated_metadata: null` on a v3 group, which zarr-python 3.0 and
  3.1 wrote, is removed. The spec [requires an object](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816).
- The `consolidated_metadata` member that zarr-python 3 writes into each
  `.zgroup` entry of a `.zmetadata` below the root is removed. The
  [v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#groups) allows no key but `zarr_format` in a `.zgroup`.
- A non-finite v2 fill value that zarr-python 3.0.0 to 3.0.6 wrote as a
  bare `NaN`, `Infinity` or `-Infinity` is written as the string the
  [v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#fill-value-encoding) uses, such as `"NaN"`.
- A non-finite float or complex fill value in a v3 group's consolidated
  metadata, which zarr-python 3.0.0 to 3.0.6 wrote as a bare token, is
  written as the string the [v3 spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#permitted-fill-values) uses, such as `"NaN"`.

## Bugs the repair readers do not undo

A document with one of these is refused:

- zarr-python 3.0.7 to 3.0.10 wrote the fill value of an `S` or `V`
  array in a `.zmetadata` entry as a JSON array of byte values. The
  array's own `.zarray` file holds the [base64 string](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#fill-value-encoding) the v2 spec
  requires, which the package reads.
- zarr-python 3.0.7 and 3.0.8 wrote the fill value of a v2 datetime
  array as an ISO 8601 string, not as a count of ticks.
- zarr-python 3.0 and 3.1 wrote a chunk length of 0 in a v3 `regular`
  chunk grid along a dimension that is not empty, for an explicit chunk
  length of 0.
- Some releases of zarr-python from 3.0 to 3.2 wrote a chunk length of
  `true` in a v3 `regular` chunk grid for `chunks=True`, and some
  releases of 3.0 and 3.1 wrote `false` for `chunks=False`.
