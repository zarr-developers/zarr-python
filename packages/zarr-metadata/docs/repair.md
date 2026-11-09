# Repairing known writer bugs

The readers are strict. If a known writer bug made a document invalid, the
readers report problems.

The repair readers in `zarr_metadata.model` handle these documents:

- `read_repaired_node_metadata_v3`
- `read_repaired_node_metadata_v2`
- `read_repaired_consolidated_metadata_v2`

Each one works in three steps:

1. Undo the known bugs.
2. Read the result strictly.
3. Return each change it made as a `Repair`.

## Bugs the repair readers undo

- **A chunk length of 0 along an empty dimension**, in a v3
  [`regular` chunk grid](https://zarr-specs.readthedocs.io/en/latest/v3/chunk-grids/regular-grid/index.html#description).
  zarr-python 3.0 and 3.1 wrote it. The repair writes 1.
- **`consolidated_metadata: null` on a v3 group.** zarr-python 3.0 and 3.1
  wrote it. The spec
  [requires an object](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816).
  The repair removes the key.
- **A `consolidated_metadata` member in a `.zgroup` entry of a
  `.zmetadata`**, below the root. zarr-python 3 writes it. The
  [v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#groups)
  allows no key but `zarr_format` in a `.zgroup`. The repair removes the
  member.
- **A bare `NaN`, `Infinity` or `-Infinity` as a v2 fill value.**
  zarr-python 3.0.0 to 3.0.6 wrote it. The repair writes the string the
  [v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#fill-value-encoding)
  uses, such as `"NaN"`.
- **A bare non-finite float or complex fill value in a v3 group's
  consolidated metadata.** zarr-python 3.0.0 to 3.0.6 wrote it. The repair
  writes the string the
  [v3 spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/data-types/index.html#permitted-fill-values)
  uses, such as `"NaN"`.

## Bugs the repair readers do not undo

A document with one of these bugs is refused:

- **The fill value of an `S` or `V` array as a JSON array of byte values**,
  in a `.zmetadata` entry. zarr-python 3.0.7 to 3.0.10 wrote it. The
  array's own `.zarray` file holds the
  [base64 string](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#fill-value-encoding)
  the v2 spec requires, and the package reads that file.
- **The fill value of a v2 datetime array as an ISO 8601 string**, not as a
  count of ticks. zarr-python 3.0.7 and 3.0.8 wrote it.
- **A chunk length of 0 along a dimension that is not empty**, in a v3
  `regular` chunk grid. zarr-python 3.0 and 3.1 wrote it for an explicit
  chunk length of 0.
- **A chunk length of `true` or `false`** in a v3 `regular` chunk grid.
  Some releases of zarr-python from 3.0 to 3.2 wrote `true` for
  `chunks=True`. Some releases of 3.0 and 3.1 wrote `false` for
  `chunks=False`.
