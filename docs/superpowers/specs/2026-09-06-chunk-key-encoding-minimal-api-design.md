# Minimal chunk-key-encoding API design

## Goal

Ship the first `zarr-chunk-key-encoding` release with the smallest public API
that implements the two core Zarr v3 chunk-key encodings, strict decoding, JSON
round-tripping, and reusable grid-bound checks.

Grid bounds are call context owned by a grid or array, not part of an
encoding's serialized identity. A deliberately thin prepared object will hold
that context so `grid_shape` is normalized and validated once rather than on
every chunk operation.

## Public API

The public package surface will contain:

- `ChunkKeyEncoding`
- `BoundedChunkKeyEncoding`
- `DefaultChunkKeyEncoding`
- `V2ChunkKeyEncoding`
- `ChunkKeyEncodingJSON`
- `Separator`
- `ChunkKeyEncodingError`
- `ChunkKeyConfigurationError`
- `UnknownChunkKeyEncodingError`
- `ChunkKeyDecodeError`
- `InvalidChunkCoordsError`
- `chunk_key_encoding_from_json`
- `__version__`

The following speculative or redundant names will be removed before the first
release:

- `BoundedChunkKeyEncodingJSON`
- `ChunkKey`
- `ChunkCoordsOutOfBoundsError` and `ChunkKeyOutOfBoundsError`
- `ChunkKeyEncodingLike` and `ChunkKeyEncodingParams`
- `CHUNK_KEY_ENCODINGS`, `SEPARATORS`, `get_chunk_key_encoding_class`,
  `parse_chunk_key_encoding`, and `parse_separator`

Closed-set dispatch and separator validation remain implementation details.
The flat parameter form accepted by `parse_chunk_key_encoding` is omitted until
an integration requires compatibility with that input form.

## Prepared bounded operations

`ChunkKeyEncoding` will provide a convenience method that prepares a bounded
view:

```python
bounded = encoding.to_bounded(grid_shape)
chunk_key = bounded.encode(chunk_coords)
decoded = bounded.decode(chunk_key)
```

`BoundedChunkKeyEncoding` is a frozen, generic dataclass containing only the
underlying encoding and a normalized `tuple[int, ...]` grid shape. Construction
validates the grid shape once. It is analogous to a compiled regular expression:
the small prepared object exists to amortize setup across repeated operations,
not to introduce a second encoding configuration format.

`grid_shape` explicitly names the number of stored chunks or shards along each
dimension. It is not an array element shape. Callers that work with sharded
arrays remain responsible for passing the shard-grid shape. It must be a
sequence of non-negative integers. An empty grid shape denotes the single
storage cell of a zero-dimensional array. A grid shape containing a zero has no
valid coordinates.

`BoundedChunkKeyEncoding.encode` normalizes `chunk_coords`, verifies equal rank
and `0 <= coordinate < extent` in every dimension, then delegates to the
underlying `encode` implementation. Per-call coordinate validation remains
necessary; grid-shape normalization does not recur.

`BoundedChunkKeyEncoding.decode` recognizes the underlying `encode(())`
directly for an empty grid shape, which resolves the core `v2` encoding's
rank-zero ambiguity. Otherwise it delegates to the underlying `decode`, verifies
equal rank and bounds, and returns the decoded coordinates.
`NotImplementedError` continues to propagate for an encoding without `decode`,
except for the directly recognizable rank-zero key.

The bounded view exposes only `encode` and `decode`. It is not a `Collection`,
does not implement membership, iteration, or length, and has no JSON form or
alternate `from_unbounded` constructor. It makes no general claim that
restricting an arbitrary encoding's domain makes it injective or gives it a
total inverse.

## Errors

- Invalid grid-shape entries passed during bounded-view construction raise
  `ChunkKeyConfigurationError`.
- Non-integer or negative coordinates passed to bounded `encode` raise
  `InvalidChunkCoordsError`.
- Valid coordinates with the wrong rank or outside the grid also raise
  `InvalidChunkCoordsError`, with a bounds-specific message.
- Malformed keys, decoded coordinates with the wrong rank, and decoded
  coordinates outside the grid raise `ChunkKeyDecodeError`.

No bounded-specific exception subclasses are retained. Callers can distinguish
configuration, encode-input, and decode-input failures without committing the
package to a larger taxonomy.

## Internal organization

- Retain `_bounded.py`, but reduce it to the thin prepared dataclass and its
  rank/bounds checks.
- Fold the small separator type and validator from `_separator.py` into
  `_parsing.py`, then delete `_separator.py`.
- Keep the concrete encodings in separate private modules because each mirrors
  a separately specified Zarr extension.
- Keep closed JSON dispatch in `_from_json.py`, but make its mapping and class
  lookup private and remove the loose parsing layer.
- Remove `__all__` declarations from private modules. The top-level package
  remains the sole declaration of public API.

## Documentation and release material

The README will provide installation, one encode/decode example, one prepared
bounded-view example, and a short statement about the closed core set. Detailed
API material will remain in the documentation site without repeating the same
design essay.

The unreleased changelog fragments for bounded objects and `ChunkKey` will be
removed. The initial feature fragment will describe only the final shipped API.
Package CI, release automation, generated lockfile, strict docs build, and
`zarr-metadata` integration remain: they are release infrastructure rather than
speculative API.

## Testing

Tests will establish:

- encode/decode behavior for both core encodings and both separators;
- strict rejection of malformed and non-canonical keys;
- JSON parsing and round-tripping for the two core encodings;
- one-time grid-shape validation at bounded-view construction;
- successful bounded encode/decode for ordinary, zero-dimensional, and
  zero-extent grids;
- one focused test for each bounded error category;
- the reduced top-level public API;
- byte-for-byte parity with zarr-python's existing core encoders.

Every behavior change will follow a red-green cycle. Completion requires the
package tests, parity tests, ruff, pyright, strict documentation build, and
`git diff --check` to pass.

## Compatibility

The package has not been released, so removed PR-only APIs require no
deprecation period. This cleanup does not change `zarr` behavior and does not
integrate the new package into zarr-python's existing classes.
