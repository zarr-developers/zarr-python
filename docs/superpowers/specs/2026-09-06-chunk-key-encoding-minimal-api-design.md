# Minimal chunk-key-encoding API design

## Goal

Ship the first `zarr-chunk-key-encoding` release with the smallest public API
that implements the two core Zarr v3 chunk-key encodings, strict decoding, JSON
round-tripping, and optional per-call grid-bound checks.

The package will not model a bounded encoding as a persistent object. Bounds
are call context owned by a grid or array, not part of an encoding's identity.

## Public API

The public package surface will contain:

- `ChunkKeyEncoding`
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

- `BoundedChunkKeyEncoding` and `BoundedChunkKeyEncodingJSON`
- `ChunkKey`
- `ChunkCoordsOutOfBoundsError` and `ChunkKeyOutOfBoundsError`
- `ChunkKeyEncodingLike` and `ChunkKeyEncodingParams`
- `CHUNK_KEY_ENCODINGS`, `SEPARATORS`, `get_chunk_key_encoding_class`,
  `parse_chunk_key_encoding`, and `parse_separator`

Closed-set dispatch and separator validation remain implementation details.
The flat parameter form accepted by `parse_chunk_key_encoding` is omitted until
an integration requires compatibility with that input form.

## Bounded operations

`ChunkKeyEncoding` will provide two concrete convenience methods:

```python
def encode_bounded(
    self,
    chunk_coords: Sequence[int],
    *,
    grid_shape: Sequence[int],
) -> str: ...

def decode_bounded(
    self,
    chunk_key: str,
    *,
    grid_shape: Sequence[int],
) -> tuple[int, ...]: ...
```

`grid_shape` is keyword-only and explicitly names the number of stored chunks
or shards along each dimension. It is not an array element shape. Callers that
work with sharded arrays remain responsible for passing the shard-grid shape.

Both methods validate `grid_shape` as a sequence of non-negative integers.
An empty grid shape denotes the single storage cell of a zero-dimensional
array. A grid shape containing a zero has no valid coordinates.

`encode_bounded` normalizes `chunk_coords`, verifies equal rank and
`0 <= coordinate < extent` in every dimension, then delegates to `encode`.

`decode_bounded` recognizes `encode(())` directly for an empty grid shape,
which resolves the core `v2` encoding's rank-zero ambiguity. Otherwise it
delegates to `decode`, verifies equal rank and bounds, and returns the decoded
coordinates. `NotImplementedError` continues to propagate for an encoding
without `decode`, except for the directly recognizable rank-zero key.

The methods make no general claim that restricting an arbitrary encoding's
domain makes it injective or gives it a total inverse.

## Errors

- Invalid grid-shape entries raise `ChunkKeyConfigurationError`.
- Non-integer or negative coordinates passed to `encode_bounded` raise
  `InvalidChunkCoordsError`.
- Valid coordinates with the wrong rank or outside the grid also raise
  `InvalidChunkCoordsError`, with a bounds-specific message.
- Malformed keys, decoded coordinates with the wrong rank, and decoded
  coordinates outside the grid raise `ChunkKeyDecodeError`.

No bounded-specific exception subclasses are retained. Callers can distinguish
configuration, encode-input, and decode-input failures without committing the
package to a larger taxonomy.

## Internal organization

- Delete `_bounded.py`.
- Fold the small separator type and validator from `_separator.py` into
  `_parsing.py`, then delete `_separator.py`.
- Keep the concrete encodings in separate private modules because each mirrors
  a separately specified Zarr extension.
- Keep closed JSON dispatch in `_from_json.py`, but make its mapping and class
  lookup private and remove the loose parsing layer.
- Remove `__all__` declarations from private modules. The top-level package
  remains the sole declaration of public API.

## Documentation and release material

The README will provide installation, one encode/decode example, one bounded
example, and a short statement about the closed core set. Detailed API material
will remain in the documentation site without repeating the same design essay.

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
