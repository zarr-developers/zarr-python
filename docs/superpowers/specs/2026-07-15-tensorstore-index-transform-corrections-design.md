# TensorStore-Aligned Index Transform Corrections

## Goal

Correct lazy and transform-based indexing so that basic, orthogonal, and vectorized selections preserve NumPy/Zarr semantics through composition, chunk resolution, reads, writes, and serialization.

## Scope

This change fixes five reviewed regressions:

1. Legacy advanced-indexing fallbacks bypass a lazy view's transform.
2. Multiple independent orthogonal index arrays are mistaken for correlated vectorized arrays.
3. Lazy selections do not consistently normalize negative indices or validate bounds.
4. Zero-dimensional identity transforms are incorrectly flattened.
5. Multidimensional vectorized reads cannot safely write directly into a caller-provided `out` buffer.

The public indexing API remains unchanged. Structured-dtype field selection remains on the legacy codec path, but it must either operate on the correct composed storage selection or reject unsupported transformed access explicitly; it must never silently access the wrong storage region.

## Transform Representation

Follow TensorStore's normalized index-transform model. Every `ArrayMap.index_array` is interpreted over the complete input domain of its containing `IndexTransform`.

- The index array rank equals the transform input rank.
- An axis on which an index map does not vary is represented as a singleton/broadcast axis.
- Independent orthogonal maps vary along different input dimensions.
- Correlated vectorized maps vary along the same broadcast input dimensions.
- Zero-rank transforms use zero-rank index arrays where applicable.

For an orthogonal selection with row indices of length 2 and column indices of length 3, the normalized maps have shapes `(2, 1)` and `(1, 3)`. For a pairwise vectorized selection of length 2, both maps have shape `(2,)`.

Correlation must be derived from the maps' input-dimension dependencies. Code must not infer vectorized semantics merely from the presence of two or more `ArrayMap` instances.

## Construction and Composition

Basic, orthogonal, and vectorized selection constructors normalize their inputs before producing transforms:

- Python and NumPy integer scalars use normal Zarr negative-index wraparound.
- Integer arrays normalize negative elements and reject values outside the selected view's bounds.
- Boolean masks must match the dimensions they consume.
- Slice steps remain positive-only, consistent with current Zarr behavior.
- Orthogonal integer arrays remain one-dimensional at the public API boundary, then become full-rank broadcast arrays in the normalized transform.
- Vectorized integer arrays are broadcast together before being expanded to full input rank.

Composition preserves full-rank dependency information. When an outer transform indexes an inner `ArrayMap`, the resulting array is evaluated and broadcast over the outer input domain without squeezing dependency axes. Composition may simplify a map to `ConstantMap` or `DimensionMap` only when the resulting mapping is provably equivalent.

## Intersection and Chunk Resolution

Intersection classifies maps by their dependency axes:

- Correlated maps are filtered with one joint mask so a coordinate survives only when all mapped storage coordinates are inside the chunk.
- Independent maps are filtered per dependency axis and retain outer-product semantics.
- Mixed `ArrayMap`, `DimensionMap`, and `ConstantMap` transforms retain their input-domain axis ordering.

Chunk resolution produces output selectors consistent with the normalized input domain. Correlated maps share scatter coordinates; independent maps produce orthogonal selectors. The resolver must not rely on array-map count as a proxy for correlation.

## Array Read and Write Paths

All non-field lazy reads and writes use the composed `IndexTransform` path. Eager advanced operations may continue using legacy indexers when operating on an identity transform, but a transformed view must not fall back to storage-relative indexing that ignores its transform.

Structured-field operations continue using the existing field-aware codec path. If a transformed structured-field selection cannot be expressed correctly through that path, raise a clear `NotImplementedError` instead of reading or writing the wrong region.

Buffer handling follows transform semantics:

- Flatten only when the transform contains correlated array maps whose chunk resolution emits flat scatter indices.
- Never treat an empty output-map tuple as vectorized; zero-dimensional identity reads and writes retain shape `()`.
- When a multidimensional vectorized read supplies `out`, allocate a flat temporary buffer, perform scatter reads into it, reshape it to the selection shape, and copy it into `out`.
- Writes validate broadcasting against the visible selection domain before any vectorized flattening.

## Serialization

JSON round-tripping preserves full-rank index-array shapes, including singleton dependency axes. No transform-wide `orthogonal` or `vectorized` mode flag is added. Existing JSON that already satisfies the rank invariant continues to load unchanged.

If compatibility with branch-local JSON fixtures containing lower-rank arrays is required, the loader may normalize them only when their dependency alignment is unambiguous. Ambiguous lower-rank arrays must be rejected rather than guessed.

## Error Handling

- Scalar and array indices outside explicit view bounds raise `IndexError` before chunk access.
- Invalid boolean mask shapes raise `IndexError` during transform construction.
- Incompatible assignment shapes raise `ValueError` before storage mutation.
- Unsupported transformed structured-field access raises `NotImplementedError` with an explanation.
- All validation is relative to the current lazy view domain, not the root array shape.

## Test Strategy

Add focused regression tests before implementation and verify that each fails for the reviewed reason:

- A sliced lazy view followed by advanced orthogonal read and write accesses the composed storage region.
- Multiple orthogonal index arrays with unequal lengths produce an outer-product result across chunks.
- Multiple vectorized arrays retain pairwise/broadcast semantics.
- Negative scalar and array indices work relative to a lazy view; overly negative and positive out-of-range values raise.
- Invalid Boolean mask shapes raise.
- Zero-dimensional reads return a scalar and zero-dimensional writes persist it.
- Multidimensional coordinate selection with a caller-provided `out` buffer matches NumPy.
- Transform and JSON unit tests assert full-rank singleton-axis normalization and round-trip preservation.

After focused tests pass, run the complete transform, lazy-indexing, indexing, and array test modules, followed by the repository's configured static checks for the modified files.

## Non-Goals

- Negative slice steps.
- New public indexing modes or APIs.
- General TensorStore dimension labels, implicit bounds, or arbitrary-origin domains.
- Unrelated restructuring of the legacy indexer or codec pipeline.
