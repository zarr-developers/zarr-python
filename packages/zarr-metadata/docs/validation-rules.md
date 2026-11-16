# What the readers check

The readers and model constructors enforce the declared document
structure. They also check a small set of consistency rules that need no
context. For example:

- fixed format literals
  ([`zarr_format`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#zarr-format)
  and
  [`node_type`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#node-type)),
- finite JSON numbers, except in `attributes` and in members that nothing
  in the context reads,
- non-negative
  [dimensions](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#array-metadata-shape),
- one
  [`dimension_names`](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#dimension-names)
  entry per array dimension.

## Extension points

In a v3 document, the readers also check each
[extension point](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#extension-points):

- the data type,
- the chunk grid,
- the chunk key encoding,
- each codec,
- each storage transformer.

The definition that claims a name in the context checks that extension
point. The context is `CORE_AND_EXTENSIONS_CONTEXT` unless you pass a
`context`.

- If the definition rejects a configuration, the reader rejects it too.
- A configuration key the definition does not declare is reported as
  `unknown_key`.
- If no definition claims a name, the reader does not check that extension
  point. The consumer decides whether to support it.

The [Context](context.md) page shows this with an example.

## Checks across the document

Some checks relate one part of a v3 document to another:

- The
  [fill value](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#fill-value)
  is checked against the data type the document names. The data type's
  definition does this check.
- The
  [chunk grid](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#chunk-grids)
  is checked against the shape. The grid's definition does this check.
- The codecs are checked as a
  [pipeline](https://github.com/zarr-developers/zarr-specs/blob/v3.1.0/docs/v3/core/index.rst#L931-L1024).
  In order, each codec's definition checks the codec against the chunk it
  receives. This includes a
  [shard's](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/codecs/sharding-indexed/index.html#configuration-parameters)
  inner codecs and index codecs.

The readers do no arithmetic on values. For example, they do not check
whether a fill value survives a `cast_value` round trip.

## Decisions the spec leaves open

The spec leaves some choices open, or answers them in two ways. Here is
what the package does.

**`attributes` may hold `NaN`, `Infinity` and `-Infinity`.**

- The spec
  [does not interpret attributes](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#attributes).
- zarr-python and xarray write these numbers there, for example as a CF
  `_FillValue`.
- The models read them. `to_key_value` writes them back as bare tokens,
  which a strict JSON parser rejects.
- The same goes for a member the spec does not define, when no definition
  in the context claims it. Nothing reads it, so it is kept as written.
- `check`, from `zarr_metadata.typed_json`, rejects a non-finite number
  anywhere, including in attributes.

**A reader walks at most 256 levels of nesting.**

- A value nested deeper is an `invalid_value` problem at the first level
  past the limit, wherever it is.
- Every reader, writer and comparison uses one stack frame per level.
  `copy.deepcopy` uses two, and so does `pickle` before Python 3.12.
- So a document at the limit uses about half of the interpreter's default
  recursion limit. The rest is left for the caller.

**`consolidated_metadata: null` is a problem.**

- A zarr-python 3.0.x bug wrote it.
- The spec
  [requires an object](https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L802-L816),
  and the package accepts only an object.
- The [repair readers](repair.md) remove the key before reading.

**An extension name must follow the spec's
[naming rule](https://github.com/zarr-developers/zarr-specs/blob/v3.1.0/docs/v3/core/index.rst#L1571-L1607)**,
`^[a-z][a-z0-9-_.]+$`, or be a URI.

- Earlier versions of the spec required URIs.
- Any other name is rejected before any definition is consulted.
- So `""` and `"foo/bar"` are problems, not unknown extensions.

**`must_understand: false` is rejected at every extension point**,
including codecs and storage transformers.

- The
  [core spec](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#must-understand)
  names only the data type, chunk grid and chunk key encoding.
- But a reader that skips a codec reads wrong bytes. That is just as bad as
  a reader that skips a data type and reads wrong values.
- On an unknown top-level member, `must_understand` keeps its meaning: a
  reader can skip that member.

**A v3 `regular` chunk grid's chunk lengths must be at least 1**, even
along a dimension of length 0.

- The
  [regular grid spec](https://zarr-specs.readthedocs.io/en/latest/v3/chunk-grids/regular-grid/index.html#description)
  says "Chunk sizes must be greater than zero".
  [zarr-specs#379](https://github.com/zarr-developers/zarr-specs/pull/379)
  added that sentence after v3.1.0.
- The
  [core spec's wording](https://zarr-specs.readthedocs.io/en/v3.1.0/v3/core/index.html#concepts-and-terminology)
  requires less: "non-zero when the corresponding dimensions of the arrays
  have non-zero length". But it does not explicitly allow anything more.
- So the package rejects a chunk length of 0. zarr-python 3.0 and 3.1 wrote
  one for an empty dimension.

**A v2 datetime or timedelta dtype must have a unit**, as in `<M8[ns]`.

- The
  [v2 spec](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding)
  requires the unit in brackets.
- zarr-python 3 writes `<M8` or `<m8` for a NumPy `datetime64` or
  `timedelta64` with no unit. The package refuses those dtypes.
- A codec's dtype parameter may leave out the unit. An example is the
  `dtype` of a `delta` filter. Numcodecs writes it that way and runs it.

**A v2 codec's dtype parameter is checked against the core v2 data
types**, whatever context the array is read in.

- Numcodecs reads the parameter with NumPy.
- So a context that widens a data type does not widen what a codec's dtype
  parameter may be. For example, a context that reads `<f16` as a dtype
  still refuses it as a codec's dtype parameter.
