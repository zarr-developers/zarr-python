---
title: The ndsel wire format
---

# The ndsel wire format

[ndsel](https://github.com/zarr-developers/ndsel) is a draft JSON
representation of NumPy-style n-dimensional selections, adapted from
TensorStore's `IndexTransform` model. This page documents the wire format, not
the coordinate model: [Coordinates are addresses](guide/index.md#coordinates-are-addresses)
introduces literal coordinates, and
[Lazy views compose](guide/index.md#lazy-views-compose) shows how views combine
before a transform is serialized. `zarr-indexing` implements ndsel in two layers:

| Layer | Module | Depends on | Job |
| --- | --- | --- | --- |
| Message | [`zarr_indexing.messages`](api/messages.md) | stdlib only | Validates and desugars JSON, removing redundant constant-map fields. |
| Engine | [`zarr_indexing.json`](api/json.md) | NumPy | Lowers a *canonical* body into an in-memory [`IndexTransform`](api/transform.md), and back. |

Constraints that only make sense for a real array — finite bounds, index
arrays as `ndarray`s — are checked during engine lowering. The message layer
also limits input rank to 32 and checks affine input-dimension references.
These checks are stricter than the draft's required message validation. As a result,
`messages` normalizes a message with `"-inf"` bounds that
`IndexTransform.from_json` refuses to lower.

## Two entry points

[`parse_ndsel`](api/messages.md#zarr_indexing.messages.parse_ndsel)
structurally validates a message of any kind and returns it unchanged. Use it
to confirm that a message is well formed while keeping it in its compact
shorthand form.

[`normalize_ndsel`](api/messages.md#zarr_indexing.messages.normalize_ndsel)
desugars a message into the single deterministic **canonical transform body**
of the spec (section 4.3): a bare `IndexTransform` body without the `kind`
discriminator.

```python
from zarr_indexing import normalize_ndsel

normalize_ndsel({"kind": "box", "inclusive_min": [10, 5], "shape": [40, 1]})
# {'input_rank': 2,
#  'input_inclusive_min': [10, 5],
#  'input_exclusive_max': [50, 6],
#  'input_labels': ['', ''],
#  'output': [{'offset': 0, 'stride': 1, 'input_dimension': 0},
#             {'offset': 0, 'stride': 1, 'input_dimension': 1}]}
```

Normalization is idempotent: re-tag the output with `kind: "transform"` and
normalizing it again returns the same body. The canonical body uses
TensorStore's `IndexTransform` field vocabulary, but normalization does not
guarantee acceptance by TensorStore. For example, the message layer leaves
index-array content unchecked, whereas TensorStore validates it; TensorStore
also requires unique nonempty labels and restricts finite index values to
`[-(2**62 - 2), 2**62 - 2]`. See
[TensorStore's index-space constraints](https://google.github.io/tensorstore/index_space.html).

Both entry points raise
[`NdselError`](api/messages.md#zarr_indexing.messages.NdselError), which
carries the spec `reason` code (`unknown_kind`, `rank_mismatch`, `step_zero`,
`output_map_conflict`, …) alongside a human-readable detail, so callers can
branch on the code rather than on message text.

## The five message kinds

Four are shorthands; the fifth is the canonical form itself.

| `kind` | Fields | Selects |
| --- | --- | --- |
| `point` | `coords` | A single element. Normalizes to rank 0 with one `constant` output map per dimension. |
| `box` | `inclusive_min`, one of `exclusive_max` / `inclusive_max` / `shape`, `labels` | A rectangular region. At most one upper-bound spelling may appear; omission gives implicit positive infinity. |
| `slice` | `start`, `stop`, `step`, `labels` | A strided region, one Python-style slice per dimension. |
| `points` | `coords` (a list of coordinate rows) | An explicit list of points — the `vindex` case. Normalizes to one `index_array` output map per dimension over a shared rank-1 input domain. |
| `transform` | `input_rank`, `input_inclusive_min`, one of the three `input_*` upper bounds, `input_labels`, `output` | The full canonical form. |

Value rules for validated fields (excluding verbatim `index_array` payloads
and discarded constant-map fields): every integer is a 64-bit
signed value; JSON booleans are **not** integers (Python's
`isinstance(True, int)` is guarded against explicitly); the `"-inf"` / `"+inf"`
sentinels are legal only in bound positions; and an implicit bound is the
one-element `[n]`-bracket form, whose implicit/explicit flag survives
normalization intact.

## Lowering to a transform

The engine layer converts between canonical bodies and `IndexTransform`s:

```python
from zarr_indexing import IndexTransform, normalize_ndsel

canonical = normalize_ndsel({"kind": "box", "shape": [2, 3]})
t = IndexTransform.from_json(canonical)
assert t.to_json() == canonical
```

`IndexDomain` carries the same pair for a bare domain body, and each output
map kind has a `to_json`; `output_index_map_from_json` dispatches the wire's
structurally discriminated union back to the right kind. Exact JSON equality
in this example is not a general round-trip guarantee: implicit flags are
removed and degenerate array maps are collapsed.

`index_array_bounds` constrains raw index-array values before the map's offset
and stride are applied. The message layer preserves these bounds, but the
engine cannot retain them through map operations. Both `IndexTransform.from_json`
and `output_index_map_from_json` therefore raise `NdselError("invalid_json", ...)`
when bounds differ from `["-inf", "+inf"]`. This includes one-sided constraints,
empty and singleton arrays, and zero-stride maps. Omitted or explicitly unbounded
bounds remain supported. Use the message layer to preserve constrained documents
for consumers that support them; lowering never silently discards an index-array
constraint.

A canonical body carrying a
`"-inf"` or `"+inf"` bound cannot be lowered — an `IndexDomain` addresses a
finite array — so `IndexTransform.from_json` raises. And implicit bounds lower
*by value*: the `[n]`-bracket flag is a message-layer concern, and the engine
keeps only the integer.

### The `index_array` round trip

ndsel and TensorStore both **reject** an output map that carries both
`input_dimension` and `index_array`. The in-memory
[`ArrayMap`](api/output_map.md#zarr_indexing.output_map.ArrayMap), though,
records its dependency axes in its full-rank array shape, with no
`input_dimension` field:

- **On serialize**, a non-degenerate `index_array` map is emitted *without*
  `input_dimension`.
- **On load**, dependency axes are the full-rank array's non-singleton axes.
  Maps sharing these axes describe correlated coordinates. The engine also
  accepts lower-rank nonempty arrays by prepending singleton axes; that
  convenience is not a guarantee of compatibility with other ndsel consumers.

An all-singleton `index_array` — size
1 — selects the same coordinate regardless of the input, so it is collapsed to
a `constant` map on serialize:

```python
from zarr_indexing import IndexTransform

IndexTransform.from_shape((100, 100)).oindex[[5], 0:2].to_json()
# {'input_rank': 2,
#  'input_inclusive_min': [0, 0],
#  'input_exclusive_max': [1, 2],
#  'input_labels': ['', ''],
#  'output': [{'offset': 5},
#             {'offset': 0, 'stride': 1, 'input_dimension': 1}]}
```

The size-1 input dimension stays in the domain, unconsumed by any output map.
The transform is still valid and the output shape is unchanged. A length-1
`oindex` selection therefore round-trips behaviorally (an `ArrayMap` comes back
as a `ConstantMap`) rather than by object identity.

Empty index arrays also serialize as constant maps with offset zero: the
empty input domain carries the fact that no coordinates are selected. This
avoids losing trailing shape information in JSON when an array has a leading
zero-length axis.

## Conformance

The package is checked against the language-agnostic ndsel conformance corpus,
vendored unmodified under
[`tests/conformance/`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing/tests/conformance)
— one JSON file per message kind plus `errors.json`, with the source commit
recorded in `PROVENANCE.md`. Each fixture is either a *success* case
(`input` + expected `normalized` body) or an *error* case (`input` + expected
reason code). Matching every fixture establishes corpus conformance; the
fixtures do not prove correctness for every possible input or universal
TensorStore compatibility. `tests/test_conformance.py` runs the whole corpus as one
parametrized test per fixture, so a corpus update reports failures fixture by
fixture rather than as a single opaque assertion.

Do not edit the vendored files; to pick up spec changes, re-vendor from a newer
ndsel commit and update the recorded SHA.

A second, optional test (`tests/test_ndsel_tensorstore.py`, skipped unless
`tensorstore` is installed) checks against TensorStore itself by loading
canonical bodies into `tensorstore.IndexTransform` and re-loading TensorStore's
own `to_json()` output back through the engine layer.
