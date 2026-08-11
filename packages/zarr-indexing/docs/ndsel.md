---
title: The ndsel wire format
---

# The ndsel wire format

[ndsel](https://github.com/zarr-developers/ndsel) is a draft JSON
representation of NumPy-style n-dimensional selections, adapted from
TensorStore's `IndexTransform` model. `zarr-indexing` implements it in two
layers, and the split between them is the thing worth understanding:

| Layer | Module | Depends on | Job |
| --- | --- | --- | --- |
| Message | [`zarr_indexing.messages`](api/messages.md) | stdlib only | JSON in, canonical JSON out. Validates and desugars. Never rounds, clamps, or drops information. |
| Engine | [`zarr_indexing.json`](api/json.md) | NumPy | Lowers a *canonical* body into an in-memory [`IndexTransform`](api/transform.md), and back. |

Constraints that only make sense for a real array — finite bounds, index
arrays as `ndarray`s — live in the engine layer and nowhere else. That is why
`messages` can happily normalize a message with `"-inf"` bounds that
`json.transform_from_canonical` will refuse to lower.

## Two entry points

[`parse_ndsel`](api/messages.md#zarr_indexing.messages.parse_ndsel)
structurally validates a message of any kind and returns it **unchanged** —
use it when you want to keep a message in its compact shorthand form but
confirm it is well formed.

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
normalizing it again returns the same body. Because the canonical body is
field-for-field a TensorStore `IndexTransform` minus `kind`, a normalized
message loads directly into `tensorstore.IndexTransform(json=...)`.

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
| `box` | `inclusive_min`, one of `exclusive_max` / `inclusive_max` / `shape`, `labels` | A rectangular region. Exactly one upper-bound spelling may appear. |
| `slice` | `start`, `stop`, `step`, `labels` | A strided region, one Python-style slice per dimension. |
| `points` | `coords` (a list of coordinate rows) | An explicit list of points — the `vindex` case. Normalizes to one `index_array` output map per dimension over a shared rank-1 input domain. |
| `transform` | `input_rank`, `input_inclusive_min`, one of the three `input_*` upper bounds, `input_labels`, `output` | The full canonical form. |

Value rules the message layer enforces throughout: every integer is a 64-bit
signed value; JSON booleans are **not** integers (Python's
`isinstance(True, int)` is guarded against explicitly); the `"-inf"` / `"+inf"`
sentinels are legal only in bound positions; and an implicit bound is the
one-element `[n]`-bracket form, whose implicit/explicit flag survives
normalization intact.

## Lowering to a transform

The engine layer converts between canonical bodies and `IndexTransform`s:

```python
from zarr_indexing import transform_from_canonical, transform_to_canonical

t = transform_from_canonical(canonical)
transform_to_canonical(t) == canonical
```

`index_transform_to_json` / `index_transform_from_json` (and the
`index_domain_*` variants) are these same converters under their historical
names.

Two engine constraints apply here and only here. A canonical body carrying a
`"-inf"` or `"+inf"` bound cannot be lowered — an `IndexDomain` addresses a
finite array — so `transform_from_canonical` raises. And implicit bounds lower
*by value*: the `[n]`-bracket flag is a message-layer concern, and the engine
keeps only the integer.

### The `index_array` round trip

ndsel and TensorStore both **reject** an output map that carries both
`input_dimension` and `index_array`. The in-memory
[`ArrayMap`](api/output_map.md#zarr_indexing.output_map.ArrayMap), though,
records an `input_dimension` to pin the axis an orthogonal (`oindex`) array
varies over. The serializer bridges that gap in both directions:

- **On serialize**, a non-degenerate `index_array` map is emitted *without*
  `input_dimension`.
- **On load**, the in-memory `input_dimension` is reconstructed from the
  full-rank array's dependency axes — its non-singleton axes. An array that
  solely owns a single non-singleton axis is orthogonal; arrays that share
  non-singleton axes, or vary over several, are correlated (`vindex`), and get
  `input_dimension = None`. A single 1-D array over a rank-1 domain is
  inherently ambiguous between the two flavours and reconstructs as
  orthogonal, which is behaviorally identical in that case.

There is one deliberate exception, worth calling out because it is the one
place a round trip changes representation rather than preserving it. An
all-singleton `index_array` — size 1 — selects the same coordinate no matter
what the input is, so it is **collapsed to a `constant` map** on serialize:

```python
from zarr_indexing import IndexTransform, transform_to_canonical

transform_to_canonical(IndexTransform.from_shape((100, 100)).oindex[[5], 0:2])
# {'input_rank': 2,
#  'input_inclusive_min': [0, 0],
#  'input_exclusive_max': [1, 2],
#  'input_labels': ['', ''],
#  'output': [{'offset': 5},
#             {'offset': 0, 'stride': 1, 'input_dimension': 1}]}
```

The size-1 input dimension stays in the domain, unconsumed by any output map —
still a valid transform, and still the right output shape. A length-1 `oindex`
selection therefore round-trips *behaviorally* (an `ArrayMap` comes back as a
`ConstantMap`) rather than by object identity.

## Conformance

The package is checked against the language-agnostic ndsel conformance corpus,
vendored unmodified under
[`tests/conformance/`](https://github.com/zarr-developers/zarr-python/tree/main/packages/zarr-indexing/tests/conformance)
— one JSON file per message kind plus `errors.json`, with the source commit
recorded in `PROVENANCE.md`. Each fixture is either a *success* case
(`input` + expected `normalized` body) or an *error* case (`input` + expected
reason code), and an implementation is conformant iff `normalize` reproduces
every one. `tests/test_conformance.py` runs the whole corpus as one
parametrized test per fixture, so a corpus update reports failures fixture by
fixture rather than as a single opaque assertion.

Do not edit the vendored files; to pick up spec changes, re-vendor from a newer
ndsel commit and update the recorded SHA.

A second, optional test (`tests/test_ndsel_tensorstore.py`, skipped unless
`tensorstore` is installed) closes the loop against a real TensorStore by
loading canonical bodies into `tensorstore.IndexTransform` and re-loading
TensorStore's own `to_json()` output back through the engine layer.
