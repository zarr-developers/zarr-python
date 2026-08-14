"""The canonical ndsel wire vocabulary, and the rules for lowering it.

This is the **engine layer**. Where `messages.py` is pure JSON→JSON and imposes
no array constraints, this module holds the JSON shapes a canonical ndsel body
takes (spec section 4.3, as produced by `zarr_indexing.messages.normalize_ndsel`)
together with the lowering rules that turn one into the numpy-backed engine
representation.

The conversions themselves are **methods on the types**, since each type owns
its one serialization: `IndexTransform.to_json` / `from_json`,
`IndexDomain.to_json` / `from_json`, and `to_json` on each output map kind,
with `output_index_map_from_json` in `zarr_indexing.output_map` dispatching the
wire's tagged union back to the right kind. This module is what they share.

Three engine constraints live **here and only here**:

- **Finite bounds.** An `IndexDomain` addresses a finite array, so a canonical
  body carrying a `"-inf"`/`"+inf"` bound cannot be lowered; `from_json` raises.
- **Implicit bounds lower by value.** The `[n]`-bracket implicit/explicit flag
  is a message-layer concern; the engine keeps only the integer value.
- **Integer `index_array` content.** The message layer carries `index_array`
  verbatim (the spec defers its shape and type), so lowering is where a float,
  boolean or string array is rejected — as an `NdselError`, rather than
  truncating `[0.9, 1.9]` to cells 0 and 1 or leaking a raw NumPy error.

## The `index_array` wire format (and the degenerate-collapse it documents)

ndsel and TensorStore both **reject** an output map that carries *both*
`input_dimension` and `index_array`; `input_dimension` belongs to affine
(`single_input_dimension`) maps. The in-memory `ArrayMap` matches: what a map
depends on is read from its full-rank array's shape (its non-singleton axes),
so there is nothing to reconstruct on load. On serialize (`to_json`):

1. An all-singleton `index_array` (size 1) selects a single coordinate
   regardless of input, so it is **collapsed to a `constant` map**
   `{offset: offset + stride*value}`. The size-1 input dimension stays in the
   domain, unconsumed — a valid transform. The selection layer already builds
   such maps as `ConstantMap` (`output_map.array_map_or_constant`); this covers
   hand-built transforms.
2. An **empty** `index_array` (size 0) collapses the same way, to
   `{offset: 0}`. It names no cell, and it can only be empty because an input
   dimension is — the full-rank invariant makes every axis either 1 or the
   domain's extent — so nothing is ever read through it and the emptiness is
   carried by the domain, which is emitted separately. TensorStore does the
   same: `t[ts.d[0][[]]]` is `out[0] = 0`, emitted as `{}`. Emitting the array
   instead would produce a document neither implementation could load, because
   `ndarray.tolist()` renders every empty array as `[]` once the leading axis
   is the zero-length one, and nested lists cannot spell the shape back —
   `[[]]` is `(1, 0)` and nothing spells `(0, 1)`.
3. Non-degenerate `index_array` maps are emitted with their array and bounds
   only.

"""

from __future__ import annotations

from typing import Any, Required, TypedDict

# ---------------------------------------------------------------------------
# TypedDict definitions (canonical JSON shapes)
# ---------------------------------------------------------------------------

# An `index_array` serializes via `ndarray.tolist()`, so it is a nested list of
# ints whose nesting depth equals the array rank.
NestedIntList = list[Any]

# A canonical *lowered* body carries only finite integer bounds, but the JSON
# shape admits the full ndsel `bound` grammar: an explicit int / sentinel, or a
# one-element implicit `[value]` array.
IndexValueJSON = int | str
BoundJSON = int | str | list[IndexValueJSON]


class IndexDomainJSON(TypedDict, total=False):
    """Canonical JSON representation of an IndexDomain.

    Examples
    --------
    >>> doc: IndexDomainJSON = {
    ...     "input_inclusive_min": [0],
    ...     "input_exclusive_max": [4],
    ...     "input_labels": ["x"],
    ... }
    >>> from zarr_indexing import IndexDomain
    >>> IndexDomain.from_json(doc).shape
    (4,)
    """

    input_inclusive_min: Required[list[BoundJSON]]
    """Per-dimension lower bounds; `"-inf"` is legal on the wire but cannot be lowered."""

    input_exclusive_max: Required[list[BoundJSON]]
    """Per-dimension exclusive upper bounds; `"+inf"` is legal on the wire but cannot be lowered."""

    input_labels: Required[list[str]]
    """Per-dimension names; the empty string marks an unlabeled dimension."""


class OutputIndexMapJSON(TypedDict, total=False):
    """Canonical JSON representation of a single output index map.

    Exactly one of three forms (distinguished by which fields are present):

    - `{"offset": 5}` — constant
    - `{"offset": 0, "stride": 1, "input_dimension": 0}` — single_input_dimension
    - `{"offset": 0, "stride": 1, "index_array": [...],
       "index_array_bounds": ["-inf", "+inf"]}` — index_array

    Examples
    --------
    >>> from zarr_indexing import output_index_map_from_json
    >>> constant: OutputIndexMapJSON = {"offset": 5}
    >>> output_index_map_from_json(constant)
    ConstantMap(offset=5)
    >>> affine: OutputIndexMapJSON = {"offset": 0, "stride": 2, "input_dimension": 1}
    >>> output_index_map_from_json(affine)
    DimensionMap(input_dimension=1, offset=0, stride=2)
    """

    offset: int
    """Constant term; alone it is the whole constant form."""

    stride: int
    """Multiplier applied to the input coordinate or to each `index_array` value."""

    input_dimension: int
    """The input dimension the single_input_dimension form reads."""

    index_array: NestedIntList
    """Nested lists of output coordinates, one nesting level per input dimension."""

    index_array_bounds: list[IndexValueJSON]
    """Bounds the `index_array` values are promised to lie in; `["-inf", "+inf"]` if unconstrained."""


class IndexTransformJSON(TypedDict, total=False):
    """Canonical JSON representation of an IndexTransform (spec section 4.3).

    Examples
    --------
    >>> doc: IndexTransformJSON = {
    ...     "input_rank": 1,
    ...     "input_inclusive_min": [0],
    ...     "input_exclusive_max": [2],
    ...     "input_labels": [""],
    ...     "output": [{"offset": 1, "stride": 2, "input_dimension": 0}],
    ... }
    >>> from zarr_indexing import IndexTransform
    >>> IndexTransform.from_json(doc).domain.shape
    (2,)
    """

    input_rank: Required[int]
    """The number of input dimensions; the bounds and labels lists match it in length."""

    input_inclusive_min: Required[list[BoundJSON]]
    """Per-dimension lower bounds; `"-inf"` is legal on the wire but cannot be lowered."""

    input_exclusive_max: Required[list[BoundJSON]]
    """Per-dimension exclusive upper bounds; `"+inf"` is legal on the wire but cannot be lowered."""

    input_labels: Required[list[str]]
    """Per-dimension names; the empty string marks an unlabeled dimension."""

    output: Required[list[OutputIndexMapJSON]]
    """One output map per output dimension."""
