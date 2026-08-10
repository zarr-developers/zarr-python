"""Lowering between canonical ndsel bodies and in-memory `IndexTransform`s.

This is the **engine layer**. Where `messages.py` is pure JSON→JSON and imposes
no array constraints, this module converts a *canonical* ndsel transform body
(spec section 4.3, as produced by `zarr_indexing.messages.normalize_ndsel`)
into the numpy-backed `IndexTransform` the chunk engine runs on, and back.

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
so there is nothing to reconstruct on load. On serialize
(`transform_to_canonical`):

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

`index_transform_to_json` / `index_transform_from_json` (and the `*_domain_*`
variants) are these canonical converters under their historical names.
"""

from __future__ import annotations

from typing import Any, Required, TypedDict

import numpy as np

from zarr_indexing.domain import IndexDomain
from zarr_indexing.messages import NdselError, normalize_ndsel
from zarr_indexing.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_indexing.transform import IndexTransform

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
    >>> index_domain_from_json(doc).shape
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
    >>> transform_from_canonical(doc).domain.shape
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


# ---------------------------------------------------------------------------
# Bound / label lowering (engine constraints)
# ---------------------------------------------------------------------------


def _lower_bound(bound: BoundJSON, where: str) -> int:
    """Lower a canonical bound to a finite integer, rejecting infinities.

    Reached only with a bound the message layer has already validated as an
    `index-value`, so the one thing left to rule out is a sentinel: an
    `IndexDomain` addresses a finite array.
    """
    value = bound[0] if isinstance(bound, list) else bound
    if value == "-inf" or value == "+inf":
        raise NdselError(
            "invalid_json",
            f"{where} is infinite ({value!r}); an IndexDomain addresses a finite "
            f"array and cannot lower an infinite bound",
        )
    return int(value)


def _lower_index_array(raw: Any, where: str) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Lower a canonical `index_array` to `intp`, rejecting non-integer content.

    The message layer carries `index_array` verbatim — the spec defers its shape
    and type to the engine — so this is where the content is checked. An index
    array names output coordinates, and nothing but an integer names one: converting
    `[0.9, 1.9]` would silently read cells 0 and 1, and `[true, false]` cells 1
    and 0. Strings raise here rather than leaking NumPy's own conversion error.
    """
    if not isinstance(raw, list):
        # A bare integer would become a rank-0 array and then be widened into a
        # length-1 map, so a document that names no cells would select one.
        raise NdselError(
            "invalid_json",
            f"{where} must be an array of integers, got {raw!r}",
        )
    try:
        arr = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise NdselError("invalid_json", f"{where} is not an array: {exc}") from exc
    if arr.size == 0 and arr.dtype.kind == "f":
        # An empty JSON list carries no element type and NumPy defaults it to
        # float64. An empty selection is legal, so take it as an empty index array.
        return np.zeros(arr.shape, dtype=np.intp)
    if arr.dtype.kind not in "iu":
        raise NdselError(
            "invalid_json",
            f"{where} must hold integers, got an array of {arr.dtype.name}; an "
            f"index array names output coordinates, which floats, booleans and "
            f"strings do not",
        )
    return np.asarray(arr, dtype=np.intp)


def _lower_labels(labels: list[str]) -> tuple[str, ...] | None:
    """All-empty labels collapse to `None` so a label-free domain round-trips."""
    return None if all(label == "" for label in labels) else tuple(labels)


def _emit_labels(labels: tuple[str, ...] | None, rank: int) -> list[str]:
    """Emit canonical labels: `[""]*rank` when the domain is unlabeled."""
    return [""] * rank if labels is None else list(labels)


# ---------------------------------------------------------------------------
# IndexDomain serialization
# ---------------------------------------------------------------------------


def index_domain_to_json(domain: IndexDomain) -> IndexDomainJSON:
    """Convert an IndexDomain to its canonical JSON representation.

    Examples
    --------
    >>> domain = IndexDomain(inclusive_min=(0,), exclusive_max=(3,))
    >>> index_domain_to_json(domain)
    {'input_inclusive_min': [0], 'input_exclusive_max': [3], 'input_labels': ['']}
    >>> index_domain_from_json(index_domain_to_json(domain)) == domain
    True
    """
    return {
        "input_inclusive_min": list(domain.inclusive_min),
        "input_exclusive_max": list(domain.exclusive_max),
        "input_labels": _emit_labels(domain.labels, domain.ndim),
    }


def index_domain_from_json(data: IndexDomainJSON) -> IndexDomain:
    """Construct an IndexDomain from its canonical JSON representation.

    The document is validated by the message layer first, exactly as a transform
    body is. Reading the keys directly would be a second, undefended way into
    the same objects: `int(value)` alone accepts `3.9`, `"3"` and `True`, and
    each of those builds a domain that is not the document's.

    Examples
    --------
    >>> domain = index_domain_from_json(
    ...     {"input_inclusive_min": [1], "input_exclusive_max": [4], "input_labels": [""]}
    ... )
    >>> (domain.inclusive_min, domain.exclusive_max)
    ((1,), (4,))
    >>> domain.shape
    (3,)
    """
    # The annotation says what a well-formed caller passes; this is a parser of
    # documents that arrive from elsewhere, so the shape is checked rather than
    # assumed.
    if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise NdselError("invalid_json", f"an index domain must be a JSON object, got {data!r}")
    body = normalize_ndsel({**data, "kind": "transform"})
    inclusive_min = tuple(
        _lower_bound(b, f"input_inclusive_min[{i}]")
        for i, b in enumerate(body["input_inclusive_min"])
    )
    exclusive_max = tuple(
        _lower_bound(b, f"input_exclusive_max[{i}]")
        for i, b in enumerate(body["input_exclusive_max"])
    )
    return IndexDomain(
        inclusive_min=inclusive_min,
        exclusive_max=exclusive_max,
        labels=_lower_labels(body["input_labels"]),
    )


# ---------------------------------------------------------------------------
# OutputIndexMap serialization
# ---------------------------------------------------------------------------


def output_index_map_to_json(m: OutputIndexMap) -> OutputIndexMapJSON:
    """Convert an output index map to its canonical JSON representation.

    A degenerate all-singleton `ArrayMap` collapses to a `constant` map (see
    the module docstring on the wire format).
    """
    if isinstance(m, ConstantMap):
        return {"offset": m.offset}

    if isinstance(m, DimensionMap):
        return {"offset": m.offset, "stride": m.stride, "input_dimension": m.input_dimension}

    # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
    if m.index_array.size == 1:
        value = int(m.index_array.reshape(-1)[0])
        return {"offset": m.offset + m.stride * value}
    if m.index_array.size == 0:
        # An empty index array names no cell, and it can only be empty because
        # an input dimension is — the full-rank invariant makes every axis
        # either 1 or the domain's extent, so a 0 there means the domain has a 0
        # there too. Nothing is ever read through this map, which makes it
        # degenerate in exactly the way a size-1 array is, and it collapses the
        # same way. The empty dimension stays in the domain, so the transform
        # that comes back describes the same (empty) selection.
        #
        # This is also what the reference implementation does: TensorStore
        # renders `t[ts.d[0][[]]]` as `out[0] = 0` and emits `{}`. Emitting the
        # array instead would produce a document neither it nor this package
        # could load, because `tolist()` renders every empty array as `[]` once
        # the leading axis is the zero-length one — and nested lists cannot
        # spell the shape back, `[[]]` being (1, 0) with nothing for (0, 1).
        return {"offset": 0}
    return {
        "offset": m.offset,
        "stride": m.stride,
        "index_array": m.index_array.tolist(),
        "index_array_bounds": ["-inf", "+inf"],
    }


def output_index_map_from_json(data: OutputIndexMapJSON) -> OutputIndexMap:
    """Construct an output index map from its canonical JSON representation."""
    if "index_array" in data:
        arr = _lower_index_array(data["index_array"], "index_array")
        return ArrayMap(
            index_array=arr,
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
        )

    if "input_dimension" in data:
        return DimensionMap(
            input_dimension=data["input_dimension"],
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
        )

    return ConstantMap(offset=data.get("offset", 0))


def _full_rank_index_array(
    arr: np.ndarray[Any, np.dtype[np.intp]],
    domain: IndexDomain,
    where: str,
) -> np.ndarray[Any, np.dtype[np.intp]]:
    """Give an incoming `index_array` the input rank the engine requires.

    ndsel leaves index-array rank unvalidated, so a conformant producer may send
    an array of lower rank that broadcasts against the domain. A non-empty one
    is aligned to the *trailing* input dimensions, which is how NumPy broadcasts
    and how a producer omitting leading singletons means it to be read.

    An empty array is a different matter: `[]` is the only spelling of every
    empty shape once the leading axis is the zero-length one, so the axis it
    varies over cannot be read off it. It is recovered from the domain, which
    can only be empty on the axis in question — and rejected when the domain
    leaves that ambiguous. This package never emits such a document (an empty
    map is degenerate and collapses to a constant, as TensorStore's does), so
    this path exists for external producers alone.
    """
    if arr.size == 0 and arr.ndim != domain.ndim:
        empty_axes = [k for k, extent in enumerate(domain.shape) if extent == 0]
        if len(empty_axes) != 1:
            raise NdselError(
                "invalid_json",
                f"{where}.index_array is empty, but the input domain has "
                f"{len(empty_axes)} zero-length dimensions, so the axis it varies "
                f"over cannot be recovered",
            )
        shape = [1] * domain.ndim
        shape[empty_axes[0]] = 0
        return arr.reshape(tuple(shape))

    if arr.ndim < domain.ndim:
        return arr.reshape((1,) * (domain.ndim - arr.ndim) + arr.shape)
    return arr


# ---------------------------------------------------------------------------
# IndexTransform serialization
# ---------------------------------------------------------------------------


def transform_to_canonical(transform: IndexTransform) -> IndexTransformJSON:
    """Convert an IndexTransform to its canonical ndsel transform body.

    The result is fully explicit (spec section 4.3): `input_rank`, fully written
    bounds and labels, and an explicit `output` with `offset`/`stride` present
    on every affine and array map.

    Examples
    --------
    >>> body = transform_to_canonical(IndexTransform.from_shape((6,))[1:5:2])
    >>> (body["input_inclusive_min"], body["input_exclusive_max"])
    ([0], [2])
    >>> body["output"]
    [{'offset': 1, 'stride': 2, 'input_dimension': 0}]
    >>> index_transform_to_json is transform_to_canonical  # the historical name
    True
    """
    return {
        "input_rank": transform.domain.ndim,
        "input_inclusive_min": list(transform.domain.inclusive_min),
        "input_exclusive_max": list(transform.domain.exclusive_max),
        "input_labels": _emit_labels(transform.domain.labels, transform.domain.ndim),
        "output": [output_index_map_to_json(m) for m in transform.output],
    }


def transform_from_canonical(data: IndexTransformJSON) -> IndexTransform:
    """Construct an IndexTransform from a canonical (or canonicalizable) body.

    The body is first run through the message layer (`normalize_ndsel`) so that
    omitted fields — identity `output`, default bounds/labels — are filled and
    validated, then lowered to the engine representation. Lower-rank
    `index_array`s are widened to the full input rank on the way in (see the
    module docstring).

    Examples
    --------
    >>> body = transform_to_canonical(IndexTransform.from_shape((6,))[1:5:2])
    >>> transform = transform_from_canonical(body)
    >>> transform.domain.shape
    (2,)
    >>> transform_to_canonical(transform) == body  # the round trip is exact
    True
    >>> index_transform_from_json is transform_from_canonical  # the historical name
    True
    """
    if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise NdselError("invalid_json", f"a transform body must be a JSON object, got {data!r}")
    kind = data.get("kind", "transform")
    if kind != "transform":
        # Spelled last below, so a body carrying its own `kind` cannot reinterpret
        # the document as some other message and return a selection this function
        # never promised.
        raise NdselError(
            "invalid_json",
            f"a transform body cannot carry kind {kind!r}",
        )
    body = normalize_ndsel({**data, "kind": "transform"})

    inclusive_min = tuple(
        _lower_bound(b, f"input_inclusive_min[{i}]")
        for i, b in enumerate(body["input_inclusive_min"])
    )
    exclusive_max = tuple(
        _lower_bound(b, f"input_exclusive_max[{i}]")
        for i, b in enumerate(body["input_exclusive_max"])
    )
    domain = IndexDomain(
        inclusive_min=inclusive_min,
        exclusive_max=exclusive_max,
        labels=_lower_labels(body["input_labels"]),
    )

    output_raw: list[dict[str, Any]] = body["output"]

    output: list[OutputIndexMap] = []
    for i, om in enumerate(output_raw):
        if "index_array" in om:
            where = f"output[{i}]"
            arr = _lower_index_array(om["index_array"], f"{where}.index_array")
            # ndsel leaves index-array rank unvalidated, so an external producer
            # may send an array of lower rank that broadcasts against the domain.
            # Widen it here, on the way in, so every transform that exists holds
            # the full-rank invariant the engine reads dependency axes from.
            arr = _full_rank_index_array(arr, domain, where)
            output.append(
                ArrayMap(
                    index_array=arr,
                    offset=om.get("offset", 0),
                    stride=om.get("stride", 1),
                )
            )
        elif "input_dimension" in om:
            output.append(
                DimensionMap(
                    input_dimension=om["input_dimension"],
                    offset=om.get("offset", 0),
                    stride=om.get("stride", 1),
                )
            )
        else:
            output.append(ConstantMap(offset=om.get("offset", 0)))

    try:
        return IndexTransform(domain=domain, output=tuple(output))
    except ValueError as exc:
        # The engine's invariants are the last gate a document passes, and they
        # speak in the engine's vocabulary. A document that fails them is invalid
        # input, so it leaves here as one — with the engine's account of what was
        # wrong kept, since it names the offending output map and axis.
        raise NdselError("rank_mismatch", str(exc)) from exc


# Historical names, now pointing at the canonical converters.
index_transform_to_json = transform_to_canonical
index_transform_from_json = transform_from_canonical
