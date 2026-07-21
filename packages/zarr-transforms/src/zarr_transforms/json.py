"""Lowering between canonical ndsel bodies and in-memory `IndexTransform`s.

This is the **engine layer**. Where `messages.py` is pure JSON→JSON and imposes
no array constraints, this module converts a *canonical* ndsel transform body
(spec section 4.3, as produced by `zarr_transforms.messages.normalize_ndsel`)
into the numpy-backed `IndexTransform` the chunk engine runs on, and back.

Two engine constraints live **here and only here**:

- **Finite bounds.** An `IndexDomain` addresses a finite array, so a canonical
  body carrying a `"-inf"`/`"+inf"` bound cannot be lowered; `from_json` raises.
- **Implicit bounds lower by value.** The `[n]`-bracket implicit/explicit flag
  is a message-layer concern; the engine keeps only the integer value.

## The `index_array` wire format (and the degenerate-collapse it documents)

ndsel and TensorStore both **reject** an output map that carries *both*
`input_dimension` and `index_array`. The in-memory `ArrayMap`, however, records
an `input_dimension` to pin the axis an orthogonal (`oindex`) array varies over.
This module bridges the gap:

- **On serialize** (`transform_to_canonical`):
  1. An all-singleton `index_array` (size 1) selects a single coordinate
     regardless of input, so it is **collapsed to a `constant` map**
     `{offset: offset + stride*value}`. The size-1 input dimension stays in the
     domain, unconsumed — a valid transform. This makes a length-1 `oindex`
     selection round-trip *behaviorally* (an `ArrayMap` becomes a `ConstantMap`)
     rather than by object identity.
  2. Non-degenerate `index_array` maps are emitted **without** `input_dimension`.

- **On load** (`transform_from_canonical`): the in-memory `input_dimension` is
  reconstructed from the full-rank array's dependency axes (its non-singleton
  axes, see `transform._array_map_dependency_axes`). An array that solely owns a
  single non-singleton axis is orthogonal (`input_dimension = that axis`); arrays
  that share non-singleton axes, or vary over several, are correlated (`vindex`,
  `input_dimension = None`). A single 1-D array over a rank-1 domain is
  inherently ambiguous between the two flavours; it reconstructs as orthogonal,
  which is behaviorally identical for the single-array case.

`index_transform_to_json` / `index_transform_from_json` (and the `*_domain_*`
variants) are these canonical converters under their historical names.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Required, TypedDict

import numpy as np

from zarr_transforms.domain import IndexDomain
from zarr_transforms.messages import normalize_ndsel
from zarr_transforms.output_map import ArrayMap, ConstantMap, DimensionMap, OutputIndexMap
from zarr_transforms.transform import (
    IndexTransform,
    _array_map_dependency_axes,  # pyright: ignore[reportPrivateUsage]
)

# `_array_map_dependency_axes` is a leading-underscore helper in `transform.py`,
# but it is deliberately shared with this module (the engine-level JSON <->
# `IndexTransform` lowering below needs the same dependency-axis logic that
# `transform.py`'s own array-reindexing helpers use). It is not part of the
# package's public API; pyright's `reportPrivateUsage` flags the cross-module
# import anyway. See `chunk_resolution.py`'s `_dimensions` suppression for the
# analogous rationale — whether to promote either symbol out of "private" is
# an open pre-publish API decision, not resolved here.

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
    """Canonical JSON representation of an IndexDomain."""

    input_inclusive_min: Required[list[BoundJSON]]
    input_exclusive_max: Required[list[BoundJSON]]
    input_labels: Required[list[str]]


class OutputIndexMapJSON(TypedDict, total=False):
    """Canonical JSON representation of a single output index map.

    Exactly one of three forms (distinguished by which fields are present):

    - `{"offset": 5}` — constant
    - `{"offset": 0, "stride": 1, "input_dimension": 0}` — single_input_dimension
    - `{"offset": 0, "stride": 1, "index_array": [...],
       "index_array_bounds": ["-inf", "+inf"]}` — index_array
    """

    offset: int
    stride: int
    input_dimension: int
    index_array: NestedIntList
    index_array_bounds: list[IndexValueJSON]


class IndexTransformJSON(TypedDict, total=False):
    """Canonical JSON representation of an IndexTransform (spec section 4.3)."""

    input_rank: Required[int]
    input_inclusive_min: Required[list[BoundJSON]]
    input_exclusive_max: Required[list[BoundJSON]]
    input_labels: Required[list[str]]
    output: Required[list[OutputIndexMapJSON]]


# ---------------------------------------------------------------------------
# Bound / label lowering (engine constraints)
# ---------------------------------------------------------------------------


def _lower_bound(bound: BoundJSON, where: str) -> int:
    """Lower a canonical bound to a finite integer, rejecting infinities."""
    value = bound[0] if isinstance(bound, list) else bound
    if value == "-inf" or value == "+inf":
        raise ValueError(
            f"{where} is infinite ({value!r}); an IndexDomain addresses a finite "
            f"array and cannot lower an infinite bound"
        )
    return int(value)


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
    """Convert an IndexDomain to its canonical JSON representation."""
    return {
        "input_inclusive_min": list(domain.inclusive_min),
        "input_exclusive_max": list(domain.exclusive_max),
        "input_labels": _emit_labels(domain.labels, domain.ndim),
    }


def index_domain_from_json(data: IndexDomainJSON) -> IndexDomain:
    """Construct an IndexDomain from its canonical JSON representation."""
    inclusive_min = tuple(
        _lower_bound(b, f"input_inclusive_min[{i}]")
        for i, b in enumerate(data["input_inclusive_min"])
    )
    exclusive_max = tuple(
        _lower_bound(b, f"input_exclusive_max[{i}]")
        for i, b in enumerate(data["input_exclusive_max"])
    )
    labels = _lower_labels(list(data["input_labels"]))
    return IndexDomain(inclusive_min=inclusive_min, exclusive_max=exclusive_max, labels=labels)


# ---------------------------------------------------------------------------
# OutputIndexMap serialization
# ---------------------------------------------------------------------------


def output_index_map_to_json(m: OutputIndexMap) -> OutputIndexMapJSON:
    """Convert an output index map to its canonical JSON representation.

    A degenerate all-singleton `ArrayMap` collapses to a `constant` map; a
    non-degenerate one is emitted without `input_dimension` (see the module
    docstring on the wire format).
    """
    if isinstance(m, ConstantMap):
        return {"offset": m.offset}

    if isinstance(m, DimensionMap):
        return {"offset": m.offset, "stride": m.stride, "input_dimension": m.input_dimension}

    # m: ArrayMap (OutputIndexMap = ConstantMap | DimensionMap | ArrayMap)
    if m.index_array.size == 1:
        value = int(m.index_array.reshape(-1)[0])
        return {"offset": m.offset + m.stride * value}
    return {
        "offset": m.offset,
        "stride": m.stride,
        "index_array": m.index_array.tolist(),
        "index_array_bounds": ["-inf", "+inf"],
    }


def output_index_map_from_json(data: OutputIndexMapJSON) -> OutputIndexMap:
    """Construct an output index map from its canonical JSON representation.

    An `index_array` map's `input_dimension` is reconstructed from the array's
    dependency axes in isolation (single non-singleton axis → orthogonal). The
    transform-level loader classifies globally; use it when several maps may
    share axes.
    """
    if "index_array" in data:
        arr = np.asarray(data["index_array"], dtype=np.intp)
        return ArrayMap(
            index_array=arr,
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
            input_dimension=_solo_dependency_axis(arr),
        )

    if "input_dimension" in data:
        return DimensionMap(
            input_dimension=data["input_dimension"],
            offset=data.get("offset", 0),
            stride=data.get("stride", 1),
        )

    return ConstantMap(offset=data.get("offset", 0))


def _solo_dependency_axis(arr: np.ndarray[Any, Any]) -> int | None:
    """The single axis a lone `index_array` varies over, or `None` if not exactly one."""
    dep = _array_map_dependency_axes(arr)
    return dep[0] if len(dep) == 1 else None


# ---------------------------------------------------------------------------
# IndexTransform serialization
# ---------------------------------------------------------------------------


def transform_to_canonical(transform: IndexTransform) -> IndexTransformJSON:
    """Convert an IndexTransform to its canonical ndsel transform body.

    The result is fully explicit (spec section 4.3): `input_rank`, fully written
    bounds and labels, and an explicit `output` with `offset`/`stride` present
    on every affine and array map.
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
    validated, then lowered to the engine representation. `index_array` maps'
    `input_dimension` values are reconstructed by global dependency-axis
    ownership (see the module docstring).
    """
    body = normalize_ndsel({"kind": "transform", **data})

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

    # Classify index_array maps globally: an axis owned by exactly one array map
    # (and the map's sole non-singleton axis) marks that map orthogonal; shared
    # or multiple non-singleton axes mark the maps correlated (vindex).
    array_axes: dict[int, tuple[int, ...]] = {}
    axis_owners: Counter[int] = Counter()
    for i, om in enumerate(output_raw):
        if "index_array" in om:
            arr = np.asarray(om["index_array"], dtype=np.intp)
            dep = _array_map_dependency_axes(arr)
            array_axes[i] = dep
            axis_owners.update(dep)

    output: list[OutputIndexMap] = []
    for i, om in enumerate(output_raw):
        if "index_array" in om:
            dep = array_axes[i]
            input_dim = dep[0] if len(dep) == 1 and axis_owners[dep[0]] == 1 else None
            output.append(
                ArrayMap(
                    index_array=np.asarray(om["index_array"], dtype=np.intp),
                    offset=om.get("offset", 0),
                    stride=om.get("stride", 1),
                    input_dimension=input_dim,
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

    return IndexTransform(domain=domain, output=tuple(output))


# Historical names, now pointing at the canonical converters.
index_transform_to_json = transform_to_canonical
index_transform_from_json = transform_from_canonical
