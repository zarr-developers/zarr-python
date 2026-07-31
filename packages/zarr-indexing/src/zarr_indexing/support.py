"""How much indexing a wrapped array can do for itself.

[`LazyArray`](lazy_array.md) reads through the object it wraps, and the objects
it can wrap differ widely in what they accept. A NumPy array takes any key NumPy
takes. A zarr array takes basic slicing through `__getitem__` and offers
`oindex` and `vindex` accessors for the two fancy dialects. An HDF5 dataset
takes integer arrays but only one axis at a time. A minimal duck array may take
nothing but integers and slices.

`IndexingSupport` names those four capabilities, using the taxonomy and the
member names of xarray's [`IndexingSupport`][1], so a reader who knows one knows
the other. `LazyArray` uses the level to decide how much of a selection to hand
to the source and how much to finish in memory: the part the source cannot
express is applied to the block that comes back, with NumPy. Every level
therefore produces the same answer — the level chooses how much data crosses the
boundary, not what the read means.

[1]: https://github.com/pydata/xarray/blob/main/xarray/core/indexing.py

Declaring a level
-----------------
A source can declare its own level by exposing a `__zarr_indexing_support__`
attribute holding an `IndexingSupport`. That attribute is read defensively — a
value of the wrong type, or an attribute access that raises, means "no
declaration" rather than an error.

Otherwise the level is inferred, conservatively:

- a `numpy.ndarray` (or a subclass) supports everything NumPy supports:
  `VECTORIZED`
- an object exposing both an `oindex` and a `vindex` accessor — zarr's surface —
  is taken at its word: `VECTORIZED`
- anything else: `BASIC`

`BASIC` is the default for an unrecognized source because it is the only
assumption that is always correct: an object that answers `shape`, `dtype`, and
`__getitem__` is guaranteed to understand integers and slices and nothing more.
Inference deliberately ignores `__array_namespace__`: the array API standard
requires neither integer-array nor boolean-mask indexing, so declaring one is no
evidence of fancy indexing.

`LazyArray.with_indexing_support` overrides both, for a source that supports more
(or less) than can be detected from the outside.
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np

__all__ = [
    "SUPPORT_ATTRIBUTE",
    "IndexingSupport",
    "infer_indexing_support",
    "resolve_indexing_support",
]

SUPPORT_ATTRIBUTE = "__zarr_indexing_support__"
"""The attribute a source sets to declare its own `IndexingSupport`."""


class IndexingSupport(enum.Enum):
    """What a source's indexing surface accepts.

    Each member describes the widest key the source can be given in one request.
    A level is a promise about `__getitem__` unless the source also exposes an
    `oindex` or `vindex` accessor, in which case the orthogonal and correlated
    forms are sent there instead — see the notes on each member.

    The four levels are not a numeric scale, and this enum is deliberately not
    ordered: `OUTER_1VECTOR` sits *between* `BASIC` and `OUTER` in capability,
    which no ordering of xarray's member names would make obvious. Compare
    members by identity.
    """

    BASIC = "basic"
    """Integers and slices only.

    `source[key]` accepts a tuple of integers and slices with a positive step.
    This is the floor every array-like meets, and the default for a source whose
    capability cannot be detected. A fancy selection against a `BASIC` source is
    read as the smallest enclosing slab and gathered from that block in memory.
    """

    OUTER = "outer"
    """Adds a one-dimensional integer array per axis, combined as an outer product.

    Several array axes can be requested at once, and each varies independently:
    the result is the outer product of the per-axis selections, which is what
    `zarr.Array.oindex` and `h5py`'s multi-list keys mean.

    A multi-array outer request is only ever sent to an `oindex` accessor.
    `__getitem__` alone is ambiguous — the same key means an outer product to
    HDF5 and a correlated gather to NumPy — so a source that declares `OUTER`
    without exposing `oindex` is given at most one array axis per request, the
    key both readings agree on.
    """

    OUTER_1VECTOR = "outer_1vector"
    """Outer indexing, but with at most one array axis per request.

    The limit some backends impose (HDF5's is the classic one). The remaining
    array axes are read as slabs and gathered in memory. The axis that keeps its
    array is the one where a slab would over-read the most.
    """

    VECTORIZED = "vectorized"
    """Full NumPy-style fancy indexing: several arrays broadcast against each other.

    `source[key]` follows NumPy's rules, so a tuple of coordinate arrays gathers
    a list of points rather than an outer product. A source that exposes
    `vindex` is given point gathers there instead, and one that exposes `oindex`
    is given outer-product requests there, since both accessors say what they
    mean without relying on NumPy's placement rules.
    """


def _read_attribute(array: Any, name: str) -> Any:
    """Read an attribute off a foreign object, treating any failure as "absent"."""
    try:
        return getattr(array, name, None)
    # Detection inspects an object we did not write; a guarded property may
    # raise anything, and that can only ever mean "no information".
    except Exception:
        return None


def has_accessor(array: Any, name: str) -> bool:
    """Whether `array` exposes a usable `oindex`/`vindex`-style accessor."""
    return _read_attribute(array, name) is not None


def declared_indexing_support(array: Any) -> IndexingSupport | None:
    """The level `array` declares for itself, or None if it declares none.

    Reads `__zarr_indexing_support__` defensively: a missing attribute, an
    attribute of the wrong type, and an attribute access that raises are all
    "no declaration".
    """
    declared = _read_attribute(array, SUPPORT_ATTRIBUTE)
    return declared if isinstance(declared, IndexingSupport) else None


def infer_indexing_support(array: Any) -> IndexingSupport:
    """Guess what `array` accepts from its type and its accessors.

    Conservative by construction: only a NumPy array and an object carrying
    zarr's full `oindex`/`vindex` surface are credited with more than `BASIC`.
    See the module docstring for why an under-estimate is the safe direction and
    why `__array_namespace__` is not evidence.
    """
    if isinstance(array, np.ndarray):
        return IndexingSupport.VECTORIZED
    if has_accessor(array, "oindex") and has_accessor(array, "vindex"):
        return IndexingSupport.VECTORIZED
    return IndexingSupport.BASIC


def resolve_indexing_support(array: Any) -> IndexingSupport:
    """The level to use for `array`: what it declares, else what can be inferred."""
    declared = declared_indexing_support(array)
    return infer_indexing_support(array) if declared is None else declared
