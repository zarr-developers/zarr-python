"""Shared lowering rules between the canonical ndsel wire form and the engine.

Package-private: the types that serialize themselves (`IndexDomain`,
`IndexTransform`, the output map kinds) all need these, so they cannot live in
any one of them, and they are not API. The three engine constraints named in
[`zarr_indexing.json`][zarr_indexing.json] — finite bounds, implicit bounds
lowering by value, integer `index_array` content — are enforced here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing.messages import NdselError

if TYPE_CHECKING:
    from zarr_indexing.domain import IndexDomain
    from zarr_indexing.json import BoundJSON


def lower_bound(bound: BoundJSON, where: str) -> int:
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


def lower_index_array(raw: Any, where: str) -> np.ndarray[Any, np.dtype[np.intp]]:
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


def lower_labels(labels: list[str]) -> tuple[str, ...] | None:
    """All-empty labels collapse to `None` so a label-free domain round-trips."""
    return None if all(label == "" for label in labels) else tuple(labels)


def emit_labels(labels: tuple[str, ...] | None, rank: int) -> list[str]:
    """Emit canonical labels: `[""]*rank` when the domain is unlabeled."""
    return [""] * rank if labels is None else list(labels)


def full_rank_index_array(
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
