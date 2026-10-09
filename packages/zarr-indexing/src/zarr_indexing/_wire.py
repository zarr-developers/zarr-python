"""Shared lowering rules between the canonical ndsel wire form and the engine.

Package-private: the types that serialize themselves (`IndexDomain`,
`IndexTransform`, the output map kinds) all need these, so they cannot live in
any one of them, and they are not API. Domain bounds, index-array bounds,
implicit bounds lowering by value, and integer `index_array` content are
handled here, as described in [`zarr_indexing.json`][zarr_indexing.json].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from zarr_indexing.messages import NdselError, validate_index_array_bounds

if TYPE_CHECKING:
    from zarr_indexing.domain import IndexDomain
    from zarr_indexing.json import BoundJSON


def check_index_array_bounds(array: np.ndarray[Any, Any], bounds: Any, where: str) -> None:
    """Validate every raw index value against an inclusive interval.

    Validate interval syntax even for empty arrays. Once checked, immutable
    index coordinates need no retained constraint. Use Python integer extrema
    to avoid overflow or floating-point rounding at integer limits.
    """
    lo, hi = validate_index_array_bounds(bounds, where)
    if array.size == 0 or (lo == "-inf" and hi == "+inf"):
        return
    minimum, maximum = int(array.min()), int(array.max())
    if (
        lo == "+inf"
        or hi == "-inf"
        or (isinstance(lo, int) and minimum < lo)
        or (isinstance(hi, int) and maximum > hi)
    ):
        raise NdselError(
            "invalid_json",
            f"{where}.index_array values [{minimum}, {maximum}] are outside "
            f"index_array_bounds {bounds!r}",
        )


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
        # The wire representation requires a nested array, not a scalar.
        raise NdselError(
            "invalid_json",
            f"{where} must be an array of integers, got {raw!r}",
        )
    # Validate before NumPy inference can coerce mixed booleans to integers or
    # conversion to intp can wrap unsigned coordinates.
    limits = np.iinfo(np.intp)
    pending = [raw]
    while pending:
        for value in pending.pop():
            if isinstance(value, list):
                pending.append(value)
            elif isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                dtype = np.asarray(value).dtype.name
                raise NdselError(
                    "invalid_json", f"{where} must hold integers, got {dtype}: {value!r}"
                )
            elif not limits.min <= value <= limits.max:
                raise NdselError(
                    "invalid_json", f"{where} coordinate {value} is outside intp range"
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

    ndsel intends index arrays to have the input rank, but defers validating
    that constraint. This engine also accepts lower-rank arrays, aligning them
    to the *trailing* input dimensions as in NumPy broadcasting. This extension
    does not imply other ndsel consumers accept the same document.

    An empty array is a different matter: `[]` is the only spelling of every
    empty shape once the leading axis is the zero-length one, so the axis it
    varies over cannot be read off it. When ranks differ, this engine recovers
    a shape with the domain's single empty axis and singleton axes elsewhere;
    it rejects recovery if the domain has zero or multiple empty axes.
    This package never emits such a document (an empty
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
