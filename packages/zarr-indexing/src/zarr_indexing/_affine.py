"""Checked affine coordinate arithmetic."""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt

_INTP_INFO = np.iinfo(np.intp)


def _fits_intp(value: int) -> bool:
    return _INTP_INFO.min <= value <= _INTP_INFO.max


@overload
def checked_affine(offset: int, stride: int, coordinates: int) -> int: ...


@overload
def checked_affine(
    offset: int,
    stride: int,
    coordinates: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.intp]: ...


def checked_affine(
    offset: int,
    stride: int,
    coordinates: int | npt.NDArray[np.integer[Any]],
) -> int | npt.NDArray[np.intp]:
    """Evaluate ``offset + stride * coordinates`` without integer overflow.

    Bounds are established with Python integers before coordinates are cast or
    NumPy performs fixed-width arithmetic. The common representable case then
    uses an ``np.intp`` fast path whose multiplication and addition were proven
    safe; cancellation cases use exact object arithmetic.

    The identity affine (``offset == 0 and stride == 1``) of an array whose
    dtype fits ``np.intp`` cannot overflow and is returned re-typed without a
    scan; `ArrayMap.__post_init__` normalizes every index array through it.
    """
    offset = int(offset)
    stride = int(stride)
    if not isinstance(coordinates, np.ndarray):
        mapped = offset + stride * int(coordinates)
        if not _fits_intp(mapped):
            raise OverflowError(f"output coordinate {mapped} is outside np.intp range")
        return mapped

    if coordinates.size == 0:
        return np.empty(coordinates.shape, dtype=np.intp)

    if offset == 0 and stride == 1 and np.can_cast(coordinates.dtype, np.intp):
        return np.asarray(coordinates, dtype=np.intp)

    coordinate_min = int(coordinates.min())
    coordinate_max = int(coordinates.max())
    product_at_min = stride * coordinate_min
    product_at_max = stride * coordinate_max
    mapped_at_min = offset + product_at_min
    mapped_at_max = offset + product_at_max
    mapped_min = min(mapped_at_min, mapped_at_max)
    mapped_max = max(mapped_at_min, mapped_at_max)
    if not _fits_intp(mapped_min) or not _fits_intp(mapped_max):
        invalid = mapped_min if not _fits_intp(mapped_min) else mapped_max
        raise OverflowError(f"output coordinate {invalid} is outside np.intp range")

    safe_fixed_width = (
        _fits_intp(coordinate_min)
        and _fits_intp(coordinate_max)
        and _fits_intp(offset)
        and _fits_intp(stride)
        and _fits_intp(product_at_min)
        and _fits_intp(product_at_max)
    )
    if safe_fixed_width:
        intp_coordinates = coordinates.astype(np.intp, copy=False)
        return np.asarray(offset + stride * intp_coordinates, dtype=np.intp)

    exact = offset + stride * coordinates.astype(object)
    return np.asarray(exact, dtype=np.intp)
