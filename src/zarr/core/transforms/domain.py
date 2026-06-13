"""Index domains — rectangular regions in N-dimensional integer space.

An ``IndexDomain`` represents the set of valid coordinates for an array or
array view. It is the cartesian product of per-dimension integer ranges::

    IndexDomain(inclusive_min=(2, 5), exclusive_max=(10, 20))
    # represents {(i, j) : 2 <= i < 10, 5 <= j < 20}

Unlike NumPy, domains can have **non-zero origins**. After slicing
``arr[5:10]``, the result has origin 5 and shape 5 — coordinates 5 through
9 are valid. This follows the TensorStore convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexDomain:
    """A rectangular region in N-dimensional index space.

    The valid coordinates are the integers in
    ``[inclusive_min[d], exclusive_max[d])`` for each dimension ``d``.
    """

    inclusive_min: tuple[int, ...]
    exclusive_max: tuple[int, ...]
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.inclusive_min) != len(self.exclusive_max):
            raise ValueError(
                f"inclusive_min and exclusive_max must have the same length. "
                f"Got {len(self.inclusive_min)} and {len(self.exclusive_max)}."
            )
        for i, (lo, hi) in enumerate(zip(self.inclusive_min, self.exclusive_max, strict=True)):
            if lo > hi:
                raise ValueError(
                    f"inclusive_min must be <= exclusive_max for all dimensions. "
                    f"Dimension {i}: {lo} > {hi}"
                )
        if self.labels is not None and len(self.labels) != len(self.inclusive_min):
            raise ValueError(
                f"labels must have the same length as dimensions. "
                f"Got {len(self.labels)} labels for {len(self.inclusive_min)} dimensions."
            )

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> IndexDomain:
        """Create a domain with origin at zero."""
        return cls(
            inclusive_min=(0,) * len(shape),
            exclusive_max=shape,
        )

    @property
    def ndim(self) -> int:
        return len(self.inclusive_min)

    @property
    def origin(self) -> tuple[int, ...]:
        return self.inclusive_min

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(hi - lo for lo, hi in zip(self.inclusive_min, self.exclusive_max, strict=True))

    def contains(self, index: tuple[int, ...]) -> bool:
        if len(index) != self.ndim:
            return False
        return all(
            lo <= idx < hi
            for lo, hi, idx in zip(self.inclusive_min, self.exclusive_max, index, strict=True)
        )

    def contains_domain(self, other: IndexDomain) -> bool:
        if other.ndim != self.ndim:
            return False
        return all(
            self_lo <= other_lo and other_hi <= self_hi
            for self_lo, self_hi, other_lo, other_hi in zip(
                self.inclusive_min,
                self.exclusive_max,
                other.inclusive_min,
                other.exclusive_max,
                strict=True,
            )
        )

    def intersect(self, other: IndexDomain) -> IndexDomain | None:
        if other.ndim != self.ndim:
            raise ValueError(
                f"Cannot intersect domains with different ranks: {self.ndim} vs {other.ndim}"
            )
        new_min = tuple(
            max(a, b) for a, b in zip(self.inclusive_min, other.inclusive_min, strict=True)
        )
        new_max = tuple(
            min(a, b) for a, b in zip(self.exclusive_max, other.exclusive_max, strict=True)
        )
        if any(lo >= hi for lo, hi in zip(new_min, new_max, strict=True)):
            return None
        return IndexDomain(inclusive_min=new_min, exclusive_max=new_max)

    def translate(self, offset: tuple[int, ...]) -> IndexDomain:
        if len(offset) != self.ndim:
            raise ValueError(
                f"Offset must have same length as domain dimensions. "
                f"Domain has {self.ndim} dimensions, offset has {len(offset)}."
            )
        new_min = tuple(lo + off for lo, off in zip(self.inclusive_min, offset, strict=True))
        new_max = tuple(hi + off for hi, off in zip(self.exclusive_max, offset, strict=True))
        return IndexDomain(inclusive_min=new_min, exclusive_max=new_max)

    def narrow(self, selection: Any) -> IndexDomain:
        """Apply a basic selection and return a narrowed domain.
        Indices are absolute coordinates. Integer indices produce length-1 extent.
        Strided slices are not supported — use IndexTransform for strides.
        """
        normalized = _normalize_selection(selection, self.ndim)
        new_inclusive_min: list[int] = []
        new_exclusive_max: list[int] = []
        for dim_idx, (sel, dim_lo, dim_hi) in enumerate(
            zip(normalized, self.inclusive_min, self.exclusive_max, strict=True)
        ):
            if isinstance(sel, int):
                if sel < dim_lo or sel >= dim_hi:
                    raise IndexError(
                        f"index {sel} is out of bounds for dimension {dim_idx} "
                        f"with domain [{dim_lo}, {dim_hi})"
                    )
                new_inclusive_min.append(sel)
                new_exclusive_max.append(sel + 1)
            else:
                start, stop, step = sel.start, sel.stop, sel.step
                if step is not None and step != 1:
                    raise IndexError(
                        "IndexDomain.narrow only supports step=1 slices. "
                        f"Got step={step}. Use IndexTransform for strided access."
                    )
                abs_start = dim_lo if start is None else start
                abs_stop = dim_hi if stop is None else stop
                abs_start = max(abs_start, dim_lo)
                abs_stop = min(abs_stop, dim_hi)
                abs_stop = max(abs_stop, abs_start)
                new_inclusive_min.append(abs_start)
                new_exclusive_max.append(abs_stop)
        return IndexDomain(
            inclusive_min=tuple(new_inclusive_min),
            exclusive_max=tuple(new_exclusive_max),
        )


def _normalize_selection(selection: Any, ndim: int) -> tuple[int | slice, ...]:
    """Normalize a basic selection to a tuple of ints/slices with length ndim."""
    if not isinstance(selection, tuple):
        selection = (selection,)
    result: list[int | slice] = []
    ellipsis_seen = False
    for sel in selection:
        if sel is Ellipsis:
            if ellipsis_seen:
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis_seen = True
            num_missing = ndim - (len(selection) - 1)
            result.extend([slice(None)] * num_missing)
        else:
            result.append(sel)
    while len(result) < ndim:
        result.append(slice(None))
    if len(result) > ndim:
        raise IndexError(
            f"too many indices for array: array has {ndim} dimensions, "
            f"but {len(result)} were indexed"
        )
    return tuple(result)
