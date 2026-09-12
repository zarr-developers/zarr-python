"""Index domains — rectangular regions in N-dimensional integer space.

An `IndexDomain` represents the set of valid coordinates for an array or
array view. It is the cartesian product of per-dimension integer ranges::

    IndexDomain(inclusive_min=(2, 5), exclusive_max=(10, 20))
    # represents {(i, j) : 2 <= i < 10, 5 <= j < 20}

Unlike NumPy, domains can have **non-zero origins**. After slicing
`arr[5:10]`, the result has origin 5 and shape 5 — coordinates 5 through
9 are valid. This follows the TensorStore convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zarr_indexing.errors import BoundsCheckError

if TYPE_CHECKING:
    from zarr_indexing.json import IndexDomainJSON


@dataclass(frozen=True, slots=True)
class IndexDomain:
    """A rectangular region in N-dimensional index space.

    The valid coordinates are the integers in
    `[inclusive_min[d], exclusive_max[d])` for each dimension `d`.

    Examples
    --------
    >>> domain = IndexDomain(inclusive_min=(2, 5), exclusive_max=(10, 20))
    >>> domain.shape
    (8, 15)

    Unlike a NumPy shape, a domain keeps literal coordinates: narrowing to
    `[5, 10)` gives a region whose valid coordinates are 5 through 9, not
    re-zeroed:

    >>> view = IndexDomain.from_shape((10,)).narrow(slice(5, 10))
    >>> view.origin, view.shape
    ((5,), (5,))
    >>> view.contains((5,)), view.contains((0,))
    (True, False)
    """

    inclusive_min: tuple[int, ...]
    """The lower corner: each dimension's smallest literal coordinate. May be negative."""

    exclusive_max: tuple[int, ...]
    """Each dimension's upper bound, excluded: valid coordinates end at `exclusive_max - 1`."""

    labels: tuple[str, ...] | None = None
    """Optional per-dimension names; carried through the wire format, never consulted by indexing."""
    # Lazily-memoized shape. Excluded from init/repr/eq/hash: it is derived
    # state, not part of the domain's identity. The domain is frozen, so the
    # value is computed at most once (see `shape`). `None` is the unset
    # sentinel; an empty shape caches as `()`.
    _shape: tuple[int, ...] | None = field(default=None, init=False, repr=False, compare=False)

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
    def _unchecked(
        cls, inclusive_min: tuple[int, ...], exclusive_max: tuple[int, ...]
    ) -> IndexDomain:
        """Build an unlabeled domain from bounds the caller has already established.

        Skips `__post_init__`. For internal producers that derive bounds from an
        already-valid domain — chunk resolution builds two domains per chunk,
        and re-validating them was a measurable share of a plan's cost.
        """
        domain = object.__new__(cls)
        object.__setattr__(domain, "inclusive_min", inclusive_min)
        object.__setattr__(domain, "exclusive_max", exclusive_max)
        object.__setattr__(domain, "labels", None)
        object.__setattr__(domain, "_shape", None)
        return domain

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> IndexDomain:
        """Create a domain with origin at zero."""
        return cls(
            inclusive_min=(0,) * len(shape),
            exclusive_max=shape,
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.inclusive_min)

    @property
    def origin(self) -> tuple[int, ...]:
        """The lower corner of the domain — an alias for `inclusive_min`, and may be negative."""
        return self.inclusive_min

    @property
    def shape(self) -> tuple[int, ...]:
        """Per-dimension extents: `exclusive_max - inclusive_min` for each dimension."""
        cached = self._shape
        if cached is None:
            cached = tuple(
                hi - lo for lo, hi in zip(self.inclusive_min, self.exclusive_max, strict=True)
            )
            object.__setattr__(self, "_shape", cached)
        return cached

    def contains(self, index: tuple[int, ...]) -> bool:
        """Whether the literal coordinate `index` lies inside this domain.

        Coordinates are literal, not NumPy-style offsets: a negative value is
        the coordinate itself, valid only if the domain's bounds include it.
        A tuple of the wrong length is simply not contained (returns `False`).
        """
        if len(index) != self.ndim:
            return False
        return all(
            lo <= idx < hi
            for lo, hi, idx in zip(self.inclusive_min, self.exclusive_max, index, strict=True)
        )

    def contains_domain(self, other: IndexDomain) -> bool:
        """Whether every coordinate of `other` lies inside this domain.

        An empty `other` within this domain's bounds is contained. A rank
        mismatch returns `False` rather than raising.
        """
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
        """Return the overlap of this domain with `other`, or `None` if they are disjoint.

        Raises
        ------
        ValueError
            If the two domains have different ranks.
        """
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
        """Return this domain shifted by `offset` per dimension; the shape is unchanged.

        Offsets may be negative, and the result may have a negative origin.

        Raises
        ------
        ValueError
            If `offset` does not have one entry per dimension.
        """
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

        Indices are absolute coordinates, not NumPy-style offsets: `-3` names
        the coordinate `-3`, and is out of bounds unless the domain contains it.
        Integer indices produce a length-1 extent. Strided slices are not
        supported — use `IndexTransform` for strides.

        Raises
        ------
        BoundsCheckError
            If a bound lies outside this domain. A slice bound used to be
            clamped instead, so `narrow(slice(-3, None))` on `[0, 10)` quietly
            returned the whole axis — reading as the NumPy spelling of "the last
            three" and answering with something else — and `narrow(slice(20,
            30))` returned a domain its own parent did not contain. The rest of
            the algebra states no clamping and no negative wrapping as an
            invariant and enforces it; this is the one place that did not.
        """
        normalized = _normalize_selection(selection, self.ndim)
        new_inclusive_min: list[int] = []
        new_exclusive_max: list[int] = []
        for dim_idx, (sel, dim_lo, dim_hi) in enumerate(
            zip(normalized, self.inclusive_min, self.exclusive_max, strict=True)
        ):
            if isinstance(sel, int):
                if sel < dim_lo or sel >= dim_hi:
                    raise BoundsCheckError(
                        f"index {sel} is out of bounds for dimension {dim_idx} "
                        f"with domain [{dim_lo}, {dim_hi})"
                    )
                new_inclusive_min.append(sel)
                new_exclusive_max.append(sel + 1)
            else:
                start, stop, step = sel.start, sel.stop, sel.step
                if step is not None and step != 1:
                    raise ValueError(
                        "IndexDomain.narrow only supports step=1 slices. "
                        f"Got step={step}. Use IndexTransform for strided access."
                    )
                abs_start = dim_lo if start is None else start
                abs_stop = dim_hi if stop is None else stop
                for bound, name in ((abs_start, "start"), (abs_stop, "stop")):
                    if bound < dim_lo or bound > dim_hi:
                        raise BoundsCheckError(
                            f"slice {name} {bound} is out of bounds for dimension "
                            f"{dim_idx} with domain [{dim_lo}, {dim_hi}); indices "
                            f"here are absolute coordinates, so they are neither "
                            f"clamped to the domain nor counted from its end"
                        )
                # An empty interval is legal; a reversed one is the same request
                # spelled backwards, and reads as empty rather than as an error.
                new_inclusive_min.append(abs_start)
                new_exclusive_max.append(max(abs_stop, abs_start))
        return IndexDomain(
            inclusive_min=tuple(new_inclusive_min),
            exclusive_max=tuple(new_exclusive_max),
        )

    # -- serialization ------------------------------------------------------

    def to_json(self) -> IndexDomainJSON:
        """Convert to the canonical ndsel JSON representation.

        Examples
        --------
        >>> IndexDomain(inclusive_min=(0,), exclusive_max=(3,)).to_json()
        {'input_inclusive_min': [0], 'input_exclusive_max': [3], 'input_labels': ['']}
        """
        from zarr_indexing._wire import emit_labels

        return {
            "input_inclusive_min": list(self.inclusive_min),
            "input_exclusive_max": list(self.exclusive_max),
            "input_labels": emit_labels(self.labels, self.ndim),
        }

    @classmethod
    def from_json(cls, data: IndexDomainJSON) -> IndexDomain:
        """Construct from the canonical ndsel JSON representation.

        The document is validated by the message layer first, exactly as a
        transform body is. Reading the keys directly would be a second,
        undefended way into the same objects: `int(value)` alone accepts
        `3.9`, `"3"` and `True`, and each of those builds a domain that is
        not the document's.

        Examples
        --------
        >>> domain = IndexDomain.from_json(
        ...     {"input_inclusive_min": [1], "input_exclusive_max": [4], "input_labels": [""]}
        ... )
        >>> (domain.inclusive_min, domain.exclusive_max, domain.shape)
        ((1,), (4,), (3,))
        >>> IndexDomain.from_json(domain.to_json()) == domain
        True
        """
        from zarr_indexing._wire import lower_bound, lower_labels
        from zarr_indexing.messages import NdselError, normalize_ndsel

        # The annotation says what a well-formed caller passes; this is a parser
        # of documents that arrive from elsewhere, so the shape is checked
        # rather than assumed.
        if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise NdselError("invalid_json", f"an index domain must be a JSON object, got {data!r}")
        body = normalize_ndsel({**data, "kind": "transform"})
        return cls(
            inclusive_min=tuple(
                lower_bound(b, f"input_inclusive_min[{i}]")
                for i, b in enumerate(body["input_inclusive_min"])
            ),
            exclusive_max=tuple(
                lower_bound(b, f"input_exclusive_max[{i}]")
                for i, b in enumerate(body["input_exclusive_max"])
            ),
            labels=lower_labels(body["input_labels"]),
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
