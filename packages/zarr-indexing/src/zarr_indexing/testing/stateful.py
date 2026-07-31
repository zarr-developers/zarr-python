"""A stateful property test for indexing an array through `LazyArray`.

`ChainedIndexingStateMachine` composes indexing steps onto a `LazyArray`
wrapping *your* array — `lazy[...]`, `lazy.oindex[...]`, `lazy.vindex[...]`,
each step applied to the view the last one produced — while applying the same
steps to a NumPy array holding the same values. After every step the view must
still agree with that model three ways: its shape, its `result()`, and the
assembly of its `parts()`.

Point it at an array by subclassing and overriding `make_source`:

```python
from zarr_indexing.testing import ChainedIndexingStateMachine, state_machine_test

class MyArrayIndexing(ChainedIndexingStateMachine):
    def make_source(self, data):
        array = my_format.create(shape=data.shape, dtype=data.dtype)
        array[:] = data
        return array

TestMyArrayIndexing = state_machine_test(MyArrayIndexing)
```

`data`, `partitionings`, and `supports` are class attributes; override any of
them to widen or narrow what is drawn. The base class needs no `make_source` at
all — left alone it wraps the NumPy array itself, which is a useful smoke test
of this package but says nothing about yours.

What it is checking
-------------------
The parts invariant is the one with teeth.
[`Partition`][zarr_indexing.lazy_array.Partition] documents
`out[part.out_selection] = part.array.result()` as the assembly procedure, so
this checks that literally: a part's values must arrive at exactly the shape its
`out_selection` addresses — not merely a shape that broadcasts into it — land
there, and cover the view once. Checking through `result()` alone would prove
only that `result()` is self-consistent.

The `declare_support` rule draws an
[`IndexingSupport`][zarr_indexing.support.IndexingSupport] level and applies it
to the view, so the level a source is read at becomes part of the chain. This is
the property a source author most needs pinned: the level chooses how much data
crosses the boundary, never what the read means, so a chain read at `BASIC` must
answer exactly what the same chain read at `VECTORIZED` answers. `BASIC` is
always honest — at that level `LazyArray` sends nothing but integers and slices
through `__getitem__` — so it is drawn for every source; the levels beyond it are
only as trustworthy as the accessors they name, which is why `supports` defaults
to the declared or inferred level alone and widening it is your call.

Requires the `testing` extra (`pip install zarr-indexing[testing]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from zarr_indexing.lazy_array import LazyArray
from zarr_indexing.support import IndexingSupport
from zarr_indexing.testing.strategies import (
    basic_selections,
    orthogonal_selections,
    slice_selections,
    vectorized_selections,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_indexing.boundary import SelectionMode

__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_PARTITIONINGS",
    "DEFAULT_SETTINGS",
    "ChainedIndexingStateMachine",
    "apply_selection",
    "outer_selection",
    "state_machine_test",
]

DEFAULT_DATA = np.arange(7 * 5 * 4, dtype=np.int64).reshape(7, 5, 4)
"""The values the source holds by default: distinct, so a misplaced cell shows."""

DEFAULT_PARTITIONINGS: tuple[Any, ...] = (
    None,
    (2, 2, 2),
    (7, 5, 4),
    (3, 2, 3),
    ((3, 3, 1), (2, 2, 1), (3, 1)),
    (4, 3, 3),
)
"""Partitionings to read under: a single whole-array part, uniform boxes of
several shapes (some of which do not divide the extent), and explicit per-axis
sizes. Boxes that straddle whatever the source declares are deliberate — they
cost extra I/O but must not change an answer."""

DEFAULT_SETTINGS = settings(
    max_examples=250,
    stateful_step_count=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.too_slow,
    ],
)
"""Enough examples to find a defect reachable only through a narrow chain, at a
few seconds per run when there is nothing to find. Every step is followed by
checks that each materialize the whole view, so the budget buys examples rather
than long chains — a chain runs out of axes to index within a few steps anyway.

`filter_too_much` is suppressed because a chain that reaches a rank-0 view
leaves only `repartition` enabled, so a run that opens there is discarded."""


# --------------------------------------------------------------------------- #
# The NumPy model
# --------------------------------------------------------------------------- #


def outer_selection(array: Any, selection: Sequence[Any]) -> Any:
    """Apply an orthogonal selection to a NumPy array: the outer product of its axes.

    NumPy has no operator for this, so the model is built from `numpy.ix_`.
    Scalar integers are basic indices — NumPy applies them first and drops the
    axis — so they are peeled off before the outer product is formed.
    """

    def is_scalar(sel: Any) -> bool:
        return isinstance(sel, (int, np.integer)) and not isinstance(sel, bool)

    scalars = tuple(sel if is_scalar(sel) else slice(None) for sel in selection)
    reduced = array[scalars]
    axes = [
        np.arange(size)[sel]
        for size, sel in zip(reduced.shape, [s for s in selection if not is_scalar(s)], strict=True)
    ]
    if len(axes) == 0:
        return reduced
    return reduced[np.ix_(*axes)]


def apply_selection(array: Any, selection: tuple[Any, ...], mode: SelectionMode) -> Any:
    """Apply a selection to a NumPy array in the given mode — the model a view is checked against.

    NumPy's own semantics *are* basic and vectorized indexing, so only the
    orthogonal mode needs building (see `outer_selection`).
    """
    if mode == "orthogonal":
        return outer_selection(array, selection)
    return array[selection]


def state_machine_test(
    machine: type[RuleBasedStateMachine], *, config: settings = DEFAULT_SETTINGS
) -> Any:
    """The pytest-collectable `TestCase` for a machine, with settings applied.

    Hypothesis builds a fresh `TestCase` per state-machine class, so settings
    set on a base class do not reach a subclass's; this applies them where they
    land. Assign the result to a module-level name beginning with `Test`.
    """
    case = machine.TestCase
    case.settings = config
    return case


# --------------------------------------------------------------------------- #
# The machine
# --------------------------------------------------------------------------- #

_SOURCE = "_zarr_indexing_cached_source"


class ChainedIndexingStateMachine(RuleBasedStateMachine):
    """Indexing steps composed onto one `LazyArray`, against NumPy as the model.

    Subclass and override `make_source` to point it at your own array. See the
    module docstring for the shape of that subclass and for what the invariants
    check.

    Attributes
    ----------
    data
        The values the source holds, and the model every step is checked
        against. Any shape and dtype NumPy supports; every axis must be
        non-empty.
    partitionings
        Drawn once per run, before any indexing: `with_parts` is a pure setter
        that carries through composition untouched and is read only when a view
        resolves, so choosing it up front reaches the same states choosing it
        mid-chain does, and spends the whole step budget on indexing.
    supports
        `IndexingSupport` levels `declare_support` may draw, beyond `BASIC`,
        which is always drawn. `None` means the level `LazyArray` detects for
        the source. Name a level here only if the source really serves it — a
        level that overstates the source is a bug in the test, not in the code.
    """

    data: ClassVar[Any] = DEFAULT_DATA
    partitionings: ClassVar[Sequence[Any]] = DEFAULT_PARTITIONINGS
    supports: ClassVar[Sequence[IndexingSupport] | None] = None

    def make_source(self, data: Any) -> Any:
        """Build the array under test, holding `data`.

        Called once per machine class and cached, not once per example: an
        example is cheap and a source may not be. The machine only ever reads,
        so the same object serves every run — but it must therefore not be
        mutated by anything else while the test runs.

        The default returns `data` itself, so an unsubclassed machine exercises
        this package against NumPy.
        """
        return data

    def __init__(self) -> None:
        super().__init__()
        cls = type(self)
        self.model: Any = np.asarray(cls.data)
        source = cls.__dict__.get(_SOURCE)
        if source is None:
            source = self.make_source(self.model)
            setattr(cls, _SOURCE, source)
        self.view = LazyArray(source)
        self.levels: tuple[IndexingSupport, ...] = _support_levels(self.view, cls.supports)
        # Genuine fancy-after-fancy raises `NotImplementedError` by design, so a
        # chain carries at most one fancy step — in either order relative to the
        # basic ones.
        self.fancy_used = False
        self.chain: list[tuple[str, Any]] = []

    def _indexable(self) -> bool:
        """Whether there is anything left to index.

        A rank-0 or empty view takes no further step — NumPy would reject one
        too — so the chain ends there, and the invariants keep checking.
        """
        return self.model.ndim > 0 and self.model.size > 0

    def _step(self, mode: SelectionMode, selection: tuple[Any, ...], *, fancy: bool) -> None:
        self.chain.append((mode, selection))
        self.model = apply_selection(self.model, selection, mode)
        if mode == "basic":
            self.view = self.view.lazy[selection]
        elif mode == "orthogonal":
            self.view = self.view.lazy.oindex[selection]
        else:
            self.view = self.view.lazy.vindex[selection]
        self.fancy_used = self.fancy_used or fancy

    # -- rules --------------------------------------------------------------

    @initialize(data=st.data())
    def choose_partitioning(self, data: st.DataObject) -> None:
        """Fix how the read is broken up, before any indexing."""
        parts = data.draw(st.sampled_from(list(type(self).partitionings)))
        self.view = self.view.with_parts(parts)
        self.chain.append(("parts", parts))

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def basic(self, data: st.DataObject) -> None:
        self._step("basic", data.draw(basic_selections(self.model.shape)), fancy=False)

    @precondition(lambda self: self._indexable() and not self.fancy_used)
    @rule(data=st.data())
    def orthogonal(self, data: st.DataObject) -> None:
        self._step("orthogonal", data.draw(orthogonal_selections(self.model.shape)), fancy=True)

    @precondition(lambda self: self._indexable() and not self.fancy_used)
    @rule(data=st.data())
    def vectorized(self, data: st.DataObject) -> None:
        self._step("vectorized", data.draw(vectorized_selections(self.model.shape)), fancy=True)

    @precondition(lambda self: self._indexable())
    @rule(data=st.data())
    def slices_only(self, data: st.DataObject) -> None:
        """An `oindex` step carrying only slices is not a fancy selection.

        It narrows the view's own axes and composes like basic indexing, so it
        is legal after a fancy step, where genuine coordinates are not.
        """
        self._step("orthogonal", data.draw(slice_selections(self.model.shape)), fancy=False)

    @rule(data=st.data())
    def declare_support(self, data: st.DataObject) -> None:
        """Read the rest of the chain at a different level. This must change no answer."""
        level = data.draw(st.sampled_from(list(self.levels)))
        self.view = self.view.with_indexing_support(level)
        self.chain.append(("support", level))

    @precondition(lambda self: not self._indexable())
    @rule(data=st.data())
    def repartition(self, data: st.DataObject) -> None:
        """Re-box a chain that has run out of axes to index.

        Something must stay enabled once the view is rank-0 or empty, or
        Hypothesis has no move to make and abandons the run. Re-boxing is the
        useful thing to do there: it changes nothing the invariants may see, and
        a rank-0 view read through every partitioning is exactly the state a
        collapsed correlated selection reaches.
        """
        parts = data.draw(st.sampled_from(list(type(self).partitionings)))
        self.view = self.view.with_parts(parts)
        self.chain.append(("parts", parts))

    # -- invariants ---------------------------------------------------------

    @invariant()
    def the_view_has_the_models_shape(self) -> None:
        assert self.view.shape == self.model.shape, self.chain

    @invariant()
    def result_matches_the_model(self) -> None:
        np.testing.assert_array_equal(
            np.asarray(self.view.result()), self.model, err_msg=str(self.chain)
        )

    @invariant()
    def parts_tile_the_view(self) -> None:
        """The documented assembly, run literally.

        Every part's values arrive at exactly the shape its `out_selection`
        addresses — not merely a shape that broadcasts into it — and together
        the parts cover the view once.
        """
        assembled = np.zeros(self.view.shape, dtype=self.view.dtype)
        hits = np.zeros(self.view.shape, dtype=np.int64)
        for part in self.view.parts():
            value = np.asarray(part.array.result())
            assert value.shape == assembled[part.out_selection].shape, (
                f"part {part.base_coords} carries {value.shape} for an out_selection "
                f"addressing {assembled[part.out_selection].shape}: {self.chain}"
            )
            assembled[part.out_selection] = value
            np.add.at(hits, part.out_selection, 1)

        np.testing.assert_array_equal(assembled, self.model, err_msg=str(self.chain))
        np.testing.assert_array_equal(
            hits, np.ones(self.view.shape, dtype=np.int64), err_msg=str(self.chain)
        )


def _support_levels(
    view: LazyArray, declared: Sequence[IndexingSupport] | None
) -> tuple[IndexingSupport, ...]:
    """The levels `declare_support` draws from, `BASIC` always among them.

    `IndexingSupport` is deliberately unordered — `OUTER_1VECTOR` sits between
    `BASIC` and `OUTER` in capability — so this is a set of levels the source is
    asserted to serve, not a ceiling to count down from.
    """
    levels = list(declared) if declared is not None else [view.indexing_support]
    if IndexingSupport.BASIC not in levels:
        levels.insert(0, IndexingSupport.BASIC)
    return tuple(dict.fromkeys(levels))
