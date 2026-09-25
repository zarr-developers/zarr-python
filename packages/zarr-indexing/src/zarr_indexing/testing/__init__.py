"""Test support for projects whose arrays are read through `LazyArray`.

A Hypothesis state machine that composes indexing steps onto a `LazyArray`
wrapping your array and checks every step against NumPy
([`stateful`][zarr_indexing.testing.stateful]), and the selection strategies it
draws from, exported on their own for a project that has its own harness
([`strategies`][zarr_indexing.testing.strategies]).

```python
from zarr_indexing.testing import ChainedIndexingStateMachine, state_machine_test

class MyArrayIndexing(ChainedIndexingStateMachine):
    def make_source(self, data):
        array = my_format.create(shape=data.shape, dtype=data.dtype)
        array[:] = data
        return array

TestMyArrayIndexing = state_machine_test(MyArrayIndexing)
```

This subpackage needs `hypothesis`, which the rest of `zarr_indexing` does not:
install it with the `testing` extra (`pip install zarr-indexing[testing]`).
"""

from zarr_indexing.testing.stateful import (
    DEFAULT_DATA,
    DEFAULT_PARTITIONINGS,
    DEFAULT_SETTINGS,
    ChainedIndexingStateMachine,
    apply_selection,
    outer_selection,
    repartition,
    state_machine_test,
)
from zarr_indexing.testing.strategies import (
    basic_selections,
    masks,
    orthogonal_selections,
    slice_selections,
    vectorized_selections,
)

__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_PARTITIONINGS",
    "DEFAULT_SETTINGS",
    "ChainedIndexingStateMachine",
    "apply_selection",
    "basic_selections",
    "masks",
    "orthogonal_selections",
    "outer_selection",
    "repartition",
    "slice_selections",
    "state_machine_test",
    "vectorized_selections",
]
