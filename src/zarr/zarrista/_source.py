"""Adapt a `zarrista.Array` to what `zarr_indexing.LazyArray` reads from.

`zarrista.Array.__getitem__` accepts exactly one selection dialect: integers,
step-1 slices, and `Ellipsis`, ndim-preserving. That is precisely the contract
`zarr_indexing.UnitStepReader` targets — it decomposes any transform into the
smallest enclosing ascending unit-step slab plus a residual applied in memory —
so the two compose with no translation beyond `shape`/`dtype` plumbing.

Wrapping the array this way is what gives the engine the full NumPy dialect
(orthogonal, vectorized, masks, negative steps, composition) over a backend
that natively offers only boxes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

__all__ = [
    "AsyncZarristaSource",
    "ZarristaSource",
    "await_on",
    "run_off_loop",
    "tensor_to_numpy",
]


def tensor_to_numpy(decoded: Any) -> npt.NDArray[Any]:
    """Convert a zarrista decoded tensor to a NumPy array.

    `zarrista.Tensor` is a union of four layouts. `FixedLengthTensor` and
    `VariableLengthTensor` both export via `to_numpy()`. The two `Optional*`
    layouts carry a validity mask and convert to `numpy.ma.MaskedArray`, which
    has no zarr-python equivalent, so they are refused rather than silently
    losing the mask.
    """
    type_name = type(decoded).__name__
    if type_name in ("OptionalFixedLengthTensor", "OptionalVariableLengthTensor"):
        raise NotImplementedError(
            f"zarrista returned a {type_name}; masked layouts have no zarr-python "
            "equivalent, so this array cannot be served by the zarrista engine"
        )
    return cast("npt.NDArray[Any]", decoded.to_numpy())


class ZarristaSource:
    """A `zarrista.Array` behind the narrow surface `LazyArray` reads from."""

    __slots__ = ("_arr", "_dtype")

    def __init__(self, zarrista_array: Any, dtype: np.dtype[Any]) -> None:
        self._arr = zarrista_array
        # Taken from zarr's own metadata rather than `zarrista.Array.dtype`,
        # which is a zarrista `DataType`, not a NumPy one.
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        # zarrista reports shape as a list.
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __getitem__(self, key: Any) -> npt.NDArray[Any]:
        return tensor_to_numpy(self._arr.retrieve_array_subset(key))


class AsyncZarristaSource:
    """`ZarristaSource` over an `AsyncArray`, driven from a worker thread.

    `LazyArray` is synchronous, so the async engine resolves a selection on a
    worker thread (see `run_off_loop`). Each read is handed back to the event
    loop that owns the `zarrista.AsyncArray` with
    `asyncio.run_coroutine_threadsafe`, and the worker blocks on the result.

    Going through `zarr.core.sync.sync()` instead does not work: it would be
    called from within a coroutine already running on that loop, and zarrista's
    pyo3 futures are bound to the loop that awaits them, so the read would
    either deadlock or fail with a cross-loop future error.
    """

    __slots__ = ("_arr", "_dtype", "_loop")

    def __init__(
        self, zarrista_array: Any, dtype: np.dtype[Any], loop: asyncio.AbstractEventLoop
    ) -> None:
        self._arr = zarrista_array
        self._dtype = dtype
        self._loop = loop

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __getitem__(self, key: Any) -> npt.NDArray[Any]:
        return tensor_to_numpy(await_on(self._loop, lambda: self._arr.retrieve_array_subset(key)))


def await_on(loop: asyncio.AbstractEventLoop, make_awaitable: Callable[[], Any]) -> Any:
    """Call `make_awaitable` on `loop` and block the worker thread on its result.

    A *factory*, not an awaitable: zarrista's pyo3 futures bind to the running
    loop at the moment they are created, so calling
    `arr.retrieve_array_subset(...)` on the worker thread raises "no running
    event loop". Deferring the call into a coroutine that `loop` runs means the
    future is both created and awaited there.
    """

    async def call() -> Any:
        return await make_awaitable()

    return asyncio.run_coroutine_threadsafe(call(), loop).result()


async def run_off_loop[T](fn: Callable[[], T]) -> T:
    """Run a blocking `LazyArray` resolution on a worker thread.

    The caller's loop stays free to service the reads `fn` issues back to it.
    """
    return await asyncio.to_thread(fn)
