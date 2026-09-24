"""Explicit eager indexing for consumers such as Dask's ``from_array``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zarr_indexing.lazy_array import LazyArray


class EagerArrayAdapter:
    """Expose an eager array interface over a lazy view.

    Constructing the adapter does not read source values. Indexing materializes
    the selected view in fresh memory, using its existing reader and partitioning.
    Tokenization delegates to the view's source-token contract and may read data.

    Examples
    --------
    >>> import numpy as np
    >>> from zarr_indexing import LazyArray
    >>> view = LazyArray(np.arange(8))[1::2]
    >>> EagerArrayAdapter(view)[1:3]
    array([3, 5])
    """

    __slots__ = ("_view",)

    def __init__(self, view: LazyArray) -> None:
        self._view = view

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the selected view."""
        return self._view.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the selected view."""
        return self._view.ndim

    @property
    def dtype(self) -> Any:
        """Data type reported by the source."""
        return self._view.dtype

    def __getitem__(self, selection: Any) -> Any:
        """Read a basic selection into fresh memory."""
        return self._view[selection].result()

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        """Materialize using the view's NumPy conversion contract."""
        return self._view.__array__(dtype=dtype, copy=copy)

    def __dask_tokenize__(self) -> tuple[Any, ...]:
        """Distinguish eager tasks while preserving the source identity contract."""
        return (type(self).__qualname__, self._view.__dask_tokenize__())
