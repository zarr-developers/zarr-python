from __future__ import annotations

from typing import Final

import pytest
from numpydoc.docscrape import NumpyDocString

from zarr.api import asynchronous, synchronous

MATCHED_EXPORT_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(set(synchronous.__all__) | set(asynchronous.__all__))
)
MATCHED_CALLABLE_NAMES: Final[tuple[str, ...]] = tuple(
    x for x in MATCHED_EXPORT_NAMES if callable(getattr(synchronous, x))
)


@pytest.mark.parametrize("callable_name", MATCHED_CALLABLE_NAMES)
def test_create_docstrings(callable_name: str) -> None:
    """
    Tests that the docstrings for the sync and async define identical parameters.
    """
    callable_a = getattr(synchronous, callable_name)
    callable_b = getattr(asynchronous, callable_name)
    if callable_a.__doc__ is None:
        assert callable_b.__doc__ is None
    else:
        params_a = NumpyDocString(callable_a.__doc__)["Parameters"]
        params_b = NumpyDocString(callable_b.__doc__)["Parameters"]
        assert params_a == params_b
