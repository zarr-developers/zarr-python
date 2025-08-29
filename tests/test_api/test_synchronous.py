from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import pytest
from numpydoc.docscrape import NumpyDocString

import zarr
from zarr.api import asynchronous, synchronous

if TYPE_CHECKING:
    from collections.abc import Callable

MATCHED_EXPORT_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(set(synchronous.__all__) | set(asynchronous.__all__))
)
"""A sorted tuple of names that are exported by both the sync and async APIs."""

MATCHED_CALLABLE_NAMES: Final[tuple[str, ...]] = tuple(
    x for x in MATCHED_EXPORT_NAMES if callable(getattr(synchronous, x))
)
"""A sorted tuple of callable names that are exported by both the sync and async APIs."""


@pytest.mark.parametrize("callable_name", MATCHED_CALLABLE_NAMES)
def test_docstring_match(callable_name: str) -> None:
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


@pytest.mark.parametrize(
    ("parameter_name", "array_creation_routines"),
    [
        (
            ("store", "path"),
            (
                asynchronous.create_array,
                synchronous.create_array,
                asynchronous.create_group,
                synchronous.create_group,
                zarr.AsyncGroup.create_array,
                zarr.Group.create_array,
            ),
        ),
        (
            (
                "store",
                "path",
            ),
            (
                asynchronous.create,
                synchronous.create,
                zarr.Group.create,
                zarr.AsyncArray.create,
                zarr.Array.create,
            ),
        ),
        (
            (
                (
                    "filters",
                    "codecs",
                    "compressors",
                    "compressor",
                    "chunks",
                    "shape",
                    "dtype",
                    "shardsfill_value",
                )
            ),
            (
                asynchronous.create,
                synchronous.create,
                asynchronous.create_array,
                synchronous.create_array,
                zarr.AsyncGroup.create_array,
                zarr.Group.create_array,
                zarr.AsyncGroup.create_dataset,
                zarr.Group.create_dataset,
            ),
        ),
    ],
    ids=str,
)
def test_docstring_consistent_parameters(
    parameter_name: str, array_creation_routines: tuple[Callable[[Any], Any], ...]
) -> None:
    """
    Tests that array and group creation routines document the same parameters consistently.
    This test inspects the docstrings of sets of callables and generates two dicts:

    - a dict where the keys are parameter descriptions and the values are the names of the routines with those
    descriptions
    - a dict where the keys are parameter types and the values are the names of the routines with those types

    If each dict has just 1 value, then the parameter description and type in the docstring must be
    identical across different routines. But if these dicts have multiple values, then there must be
    routines that use the same parameter but document it differently, which will trigger a test failure.
    """
    descs: dict[tuple[str, ...], tuple[str, ...]] = {}
    types: dict[str, tuple[str, ...]] = {}
    for routine in array_creation_routines:
        key = f"{routine.__module__}.{routine.__qualname__}"
        docstring = NumpyDocString(routine.__doc__)
        param_dict = {d.name: d for d in docstring["Parameters"]}
        if parameter_name in param_dict:
            val = param_dict[parameter_name]
            if tuple(val.desc) in descs:
                descs[tuple(val.desc)] = descs[tuple(val.desc)] + (key,)
            else:
                descs[tuple(val.desc)] = (key,)
            if val.type in types:
                types[val.type] = types[val.type] + (key,)
            else:
                types[val.type] = (key,)
    assert len(descs) <= 1
    assert len(types) <= 1
