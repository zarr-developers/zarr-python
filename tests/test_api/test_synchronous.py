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
def test_docstrings_match(callable_name: str) -> None:
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
        mismatch = []
        for idx, (a, b) in enumerate(zip(params_a, params_b, strict=False)):
            if a != b:
                mismatch.append((idx, (a, b)))
        assert mismatch == []


CREATE_ARRAY_ROUTINES: Final[tuple[Callable[..., Any], ...]] = (
    asynchronous.create_array,
    synchronous.create_array,
    zarr.AsyncGroup.create_array,
    zarr.Group.create_array,
    zarr.Group.create,
)
"""
Routines that share the `create_array` signature and must document it identically.
The legacy `zarr.create` documents a different signature (e.g. `chunks : int or tuple of ints`)
and is only checked for the parameters it genuinely shares, see the cases below.
"""

CREATE_ARRAY_EXEMPT_PARAMETERS: Final[frozenset[str]] = frozenset({"name"})
"""
Parameters that the `create_array` routines document differently on purpose:
`name` is relative to the store for the module-level functions and relative to the group
for the group methods.
"""


def _documented_parameters(routines: tuple[Callable[..., Any], ...]) -> tuple[str, ...]:
    """The sorted union of the parameter names documented by `routines`."""
    names: set[str] = set()
    for routine in routines:
        names.update(param.name for param in NumpyDocString(routine.__doc__)["Parameters"])
    return tuple(sorted(names))


def _consistency_cases() -> list[Any]:
    """One test case per (parameter, routine set), so every mismatch is reported separately."""
    groups: list[tuple[str, tuple[str, ...], tuple[Callable[..., Any], ...]]] = [
        (
            "store-path-create_array_group",
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
            "store-path-create",
            ("store", "path"),
            (asynchronous.create, synchronous.create, zarr.Group.create),
        ),
        (
            "create_array_variants",
            tuple(
                name
                for name in _documented_parameters(CREATE_ARRAY_ROUTINES)
                if name not in CREATE_ARRAY_EXEMPT_PARAMETERS
            ),
            CREATE_ARRAY_ROUTINES,
        ),
    ]
    return [
        pytest.param(name, routines, id=f"{group_id}-{name}")
        for group_id, names, routines in groups
        for name in names
    ]


@pytest.mark.parametrize(("parameter_name", "array_creation_routines"), _consistency_cases())
def test_docstring_consistent_parameters(
    parameter_name: str, array_creation_routines: tuple[Callable[..., Any], ...]
) -> None:
    """
    Tests that array and group creation routines document the same parameter consistently.
    This test inspects the docstrings of a set of callables and generates two dicts:

    - a dict where the keys are parameter descriptions and the values are the names of the routines with those
    descriptions
    - a dict where the keys are parameter types and the values are the names of the routines with those types

    If each dict has at most 1 value, then the parameter description and type in the docstring must be
    identical across different routines. But if these dicts have multiple values, then there must be
    routines that use the same parameter but document it differently, which will trigger a test failure.
    """
    descs: dict[tuple[str, ...], tuple[str, ...]] = {}
    types: dict[str, tuple[str, ...]] = {}
    for routine in array_creation_routines:
        key = f"{routine.__module__}.{routine.__qualname__}"
        param_dict = {d.name: d for d in NumpyDocString(routine.__doc__)["Parameters"]}
        if parameter_name in param_dict:
            val = param_dict[parameter_name]
            descs[tuple(val.desc)] = descs.get(tuple(val.desc), ()) + (key,)
            types[val.type] = types.get(val.type, ()) + (key,)
    assert len(descs) <= 1, f"parameter {parameter_name!r} has inconsistent descriptions: {descs}"
    assert len(types) <= 1, f"parameter {parameter_name!r} has inconsistent types: {types}"
