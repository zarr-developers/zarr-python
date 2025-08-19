from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pytest
from numpydoc.docscrape import NumpyDocString

from zarr.api import asynchronous, synchronous


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    desc: tuple[str, ...]
    type: str


all_docstrings: dict[str, NumpyDocString] = {}

for name in asynchronous.__all__:
    obj = getattr(asynchronous, name)
    if callable(obj) and obj.__doc__ is not None:
        all_docstrings[f"asynchronous.{name}"] = NumpyDocString(obj.__doc__)

for name in synchronous.__all__:
    obj = getattr(synchronous, name)
    if callable(obj) and obj.__doc__ is not None:
        all_docstrings[f"synchronous.{name}"] = NumpyDocString(obj.__doc__)

MATCHED_EXPORT_NAMES: Final[tuple[str, ...]] = tuple(
    sorted(set(synchronous.__all__) | set(asynchronous.__all__))
)
MATCHED_CALLABLE_NAMES: Final[tuple[str, ...]] = tuple(
    x for x in MATCHED_EXPORT_NAMES if callable(getattr(synchronous, x))
)


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
    "parameter_name",
    [
        "store",
        "path",
        "filters",
        "codecs",
        "compressors",
        "compressor",
        "chunks",
        "shape",
        "dtype",
        "data_type",
        "fill_value",
    ],
)
def test_docstring_consistent_parameters(parameter_name: str) -> None:
    """
    Tests that callable exports from ``zarr.api.synchronous`` and ``zarr.api.asynchronous``
    document the same parameters consistently.
    """
    matches: dict[str, Param] = {}
    for name in all_docstrings:
        docstring = all_docstrings[name]
        param_dict = {d.name: d for d in docstring["Parameters"]}
        if parameter_name in param_dict:
            val = param_dict[parameter_name]
            matches[name] = Param(name=val.name, desc=tuple(val.desc), type=val.type)
    assert len(set(matches.values())) == 1
