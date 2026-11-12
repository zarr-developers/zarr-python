"""Test that no module that holds types imports a module that holds a definition.

A definition of one kind can then import the types of another kind at run
time: a data type can name its default codec's object form, and a codec
can name the data types it accepts, without an import cycle.

Python runs a package's `__init__` before any module in it, and the
`__init__` of `zarr_metadata`, `zarr_metadata.v2` and `zarr_metadata.v3`
imports every definition. So each module is imported in a fresh
interpreter in which those three packages are empty stand-ins. Every other
package on the way runs its `__init__` for real, so a types module placed in
a package that exports definitions, such as `zarr_metadata.v3.codec`, fails.
"""

import importlib
import json
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import zarr_metadata
import zarr_metadata.v2._types
import zarr_metadata.v3._types
from zarr_metadata.definition import Definition

_STAND_INS = ("zarr_metadata", "zarr_metadata.v2", "zarr_metadata.v3")
"""The packages whose `__init__` imports every definition. The probe replaces each with an empty package."""

_DOCUMENT_TYPES_MODULES = (
    "zarr_metadata._common",
    "zarr_metadata._numpy",
    "zarr_metadata.v2._declaration_json",
    "zarr_metadata.v2.array",
    "zarr_metadata.v2.attributes",
    "zarr_metadata.v2.consolidated",
    "zarr_metadata.v2.group",
    "zarr_metadata.v3._common",
    "zarr_metadata.v3.array",
    "zarr_metadata.v3.group",
)
"""The modules outside the `_types` packages that hold types only: the documents' types, and the types they share with the extensions."""

TYPES_MODULES = (
    *_DOCUMENT_TYPES_MODULES,
    *(
        info.name
        for package in (zarr_metadata.v2._types, zarr_metadata.v3._types)
        for info in pkgutil.iter_modules(package.__path__, f"{package.__name__}.")
    ),
)
"""Every module that holds types only: the document modules, and each module of the extensions' `_types` packages."""

_PROBE = """
import importlib, json, sys, types

for name, path in json.loads(sys.argv[1]).items():
    package = types.ModuleType(name)
    package.__path__ = [path]
    sys.modules[name] = package
importlib.import_module(sys.argv[2])
print(json.dumps(sorted(
    name for name, module in sys.modules.items()
    if name.startswith("zarr_metadata") and getattr(module, "__file__", None) is not None
)))
"""
"""Import one module with the stand-ins in place, and print every `zarr_metadata` module that ran."""


def _holds_definition(module_name: str) -> bool:
    """Whether the module holds a definition, or a kind of definition, in its namespace."""
    module = importlib.import_module(module_name)
    return any(
        isinstance(value, Definition) or (isinstance(value, type) and issubclass(value, Definition))
        for value in vars(module).values()
    )


DEFINITION_MODULES = frozenset(
    info.name
    for info in pkgutil.walk_packages(zarr_metadata.__path__, "zarr_metadata.")
    if _holds_definition(info.name)
) | frozenset(name for name in _STAND_INS if _holds_definition(name))
"""Every module of the package that holds a definition or a kind of definition, including a package that re-exports one."""


def _loaded_by(module_name: str) -> frozenset[str]:
    """The `zarr_metadata` modules that run when `module_name` is imported in a fresh interpreter with the stand-ins in place."""
    root = Path(zarr_metadata.__path__[0])
    paths = {name: str(root.joinpath(*name.split(".")[1:])) for name in _STAND_INS}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(paths), module_name],
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(json.loads(result.stdout))


@pytest.mark.parametrize("module_name", TYPES_MODULES)
def test_types_module_loads_no_definition(module_name: str) -> None:
    loaded = _loaded_by(module_name)
    assert module_name in loaded
    assert sorted(loaded & DEFINITION_MODULES) == []


def test_probe_sees_a_definition_module() -> None:
    """The probe reports a definition module that an extension module imports, so an empty result above is not a probe that sees nothing."""
    assert "zarr_metadata.v3._definition" in _loaded_by("zarr_metadata.v3.codec.gzip")
