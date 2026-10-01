"""Deprecations are declared once and the docs are generated from the declarations.

The deprecation policy in the contributing guide has two ways to warn: a `@deprecated`
decorator on a function, class, or property, or an entry in `zarr._deprecations` for
everything else. The Deprecations page is built from those two sources, so a warning
raised any other way would be an undocumented deprecation. These tests keep the
source honest: every deprecation warning in `src/zarr` goes through one of the two
mechanisms, and every decorator message is a string the docs build can read.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any

import pytest

import zarr
from zarr import _deprecations
from zarr._deprecations import DEPRECATIONS

SRC = Path(zarr.__file__).parent

# Warning classes that announce a deprecation. Raising one of these directly from the
# source, rather than through a declaration, leaves the deprecation off the docs.
DEPRECATION_CATEGORIES = {
    "DeprecationWarning",
    "FutureWarning",
    "PendingDeprecationWarning",
    "ZarrDeprecationWarning",
    "ZarrFutureWarning",
}

# Fields that fill each declaration's message template. A declaration with no entry
# here fails `test_every_declaration_has_an_example`, so a new template is always
# exercised once.
EXAMPLE_FIELDS: dict[str, dict[str, Any]] = {
    "data-type-validation-error-import": {"module": "zarr.dtype"},
    "storage-default-compressor": {},
    "v2-metadata-chunk-grid": {},
    "codec-enum-member": {"cls": "BloscShuffle", "name": "shuffle", "value": "shuffle"},
    "codec-enum-parameter": {"codec": "BloscCodec", "param": "shuffle"},
}


def _name(node: ast.expr) -> str | None:
    """The bare name of a `Name` or `Attribute` node, e.g. `ZarrDeprecationWarning`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _warn_category(call: ast.Call) -> str | None:
    """The category a `warnings.warn(...)` call passes, or None if it is not one."""
    if _name(call.func) != "warn":
        return None
    for keyword in call.keywords:
        if keyword.arg == "category":
            return _name(keyword.value)
    if len(call.args) >= 2:
        return _name(call.args[1])
    return None


def _source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name != "_deprecations.py")


def test_every_declaration_has_an_example() -> None:
    assert set(EXAMPLE_FIELDS) == set(DEPRECATIONS)


@pytest.mark.parametrize("key", sorted(DEPRECATIONS))
def test_warn_emits_the_declared_warning(key: str) -> None:
    """`warn` raises the declared category with the formatted message, pointing at the caller."""
    deprecation = DEPRECATIONS[key]
    fields = EXAMPLE_FIELDS[key]

    def deprecated_api() -> None:
        # Stands in for the function that handles the deprecated usage; the warning
        # must point at its caller, which is this test.
        _deprecations.warn(key, **fields)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_api()
    assert len(caught) == 1
    (record,) = caught
    assert record.category is deprecation.category
    assert str(record.message) == deprecation.message.format(**fields)
    assert record.filename == __file__


def test_warn_unknown_key() -> None:
    with pytest.raises(KeyError):
        _deprecations.warn("not-a-deprecation")


def test_deprecation_warnings_are_declared() -> None:
    """No source file raises a deprecation category with `warnings.warn` directly."""
    undeclared = [
        f"{path.relative_to(SRC)}:{node.lineno} raises {category} directly"
        for path in _source_files()
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Call)
        and (category := _warn_category(node)) in DEPRECATION_CATEGORIES
    ]
    assert undeclared == [], (
        "Deprecation warnings must go through `@deprecated` or `zarr._deprecations.warn`, "
        "so the Deprecations page lists them:\n" + "\n".join(undeclared)
    )


def test_deprecated_decorator_messages_are_static() -> None:
    """Every `@deprecated(...)` message is a string literal the docs build can read."""
    dynamic = [
        f"{path.relative_to(SRC)}:{decorator.lineno} on {node.name}"
        for path in _source_files()
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and _name(decorator.func) == "deprecated"
        and not (
            decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        )
    ]
    assert dynamic == [], (
        "The docs render the `@deprecated` message from the source, so it must be a "
        "string literal:\n" + "\n".join(dynamic)
    )
