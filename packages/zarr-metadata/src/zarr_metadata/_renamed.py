"""Scopes this package renamed, which keep their old names for now with a `DeprecationWarning`."""

import warnings
from collections.abc import Mapping


def renamed_scope(module: str, name: str, renames: Mapping[str, tuple[str, object]]) -> object:
    """The scope that `name`, an old name in `renames`, refers to, with a `DeprecationWarning`.

    `renames` maps each old name to the scope's name now and the scope. A
    module that exported the old names calls this from its `__getattr__`,
    so an import of an old name keeps working. Raises `AttributeError`, as
    attribute lookup does, when `name` is not an old name.
    """
    if name not in renames:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    new, scope = renames[name]
    warnings.warn(
        f"`{name}` is deprecated and will be removed; use `{new}`, the same scope",
        DeprecationWarning,
        stacklevel=3,
    )
    return scope


__all__ = ["renamed_scope"]
