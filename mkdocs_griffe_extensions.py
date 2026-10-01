"""Griffe extensions used when building the API reference.

``DeprecatedExtension`` extends ``griffe-warnings-deprecated`` so that a
``@deprecated`` decorator (PEP 702) is rendered as a "Deprecated" admonition
for properties as well as functions and classes. Griffe turns a ``@property``
function into an ``Attribute`` before any function hook runs, and the
attribute keeps no decorator list, so the upstream extension never sees the
decorator; this subclass reads it from the AST node instead.

The deprecation policy in the contributing guide relies on this: the decorator
message is the single source of truth for the warning, the API reference, and
the deprecations index page. The page's second table comes from the declarations
in ``zarr._deprecations`` the same way.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from griffe_warnings_deprecated import WarningsDeprecatedExtension

if TYPE_CHECKING:
    from griffe import Attribute


def _deprecated_message(node: ast.AST) -> str | None:
    """Return the static message of a ``@deprecated(...)`` decorator on `node`, if any."""
    for decorator in getattr(node, "decorator_list", ()):
        if not isinstance(decorator, ast.Call) or not decorator.args:
            continue
        # The decorator is imported by name (``from typing_extensions import deprecated``)
        # or used through its module (``typing_extensions.deprecated``).
        func = decorator.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "deprecated":
            continue
        try:
            message = ast.literal_eval(decorator.args[0])
        except ValueError:
            return None
        return message if isinstance(message, str) else None
    return None


class DeprecatedExtension(WarningsDeprecatedExtension):
    """``griffe-warnings-deprecated`` plus support for deprecated properties."""

    def on_attribute_instance(self, *, node: ast.AST, attr: Attribute, **kwargs: Any) -> None:
        """Add the admonition to a property carrying ``@deprecated``."""
        if "property" not in attr.labels:
            return
        if message := _deprecated_message(node):
            attr.deprecated = message
            # _insert_message is typed for functions and classes; attributes carry
            # the same docstring machinery.
            self._insert_message(attr, message)  # type: ignore[arg-type]
            if self.label:
                attr.labels.add(self.label)


def deprecated_objects(package: str = "zarr") -> list[tuple[str, str]]:
    """Return ``(public path, message)`` for every public deprecated object in `package`.

    An object is named by the shortest public path that reaches it: ``zarr.Array.compressor``
    rather than ``zarr.core.array.Array.compressor`` when ``Array`` is in ``zarr.__all__``.
    Objects the package does not export keep their canonical path. Private objects
    (a ``_`` prefix anywhere in the path) are left out, as the deprecation policy does
    not cover them.
    """
    import griffe

    pkg = griffe.load(package, extensions=griffe.load_extensions(DeprecatedExtension))

    found: dict[str, str] = {}
    seen: set[str] = set()

    def walk(obj: griffe.Object) -> None:
        if obj.path in seen:
            return
        seen.add(obj.path)
        for member in obj.members.values():
            if member.is_alias:
                continue
            # Recurse through every module and class, public or not: `zarr.core` is not
            # in `zarr.__all__`, yet it defines the classes that `zarr` re-exports.
            if member.is_module or member.is_class:
                walk(member)
            if member.is_public and (message := getattr(member, "deprecated", None)):
                found[member.path] = message

    walk(pkg)

    # Canonical path -> the path a user would write, for the package's exports and
    # the members of exported classes.
    public: dict[str, str] = {}
    for name in pkg.exports or ():
        try:
            export = pkg[name]
        except KeyError:
            continue
        target = export.final_target if export.is_alias else export
        public.setdefault(target.path, f"{package}.{name}")
        if target.is_class:
            for member_name, member in target.members.items():
                if not member.is_alias and member.is_public:
                    public.setdefault(member.path, f"{package}.{name}.{member_name}")

    return sorted((public.get(path, path), message) for path, message in found.items())


def deprecated_objects_table(package: str = "zarr") -> str:
    """Render `deprecated_objects` as a Markdown table."""
    rows = [f"| `{path}` | {message} |" for path, message in deprecated_objects(package)]
    if not rows:
        return "There are no deprecated functions, classes, or properties in this release."
    return "\n".join(["| Deprecated | Message |", "| --- | --- |", *rows])


def declared_deprecations_table() -> str:
    """Render `zarr._deprecations.DEPRECATIONS` as a Markdown table."""
    from zarr._deprecations import DEPRECATIONS

    rows = [f"| {d.deprecated} | {d.replacement} | {d.removal} |" for d in DEPRECATIONS.values()]
    if not rows:
        return "There are no other deprecations in this release."
    return "\n".join(
        ["| Deprecated | Use instead | Planned removal |", "| --- | --- | --- |", *rows]
    )
