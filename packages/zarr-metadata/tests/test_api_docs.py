"""Every public module has an API reference page, and every public object the docstring it renders.

The reference pages render every public name with mkdocstrings, which reads
docstrings statically, through griffe: a field's docstring is the string
literal under it, which the interpreter discards. So these tests read the
package the way the pages do, rather than through `__doc__`.

Public means: a module whose dotted path has no segment starting with `_`,
each name in its `__all__`, and each member a public class declares itself
whose name does not start with `_`.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path

import griffe

import zarr_metadata

API_PAGES = Path(__file__).parent.parent / "docs" / "api"

PACKAGE = griffe.load("zarr_metadata", resolve_aliases=True, resolve_external=False)

PUBLIC_MODULES = [
    name
    for name in [
        "zarr_metadata",
        *(
            module.name
            for module in pkgutil.walk_packages(zarr_metadata.__path__, "zarr_metadata.")
        ),
    ]
    if not any(part.startswith("_") for part in name.split("."))
]

# A strict on-disk v2 type whose fields mean what the merged in-memory
# type's same-named fields mean. Their field docstrings are the same text.
MIRRORS = [
    ("zarr_metadata.v2.array.ZarrV2ArrayMetadataJSON", "zarr_metadata.v2.array.ZarrV2ZArrayJSON"),
    ("zarr_metadata.v2.group.ZarrV2GroupMetadataJSON", "zarr_metadata.v2.group.ZarrV2ZGroupJSON"),
]


def _griffe(path: str) -> griffe.Object:
    """The object at a dotted path, followed through any re-export to its definition."""
    found = PACKAGE if path == "zarr_metadata" else PACKAGE[path.removeprefix("zarr_metadata.")]
    return found.final_target if isinstance(found, griffe.Alias) else found


def _public_objects() -> dict[str, griffe.Object]:
    """Each public module, name and class member, keyed by the path it is defined at."""
    found: dict[str, griffe.Object] = {}
    for module_name in PUBLIC_MODULES:
        module = _griffe(module_name)
        found[module.path] = module
        for name in importlib.import_module(module_name).__all__:
            target = _griffe(f"{module_name}.{name}")
            found[target.path] = target
            if target.is_class:
                for member_name, member in target.members.items():
                    if not member_name.startswith("_") and not isinstance(member, griffe.Alias):
                        found[member.path] = member
    return found


def test_public_modules_have_reference_pages() -> None:
    # The top-level namespace re-exports names whose pages are their defining
    # modules'; the API index renders the few it defines itself.
    rendered = {
        match
        for page in API_PAGES.rglob("*.md")
        for match in re.findall(r"^::: (\S+)$", page.read_text(), flags=re.MULTILINE)
    }
    assert sorted(set(PUBLIC_MODULES) - rendered - {"zarr_metadata"}) == []


def test_public_objects_have_docstrings() -> None:
    undocumented = sorted(
        path
        for path, obj in _public_objects().items()
        if obj.docstring is None or not obj.docstring.value.strip()
    )
    assert undocumented == []


def test_mirrored_fields_share_docstrings() -> None:
    drifted = []
    for source_path, mirror_path in MIRRORS:
        source, mirror = _griffe(source_path), _griffe(mirror_path)
        for name, field in mirror.members.items():
            if (
                name in source.members
                and field.is_attribute
                and getattr(field.docstring, "value", None)
                != getattr(source.members[name].docstring, "value", None)
            ):
                drifted.append(f"{mirror_path}.{name}")
    assert drifted == []
