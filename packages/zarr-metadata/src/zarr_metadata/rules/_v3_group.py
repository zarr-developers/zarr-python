"""Semantic checks for v3 group metadata documents.

A group says almost nothing that can be wrong on its own. The one thing
it can carry is consolidated metadata -- the child documents of a whole
subtree, inline -- and each of those is judged exactly as it would be
standing alone, at its own path.

`consolidated_metadata` is not a declared member of the group TypedDict:
the spec grandfathers it as a convention that "lacks the name member
required of extension objects".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._document import array_problems_v3
from zarr_metadata.v3._registry import CORE_AND_EXTENSIONS, Context

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zarr_metadata.v3._entity import Loc


def _prefixed(loc: Loc, problems: Sequence[ValidationProblem]) -> tuple[ValidationProblem, ...]:
    """Re-base every problem's `loc` under `loc`, for a nested document."""
    return tuple(
        ValidationProblem((*loc, *found.loc), found.message, found.kind) for found in problems
    )


def _as_string_mapping(value: object) -> Mapping[str, object] | None:
    """`value` as a string-keyed mapping, or None if it is not one."""
    if not isinstance(value, Mapping):
        return None
    mapping = cast("Mapping[object, object]", value)
    if any(not isinstance(key, str) for key in mapping):
        return None
    return cast("Mapping[str, object]", mapping)


def group_problems_v3(
    document: Mapping[str, object], context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Every semantic problem in a v3 group document."""
    if "consolidated_metadata" not in document:
        return ()
    return consolidated_entries_problems(
        document["consolidated_metadata"], ("consolidated_metadata",), context
    )


def consolidated_entries_problems(
    value: object, loc: Loc = (), context: Context = CORE_AND_EXTENSIONS
) -> tuple[ValidationProblem, ...]:
    """Semantic problems in an inline consolidated envelope's children.

    Structural validity of the envelope and its entries is the model
    layer's job; an entry that is not interpretable as a node document
    declines in its favour.
    """
    consolidated = _as_string_mapping(value)
    if consolidated is None:
        return ()
    metadata = _as_string_mapping(consolidated.get("metadata"))
    if metadata is None:
        return ()
    problems: list[ValidationProblem] = []
    for path, entry in metadata.items():
        node = _as_string_mapping(entry)
        if node is None:
            continue
        entry_loc = (*loc, "metadata", path)
        node_type = node.get("node_type")
        if node_type == "array":
            problems.extend(_prefixed(entry_loc, array_problems_v3(node, context)))
        elif node_type == "group":
            problems.extend(_prefixed(entry_loc, group_problems_v3(node, context)))
    return tuple(problems)


__all__ = [
    "consolidated_entries_problems",
    "group_problems_v3",
]
