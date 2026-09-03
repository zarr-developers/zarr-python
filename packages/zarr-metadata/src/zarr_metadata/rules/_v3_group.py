"""Composition rules for v3 group metadata documents.

A group document's own fields carry no cross-field constraints, but the
inline consolidated-metadata convention embeds whole child documents —
and a composition-invalid child makes the consolidated view lie about
the store. The group rule set therefore recurses: every array entry is
judged by the v3 array rules, and every group entry (which may itself
carry consolidated metadata) by this rule set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from zarr_metadata.model._validation import GROUP_METADATA_STANDARD_KEYS_V3
from zarr_metadata.rules._engine import Rule, as_string_mapping, prefixed, run_rules
from zarr_metadata.rules._registry import document_rule, document_rules, register_document_type
from zarr_metadata.rules._v3_array import ZARR_V3_ARRAY_RULES
from zarr_metadata.v3.consolidated import ZARR_V3_CONSOLIDATED_METADATA_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata.model._validation import ValidationProblem


ZARR_V3_GROUP = "zarr_v3_group"
"""Document-type key under which this module's rules are registered."""

# `consolidated_metadata` is not a declared member of the group TypedDict:
# the spec grandfathers it as a convention that 'lacks the name member
# required of extension objects'. It is declared here so the rule that
# reads it passes the typo check without exempting unknown keys.
register_document_type(
    ZARR_V3_GROUP,
    GROUP_METADATA_STANDARD_KEYS_V3,
    extension_keys=frozenset({ZARR_V3_CONSOLIDATED_METADATA_KEY}),
)


@document_rule(ZARR_V3_GROUP, frozenset({"consolidated_metadata"}))
def check_consolidated_entries(document: Mapping[str, object]) -> tuple[ValidationProblem, ...]:
    """Consolidated child documents must satisfy their own composition rules.

    Structural validity of the consolidated envelope and its entries is
    the model layer's job; entries that are not interpretable as node
    documents decline in its favor.
    """
    return consolidated_entries_problems(
        document["consolidated_metadata"], ("consolidated_metadata",)
    )


def consolidated_entries_problems(
    value: object, loc: tuple[str | int, ...] = ()
) -> tuple[ValidationProblem, ...]:
    """Composition problems in an inline consolidated envelope's children."""
    consolidated = as_string_mapping(value)
    if consolidated is None:
        return ()
    metadata = as_string_mapping(consolidated.get("metadata"))
    if metadata is None:
        return ()
    problems: list[ValidationProblem] = []
    for path, entry in metadata.items():
        node = as_string_mapping(entry)
        if node is None:
            continue
        entry_loc = (*loc, "metadata", path)
        node_type = node.get("node_type")
        if node_type == "array":
            problems.extend(prefixed(entry_loc, run_rules(ZARR_V3_ARRAY_RULES, node)))
        elif node_type == "group":
            problems.extend(prefixed(entry_loc, run_rules(ZARR_V3_GROUP_RULES, node)))
    return tuple(problems)


ZARR_V3_GROUP_RULES: Final[tuple[Rule, ...]] = document_rules(ZARR_V3_GROUP)
"""The composition rule set for v3 group metadata documents."""


__all__ = [
    "ZARR_V3_GROUP",
    "ZARR_V3_GROUP_RULES",
]
