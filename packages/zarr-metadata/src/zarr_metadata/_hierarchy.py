"""The tree that the nodes of a Zarr hierarchy form, in either format.

"A Zarr hierarchy is a tree structure, where each node in the tree is
either a group or an array. Group nodes may have children but array
nodes may not"
(https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L177-L181).
Zarr v2 hierarchies have the same structure. The rules for node names and
paths differ between the formats, so each format checks its paths before
it checks the tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from zarr_metadata._json import ValidationProblem, shown

if TYPE_CHECKING:
    from collections.abc import Mapping

NodeType = Literal["array", "group"]
"""The two kinds of node in a hierarchy."""


def tree_problems(nodes: Mapping[str, NodeType | None]) -> tuple[ValidationProblem, ...]:
    """The reasons `nodes` (the node type of each node, keyed by a valid node path) do not form a tree. Reports one problem per affected node, at that node's path.

    Each node except the root must have its parent in `nodes`, and that
    parent must be a group. A node below an array is an `invalid_value` at
    the node's path. If groups are missing above a node, the nearest
    missing group is reported once as a `missing_key` at that group's path.
    The message says how many more groups are missing above it, including
    the root. A node whose type is unknown (None) is treated as a group.
    The paths are not checked; each format checks its own path rules first.
    """
    problems: list[ValidationProblem] = []
    tree = _Tree()
    for path, node_type in nodes.items():
        tree.hold(path, node_type)
    reported: set[str] = set()
    for path in nodes:
        if path == "/":
            continue
        holder, missing = tree.above(path)
        if holder == "array":
            # No node may be below an array, no matter how many groups are
            # missing between them.
            message = (
                f"expected a node below a group, got {shown(path)}, below the array "
                f"{shown(tree.holder_of(path))}"
            )
            problems.append(ValidationProblem((path,), message, "invalid_value"))
        elif missing != 0:
            nearest = path.rpartition("/")[0] or "/"
            if nearest not in reported:
                reported.add(nearest)
                above = missing - 1
                more = (
                    "" if above == 0 else f", and {above} group{'s' if above != 1 else ''} above it"
                )
                message = f"missing the group holding {shown(path)}{more}"
                problems.append(ValidationProblem((nearest,), message, "missing_key"))
    return tuple(problems)


class _Branch:
    """A name in the tree of present paths: whether a node is present at it, the node's type, and the names below it."""

    __slots__ = ("children", "held", "node_type")

    def __init__(self) -> None:
        self.held = False
        self.node_type: NodeType | None = None
        self.children: dict[str, _Branch] = {}


class _Tree:
    """The present paths as a tree of their names, so each path is walked once, in time proportional to its length.

    Walking up a path by building each ancestor's string costs the path's
    length for each ancestor. A hostile key of a million characters would
    make that quadratic. Here a path is split once and walked name by name.
    """

    __slots__ = ("root",)

    def __init__(self) -> None:
        self.root = _Branch()

    def hold(self, path: str, node_type: NodeType | None) -> None:
        """Mark the node at `path` (a node path) as present, with type `node_type`."""
        branch = self.root
        for name in _names(path):
            branch = branch.children.setdefault(name, _Branch())
        branch.held = True
        branch.node_type = node_type

    def above(self, path: str) -> tuple[NodeType | None, int]:
        """For the node at `path` (a node path below the root), return the type of the nearest present node above it and the number of groups missing between them. The type is `"group"` for a node of unknown type, and None if no node above is present."""
        names = _names(path)
        branch: _Branch | None = self.root
        holder: NodeType | None = None
        held_depth = -1
        for depth, name in enumerate(names[:-1], start=0):
            if branch is not None and branch.held:
                holder, held_depth = branch.node_type or "group", depth
            branch = None if branch is None else branch.children.get(name)
        # The parent, at the last depth walked, may also be present.
        if branch is not None and branch.held:
            holder, held_depth = branch.node_type or "group", len(names) - 1
        return holder, len(names) - 1 - held_depth

    def holder_of(self, path: str) -> str:
        """The path of the nearest present node above the node at `path`. `path` must be a node path below the root, with a present node above it."""
        names = _names(path)
        branch: _Branch | None = self.root
        holder = "/"
        for depth, name in enumerate(names[:-1]):
            if branch is not None and branch.held:
                holder = "/" + "/".join(names[:depth]) if depth != 0 else "/"
            branch = None if branch is None else branch.children.get(name)
        if branch is not None and branch.held:
            holder = "/" + "/".join(names[:-1])
        return holder


def _names(path: str) -> list[str]:
    """The names between a node path's `/` characters. Empty for the root's path."""
    return [] if path == "/" else path[1:].split("/")


__all__ = ["NodeType", "tree_problems"]
