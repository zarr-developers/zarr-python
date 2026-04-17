"""Models for comparison testing.

The tree descriptors (GroupNode / ArrayNode) are pure data structures.
Materialization writes it into any zarr store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import zarr
import zarr.abc.store
import zarr.api.asynchronous
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROTOTYPE = default_buffer_prototype()


@dataclass(frozen=True)
class ArrayNode:
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class GroupNode:
    children: dict[str, ArrayNode | GroupNode] = field(default_factory=dict)

    def walk(self, prefix: str = "") -> Iterator[tuple[str, Node]]:
        """Yield ``(path, child)`` for every node, depth-first."""
        for name, child in self.children.items():
            p = f"{prefix}/{name}" if prefix else name
            yield p, child
            if isinstance(child, GroupNode):
                yield from child.walk(p)

    def nodes(self, prefix: str = "", *, include_root: bool = False) -> list[str]:
        """Return paths of all nodes, optionally including root."""
        root = [prefix] if include_root else []
        return root + [p for p, _ in self.walk(prefix)]

    def groups(self, prefix: str = "", *, include_root: bool = False) -> list[str]:
        """Return paths of all group nodes, optionally including root."""
        root = [prefix] if include_root else []
        return root + [p for p, c in self.walk(prefix) if isinstance(c, GroupNode)]

    def arrays(self, prefix: str = "") -> list[str]:
        """Return paths of all array nodes."""
        return [p for p, c in self.walk(prefix) if isinstance(c, ArrayNode)]

    def materialize(
        self,
        store: zarr.abc.store.Store,
        *,
        zarr_format: Literal[2, 3] = 3,
        mode: Literal["w", "a"] = "w",
    ) -> zarr.Group:
        """Write this tree into *store* and return the root group.

        ``mode`` is forwarded to :func:`zarr.open_group` when opening the root.
        """
        root = zarr.open_group(store, mode=mode, zarr_format=zarr_format)

        def _write(group: zarr.Group, node: GroupNode) -> None:
            for name, child in node.children.items():
                if isinstance(child, ArrayNode):
                    group.create_array(name, shape=child.shape, dtype=child.dtype)
                else:
                    _write(group.create_group(name), child)

        _write(root, self)
        return root

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GroupNode:
        """Convert a nested dict (with ArrayNode leaves) to a GroupNode tree."""
        children: dict[str, ArrayNode | GroupNode] = {}
        for name, value in d.items():
            if isinstance(value, ArrayNode):
                children[name] = value
            else:
                children[name] = cls.from_dict(value)
        return cls(children=children)

    @classmethod
    def from_paths(cls, arrays: set[str], groups: set[str]) -> GroupNode:
        """Build a GroupNode from flat sets of array and group paths.

        Example::

            GroupNode.from_paths(
                arrays={"a/x", "b"},
                groups={"a"},
            )
        """
        tree: dict[str, Any] = {}
        for path in sorted(groups - {""}):
            current = tree
            for part in path.split("/"):
                current = current.setdefault(part, {})
        for path in sorted(arrays):
            parts = path.split("/")
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = ArrayNode(shape=(1,), dtype=np.dtype("i4"))
        return cls.from_dict(tree)

    @classmethod
    async def from_store_async(cls, store: zarr.abc.store.Store) -> GroupNode:
        """Build a GroupNode by reading a zarr store's structure.

        Example::

            await GroupNode.from_store_async(some_memory_store)
        """
        root = await zarr.api.asynchronous.open_group(store, mode="r")
        tree: dict[str, Any] = {}
        async for path, obj in root.members(max_depth=None):
            parts = path.split("/")
            current = tree
            if isinstance(obj, zarr.AsyncArray):
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                current[parts[-1]] = ArrayNode(shape=obj.shape, dtype=obj.dtype)
            else:
                for part in parts:
                    current = current.setdefault(part, {})
        return cls.from_dict(tree)

    @classmethod
    def from_store(cls, store: zarr.abc.store.Store) -> GroupNode:
        """Build a GroupNode by reading a zarr store's structure.

        Example::

            GroupNode.from_store(some_memory_store)
        """
        return sync(cls.from_store_async(store))


Node = ArrayNode | GroupNode
