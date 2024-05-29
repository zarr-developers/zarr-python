from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING

from zarr.common import JSON

if TYPE_CHECKING:
    from zarr.array import Array
    from zarr.group import Group


class Attributes(MutableMapping[str, JSON]):
    def __init__(self, obj: Array | Group):
        # key=".zattrs", read_only=False, cache=True, synchronizer=None
        self._obj = obj

    def __getitem__(self, key: str) -> JSON:
        return self._obj.metadata.attributes[key]

    def __setitem__(self, key: str, value: JSON) -> None:
        new_attrs = dict(self._obj.metadata.attributes)
        new_attrs[key] = value
        self._obj = self._obj.update_attributes(new_attrs)

    def __delitem__(self, key: str) -> None:
        new_attrs = dict(self._obj.metadata.attributes)
        del new_attrs[key]
        self._obj = self._obj.update_attributes(new_attrs)

    def __iter__(self) -> Iterator[str]:
        return iter(self._obj.metadata.attributes)

    def __len__(self) -> int:
        return len(self._obj.metadata.attributes)
