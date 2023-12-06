from __future__ import annotations
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from zarr.v3.group import Group
    from zarr.v3.array import Array


class Attributes(MutableMapping[str, Any]):
    def __init__(self, obj: Union[Array, Group]):
        # key=".zattrs", read_only=False, cache=True, synchronizer=None
        self._obj = obj

    def __getitem__(self, key):
        return self._obj.metadata.attributes[key]

    def __setitem__(self, key, value):
        new_attrs = dict(self._obj.metadata.attributes)
        new_attrs[key] = value
        self._obj = self._obj.update_attributes(new_attrs)

    def __delitem__(self, key):
        new_attrs = dict(self._obj.metadata.attributes)
        del new_attrs[key]
        self._obj = self._obj.update_attributes(new_attrs)

    def __iter__(self):
        return iter(self._obj.metadata.attributes)

    def __len__(self):
        return len(self._obj.metadata.attributes)
