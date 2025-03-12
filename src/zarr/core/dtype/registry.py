from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from zarr.core.dtype.common import DataTypeValidationError

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from zarr.core.common import JSON
    from zarr.core.dtype.wrapper import DTypeWrapper, TDType


@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    contents: dict[str, type[DTypeWrapper[Any, Any]]] = field(default_factory=dict, init=False)
    lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.name, e.load())

        self.lazy_load_list.clear()

    def register(self: Self, key: str, cls: type[DTypeWrapper[Any, Any]]) -> None:
        # don't register the same dtype twice
        if key not in self.contents or self.contents[key] != cls:
            self.contents[key] = cls

    def get(self, key: str) -> type[DTypeWrapper[Any, Any]]:
        return self.contents[key]

    def match_dtype(self, dtype: TDType) -> DTypeWrapper[Any, Any]:
        self.lazy_load()
        for val in self.contents.values():
            try:
                return val.from_dtype(dtype)
            except DataTypeValidationError:
                pass
        raise ValueError(f"No data type wrapper found that matches dtype '{dtype}'")

    def match_json(self, data: JSON) -> DTypeWrapper[Any, Any]:
        self.lazy_load()
        for val in self.contents.values():
            try:
                return val.from_dict(data)
            except DataTypeValidationError:
                pass
        raise ValueError(f"No data type wrapper found that matches {data}")
