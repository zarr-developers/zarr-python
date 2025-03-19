from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from zarr.core.dtype.common import DataTypeValidationError

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from zarr.core.common import JSON, ZarrFormat
    from zarr.core.dtype.wrapper import ZDType, _BaseDType, _BaseScalar


@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    contents: dict[str, type[ZDType[_BaseDType, _BaseScalar]]] = field(
        default_factory=dict, init=False
    )
    lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.name, e.load())

        self.lazy_load_list.clear()

    def register(self: Self, key: str, cls: type[ZDType[_BaseDType, _BaseScalar]]) -> None:
        # don't register the same dtype twice
        if key not in self.contents or self.contents[key] != cls:
            self.contents[key] = cls

    def get(self, key: str) -> type[ZDType[_BaseDType, _BaseScalar]]:
        return self.contents[key]

    def match_dtype(self, dtype: _BaseDType) -> ZDType[_BaseDType, _BaseScalar]:
        self.lazy_load()
        for val in self.contents.values():
            try:
                return val.from_dtype(dtype)
            except DataTypeValidationError:
                pass
        raise ValueError(f"No data type wrapper found that matches dtype '{dtype}'")

    def match_json(self, data: JSON, zarr_format: ZarrFormat) -> ZDType[_BaseDType, _BaseScalar]:
        self.lazy_load()
        for val in self.contents.values():
            try:
                return val.from_json(data, zarr_format=zarr_format)
            except DataTypeValidationError:
                pass
        raise ValueError(f"No data type wrapper found that matches {data}")
