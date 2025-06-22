from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import numpy as np

from zarr.core.dtype.common import (
    DataTypeValidationError,
    DTypeJSON,
)

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


# This class is different from the other registry classes, which inherit from
# dict. IMO it's simpler to just do a dataclass. But long-term we should
# have just 1 registry class in use.
@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    contents: dict[str, type[ZDType[TBaseDType, TBaseScalar]]] = field(
        default_factory=dict, init=False
    )

    lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def lazy_load(self) -> None:
        for e in self.lazy_load_list:
            self.register(e.load()._zarr_v3_name, e.load())

        self.lazy_load_list.clear()

    def register(self: Self, key: str, cls: type[ZDType[TBaseDType, TBaseScalar]]) -> None:
        # don't register the same dtype twice
        if key not in self.contents or self.contents[key] != cls:
            self.contents[key] = cls

    def unregister(self, key: str) -> None:
        """Unregister a data type by its key."""
        if key in self.contents:
            del self.contents[key]
        else:
            raise KeyError(f"Data type '{key}' not found in registry.")

    def get(self, key: str) -> type[ZDType[TBaseDType, TBaseScalar]]:
        return self.contents[key]

    def match_dtype(self, dtype: TBaseDType) -> ZDType[TBaseDType, TBaseScalar]:
        if dtype == np.dtype("O"):
            msg = (
                f"Zarr data type resolution from {dtype} failed. "
                'Attempted to resolve a zarr data type from a numpy "Object" data type, which is '
                'ambiguous, as multiple zarr data types can be represented by the numpy "Object" '
                "data type. "
                "In this case you should construct your array by providing a specific Zarr data "
                'type. For a list of Zarr data types that are compatible with the numpy "Object"'
                "data type, see https://github.com/zarr-developers/zarr-python/issues/3117"
            )
            raise ValueError(msg)
        matched: list[ZDType[TBaseDType, TBaseScalar]] = []
        for val in self.contents.values():
            with contextlib.suppress(DataTypeValidationError):
                matched.append(val.from_native_dtype(dtype))
        if len(matched) == 1:
            return matched[0]
        elif len(matched) > 1:
            msg = (
                f"Zarr data type resolution from {dtype} failed. "
                f"Multiple data type wrappers found that match dtype '{dtype}': {matched}. "
                "You should unregister one of these data types, or avoid Zarr data type inference "
                "entirely by providing a specific Zarr data type when creating your array."
                "For more information, see https://github.com/zarr-developers/zarr-python/issues/3117"
            )
            raise ValueError(msg)
        raise ValueError(f"No Zarr data type found that matches dtype '{dtype!r}'")

    def match_json(
        self, data: DTypeJSON, *, zarr_format: ZarrFormat
    ) -> ZDType[TBaseDType, TBaseScalar]:
        for val in self.contents.values():
            try:
                return val.from_json(data, zarr_format=zarr_format)
            except DataTypeValidationError:
                pass
        raise ValueError(f"No Zarr data type found that matches {data!r}")
