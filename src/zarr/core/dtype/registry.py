from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from zarr.core.dtype.common import HasNestedDTypes
from zarr.errors import DataTypeValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.metadata import EntryPoint

    from zarr.core.common import ZarrFormat
    from zarr.core.dtype.common import DTypeJSON
    from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType


# Zarr V2 data type names are NumPy typestrs, whose first character is the byte order: "<", ">", or
# "|" (not relevant). For the data types below the byte order is not relevant, so their "<" and ">"
# spellings name the same data type as the canonical "|" spelling. Other implementations write
# them: netCDF-C writes "<i1" and "<u1", Zarr.jl wrote "<b1", "<i1" and "<u1", and jzarr wrote
# ">i1" and ">u1".

# Single-byte data types, keyed by their non-canonical spellings.
_V2_SINGLE_BYTE_ALIASES: Final[Mapping[str, str]] = {
    "<b1": "|b1",
    ">b1": "|b1",
    "<i1": "|i1",
    ">i1": "|i1",
    "<u1": "|u1",
    ">u1": "|u1",
}
# Fixed-length bytes data types, whose names are the kind followed by the length in bytes.
_V2_BYTES_KINDS: Final = ("V",)


def _v2_canonical_name(name: str) -> str:
    """
    The canonical spelling of a Zarr V2 data type name: `"|"` for a byte order that is not
    relevant, and the name as written otherwise.
    """
    if name in _V2_SINGLE_BYTE_ALIASES:
        return _V2_SINGLE_BYTE_ALIASES[name]
    byte_order, kind, length = name[:1], name[1:2], name[2:]
    if (
        byte_order in ("<", ">")
        and kind in _V2_BYTES_KINDS
        and length.isascii()
        and length.isdigit()
    ):
        return f"|{kind}{length}"
    return name


def _v2_spellings(data: DTypeJSON) -> tuple[DTypeJSON, ...]:
    """
    The spellings of a Zarr V2 data type JSON to match, in order: the name as written, then its
    canonical spelling if it has another. A data type that declares a non-canonical spelling
    (e.g. `">S1"`) takes precedence over the canonical alias, because the name as written is
    tried first.
    """
    if (
        isinstance(data, dict)
        and isinstance(name := data.get("name"), str)
        and (canonical := _v2_canonical_name(name)) != name
    ):
        return (data, {**data, "name": canonical})
    return (data,)


# This class is different from the other registry classes, which inherit from
# dict. IMO it's simpler to just do a dataclass. But long-term we should
# have just 1 registry class in use.
@dataclass(frozen=True, kw_only=True)
class DataTypeRegistry:
    """
    A registry for ZDType classes.

    This registry is a mapping from Zarr data type names to their
    corresponding ZDType classes.

    Attributes
    ----------
    contents : dict[str, type[ZDType[TBaseDType, TBaseScalar]]]
        The mapping from Zarr data type names to their corresponding
        ZDType classes.
    """

    contents: dict[str, type[ZDType[TBaseDType, TBaseScalar]]] = field(
        default_factory=dict, init=False
    )

    _lazy_load_list: list[EntryPoint] = field(default_factory=list, init=False)

    def _lazy_load(self) -> None:
        """
        Load all data types from the lazy load list and register them with
        the registry. After loading, clear the lazy load list.
        """
        for e in self._lazy_load_list:
            cls = e.load()
            self.register(cls._zarr_v3_name, cls)

        self._lazy_load_list.clear()

    def register(self: Self, key: str, cls: type[ZDType[TBaseDType, TBaseScalar]]) -> None:
        """
        Register a data type with the registry.

        Parameters
        ----------
        key : str
            The Zarr V3 name of the data type.
        cls : type[ZDType[TBaseDType, TBaseScalar]]
            The class of the data type to register.

        Notes
        -----
        This method is idempotent. If the data type is already registered, this
        method does nothing.
        """
        if key not in self.contents or self.contents[key] != cls:
            self.contents[key] = cls

    def unregister(self, key: str) -> None:
        """
        Unregister a data type from the registry.

        Parameters
        ----------
        key : str
            The key associated with the ZDType class to be unregistered.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the data type is not found in the registry.
        """
        if key in self.contents:
            del self.contents[key]
        else:
            raise KeyError(f"Data type '{key}' not found in registry.")

    def get(self, key: str) -> type[ZDType[TBaseDType, TBaseScalar]]:
        """
        Retrieve a registered ZDType class by its key.

        Parameters
        ----------
        key : str
            The key associated with the desired ZDType class.

        Returns
        -------
        type[ZDType[TBaseDType, TBaseScalar]]
            The ZDType class registered under the given key.

        Raises
        ------
        KeyError
            If the key is not found in the registry.
        """

        self._lazy_load()
        return self.contents[key]

    def match_dtype(self, dtype: TBaseDType) -> ZDType[TBaseDType, TBaseScalar]:
        """
        Match a native data type, e.g. a NumPy data type, to a registered ZDType.

        Parameters
        ----------
        dtype : TBaseDType
            The native data type to match.

        Returns
        -------
        ZDType[TBaseDType, TBaseScalar]
            The matched ZDType corresponding to the provided NumPy data type.

        Raises
        ------
        ValueError
            If the data type is a NumPy "Object" type, which is ambiguous, or if multiple
            or no Zarr data types are found that match the provided dtype.

        Notes
        -----
        This function attempts to resolve a Zarr data type from a given native data type.
        If the dtype is a NumPy "Object" data type, it raises a ValueError, as this type
        can represent multiple Zarr data types. In such cases, a specific Zarr data type
        should be explicitly constructed instead of relying on dynamic resolution.

        If multiple matches are found, it will also raise a ValueError. In this case
        conflicting data types must be unregistered, or the Zarr data type should be explicitly
        constructed.
        """

        self._lazy_load()
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
            # DataTypeValidationError means "this dtype doesn't match me", which is
            # expected and suppressed. Other exceptions (e.g. ValueError for a dtype
            # that matches the type but has an invalid configuration) are propagated
            # to the caller.
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
        """
        Match a JSON representation of a data type to a registered ZDType.

        Parameters
        ----------
        data : DTypeJSON
            The JSON representation of a data type to match.
        zarr_format : ZarrFormat
            The Zarr format version to consider when matching data types.

        Returns
        -------
        ZDType[TBaseDType, TBaseScalar]
            The matched ZDType corresponding to the JSON representation.

        Raises
        ------
        ValueError
            If no matching Zarr data type is found for the given JSON data.
        """

        self._lazy_load()
        candidates = _v2_spellings(data) if zarr_format == 2 else (data,)
        for candidate in candidates:
            for val in self.contents.values():
                try:
                    if issubclass(val, HasNestedDTypes):
                        # the data types it contains are resolved with this registry
                        return val._from_json_nested(
                            candidate, zarr_format=zarr_format, resolver=self.match_json
                        )
                    return val.from_json(candidate, zarr_format=zarr_format)
                except DataTypeValidationError:
                    pass
        raise ValueError(f"No Zarr data type found that matches {data!r}")
