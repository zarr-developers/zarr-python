from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final, Literal

Endianness = Literal["little", "big"]
SpecialFloatStrings = Literal["NaN", "Infinity", "-Infinity"]
SPECIAL_FLOAT_STRINGS: Final = ("NaN", "Infinity", "-Infinity")
JSONFloatV2 = float | SpecialFloatStrings
JSONFloatV3 = float | SpecialFloatStrings | str


class DataTypeValidationError(ValueError): ...


@dataclass(frozen=True)
class HasLength:
    """
    A mix-in class for data types with a length attribute, such as fixed-size collections
    of unicode strings, or bytes.
    """

    length: int


@dataclass(frozen=True)
class HasEndianness:
    """
    A mix-in class for data types with an endianness attribute
    """

    endianness: Endianness | None = "little"


@dataclass(frozen=True)
class HasItemSize:
    """
    A mix-in class for data types with an item size attribute.
    This mix-in bears a property ``item_size``, which denotes the size of each element of the data
    type, in bytes.
    """

    @property
    def item_size(self) -> int:
        raise NotImplementedError


class UnstableSpecificationWarning(FutureWarning): ...


def v3_unstable_dtype_warning(dtype: object) -> None:
    """
    Emit this warning when a data type does not have a stable zarr v3 spec
    """
    msg = (
        f"The data type ({dtype}) does not have a Zarr V3 specification. "
        "That means that the representation of data saved with this data type may change without "
        "warning in a future version of Zarr Python. "
        "Arrays stored with this data type may be unreadable by other Zarr libraries "
        "Use this data type at your own risk! "
        "Check https://github.com/zarr-developers/zarr-extensions/tree/main/data-types for the "
        "status of data type specifications for Zarr V3."
    )
    warnings.warn(msg, category=UnstableSpecificationWarning, stacklevel=2)
