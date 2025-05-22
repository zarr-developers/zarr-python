from __future__ import annotations

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
