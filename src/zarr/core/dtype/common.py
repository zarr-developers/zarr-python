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
