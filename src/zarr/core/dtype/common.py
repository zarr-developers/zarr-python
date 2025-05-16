from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Endianness = Literal["little", "big"]
SpecialFloats = Literal["NaN", "Infinity", "-Infinity"]
JSONFloat = float | SpecialFloats


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
