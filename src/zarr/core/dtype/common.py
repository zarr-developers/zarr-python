from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar, Final, Literal

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


@dataclass(frozen=True)
class HasObjectCodec:
    """
    A mix-in class for data types that require an object codec id.
    This class bears the property ``object_codec_id``, which is the string name of an object
    codec that is required to encode and decode the data type.

    In zarr-python 2.x certain data types like variable-length strings or variable-length arrays
    used the catch-all numpy "object" data type for their in-memory representation. But these data
    types cannot be stored as numpy object data types, because the object data type does not define
    a fixed memory layout. So these data types required a special codec, called an "object codec",
    that effectively defined a compact representation for the data type, which was used to encode
    and decode the data type.

    Zarr-python 2.x would not allow the creation of arrays with the "object" data type if an object
    codec was not specified, and thus the name of the object codec is effectively part of the data
    type model.
    """

    object_codec_id: ClassVar[str]


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
