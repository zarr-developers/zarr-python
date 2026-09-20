"""How a data type lays out in bytes, and what that demands of a codec.

Two rules in this package need the same fact about a data type: whether
one of its scalars occupies a fixed number of bytes, and if so whether
that number is one (no byte order to declare) or more (a byte order the
`bytes` codec must declare).

The `bytes` codec spec makes `endian` "Required for data types for which
endianness is applicable... multi-byte data types, such as `uint16` and
`int32`, but not single-byte data types, such as `uint8` or `bool`", and
addresses fixed-size numeric types only.

The `struct` spec then states its own constraints in those same terms
rather than inventing new ones: a field's data type must be one "whose
size in bytes is fixed and known at the time the array is opened", since
variable-length types "do not have a fixed encoded size"; and "When a
`struct` type contains multi-byte numeric fields, the `bytes` codec MUST
be configured with an explicit `endian` setting", while a struct
"composed entirely of single-byte fields... MAY omit the `endian`
configuration".

So a struct's own storage class is the widest class among its fields,
recursively, and "valid as a struct field" is simply "not
variable-length". One classifier answers both.

- https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
- https://github.com/zarr-developers/zarr-extensions/blob/main/data-types/struct/README.md
"""

from __future__ import annotations

from typing import Literal, cast

from zarr_metadata.rules._engine import as_string_mapping
from zarr_metadata.v3.data_type.bool import BOOL_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.bytes import BYTES_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex64 import COMPLEX64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.complex128 import COMPLEX128_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float16 import FLOAT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float32 import FLOAT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.float64 import FLOAT64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int8 import INT8_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int16 import INT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int32 import INT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.int64 import INT64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.numpy_datetime64 import NUMPY_DATETIME64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.numpy_timedelta64 import NUMPY_TIMEDELTA64_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.raw import RAW_BYTES_NAME_PATTERN
from zarr_metadata.v3.data_type.string import STRING_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.struct import STRUCT_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint8 import UINT8_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint16 import UINT16_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint32 import UINT32_DATA_TYPE_NAME
from zarr_metadata.v3.data_type.uint64 import UINT64_DATA_TYPE_NAME

StorageClass = Literal["single_byte", "multi_byte", "variable_length"]
"""How one scalar of a data type occupies bytes.

`single_byte` and `multi_byte` are both fixed-size; they differ only in
whether a byte order applies.
"""

_SINGLE_BYTE = frozenset({BOOL_DATA_TYPE_NAME, INT8_DATA_TYPE_NAME, UINT8_DATA_TYPE_NAME})
_MULTI_BYTE = frozenset(
    {
        INT16_DATA_TYPE_NAME,
        INT32_DATA_TYPE_NAME,
        INT64_DATA_TYPE_NAME,
        UINT16_DATA_TYPE_NAME,
        UINT32_DATA_TYPE_NAME,
        UINT64_DATA_TYPE_NAME,
        FLOAT16_DATA_TYPE_NAME,
        FLOAT32_DATA_TYPE_NAME,
        FLOAT64_DATA_TYPE_NAME,
        COMPLEX64_DATA_TYPE_NAME,
        COMPLEX128_DATA_TYPE_NAME,
        NUMPY_DATETIME64_DATA_TYPE_NAME,
        NUMPY_TIMEDELTA64_DATA_TYPE_NAME,
    }
)
_VARIABLE_LENGTH = frozenset({BYTES_DATA_TYPE_NAME, STRING_DATA_TYPE_NAME})


def data_type_name(data_type: object) -> str | None:
    """The name a data-type metadata field carries, or None if it has none."""
    if isinstance(data_type, str):
        return data_type
    mapping = as_string_mapping(data_type)
    if mapping is None:
        return None
    name = mapping.get("name")
    return name if isinstance(name, str) else None


def storage_class(data_type: object) -> StorageClass | None:
    """Classify a known data type by its raw byte representation.

    None means undetermined — an unknown name, or a `struct` whose fields
    this package cannot read — and every rule declines rather than
    guessing. A `struct` takes the widest class among its fields, so a
    struct of `uint8` and `int32` is `multi_byte` and one containing a
    `string` is `variable_length`, recursively.
    """
    name = data_type_name(data_type)
    if name in _SINGLE_BYTE or (name is not None and RAW_BYTES_NAME_PATTERN.fullmatch(name)):
        return "single_byte"
    if name in _MULTI_BYTE:
        return "multi_byte"
    if name in _VARIABLE_LENGTH:
        return "variable_length"
    if name != STRUCT_DATA_TYPE_NAME:
        return None

    envelope = as_string_mapping(data_type)
    configuration = (
        as_string_mapping(envelope.get("configuration")) if envelope is not None else None
    )
    fields = configuration.get("fields") if configuration is not None else None
    if not isinstance(fields, tuple):
        return None
    classes: set[StorageClass] = set()
    for field in cast("tuple[object, ...]", fields):
        field_mapping = as_string_mapping(field)
        if field_mapping is None or "data_type" not in field_mapping:
            return None
        field_class = storage_class(field_mapping["data_type"])
        if field_class is None:
            return None
        classes.add(field_class)
    if "variable_length" in classes:
        return "variable_length"
    if "multi_byte" in classes:
        return "multi_byte"
    return "single_byte"


__all__ = [
    "StorageClass",
    "data_type_name",
    "storage_class",
]
