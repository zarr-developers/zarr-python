"""The JSON shapes of a v2 dtype and a v2 codec. They are in their own module so that both the array document and the definitions can import them without a cycle; `zarr_metadata.v2.data_type` and `zarr_metadata.v2.codec` export them."""

from typing_extensions import TypeAliasType, TypedDict

from zarr_metadata._common import JSONValue

ZarrV2DataTypeMetadata = TypeAliasType(
    "ZarrV2DataTypeMetadata",
    str
    | tuple[
        tuple[str, "ZarrV2DataTypeMetadata"]
        | tuple[str, "ZarrV2DataTypeMetadata", tuple[int, ...]],
        ...,
    ],
)
"""The v2 dtype representation.

Either a numpy-style dtype string (e.g. `"<f8"`, `"|S10"`) or a tuple of
field records describing a structured dtype. Each field record is either
a 2-tuple `(name, datatype)` or a 3-tuple `(name, datatype, shape)`
(the 3-tuple form indicates a subarray field). A field datatype may itself
be another structured dtype.

Endianness is encoded in the prefix character of the dtype string;
parsing it out is a caller concern, not part of this type.

See https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#data-type-encoding
"""


class ZarrV2CodecMetadata(TypedDict, extra_items=JSONValue):
    """
    A numcodecs configuration dict, used as a v2 compressor or filter.

    The required `id` field names the codec; codec-specific parameters
    (e.g. `cname`, `clevel` for blosc) appear as extra fields.

    See the "compressor" and "filters" sections of
    https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    id: str
    """The numcodecs id that names the codec, such as `"blosc"`."""


__all__ = ["ZarrV2CodecMetadata", "ZarrV2DataTypeMetadata"]
