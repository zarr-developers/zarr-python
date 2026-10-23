"""The JSON shape of a v2 codec. It is in its own module so that both the array document and the codec definitions can import it."""

from typing_extensions import TypedDict

from zarr_metadata._common import JSONValue


class ZarrV2CodecMetadata(TypedDict, extra_items=JSONValue):
    """
    A numcodecs configuration dict, used as a v2 compressor or filter.

    The required `id` field names the codec; codec-specific parameters
    (e.g. `cname`, `clevel` for blosc) appear as extra fields.

    See the "compressor" and "filters" sections of
    https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    id: str


__all__ = ["ZarrV2CodecMetadata"]
