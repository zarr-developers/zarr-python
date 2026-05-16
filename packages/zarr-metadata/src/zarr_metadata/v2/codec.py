"""
Zarr v2 codec configuration shape.

In v2, compressors and filters are numcodecs configuration dicts: a required
`id` field naming the codec, plus arbitrary codec-specific extra fields.
"""

from typing_extensions import TypedDict


class CodecMetadataV2(TypedDict, extra_items=object):  # type: ignore[call-arg]
    """
    A numcodecs configuration dict, used as a v2 compressor or filter.

    The required `id` field names the codec; codec-specific parameters
    (e.g. `cname`, `clevel` for blosc) appear as extra fields.

    See the "compressor" and "filters" sections of
    https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    id: str


__all__ = [
    "CodecMetadataV2",
]
