"""
Zarr v2 codec configuration shape.

V2 compressors and filters are numcodecs configuration dicts: a required
`id` field naming the codec, plus arbitrary codec-specific extra fields.
"""

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.common import JSON


class NumcodecsConfig(TypedDict, extra_items=JSON):  # type: ignore[call-arg]
    """
    A numcodecs configuration dict, used as a v2 compressor or filter.

    The required `id` field names the codec; codec-specific parameters
    (e.g. `cname`, `clevel` for blosc) appear as extra fields.

    See the "compressor" and "filters" sections of
    https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
    """

    id: ReadOnly[str]


__all__ = [
    "NumcodecsConfig",
]
