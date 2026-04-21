"""
Zarr v2 codec configuration shape.

V2 compressors and filters are numcodecs configuration dicts: a required
``id`` field naming the codec, plus arbitrary codec-specific extra fields.
"""

from typing_extensions import ReadOnly, TypedDict

from zarr_metadata import JSON


class NumcodecsConfig(TypedDict, extra_items=JSON):  # type: ignore[call-arg]
    """
    A numcodecs configuration dict, used as a v2 compressor or filter.

    The required ``id`` field names the codec; codec-specific parameters
    (e.g. ``cname``, ``clevel`` for blosc) appear as extra fields.
    """

    id: ReadOnly[str]


__all__ = [
    "NumcodecsConfig",
]
