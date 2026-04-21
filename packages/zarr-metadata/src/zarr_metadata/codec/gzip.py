"""
Gzip codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
"""

from typing import Final, Literal, NotRequired, TypedDict

GZIP_CODEC_NAME: Final = "gzip"
"""The ``name`` field value of the ``gzip`` codec."""

GzipCodecName = Literal["gzip"]
"""Literal type of the ``name`` field of the ``gzip`` codec."""


class GzipCodecConfiguration(TypedDict):
    """
    Configuration for the Zarr v3 ``gzip`` codec.

    ``level`` is an integer in the range 0-9; 0 disables compression and 9
    is slowest with the best compression ratio. The spec does not mandate
    a default.
    """

    level: NotRequired[int]


class GzipCodec(TypedDict):
    """``gzip`` codec metadata."""

    name: GzipCodecName
    configuration: GzipCodecConfiguration


__all__ = [
    "GZIP_CODEC_NAME",
    "GzipCodec",
    "GzipCodecConfiguration",
    "GzipCodecName",
]
