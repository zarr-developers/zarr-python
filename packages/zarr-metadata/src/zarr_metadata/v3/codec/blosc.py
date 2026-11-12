"""
Blosc codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
"""

from collections.abc import Iterator
from typing import Final

from zarr_metadata._json import ValidationProblem
from zarr_metadata.definition._definition import Nested
from zarr_metadata.v3._definition import CodecDefinition
from zarr_metadata.v3._types.codec import (
    BLOSC_CNAME,
    BLOSC_CODEC_NAME,
    BLOSC_NO_SHUFFLE,
    BLOSC_SHUFFLE,
    BloscCName,
    BloscCodecConfiguration,
    BloscCodecMetadata,
    BloscCodecName,
    BloscCodecObject,
    BloscShuffle,
)


def _rules(configuration: BloscCodecConfiguration, nested: Nested) -> Iterator[ValidationProblem]:
    """Check `typesize` against `shuffle`.

    For `noshuffle`, the spec says of `typesize` that "the value is
    ignored", and the canonical form drops it. For either other shuffle,
    `typesize` is required and must be positive. The type cannot express
    whether `typesize` is required, so it does not express the bound
    either. These rules check both.
    """
    shuffle = configuration["shuffle"]
    if shuffle != BLOSC_NO_SHUFFLE:
        typesize = configuration.get("typesize")
        if typesize is None:
            yield ValidationProblem(
                ("typesize",), f"typesize is required when shuffle is {shuffle!r}", "missing_key"
            )
        elif typesize < 1:
            yield ValidationProblem(
                ("typesize",),
                f"expected a positive integer, got {typesize}",
                "invalid_value",
                ctx={"ge": 1},
            )


def _canonical(configuration: BloscCodecConfiguration) -> BloscCodecConfiguration:
    """Remove `typesize` when `shuffle` is `noshuffle`, because it has no meaning there.

    The spec says of that case that "the value is ignored", so two
    configurations that differ only in `typesize` describe the same codec.
    """
    if configuration["shuffle"] != BLOSC_NO_SHUFFLE or "typesize" not in configuration:
        return configuration
    simpler = configuration.copy()
    del simpler["typesize"]
    return simpler


BLOSC_CODEC: Final = CodecDefinition(
    name=BLOSC_CODEC_NAME,
    configuration=BloscCodecConfiguration,
    kind="bytes_bytes",
    size="dynamic",
    rules=_rules,
    canonical=_canonical,
)
"""The `blosc` codec."""


__all__ = [
    "BLOSC_CNAME",
    "BLOSC_CODEC",
    "BLOSC_CODEC_NAME",
    "BLOSC_NO_SHUFFLE",
    "BLOSC_SHUFFLE",
    "BloscCName",
    "BloscCodecConfiguration",
    "BloscCodecMetadata",
    "BloscCodecName",
    "BloscCodecObject",
    "BloscShuffle",
]
