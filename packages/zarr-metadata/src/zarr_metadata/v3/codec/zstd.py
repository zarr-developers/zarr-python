"""
Zstandard codec types.

See https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md
(the zarr-extensions registry entry; zarr-specs PR #256, which first
proposed the codec, was never merged).
"""

from typing import Final

from zarr_metadata.v3._definition import CodecDefinition
from zarr_metadata.v3._types.codec import (
    ZSTD_CODEC_NAME,
    ZSTD_MAX_LEVEL,
    ZSTD_MIN_LEVEL,
    ZstdCodecConfiguration,
    ZstdCodecMetadata,
    ZstdCodecName,
    ZstdCodecObject,
)


def _canonical(configuration: ZstdCodecConfiguration) -> ZstdCodecConfiguration:
    """Remove a `checksum` of `false`, because an absent `checksum` means the same thing.

    The spec says of `checksum` that it "should be omitted if false"
    (https://github.com/zarr-developers/zarr-extensions/blob/4da7b37a84f76e660902f6d3de3eaef0e0febae6/codecs/zstd/README.md?plain=1#L17-L19),
    so the two spellings describe the same codec.
    """
    if configuration.get("checksum") is not False:
        return configuration
    simpler = configuration.copy()
    del simpler["checksum"]
    return simpler


ZSTD_CODEC: Final = CodecDefinition(
    name=ZSTD_CODEC_NAME,
    configuration=ZstdCodecConfiguration,
    canonical=_canonical,
    kind="bytes_bytes",
    size="dynamic",
)
"""The `zstd` codec."""


__all__ = [
    "ZSTD_CODEC",
    "ZSTD_CODEC_NAME",
    "ZSTD_MAX_LEVEL",
    "ZSTD_MIN_LEVEL",
    "ZstdCodecConfiguration",
    "ZstdCodecMetadata",
    "ZstdCodecName",
    "ZstdCodecObject",
]
