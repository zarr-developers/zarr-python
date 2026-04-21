"""
Smoke test: every public name is importable.
"""

from __future__ import annotations


def test_top_level_imports() -> None:
    from zarr_metadata import (
        JSON,
        ArrayMetadata,
        ArrayMetadataV2,
        ArrayMetadataV3,
        GroupMetadata,
        GroupMetadataV2,
        GroupMetadataV3,
        NamedConfig,
        NamedRequiredConfig,
    )

    _ = (
        JSON,
        ArrayMetadata,
        ArrayMetadataV2,
        ArrayMetadataV3,
        GroupMetadata,
        GroupMetadataV2,
        GroupMetadataV3,
        NamedConfig,
        NamedRequiredConfig,
    )


def test_v2_imports() -> None:
    from zarr_metadata.v2 import (
        ArrayMetadataV2,
        ConsolidatedMetadataV2,
        DataTypeV2,
        DataTypeV2Structured,
        GroupMetadataV2,
    )

    _ = (
        ArrayMetadataV2,
        ConsolidatedMetadataV2,
        DataTypeV2,
        DataTypeV2Structured,
        GroupMetadataV2,
    )


def test_v3_imports() -> None:
    from zarr_metadata.v3 import (
        AllowedExtraField,
        ArrayMetadataV3,
        ConsolidatedMetadataV3,
        GroupMetadataV3,
        RectilinearChunkGrid,
        RectilinearChunkGridConfig,
        RectilinearDimSpec,
        RegularChunkGrid,
        RegularChunkGridConfig,
    )

    _ = (
        AllowedExtraField,
        ArrayMetadataV3,
        ConsolidatedMetadataV3,
        GroupMetadataV3,
        RectilinearChunkGrid,
        RectilinearChunkGridConfig,
        RectilinearDimSpec,
        RegularChunkGrid,
        RegularChunkGridConfig,
    )


def test_codec_imports() -> None:
    from zarr_metadata.codec import Codec
    from zarr_metadata.codec.blosc import (
        BloscCodecConfiguration,
        BloscCodecConfigurationNumcodecs,
        BloscCodecConfigurationV1,
        CName,
        Shuffle,
    )

    _ = (
        Codec,
        BloscCodecConfiguration,
        BloscCodecConfigurationNumcodecs,
        BloscCodecConfigurationV1,
        CName,
        Shuffle,
    )


def test_dtype_imports() -> None:
    from zarr_metadata.dtype import DType
    from zarr_metadata.dtype.bytes import FixedLengthBytesConfig
    from zarr_metadata.dtype.string import LengthBytesConfig
    from zarr_metadata.dtype.time import DateTimeUnit, TimeConfig

    _ = (
        DType,
        FixedLengthBytesConfig,
        LengthBytesConfig,
        DateTimeUnit,
        TimeConfig,
    )
