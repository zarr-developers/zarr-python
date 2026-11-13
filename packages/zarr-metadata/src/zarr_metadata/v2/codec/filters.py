"""The v2 filters that numcodecs 0.16 configures: `shuffle`, `delta`, `fixedscaleoffset`, `quantize`, `bitround`, `astype` and `packbits`.

A dtype parameter is a NumPy typestr, as numcodecs writes it. A
parameter that numcodecs gives a default is optional, and the canonical
form of each codec fills it in, as `get_config()` writes it. The
canonical form writes each dtype parameter as NumPy spells it, as an
array's dtype compares, so `<i1` and `|i1` are one parameter.
"""

from __future__ import annotations

from typing import Final

from zarr_metadata._common import EmptyConfiguration
from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2._types.codec import (
    ZarrV2AsTypeParameters,
    ZarrV2BitRoundParameters,
    ZarrV2DeltaParameters,
    ZarrV2FixedScaleOffsetParameters,
    ZarrV2QuantizeParameters,
    ZarrV2ShuffleParameters,
)
from zarr_metadata.v2.codec._dtype import canonical_typestr, dtype_parameter


def _shuffle_canonical(configuration: ZarrV2ShuffleParameters) -> ZarrV2ShuffleParameters:
    """Fill in the default `elementsize`, 4, as numcodecs writes it."""
    return {"elementsize": 4, **configuration}


def _typestrs(dtype: str, astype: str | None) -> tuple[str, str]:
    """`dtype` and `astype` as NumPy spells them, with an `astype` that is omitted or null filled in with the `dtype`, as numcodecs writes them."""
    canonical = canonical_typestr(dtype)
    return canonical, canonical if astype is None else canonical_typestr(astype)


def _delta_canonical(configuration: ZarrV2DeltaParameters) -> ZarrV2DeltaParameters:
    """Fill in an `astype` that is omitted or null with the `dtype`, and write each dtype as NumPy spells it, as numcodecs writes them."""
    dtype, astype = _typestrs(configuration["dtype"], configuration.get("astype"))
    return {**configuration, "dtype": dtype, "astype": astype}


def _fixedscaleoffset_canonical(
    configuration: ZarrV2FixedScaleOffsetParameters,
) -> ZarrV2FixedScaleOffsetParameters:
    """Fill in an `astype` that is omitted or null with the `dtype`, and write each dtype as NumPy spells it, as numcodecs writes them.

    `offset` and `scale` are kept as written. NumPy computes with an
    integer and a float differently for integer and half-precision dtypes,
    so numcodecs encodes a `scale` of 10 and of 10.0 differently.
    """
    dtype, astype = _typestrs(configuration["dtype"], configuration.get("astype"))
    return {**configuration, "dtype": dtype, "astype": astype}


def _quantize_canonical(configuration: ZarrV2QuantizeParameters) -> ZarrV2QuantizeParameters:
    """Fill in an `astype` that is omitted or null with the `dtype`, and write each dtype as NumPy spells it, as numcodecs writes them."""
    dtype, astype = _typestrs(configuration["dtype"], configuration.get("astype"))
    return {**configuration, "dtype": dtype, "astype": astype}


def _astype_canonical(configuration: ZarrV2AsTypeParameters) -> ZarrV2AsTypeParameters:
    """Write each dtype as NumPy spells it, as numcodecs writes them."""
    return {
        **configuration,
        "encode_dtype": canonical_typestr(configuration["encode_dtype"]),
        "decode_dtype": canonical_typestr(configuration["decode_dtype"]),
    }


SHUFFLE_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="shuffle", configuration=ZarrV2ShuffleParameters, canonical=_shuffle_canonical
)
"""`numcodecs.Shuffle`."""
DELTA_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="delta",
    configuration=ZarrV2DeltaParameters,
    rules=dtype_parameter("dtype", "astype"),
    canonical=_delta_canonical,
)
"""`numcodecs.Delta`."""
FIXEDSCALEOFFSET_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="fixedscaleoffset",
    configuration=ZarrV2FixedScaleOffsetParameters,
    rules=dtype_parameter("dtype", "astype"),
    canonical=_fixedscaleoffset_canonical,
)
"""`numcodecs.FixedScaleOffset`."""
QUANTIZE_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="quantize",
    configuration=ZarrV2QuantizeParameters,
    rules=dtype_parameter("dtype", "astype", float_only=True),
    canonical=_quantize_canonical,
)
"""`numcodecs.Quantize`."""
BITROUND_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="bitround", configuration=ZarrV2BitRoundParameters
)
"""`numcodecs.BitRound`."""
ASTYPE_CODEC_V2: Final = ZarrV2CodecDefinition(
    name="astype",
    configuration=ZarrV2AsTypeParameters,
    rules=dtype_parameter("encode_dtype", "decode_dtype"),
    canonical=_astype_canonical,
)
"""`numcodecs.AsType`."""
PACKBITS_CODEC_V2: Final = ZarrV2CodecDefinition(name="packbits", configuration=EmptyConfiguration)
"""`numcodecs.PackBits`. It has no configuration."""

__all__ = [
    "ASTYPE_CODEC_V2",
    "BITROUND_CODEC_V2",
    "DELTA_CODEC_V2",
    "FIXEDSCALEOFFSET_CODEC_V2",
    "PACKBITS_CODEC_V2",
    "QUANTIZE_CODEC_V2",
    "SHUFFLE_CODEC_V2",
    "ZarrV2AsTypeParameters",
    "ZarrV2BitRoundParameters",
    "ZarrV2DeltaParameters",
    "ZarrV2FixedScaleOffsetParameters",
    "ZarrV2QuantizeParameters",
    "ZarrV2ShuffleParameters",
]
