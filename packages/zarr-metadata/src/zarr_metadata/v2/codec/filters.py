"""The v2 filters numcodecs 0.16 configures: `shuffle`, `delta`, `fixedscaleoffset`, `quantize`, `bitround`, `astype` and `packbits`.

A dtype parameter is a NumPy typestr, as numcodecs writes one; a
parameter numcodecs defaults is optional.
"""

from __future__ import annotations

from typing import Annotated, Final, NotRequired

from annotated_types import Ge
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata.v2._definition import ZarrV2CodecDefinition
from zarr_metadata.v2.codec._dtype import dtype_parameter
from zarr_metadata.v3._definition import EmptyConfiguration


class ZarrV2ShuffleParameters(TypedDict, closed=True):
    """`numcodecs.Shuffle(elementsize=4)`."""

    elementsize: NotRequired[ReadOnly[Annotated[int, Ge(1)]]]


class ZarrV2DeltaParameters(TypedDict, closed=True):
    """`numcodecs.Delta(dtype, astype=None)`: `astype` is `dtype` when left out."""

    dtype: ReadOnly[str]
    astype: NotRequired[ReadOnly[str]]


class ZarrV2FixedScaleOffsetParameters(TypedDict, closed=True):
    """`numcodecs.FixedScaleOffset(offset, scale, dtype, astype=None)`."""

    offset: ReadOnly[float | int]
    scale: ReadOnly[float | int]
    dtype: ReadOnly[str]
    astype: NotRequired[ReadOnly[str]]


class ZarrV2QuantizeParameters(TypedDict, closed=True):
    """`numcodecs.Quantize(digits, dtype, astype=None)`: `dtype` names a float."""

    digits: ReadOnly[Annotated[int, Ge(0)]]
    dtype: ReadOnly[str]
    astype: NotRequired[ReadOnly[str]]


class ZarrV2BitRoundParameters(TypedDict, closed=True):
    """`numcodecs.BitRound(keepbits)`."""

    keepbits: ReadOnly[Annotated[int, Ge(0)]]


class ZarrV2AsTypeParameters(TypedDict, closed=True):
    """`numcodecs.AsType(encode_dtype, decode_dtype)`."""

    encode_dtype: ReadOnly[str]
    decode_dtype: ReadOnly[str]


SHUFFLE_V2: Final = ZarrV2CodecDefinition(name="shuffle", configuration=ZarrV2ShuffleParameters)
"""`numcodecs.Shuffle`."""
DELTA_V2: Final = ZarrV2CodecDefinition(
    name="delta", configuration=ZarrV2DeltaParameters, rules=dtype_parameter("dtype", "astype")
)
"""`numcodecs.Delta`."""
FIXEDSCALEOFFSET_V2: Final = ZarrV2CodecDefinition(
    name="fixedscaleoffset",
    configuration=ZarrV2FixedScaleOffsetParameters,
    rules=dtype_parameter("dtype", "astype"),
)
"""`numcodecs.FixedScaleOffset`."""
QUANTIZE_V2: Final = ZarrV2CodecDefinition(
    name="quantize",
    configuration=ZarrV2QuantizeParameters,
    rules=dtype_parameter("dtype", "astype", float_only=True),
)
"""`numcodecs.Quantize`."""
BITROUND_V2: Final = ZarrV2CodecDefinition(name="bitround", configuration=ZarrV2BitRoundParameters)
"""`numcodecs.BitRound`."""
ASTYPE_V2: Final = ZarrV2CodecDefinition(
    name="astype",
    configuration=ZarrV2AsTypeParameters,
    rules=dtype_parameter("encode_dtype", "decode_dtype"),
)
"""`numcodecs.AsType`."""
PACKBITS_V2: Final = ZarrV2CodecDefinition(name="packbits", configuration=EmptyConfiguration)
"""`numcodecs.PackBits`: nothing to configure."""

__all__ = [
    "ASTYPE_V2",
    "BITROUND_V2",
    "DELTA_V2",
    "FIXEDSCALEOFFSET_V2",
    "PACKBITS_V2",
    "QUANTIZE_V2",
    "SHUFFLE_V2",
    "ZarrV2AsTypeParameters",
    "ZarrV2BitRoundParameters",
    "ZarrV2DeltaParameters",
    "ZarrV2FixedScaleOffsetParameters",
    "ZarrV2QuantizeParameters",
    "ZarrV2ShuffleParameters",
]
