"""The types of the v2 codecs: the parameters of each codec that numcodecs configures.

This module holds types only and imports no definition. Each codec's
submodule of `zarr_metadata.v2.codec` re-exports its names, and holds the
codec's definition and rules.
"""

from typing import Annotated, Literal, NotRequired

from annotated_types import Ge, Interval, Le
from typing_extensions import ReadOnly, TypedDict

from zarr_metadata._common import JSONValue

# checksum


class ZarrV2Checksum32Parameters(TypedDict, closed=True):
    """`numcodecs.CRC32(location=None)` and the other 32-bit checksums. `location` is where the checksum is placed. When it is omitted or null, numcodecs places it at the start for crc32 and adler32, and at the end for crc32c."""

    location: NotRequired[ReadOnly[Literal["start", "end"] | None]]
    """Where the checksum is placed in the encoded chunk, `"start"` or `"end"`.

    Absent or null means the start for `crc32` and `adler32`, and the end for `crc32c`.
    """


# compression

ZarrV2CompressionLevel = Annotated[int, Interval(ge=-1, le=9)]
"""A zlib-style compression level: 0 to 9, or -1 for zlib's default. numcodecs writes the level as given."""

_CInt = Annotated[int, Interval(ge=-(2**31), le=2**31 - 1)]
"""An integer that numcodecs passes to C as an `int`: one that fits in 32 bits, with a sign. numcodecs raises `OverflowError` for any other."""


class ZarrV2ZlibParameters(TypedDict, closed=True):
    """`numcodecs.Zlib(level=1)`."""

    level: NotRequired[ReadOnly[ZarrV2CompressionLevel]]
    """The zlib compression level. Absent means 1."""


class ZarrV2GzipParameters(TypedDict, closed=True):
    """`numcodecs.GZip(level=1)`."""

    level: NotRequired[ReadOnly[ZarrV2CompressionLevel]]
    """The zlib compression level. Absent means 1."""


class ZarrV2Bz2Parameters(TypedDict, closed=True):
    """`numcodecs.BZ2(level=1)`: 1 to 9."""

    level: NotRequired[ReadOnly[Annotated[int, Interval(ge=1, le=9)]]]
    """The bzip2 compression level. Absent means 1."""


class ZarrV2LzmaParameters(TypedDict, closed=True):
    """`numcodecs.LZMA(format=1, check=-1, preset=None, filters=None)`, with the parameters as the `lzma` module accepts them."""

    format: NotRequired[ReadOnly[Annotated[int, Interval(ge=0, le=3)]]]
    """The container format, an `lzma.FORMAT_*` code. Absent means 1, `FORMAT_XZ`.

    2 is `FORMAT_ALONE`, and 3 is `FORMAT_RAW`, which requires `filters`. 0,
    `FORMAT_AUTO`, only decompresses: `lzma` refuses it when compressing.
    """
    check: NotRequired[ReadOnly[int]]
    """The integrity check, an `lzma.CHECK_*` code, or -1 for the format's default. Absent means -1.

    Only `FORMAT_XZ` takes a check other than -1 or `CHECK_NONE` (0).
    """
    preset: NotRequired[ReadOnly[Annotated[int, Ge(0)] | None]]
    """The compression preset, 0 to 9, optionally ORed with `lzma.PRESET_EXTREME`, or null for lzma's default, 6.

    Absent means null. `lzma` refuses a preset together with `filters`.
    """
    filters: NotRequired[ReadOnly[tuple[JSONValue, ...] | None]]
    """A custom filter chain, as `lzma` filter specifiers, or null to use `preset`. Absent means null."""


ZarrV2BloscCName = Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
"""The compressors inside Blosc."""


class ZarrV2BloscParameters(TypedDict, closed=True):
    """`numcodecs.Blosc(cname='lz4', clevel=5, shuffle=1, blocksize=0, typesize=None)`.

    For `shuffle`, -1 is automatic, 0 is none, 1 is byte shuffle, and 2 is
    bit shuffle. numcodecs writes a `shuffle` given as `True` or `False`
    as it was given, and reads it as 1 or 0. A `blocksize` of 0 lets Blosc
    choose the block size. numcodecs reads any `blocksize` that fits in a
    C `int`, negative ones included. Blosc reads any `blocksize` other
    than 0 as a size in bytes, and raises one below its minimum of 128
    bytes to that minimum, so a negative `blocksize` does not mean
    automatic.
    """

    cname: NotRequired[ReadOnly[ZarrV2BloscCName]]
    """The compressor that Blosc runs. Absent means `"lz4"`."""
    clevel: NotRequired[ReadOnly[Annotated[int, Interval(ge=0, le=9)]]]
    """The compression level. Absent means 5. At 0, Blosc stores the data uncompressed."""
    shuffle: NotRequired[ReadOnly[Literal[-1, 0, 1, 2] | bool]]
    """How Blosc shuffles the data before compressing it. Absent means 1, byte shuffle."""
    blocksize: NotRequired[ReadOnly[_CInt]]
    """The size in bytes of the blocks that Blosc compresses each chunk in. Absent means 0, which lets Blosc choose."""
    typesize: NotRequired[ReadOnly[Annotated[int, Interval(ge=1, le=2**31 - 1)] | None]]
    """The size in bytes of one element, the unit that Blosc shuffles by.

    Absent or null means the item size of the buffer that numcodecs is handed.
    """


class ZarrV2ZstdParameters(TypedDict, closed=True):
    """`numcodecs.Zstd(level=0, checksum=False)`. Level 0 is zstd's default level, and negative levels give up compression ratio for speed. zstd clamps a level outside its own range, -131072 to 22, to that range, so numcodecs reads any level that fits in a C `int`."""

    level: NotRequired[ReadOnly[_CInt]]
    """The zstd compression level. Absent means 0, zstd's default level."""
    checksum: NotRequired[ReadOnly[bool]]
    """Whether zstd writes a checksum of the content into each frame. Absent means false."""


class ZarrV2Lz4Parameters(TypedDict, closed=True):
    """`numcodecs.LZ4(acceleration=1)`. numcodecs passes `acceleration` to C as an `int`, and lz4 reads one below 1 as 1."""

    acceleration: NotRequired[ReadOnly[_CInt]]
    """How far lz4 trades compression ratio for speed: a larger value is faster. Absent means 1."""


# filters


class ZarrV2ShuffleParameters(TypedDict, closed=True):
    """`numcodecs.Shuffle(elementsize=4)`. An element size of 1 or less leaves the bytes as they are. numcodecs passes a larger one to C as a `Py_ssize_t`, so it is at most 2**63 - 1 on a 64-bit platform, and raises `OverflowError` for any other."""

    elementsize: NotRequired[ReadOnly[Annotated[int, Le(2**63 - 1)]]]
    """The size in bytes of the elements whose bytes are shuffled. Absent means 4."""


class ZarrV2DeltaParameters(TypedDict, closed=True):
    """`numcodecs.Delta(dtype, astype=None)`. When `astype` is omitted or null, it equals `dtype`."""

    dtype: ReadOnly[str]
    """The dtype of the decoded data."""
    astype: NotRequired[ReadOnly[str | None]]
    """The dtype of the encoded data. Absent or null means `dtype`."""


class ZarrV2FixedScaleOffsetParameters(TypedDict, closed=True):
    """`numcodecs.FixedScaleOffset(offset, scale, dtype, astype=None)`. When `astype` is omitted or null, it equals `dtype`."""

    offset: ReadOnly[float | int]
    """The value subtracted from each element before it is scaled."""
    scale: ReadOnly[float | int]
    """The factor that each element is multiplied by after the offset is subtracted. The product is rounded to the nearest integer."""
    dtype: ReadOnly[str]
    """The dtype of the decoded data."""
    astype: NotRequired[ReadOnly[str | None]]
    """The dtype of the encoded data. Absent or null means `dtype`."""


class ZarrV2QuantizeParameters(TypedDict, closed=True):
    """`numcodecs.Quantize(digits, dtype, astype=None)`: `dtype`, and `astype` when present, name a float. When `astype` is omitted or null, it equals `dtype`.

    `digits` is the number of decimal digits to keep, and may be negative.
    numcodecs computes `10.0 ** -digits` and a power of two above it, which
    overflow a float outside -308 to 307.
    """

    digits: ReadOnly[Annotated[int, Interval(ge=-308, le=307)]]
    """The number of decimal digits of precision to keep."""
    dtype: ReadOnly[str]
    """The float dtype of the decoded data."""
    astype: NotRequired[ReadOnly[str | None]]
    """The float dtype of the encoded data. Absent or null means `dtype`."""


class ZarrV2BitRoundParameters(TypedDict, closed=True):
    """`numcodecs.BitRound(keepbits)`."""

    keepbits: ReadOnly[Annotated[int, Ge(0)]]
    """The number of mantissa bits to keep.

    numcodecs refuses, when encoding, a value above the mantissa width of the
    data's float type: 10, 23 or 52 bits.
    """


class ZarrV2AsTypeParameters(TypedDict, closed=True):
    """`numcodecs.AsType(encode_dtype, decode_dtype)`."""

    encode_dtype: ReadOnly[str]
    """The dtype of the encoded data."""
    decode_dtype: ReadOnly[str]
    """The dtype of the decoded data."""


# vlen


class ZarrV2VLenArrayParameters(TypedDict, closed=True):
    """`numcodecs.VLenArray(dtype)`: the element type of each array."""

    dtype: ReadOnly[str]
    """The dtype of the elements of each variable-length array."""


__all__ = [
    "ZarrV2AsTypeParameters",
    "ZarrV2BitRoundParameters",
    "ZarrV2BloscCName",
    "ZarrV2BloscParameters",
    "ZarrV2Bz2Parameters",
    "ZarrV2Checksum32Parameters",
    "ZarrV2CompressionLevel",
    "ZarrV2DeltaParameters",
    "ZarrV2FixedScaleOffsetParameters",
    "ZarrV2GzipParameters",
    "ZarrV2Lz4Parameters",
    "ZarrV2LzmaParameters",
    "ZarrV2QuantizeParameters",
    "ZarrV2ShuffleParameters",
    "ZarrV2VLenArrayParameters",
    "ZarrV2ZlibParameters",
    "ZarrV2ZstdParameters",
]
