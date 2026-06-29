from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, TypedDict

import numcodecs
from numcodecs.blosc import Blosc
from packaging.version import Version

from zarr.abc.codec import BytesBytesCodec
from zarr.codecs._deprecated_enum import _coerce_enum_input, _DeprecatedStrEnumMeta
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, NamedRequiredConfig, parse_named_configuration
from zarr.core.dtype.common import HasItemSize

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

BloscShuffleLiteral = Literal["noshuffle", "shuffle", "bitshuffle"]
"""The shuffle values permitted for the blosc codec"""

BLOSC_SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")

BloscCnameLiteral = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""The codec identifiers used in the blosc codec"""

BLOSC_CNAME: Final = ("lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd")


class BloscConfigV2(TypedDict):
    """Configuration for the V2 Blosc codec"""

    cname: BloscCnameLiteral
    clevel: int
    shuffle: int
    blocksize: int
    typesize: NotRequired[int]


class BloscConfigV3(TypedDict):
    """Configuration for the V3 Blosc codec"""

    cname: BloscCnameLiteral
    clevel: int
    shuffle: BloscShuffleLiteral
    blocksize: int
    typesize: int


class BloscJSON_V3(NamedRequiredConfig[Literal["blosc"], BloscConfigV3]):
    """
    The JSON form of the Blosc codec in Zarr V3.
    """


class BloscShuffle(metaclass=_DeprecatedStrEnumMeta):
    """
    Deprecated. Pass a literal string (`"noshuffle"`, `"shuffle"`, or
    `"bitshuffle"`) directly to `BloscCodec` instead.
    """

    _members: ClassVar[dict[str, str]] = {
        "noshuffle": "noshuffle",
        "shuffle": "shuffle",
        "bitshuffle": "bitshuffle",
    }

    @staticmethod
    def from_int(num: int) -> BloscShuffleLiteral:
        mapping: dict[int, BloscShuffleLiteral] = {
            0: "noshuffle",
            1: "shuffle",
            2: "bitshuffle",
        }
        if num not in mapping:
            raise ValueError(f"Value must be between 0 and 2. Got {num}.")
        return mapping[num]


class BloscCname(metaclass=_DeprecatedStrEnumMeta):
    """
    Deprecated. Pass a literal string (one of `"lz4"`, `"lz4hc"`,
    `"blosclz"`, `"snappy"`, `"zlib"`, `"zstd"`) directly to
    `BloscCodec` instead.
    """

    _members: ClassVar[dict[str, str]] = {
        "lz4": "lz4",
        "lz4hc": "lz4hc",
        "blosclz": "blosclz",
        "snappy": "snappy",
        "zstd": "zstd",
        "zlib": "zlib",
    }


# See https://zarr.readthedocs.io/en/stable/user-guide/performance.html#configuring-blosc
numcodecs.blosc.use_threads = False


def parse_typesize(data: JSON) -> int:
    from zarr.core.json_parse import convert

    try:
        parsed: int = convert(data, int)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Value must be an int. Got {type(data)} instead.") from exc
    if parsed > 0:
        return parsed
    else:
        raise ValueError(
            f"Value must be greater than 0. Got {parsed}, which is less or equal to 0."
        )


# todo: real validation
def parse_clevel(data: JSON) -> int:
    from zarr.core.json_parse import convert

    try:
        parsed: int = convert(data, int)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Value should be an int. Got {type(data)} instead.") from exc
    return parsed


def parse_blocksize(data: JSON) -> int:
    from zarr.core.json_parse import convert

    try:
        parsed: int = convert(data, int)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Value should be an int. Got {type(data)} instead.") from exc
    return parsed


def _parse_cname(data: object) -> BloscCnameLiteral:
    if isinstance(data, str) and data in BLOSC_CNAME:
        return data  # type: ignore[return-value]
    raise ValueError(f"cname must be one of {list(BLOSC_CNAME)!r}. Got {data!r}.")


def _parse_shuffle(data: object) -> BloscShuffleLiteral:
    if isinstance(data, str) and data in BLOSC_SHUFFLE:
        return data  # type: ignore[return-value]
    raise ValueError(f"shuffle must be one of {list(BLOSC_SHUFFLE)!r}. Got {data!r}.")


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    """
    Blosc compression codec for zarr.

    Blosc is a high-performance compressor optimized for binary data. It uses a
    combination of blocking, shuffling, and fast compression algorithms to achieve
    excellent compression ratios and speed.

    Attributes
    ----------
    is_fixed_size : bool
        Always False for Blosc codec, as compression produces variable-sized output.
    typesize : int
        The data type size in bytes used for shuffle filtering.
    cname : BloscCnameLiteral
        The compression algorithm being used; one of "lz4", "lz4hc",
        "blosclz", "snappy", "zlib", or "zstd".
    clevel : int
        The compression level (0-9).
    shuffle : BloscShuffleLiteral
        The shuffle filter mode; one of "noshuffle", "shuffle", or
        "bitshuffle".
    blocksize : int
        The size of compressed blocks in bytes (0 for automatic).

    Parameters
    ----------
    typesize : int, optional
        The data type size in bytes. This affects how the shuffle filter processes
        the data. If None, defaults to 1 and the attribute is marked as tunable.
        Default: 1.
    cname : BloscCnameLiteral, optional
        The compression algorithm to use; one of "lz4", "lz4hc", "blosclz",
        "snappy", "zlib", or "zstd". Default is "zstd". Passing a `BloscCname`
        enum is deprecated.
    clevel : int, optional
        The compression level, from 0 (no compression) to 9 (maximum compression).
        Higher values provide better compression at the cost of speed. Default: 5.
    shuffle : BloscShuffleLiteral or None, optional
        The shuffle filter to apply before compression; one of "noshuffle",
        "shuffle", or "bitshuffle":

        - 'noshuffle': No shuffling
        - 'shuffle': Byte shuffling (better for typesize > 1)
        - 'bitshuffle': Bit shuffling (better for typesize == 1)

        If None, defaults to 'bitshuffle' and the attribute is marked
        as tunable. Default: 'bitshuffle'.
    blocksize : int, optional
        The requested size of compressed blocks in bytes. A value of 0 means
        automatic block size selection. Default: 0.

    Notes
    -----
    **Tunable attributes**: If `typesize` or `shuffle` are set to None during
    initialization, they are marked as tunable attributes. This means they can be
    adjusted later based on the data type of the array being compressed.

    **Thread Safety**: This codec sets `numcodecs.blosc.use_threads = False` at
    module import time to avoid threading issues in asyncio contexts.

    Examples
    --------
    Create a Blosc codec with default settings:

    >>> codec = BloscCodec()
    >>> codec.typesize
    1
    >>> codec.shuffle
    'bitshuffle'

    Create a codec with specific compression settings:

    >>> codec = BloscCodec(cname='zstd', clevel=9, shuffle='shuffle')
    >>> codec.cname
    'zstd'
    """

    # This attribute tracks parameters were set to None at init time, and thus tunable
    _tunable_attrs: set[Literal["typesize", "shuffle"]] = field(init=False)
    is_fixed_size = False

    typesize: int
    cname: BloscCnameLiteral
    clevel: int
    shuffle: BloscShuffleLiteral
    blocksize: int

    def __init__(
        self,
        *,
        typesize: int | None = None,
        cname: BloscCname | BloscCnameLiteral = "zstd",
        clevel: int = 5,
        shuffle: BloscShuffle | BloscShuffleLiteral | None = None,
        blocksize: int = 0,
    ) -> None:
        object.__setattr__(self, "_tunable_attrs", set())

        if typesize is None:
            typesize = 1
            self._tunable_attrs.update({"typesize"})

        if shuffle is None:
            shuffle = "bitshuffle"
            self._tunable_attrs.update({"shuffle"})

        cname = _coerce_enum_input(cname, "cname", "BloscCodec")  # type: ignore[assignment]
        shuffle = _coerce_enum_input(shuffle, "shuffle", "BloscCodec")  # type: ignore[assignment]

        typesize_parsed = parse_typesize(typesize)
        cname_parsed = _parse_cname(cname)
        clevel_parsed = parse_clevel(clevel)
        shuffle_parsed = _parse_shuffle(shuffle)
        blocksize_parsed = parse_blocksize(blocksize)

        object.__setattr__(self, "typesize", typesize_parsed)
        object.__setattr__(self, "cname", cname_parsed)
        object.__setattr__(self, "clevel", clevel_parsed)
        object.__setattr__(self, "shuffle", shuffle_parsed)
        object.__setattr__(self, "blocksize", blocksize_parsed)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(data, "blosc")
        return cls(**configuration_parsed)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        result: BloscJSON_V3 = {
            "name": "blosc",
            "configuration": {
                "typesize": self.typesize,
                "cname": self.cname,
                "clevel": self.clevel,
                "shuffle": self.shuffle,
                "blocksize": self.blocksize,
            },
        }
        return result  # type: ignore[return-value]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """
        Create a new codec with typesize and shuffle parameters adjusted
        according to the size of each element in the data type
        associated with array_spec. Parameters are only updated if they were set to
        None when self.__init__ was called.
        """
        item_size = 1
        if isinstance(array_spec.dtype, HasItemSize):
            item_size = array_spec.dtype.item_size
        new_codec = self
        if "typesize" in self._tunable_attrs:
            new_codec = replace(new_codec, typesize=item_size)
        if "shuffle" in self._tunable_attrs:
            new_codec = replace(
                new_codec,
                shuffle=("bitshuffle" if item_size == 1 else "shuffle"),
            )

        return new_codec

    @cached_property
    def _blosc_codec(self) -> Blosc:
        map_shuffle_str_to_int: dict[BloscShuffleLiteral, int] = {
            "noshuffle": 0,
            "shuffle": 1,
            "bitshuffle": 2,
        }
        config_dict: BloscConfigV2 = {
            "cname": self.cname,
            "clevel": self.clevel,
            "shuffle": map_shuffle_str_to_int[self.shuffle],
            "blocksize": self.blocksize,
        }
        # See https://github.com/zarr-developers/numcodecs/pull/713
        if Version(numcodecs.__version__) >= Version("0.16.0"):
            config_dict["typesize"] = self.typesize
        return Blosc.from_config(config_dict)

    def _decode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return as_numpy_array_wrapper(self._blosc_codec.decode, chunk_bytes, chunk_spec.prototype)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(self._decode_sync, chunk_bytes, chunk_spec)

    def _encode_sync(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        # Since blosc only support host memory, we convert the input and output of the encoding
        # between numpy array and buffer
        return chunk_spec.prototype.buffer.from_bytes(
            self._blosc_codec.encode(chunk_bytes.as_numpy_array())
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        return await asyncio.to_thread(self._encode_sync, chunk_bytes, chunk_spec)

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
