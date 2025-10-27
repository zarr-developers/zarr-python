from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, replace
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict

import numcodecs
from numcodecs.blosc import Blosc
from packaging.version import Version

from zarr.abc.codec import BytesBytesCodec
from zarr.core.buffer.cpu import as_numpy_array_wrapper
from zarr.core.common import JSON, NamedRequiredConfig, parse_enum, parse_named_configuration
from zarr.core.dtype.common import HasItemSize
from zarr.errors import ZarrDeprecationWarning

if TYPE_CHECKING:
    from typing import Self

    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer

Shuffle = Literal["noshuffle", "shuffle", "bitshuffle"]
"""The shuffle values permitted for the blosc codec"""

SHUFFLE: Final = ("noshuffle", "shuffle", "bitshuffle")

CName = Literal["lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd"]
"""The codec identifiers used in the blosc codec """


class BloscConfigV2(TypedDict):
    """Configuration for the V2 Blosc codec"""

    cname: CName
    clevel: int
    shuffle: int
    blocksize: int
    typesize: NotRequired[int]


class BloscConfigV3(TypedDict):
    """Configuration for the V3 Blosc codec"""

    cname: CName
    clevel: int
    shuffle: Shuffle
    blocksize: int
    typesize: int


class BloscJSON_V3(NamedRequiredConfig[Literal["blosc"], BloscConfigV3]):
    """
    The JSON form of the Blosc codec in Zarr V3.
    """


class BloscShuffle(Enum):
    """
    Enum for shuffle filter used by blosc.
    """

    noshuffle = "noshuffle"
    shuffle = "shuffle"
    bitshuffle = "bitshuffle"

    @classmethod
    def from_int(cls, num: int) -> BloscShuffle:
        blosc_shuffle_int_to_str = {
            0: "noshuffle",
            1: "shuffle",
            2: "bitshuffle",
        }
        if num not in blosc_shuffle_int_to_str:
            raise ValueError(f"Value must be between 0 and 2. Got {num}.")
        return BloscShuffle[blosc_shuffle_int_to_str[num]]


class BloscCname(Enum):
    """
    Enum for compression library used by blosc.
    """

    lz4 = "lz4"
    lz4hc = "lz4hc"
    blosclz = "blosclz"
    zstd = "zstd"
    snappy = "snappy"
    zlib = "zlib"


# See https://zarr.readthedocs.io/en/stable/user-guide/performance.html#configuring-blosc
numcodecs.blosc.use_threads = False


def parse_typesize(data: JSON) -> int:
    if isinstance(data, int):
        if data > 0:
            return data
        else:
            raise ValueError(
                f"Value must be greater than 0. Got {data}, which is less or equal to 0."
            )
    raise TypeError(f"Value must be an int. Got {type(data)} instead.")


# todo: real validation
def parse_clevel(data: JSON) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Value should be an int. Got {type(data)} instead.")


def parse_blocksize(data: JSON) -> int:
    if isinstance(data, int):
        return data
    raise TypeError(f"Value should be an int. Got {type(data)} instead.")


@dataclass(frozen=True)
class BloscCodec(BytesBytesCodec):
    """
    Blosc compression codec for zarr.

    Blosc is a high-performance compressor optimized for binary data. It uses a
    combination of blocking, shuffling, and fast compression algorithms to achieve
    excellent compression ratios and speed.

    Attributes
    ----------
    tunable_attrs : set of {'typesize', 'shuffle'}
        Attributes that will be automatically tuned when `evolve_from_array_spec()`
        is called. By default, contains {'typesize', 'shuffle'}. When either
        `typesize` or `shuffle` is explicitly set to None during initialization,
        the corresponding attribute is added to this set (if not already present),
        allowing it to be overridden based on the array's dtype.
    is_fixed_size : bool
        Always False for Blosc codec, as compression produces variable-sized output.
    typesize : int
        The data type size in bytes used for shuffle filtering.
    cname : BloscCname
        The compression algorithm being used (lz4, lz4hc, blosclz, snappy, zlib, or zstd).
    clevel : int
        The compression level (0-9).
    shuffle : BloscShuffle
        The shuffle filter mode (noshuffle, shuffle, or bitshuffle).
    blocksize : int
        The size of compressed blocks in bytes (0 for automatic).

    Parameters
    ----------
    typesize : int, optional
        The data type size in bytes. This affects how the shuffle filter processes
        the data. If None (deprecated), defaults to 1 and the attribute is marked
        as tunable. Default: 1.
    cname : BloscCname or {'lz4', 'lz4hc', 'blosclz', 'snappy', 'zlib', 'zstd'}, optional
        The compression algorithm to use. Default: 'zstd'.
    clevel : int, optional
        The compression level, from 0 (no compression) to 9 (maximum compression).
        Higher values provide better compression at the cost of speed. Default: 5.
    shuffle : BloscShuffle or {'noshuffle', 'shuffle', 'bitshuffle'}, optional
        The shuffle filter to apply before compression:

        - 'noshuffle': No shuffling
        - 'shuffle': Byte shuffling (better for typesize > 1)
        - 'bitshuffle': Bit shuffling (better for typesize == 1)

        If None (deprecated), defaults to 'bitshuffle' and the attribute is marked
        as tunable. Default: 'bitshuffle'.
    blocksize : int, optional
        The requested size of compressed blocks in bytes. A value of 0 means
        automatic block size selection. Default: 0.
    tunable_attrs : set of {'typesize', 'shuffle'}, optional
        Names of attributes that can be automatically adjusted by
        `evolve_from_array_spec()`. This allows the codec to adapt its parameters
        based on the array's data type when the array is created. If None, defaults
        to {'typesize', 'shuffle'}.

    Notes
    -----
    **Tunable Attributes Logic**:

    The `tunable_attrs` mechanism allows codec parameters to be automatically
    adjusted based on the array's data type:

    1. **Initialization**: During `__init__`, if `tunable_attrs` is None, it
       defaults to {'typesize', 'shuffle'}. This means both attributes can be
       tuned by default.

    2. **Deprecated None Values**: If `typesize` or `shuffle` is explicitly set
       to None:

       - A deprecation warning is issued
       - The parameter is set to a default value (1 for typesize, 'bitshuffle'
         for shuffle)
       - The attribute name is added to `tunable_attrs`

    3. **Evolution**: When `evolve_from_array_spec()` is called (typically during
       array creation), it creates a new codec instance with updated parameters:

       - If 'typesize' is in `tunable_attrs`, it's set to the array dtype's
         item size
       - If 'shuffle' is in `tunable_attrs`, it's set to 'bitshuffle' if
         item_size == 1, otherwise 'shuffle'

    4. **Explicit Values**: If you explicitly set `typesize=4` or
       `shuffle='noshuffle'`, these values are NOT in `tunable_attrs` by default
       and will not be changed by `evolve_from_array_spec()`.

    **Thread Safety**: This codec sets `numcodecs.blosc.use_threads = False` at
    module import time to avoid threading issues in asyncio contexts.

    Examples
    --------
    Create a Blosc codec with default settings:

    >>> codec = BloscCodec()
    >>> codec.typesize
    1
    >>> codec.shuffle
    <BloscShuffle.bitshuffle: 'bitshuffle'>

    Create a codec with specific compression settings:

    >>> codec = BloscCodec(cname='zstd', clevel=9, shuffle='shuffle')
    >>> codec.cname
    <BloscCname.zstd: 'zstd'>

    Use deprecated None values (will be tuned automatically):

    >>> codec = BloscCodec(typesize=None, shuffle=None)  # doctest: +SKIP
    DeprecationWarning: The typesize parameter was set to None...
    >>> 'typesize' in codec.tunable_attrs
    True
    >>> 'shuffle' in codec.tunable_attrs
    True

    Prevent automatic tuning:

    >>> codec = BloscCodec(typesize=4, shuffle='noshuffle', tunable_attrs=set())
    >>> codec.tunable_attrs
    set()

    See Also
    --------
    BloscShuffle : Enum for shuffle filter options
    BloscCname : Enum for compression algorithm options
    """

    tunable_attrs: set[Literal["typesize", "shuffle"]]
    is_fixed_size = False

    typesize: int
    cname: BloscCname
    clevel: int
    shuffle: BloscShuffle
    blocksize: int

    def __init__(
        self,
        *,
        typesize: int | None = 1,
        cname: BloscCname | CName = BloscCname.zstd,
        clevel: int = 5,
        shuffle: BloscShuffle | Shuffle | None = "bitshuffle",
        blocksize: int = 0,
        tunable_attrs: set[Literal["typesize", "shuffle"]] | None = None,
    ) -> None:
        if tunable_attrs is None:
            object.__setattr__(self, "tunable_attrs", set())
        else:
            object.__setattr__(self, "tunable_attrs", tunable_attrs)

        # If typesize was set to None: warn, replace it with a valid typesize
        # and flag the typesize attribute as safe to replace later
        if typesize in (None, 1):
            if typesize is None:
                msg = (
                    "The typesize parameter was set to None. This is deprecated. "
                    "Provide a positive int for the typesize parameter instead. "
                )
                warnings.warn(msg, ZarrDeprecationWarning, stacklevel=2)
                typesize = 1
            self.tunable_attrs.update({"typesize"})

        # If shuffle was set to None: warn, replace it with a valid typesize
        # and flag the shuffle attribute as safe to replace later
        if shuffle is None or shuffle == "bitshuffle" or shuffle == BloscShuffle.bitshuffle:
            if shuffle is None:
                msg = (
                    "The shuffle parameter was set to None. This is deprecated. "
                    "Provide a valid shuffle literal string -- "
                    f"one of {SHUFFLE!r} -- instead."
                )
                warnings.warn(msg, ZarrDeprecationWarning, stacklevel=2)
                shuffle = BloscShuffle.bitshuffle
            self.tunable_attrs.update({"shuffle"})

        typesize_parsed = parse_typesize(typesize)
        cname_parsed = parse_enum(cname, BloscCname)
        clevel_parsed = parse_clevel(clevel)
        shuffle_parsed = parse_enum(shuffle, BloscShuffle)
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
                "cname": self.cname.value,
                "clevel": self.clevel,
                "shuffle": self.shuffle.value,
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
        if "typesize" in self.tunable_attrs:
            new_codec = replace(new_codec, typesize=item_size)
        if "shuffle" in self.tunable_attrs:
            new_codec = replace(
                new_codec,
                shuffle=(BloscShuffle.bitshuffle if item_size == 1 else BloscShuffle.shuffle),
            )

        return new_codec

    @cached_property
    def _blosc_codec(self) -> Blosc:
        map_shuffle_str_to_int = {
            BloscShuffle.noshuffle: 0,
            BloscShuffle.shuffle: 1,
            BloscShuffle.bitshuffle: 2,
        }
        config_dict: BloscConfigV2 = {
            "cname": self.cname.name,  # type: ignore[typeddict-item]
            "clevel": self.clevel,
            "shuffle": map_shuffle_str_to_int[self.shuffle],
            "blocksize": self.blocksize,
        }
        # See https://github.com/zarr-developers/numcodecs/pull/713
        if Version(numcodecs.__version__) >= Version("0.16.0"):
            config_dict["typesize"] = self.typesize
        return Blosc.from_config(config_dict)

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        return await asyncio.to_thread(
            as_numpy_array_wrapper, self._blosc_codec.decode, chunk_bytes, chunk_spec.prototype
        )

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        # Since blosc only support host memory, we convert the input and output of the encoding
        # between numpy array and buffer
        return await asyncio.to_thread(
            lambda chunk: chunk_spec.prototype.buffer.from_bytes(
                self._blosc_codec.encode(chunk.as_numpy_array())
            ),
            chunk_bytes,
        )

    def compute_encoded_size(self, _input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        raise NotImplementedError
