"""
Sharding-indexed codec types.

See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NotRequired, Self

from typing_extensions import TypedDict

from zarr_metadata.model._sentinel import UNSET
from zarr_metadata.model._validation import ValidationProblem
from zarr_metadata.v3._chain import chain_problems
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._entity import (
    ArrayBytesCodec,
    CodecEntity,
    Opaque,
    problem,
)
from zarr_metadata.v3._parts import (
    UNKNOWN_GRID,
    ArrayParts,
    ChunkGrid,
    shard_index_grid,
)
from zarr_metadata.v3.data_type.uint64 import Uint64DataType

if TYPE_CHECKING:
    from collections.abc import Iterator

SHARDING_INDEXED_CODEC_NAME: Final = "sharding_indexed"
"""The `name` field value of the `sharding_indexed` codec."""

ShardingIndexedCodecName = Literal["sharding_indexed"]
"""Literal type of the `name` field of the `sharding_indexed` codec."""

ShardingIndexLocation = Literal["start", "end"]
"""Literal type of the position of the shard index within the encoded shard."""

SHARDING_INDEX_LOCATION: Final = ("start", "end")
"""Tuple of permitted values for the `index_location` field of the `sharding_indexed` codec."""


class ShardingIndexedCodecConfiguration(TypedDict, closed=True):
    """
    Configuration for the Zarr v3 `sharding_indexed` codec.

    `chunk_shape` is the shape of inner chunks along each dimension;
    it must evenly divide the shard shape.

    `codecs` is the codec pipeline applied to each inner chunk; exactly
    one array-to-bytes codec is required.

    `index_codecs` is the codec pipeline applied to the shard index;
    it must be deterministic (no variable-size compression).
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L147-L155

    `index_location` defaults to `"end"` per the spec.
      https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L157-L161
    """

    chunk_shape: tuple[int, ...]
    codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    index_codecs: tuple[ZarrV3MetadataFieldJSON, ...]
    index_location: NotRequired[ShardingIndexLocation]


class ShardingIndexedCodecObject(TypedDict, closed=True):
    """`sharding_indexed` codec metadata in object form."""

    name: ShardingIndexedCodecName
    configuration: ShardingIndexedCodecConfiguration
    must_understand: NotRequired[bool]


ShardingIndexedCodecMetadata = ShardingIndexedCodecObject
"""Permitted JSON shape for `sharding_indexed` codec metadata.

The configuration has multiple required keys (`chunk_shape`, `codecs`,
`index_codecs`), so only the object form is valid; the short-hand-name
form is not permitted by the spec for this codec.
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/codecs/sharding-indexed/index.rst#L141-L155 (required members)
  https://github.com/zarr-developers/zarr-specs/blob/fc7dd9c9beb5a50b87f9b08b00bf50fc0048482f/docs/v3/core/index.rst#L1562-L1564 (short-hand names only "if no configuration metadata is required")
"""

__all__ = [
    "SHARDING_INDEXED_CODEC_NAME",
    "SHARDING_INDEX_LOCATION",
    "ShardingIndexLocation",
    "ShardingIndexedCodec",
    "ShardingIndexedCodecConfiguration",
    "ShardingIndexedCodecMetadata",
    "ShardingIndexedCodecName",
    "ShardingIndexedCodecObject",
]


@dataclass(frozen=True)
class ShardingIndexedOptions:
    """What `sharding_indexed` is configured with."""

    chunk_shape: tuple[int, ...]
    codecs: tuple[CodecEntity | Opaque, ...]
    index_codecs: tuple[CodecEntity | Opaque, ...]
    index_location: ShardingIndexLocation | UNSET = UNSET


def sharding_problems(codec: "ShardingIndexedCodec", /) -> "Iterator[ValidationProblem]":
    for index, extent in enumerate(codec.chunk_shape):
        if extent < 1:
            yield ValidationProblem(
                ("chunk_shape", index), f"expected an integer >= 1, got {extent}", "invalid_value"
            )


@dataclass(frozen=True)
class ShardingIndexedCodec(ArrayBytesCodec):
    """The `sharding_indexed` codec, coerced from its metadata.

    Holds two codec pipelines, so it is one of the few entities that
    needs the scope it is being read in: an entry of either pipeline is
    itself an entity, read the same way this one was.
    """

    configuration: ShardingIndexedOptions

    identifier: ClassVar[str] = SHARDING_INDEXED_CODEC_NAME
    variable_size: ClassVar[bool] = True

    problems = sharding_problems

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        return self.configuration.chunk_shape

    @property
    def codecs(self) -> tuple[CodecEntity | Opaque, ...]:
        return self.configuration.codecs

    @property
    def index_codecs(self) -> tuple[CodecEntity | Opaque, ...]:
        return self.configuration.index_codecs

    @property
    def index_location(self) -> ShardingIndexLocation | UNSET:
        return self.configuration.index_location

    def canonical(self) -> Self:
        """Each pipeline's codecs in their own canonical form."""
        return self.with_configuration(
            codecs=tuple(codec.canonical() for codec in self.codecs),
            index_codecs=tuple(codec.canonical() for codec in self.index_codecs),
        )

    def incoming_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """This shard against the array reaching it, and its two pipelines.

        One sharding configuration encodes every chunk, so its inner
        shape has to divide all of them. Under a rectilinear grid an axis
        has several lengths and the inner extent must divide each; an axis
        whose lengths are unknown declines while the others are judged.
        """
        found = list(self._inner_chunk_problems(incoming))
        # Both pipelines start from this codec's own configuration and
        # from the spec, so neither waits on what reached the codec. An
        # unreadable codec upstream costs the element type and the
        # enclosing extents; it does not make the inner chunk shape
        # unknown, and the index is a `uint64` array whatever precedes it.
        outer = incoming.grid if incoming is not None else UNKNOWN_GRID
        found.extend(
            chain_problems(
                self.codecs,
                ArrayParts(
                    ChunkGrid.regular(self.chunk_shape),
                    incoming.data_type if incoming is not None else None,
                ),
                ("codecs",),
            )
        )
        found.extend(
            chain_problems(
                self.index_codecs,
                ArrayParts(shard_index_grid(outer, self.chunk_shape), Uint64DataType()),
                ("index_codecs",),
            )
        )
        found.extend(
            ValidationProblem(
                ("index_codecs", index),
                f"{type(codec).identifier!r} produces variable-size output; "
                "index_codecs must be fixed-size",
                "invalid_value",
            )
            for index, codec in enumerate(self.index_codecs)
            if isinstance(codec, CodecEntity) and type(codec).variable_size
        )
        return tuple(found)

    def _inner_chunk_problems(self, incoming: ArrayParts | None) -> tuple[ValidationProblem, ...]:
        """Whether the inner chunk divides every chunk this shard receives."""
        if incoming is None or incoming.grid.rank is None:
            return ()
        if len(self.chunk_shape) != incoming.grid.rank:
            return problem(
                ("chunk_shape",),
                f"chunk_shape has {len(self.chunk_shape)} entries but the incoming array "
                f"has {incoming.grid.rank} dimensions",
                "invalid_value",
            )
        found: list[ValidationProblem] = []
        for position, extent in enumerate(self.chunk_shape):
            lengths = incoming.grid.axis(position)
            if lengths is None or extent < 1:
                continue
            indivisible = sorted(length for length in lengths if length % extent != 0)
            if len(indivisible) != 0:
                found.extend(
                    problem(
                        ("chunk_shape", position),
                        f"inner chunk extent {extent} does not evenly divide the incoming "
                        f"extent {indivisible[0]}",
                        "invalid_value",
                    )
                )
        return tuple(found)
