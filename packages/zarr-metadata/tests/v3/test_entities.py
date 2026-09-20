"""The correspondence between an entity's dataclass and its TypedDicts.

A configuration TypedDict unpacked is exactly the dataclass constructor's
signature, and the object TypedDict is exactly what `to_json` returns.
Asserting the first is what lets the dataclass carry its own checks
instead of a hand-written per-member table elsewhere: the two cannot
drift, because one drifting makes this fail.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, get_type_hints

import pytest

from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RectilinearChunkGrid,
    RectilinearChunkGridConfiguration,
)
from zarr_metadata.v3.chunk_grid.regular import RegularChunkGrid, RegularChunkGridConfiguration
from zarr_metadata.v3.chunk_key_encoding.default import (
    DefaultChunkKeyEncoding,
    DefaultChunkKeyEncodingConfiguration,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2ChunkKeyEncoding,
    V2ChunkKeyEncodingConfiguration,
)
from zarr_metadata.v3.codec.blosc import BloscCodec, BloscCodecConfiguration
from zarr_metadata.v3.codec.bytes import BytesCodec, BytesCodecConfiguration
from zarr_metadata.v3.codec.crc32c import Crc32cCodec, Empty
from zarr_metadata.v3.codec.gzip import GzipCodec, GzipCodecConfiguration
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodec, ScaleOffsetCodecConfiguration
from zarr_metadata.v3.codec.transpose import TransposeCodec, TransposeCodecConfiguration
from zarr_metadata.v3.codec.zstd import ZstdCodec, ZstdCodecConfiguration
from zarr_metadata.v3.data_type.bool import BoolDataType
from zarr_metadata.v3.data_type.bytes import BytesDataType
from zarr_metadata.v3.data_type.complex64 import Complex64DataType
from zarr_metadata.v3.data_type.complex128 import Complex128DataType
from zarr_metadata.v3.data_type.float16 import Float16DataType
from zarr_metadata.v3.data_type.float32 import Float32DataType
from zarr_metadata.v3.data_type.float64 import Float64DataType
from zarr_metadata.v3.data_type.int8 import Int8DataType
from zarr_metadata.v3.data_type.int16 import Int16DataType
from zarr_metadata.v3.data_type.int32 import Int32DataType
from zarr_metadata.v3.data_type.int64 import Int64DataType
from zarr_metadata.v3.data_type.numpy_datetime64 import (
    NumpyDatetime64Configuration,
    NumpyDatetime64DataType,
)
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NumpyTimedelta64Configuration,
    NumpyTimedelta64DataType,
)
from zarr_metadata.v3.data_type.raw import RawBytesDataType
from zarr_metadata.v3.data_type.string import StringDataType
from zarr_metadata.v3.data_type.uint8 import Uint8DataType
from zarr_metadata.v3.data_type.uint16 import Uint16DataType
from zarr_metadata.v3.data_type.uint32 import Uint32DataType
from zarr_metadata.v3.data_type.uint64 import Uint64DataType

if TYPE_CHECKING:
    from zarr_metadata.v3._entity import MetadataEntity

# Each registered entity, paired with the TypedDict its constructor
# mirrors. Keyed by `<field>:<identifier>`, because an identifier is only
# unique within its extension point -- `bytes` is both a codec and a data
# type.
CONFIGURATIONS: dict[str, tuple[type[MetadataEntity], type | None]] = {
    "codecs:blosc": (BloscCodec, BloscCodecConfiguration),
    "codecs:bytes": (BytesCodec, BytesCodecConfiguration),
    "codecs:crc32c": (Crc32cCodec, Empty),
    "codecs:gzip": (GzipCodec, GzipCodecConfiguration),
    "codecs:scale_offset": (ScaleOffsetCodec, ScaleOffsetCodecConfiguration),
    "codecs:transpose": (TransposeCodec, TransposeCodecConfiguration),
    "codecs:zstd": (ZstdCodec, ZstdCodecConfiguration),
    "chunk_grid:regular": (RegularChunkGrid, RegularChunkGridConfiguration),
    "chunk_grid:rectilinear": (RectilinearChunkGrid, RectilinearChunkGridConfiguration),
    "chunk_key_encoding:default": (DefaultChunkKeyEncoding, DefaultChunkKeyEncodingConfiguration),
    "chunk_key_encoding:v2": (V2ChunkKeyEncoding, V2ChunkKeyEncodingConfiguration),
    "data_type:numpy.datetime64": (NumpyDatetime64DataType, NumpyDatetime64Configuration),
    "data_type:numpy.timedelta64": (NumpyTimedelta64DataType, NumpyTimedelta64Configuration),
    # Bare entities: the name says everything, so there is no
    # configuration TypedDict for the constructor to mirror.
    "data_type:bool": (BoolDataType, None),
    "data_type:int8": (Int8DataType, None),
    "data_type:int16": (Int16DataType, None),
    "data_type:int32": (Int32DataType, None),
    "data_type:int64": (Int64DataType, None),
    "data_type:uint8": (Uint8DataType, None),
    "data_type:uint16": (Uint16DataType, None),
    "data_type:uint32": (Uint32DataType, None),
    "data_type:uint64": (Uint64DataType, None),
    "data_type:float16": (Float16DataType, None),
    "data_type:float32": (Float32DataType, None),
    "data_type:float64": (Float64DataType, None),
    "data_type:complex64": (Complex64DataType, None),
    "data_type:complex128": (Complex128DataType, None),
    "data_type:bytes": (BytesDataType, None),
    "data_type:string": (StringDataType, None),
    # The one exception. `r<N>` is a family, so the class holds the
    # spelling that picks a member of it -- a field with no configuration
    # member behind it, because the name carries the information.
    "data_type:r<N>": (RawBytesDataType, None),
}


@pytest.mark.parametrize(
    ("entity", "configuration"), CONFIGURATIONS.values(), ids=list(CONFIGURATIONS)
)
def test_the_constructor_mirrors_the_configuration(
    entity: type[MetadataEntity], configuration: type | None
) -> None:
    # `must_understand` belongs to the object, not the configuration, so it
    # is the one field the two deliberately do not share.
    if configuration is None:
        expected = {"data_type_name"} if entity is RawBytesDataType else set()
        assert {field.name for field in dataclasses.fields(entity)} - {
            "must_understand"
        } == expected
        return
    fields = {field.name for field in dataclasses.fields(entity)} - {"must_understand"}
    assert fields == set(get_type_hints(configuration))


@pytest.mark.parametrize(
    ("entity", "configuration"), CONFIGURATIONS.values(), ids=list(CONFIGURATIONS)
)
def test_the_member_table_mirrors_the_configuration(
    entity: type[MetadataEntity], configuration: type | None
) -> None:
    # The third spelling of the same set. Which members are *required* is
    # in the TypedDict too, so that cannot drift either.
    if configuration is None:
        assert entity.member_types == {}
        return
    assert set(entity.member_types) == set(get_type_hints(configuration))
    required = {key for key, (needed, _) in entity.member_types.items() if needed}
    assert required == set(configuration.__required_keys__)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("entity", "configuration"), CONFIGURATIONS.values(), ids=list(CONFIGURATIONS)
)
def test_a_required_member_rules_out_the_bare_spelling(
    entity: type[MetadataEntity], configuration: type | None
) -> None:
    # The spec permits a bare name only "if no configuration metadata is
    # required", so one flag follows from the other.
    required = 0 if configuration is None else len(configuration.__required_keys__)  # type: ignore[attr-defined]
    assert entity.configuration_required == (required != 0)


def test_every_registered_entity_is_checked_here() -> None:
    registered = {
        f"{field}:{identifier}"
        for field, entities in CORE_AND_EXTENSIONS.entities.items()
        for identifier in entities
    }
    assert registered == set(CONFIGURATIONS)


def test_core_is_a_subset_of_core_and_extensions() -> None:
    for field, entities in CORE.entities.items():
        assert entities.items() <= CORE_AND_EXTENSIONS.entities[field].items()


def test_a_name_out_of_scope_resolves_to_nothing() -> None:
    # Not an error: an unmodelled extension is left unjudged, not rejected.
    assert CORE.resolve("codecs", "mycorp.secret") is None
    assert CORE.resolve("codecs", "blosc") is BloscCodec


def test_an_entity_round_trips_through_its_json_form() -> None:
    original = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": "shuffle",
            "blocksize": 0,
            "typesize": 4,
        },
    }
    codec, problems = BloscCodec.coerce(original, CORE)
    assert problems == ()
    assert codec is not None
    assert codec.to_json() == original
