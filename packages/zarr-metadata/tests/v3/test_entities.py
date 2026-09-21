"""The correspondence between an entity's dataclass and its TypedDicts.

A configuration TypedDict unpacked is exactly the dataclass constructor's
signature, and the object TypedDict is exactly what `to_json` returns.
Asserting the first is what lets the dataclass carry its own checks
instead of a hand-written per-member table elsewhere: the two cannot
drift, because one drifting makes this fail.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from typing import (
    Any,
    ClassVar,
    Self,
    cast,
    get_type_hints,
)

import pytest
from hypothesis import given, settings
from typing_extensions import is_typeddict

from tests.helpers import configuration_of, entry_at
from tests.rules.strategies import valid_documents
from zarr_metadata.model import UNSET, MetadataValidationError
from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3._document import read_array_v3
from zarr_metadata.v3._entity import is_from_name
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS
from zarr_metadata.v3._typed_json import field_hints, is_not_required, is_optional
from zarr_metadata.v3.chunk_grid.rectilinear import (
    RectilinearChunkGrid,
)
from zarr_metadata.v3.chunk_grid.regular import RegularChunkGrid
from zarr_metadata.v3.chunk_key_encoding.default import (
    DefaultChunkKeyEncoding,
)
from zarr_metadata.v3.chunk_key_encoding.v2 import (
    V2ChunkKeyEncoding,
)
from zarr_metadata.v3.codec.blosc import BloscCodec
from zarr_metadata.v3.codec.bytes import BytesCodec
from zarr_metadata.v3.codec.cast_value import CastValueCodec
from zarr_metadata.v3.codec.crc32c import Crc32cCodec
from zarr_metadata.v3.codec.gzip import GzipCodec
from zarr_metadata.v3.codec.scale_offset import ScaleOffsetCodec
from zarr_metadata.v3.codec.sharding_indexed import (
    ShardingIndexedCodec,
)
from zarr_metadata.v3.codec.transpose import TransposeCodec
from zarr_metadata.v3.codec.zstd import ZstdCodec
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
    NumpyDatetime64DataType,
)
from zarr_metadata.v3.data_type.numpy_timedelta64 import (
    NumpyTimedelta64DataType,
)
from zarr_metadata.v3.data_type.raw import RawBytesDataType
from zarr_metadata.v3.data_type.string import StringDataType
from zarr_metadata.v3.data_type.struct import StructDataType
from zarr_metadata.v3.data_type.uint8 import Uint8DataType
from zarr_metadata.v3.data_type.uint16 import Uint16DataType
from zarr_metadata.v3.data_type.uint32 import Uint32DataType
from zarr_metadata.v3.data_type.uint64 import Uint64DataType
from zarr_metadata.v3.entity import (
    ArrayDocumentV3,
    ChunkGridEntity,
    ChunkKeyEncodingEntity,
    CodecEntity,
    Configuration,
    DataTypeEntity,
    MetadataEntity,
    StorageTransformerEntity,
)

# Every registered entity, keyed by `<field>:<identifier>` -- an identifier
# is unique only within its extension point, and `bytes` is both a codec
# and a data type.
# The field of a v3 array document each kind is read from: the test ids.
POINT: dict[type[MetadataEntity], str] = {
    DataTypeEntity: "data_type",
    ChunkGridEntity: "chunk_grid",
    ChunkKeyEncodingEntity: "chunk_key_encoding",
    CodecEntity: "codecs",
    StorageTransformerEntity: "storage_transformers",
}

ENTITIES: dict[str, type[MetadataEntity]] = {
    "codecs:blosc": BloscCodec,
    "codecs:bytes": BytesCodec,
    "codecs:cast_value": CastValueCodec,
    "codecs:crc32c": Crc32cCodec,
    "codecs:gzip": GzipCodec,
    "codecs:scale_offset": ScaleOffsetCodec,
    "codecs:sharding_indexed": ShardingIndexedCodec,
    "codecs:transpose": TransposeCodec,
    "codecs:zstd": ZstdCodec,
    "chunk_grid:regular": RegularChunkGrid,
    "chunk_grid:rectilinear": RectilinearChunkGrid,
    "chunk_key_encoding:default": DefaultChunkKeyEncoding,
    "chunk_key_encoding:v2": V2ChunkKeyEncoding,
    "data_type:numpy.datetime64": NumpyDatetime64DataType,
    "data_type:numpy.timedelta64": NumpyTimedelta64DataType,
    "data_type:bool": BoolDataType,
    "data_type:int8": Int8DataType,
    "data_type:int16": Int16DataType,
    "data_type:int32": Int32DataType,
    "data_type:int64": Int64DataType,
    "data_type:uint8": Uint8DataType,
    "data_type:uint16": Uint16DataType,
    "data_type:uint32": Uint32DataType,
    "data_type:uint64": Uint64DataType,
    "data_type:float16": Float16DataType,
    "data_type:float32": Float32DataType,
    "data_type:float64": Float64DataType,
    "data_type:complex64": Complex64DataType,
    "data_type:complex128": Complex128DataType,
    "data_type:bytes": BytesDataType,
    "data_type:struct": StructDataType,
    "data_type:string": StringDataType,
    "data_type:r<N>": RawBytesDataType,
}


# One or more documents each entity reads, spelled to reach both shapes
# where the entity has both: the bare name when every member is absent,
# the object otherwise. `st.from_type` over the named types cannot serve
# here -- measured, it reaches a valid gzip or blosc in under 1% of draws.
EXAMPLES: dict[str, tuple[object, ...]] = {
    "codecs:blosc": (
        {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "shuffle",
                "typesize": 4,
                "blocksize": 0,
            },
        },
    ),
    "codecs:bytes": ("bytes", {"name": "bytes", "configuration": {"endian": "little"}}),
    "codecs:cast_value": (
        {"name": "cast_value", "configuration": {"data_type": "int8"}},
        {
            "name": "cast_value",
            "configuration": {"data_type": "int8", "scalar_map": {"encode": (("NaN", 0),)}},
        },
    ),
    "codecs:crc32c": ("crc32c", {"name": "crc32c"}),
    "codecs:gzip": ({"name": "gzip", "configuration": {"level": 5}},),
    "codecs:scale_offset": (
        "scale_offset",
        {"name": "scale_offset", "configuration": {"offset": 2, "scale": 0.5}},
    ),
    "codecs:sharding_indexed": (
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": (4,),
                "codecs": ("bytes",),
                "index_codecs": (
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    "crc32c",
                ),
                "index_location": "start",
            },
        },
    ),
    "codecs:transpose": ({"name": "transpose", "configuration": {"order": (2, 1, 0)}},),
    "codecs:zstd": ({"name": "zstd", "configuration": {"level": 3, "checksum": False}},),
    "chunk_grid:regular": ({"name": "regular", "configuration": {"chunk_shape": (4, 4)}},),
    "chunk_grid:rectilinear": (
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((32, 32, 32),)},
        },
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (((32, 3),),)},
        },
    ),
    "chunk_key_encoding:default": (
        "default",
        {"name": "default", "configuration": {"separator": "."}},
    ),
    "chunk_key_encoding:v2": ("v2", {"name": "v2", "configuration": {"separator": "/"}}),
    "data_type:numpy.datetime64": (
        {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}},
    ),
    "data_type:numpy.timedelta64": (
        {"name": "numpy.timedelta64", "configuration": {"unit": "ms", "scale_factor": 10}},
    ),
    "data_type:bool": ("bool",),
    "data_type:int8": ("int8",),
    "data_type:int16": ("int16",),
    "data_type:int32": ("int32",),
    "data_type:int64": ("int64",),
    "data_type:uint8": ("uint8",),
    "data_type:uint16": ("uint16",),
    "data_type:uint32": ("uint32",),
    "data_type:uint64": ("uint64",),
    "data_type:float16": ("float16",),
    "data_type:float32": ("float32",),
    "data_type:float64": ("float64",),
    "data_type:complex64": ("complex64",),
    "data_type:complex128": ("complex128",),
    "data_type:bytes": ("bytes",),
    "data_type:struct": (
        {
            "name": "struct",
            "configuration": {
                "fields": (
                    {"name": "a", "data_type": "uint8"},
                    {
                        "name": "b",
                        "data_type": {
                            "name": "numpy.datetime64",
                            "configuration": {"unit": "s", "scale_factor": 1},
                        },
                    },
                ),
            },
        },
    ),
    "data_type:string": ("string",),
    "data_type:r<N>": ("r16", "r008"),
}


def _round_trips(entity: type[MetadataEntity], document: object) -> MetadataEntity:
    read, problems = entity.coerce(document, CORE_AND_EXTENSIONS)
    assert problems == ()
    assert read is not None
    again, problems = entity.coerce(read.to_json(), CORE_AND_EXTENSIONS)
    assert problems == ()
    assert again == read
    return read


@pytest.mark.parametrize(
    ("entity", "document"),
    [(ENTITIES[key], document) for key, documents in EXAMPLES.items() for document in documents],
    ids=[
        f"{key}:{index}" for key, documents in EXAMPLES.items() for index in range(len(documents))
    ],
)
def test_to_json_reads_back_to_the_same_entity(
    entity: type[MetadataEntity], document: object
) -> None:
    # What `to_json` writes, `coerce` reads to the entity that wrote it:
    # every member is written, in the spelling the reader expects.
    _round_trips(entity, document)


@given(document=valid_documents())
@settings(max_examples=50, deadline=None)
def test_to_json_reads_back_across_a_valid_document(document: dict[str, object]) -> None:
    array, problems = read_array_v3(document, CORE_AND_EXTENSIONS)
    assert problems == ()
    for entity in (array.data_type, array.chunk_grid, array.chunk_key_encoding, *array.codecs):
        assert isinstance(entity, MetadataEntity)
        _round_trips(type(entity), entity.to_json())


def test_every_registered_entity_is_checked_here() -> None:
    registered = {
        f"{POINT[kind]}:{identifier}"
        for kind, entities in CORE_AND_EXTENSIONS.tables.items()
        for identifier in entities
    }
    assert registered == set(ENTITIES)
    assert set(EXAMPLES) == set(ENTITIES)


def test_core_is_a_subset_of_core_and_extensions() -> None:
    both = CORE_AND_EXTENSIONS.tables
    for kind, entities in CORE.tables.items():
        assert entities.items() <= both[kind].items()


def test_a_name_out_of_scope_resolves_to_nothing() -> None:
    # Not an error: an unmodelled extension is left unjudged, not rejected.
    assert CORE.resolve(CodecEntity, "mycorp.secret") is None
    assert CORE.resolve(CodecEntity, "blosc") is BloscCodec


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


def test_an_unknown_key_is_reported_without_losing_the_member() -> None:
    # The document is invalid either way, but dropping the member would
    # make the entity describe something the document does not say.
    codec, problems = CastValueCodec.coerce(
        {
            "name": "cast_value",
            "configuration": {"data_type": "int8", "scalar_map": {"encode": (), "enc": ()}},
        },
        CORE_AND_EXTENSIONS,
    )
    assert [problem.kind for problem in problems] == ["unknown_key"]
    assert codec is not None
    assert codec.scalar_map == {"encode": (), "enc": ()}


# (a defect in a nested metadata field, the location it belongs at)
NESTED_ENVELOPES: dict[str, tuple[dict[str, object], tuple[str | int, ...]]] = {
    "unexpected-member": ({"name": "bytes", "typo": 1}, ("typo",)),
    "configuration-not-an-object": ({"name": "bytes", "configuration": 42}, ("configuration",)),
    "must-understand-not-a-boolean": (
        {"name": "bytes", "must_understand": "no"},
        ("must_understand",),
    ),
}


@pytest.mark.parametrize(
    ("codec", "inner_loc"), NESTED_ENVELOPES.values(), ids=list(NESTED_ENVELOPES)
)
def test_a_nested_metadata_field_is_judged_like_a_top_level_one(
    codec: dict[str, object], inner_loc: tuple[str | int, ...]
) -> None:
    # A metadata field is a metadata field wherever it appears. These sit
    # inside a shard's pipeline, which only the entity layer reads, so
    # nothing else is going to notice them.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (8,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8,)}},
        "chunk_key_encoding": "default",
        "codecs": (
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": (4,),
                    "codecs": (codec,),
                    "index_codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                },
            },
        ),
    }
    problems = validate_array_metadata_v3(document)
    assert [problem.loc for problem in problems] == [
        ("codecs", 0, "configuration", "codecs", 0, *inner_loc)
    ]


def test_every_problem_location_indexes_into_the_document() -> None:
    # Two defects in one shard, at different depths. A location that does
    # not resolve is a location a consumer cannot use.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (8, 8),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (8, 8)}},
        "chunk_key_encoding": "default",
        "codecs": (
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": (4, 4),
                    "codecs": (
                        {"name": "transpose", "configuration": {"order": (0, 0)}},
                        {"name": "bytes", "configuration": {"endian": "little"}},
                    ),
                    "index_codecs": ({"name": "bytes"},),
                },
            },
        ),
    }
    problems = validate_array_metadata_v3(document)
    assert len(problems) != 0
    for problem in problems:
        node: object = document
        for step in problem.loc:
            # Two branches rather than one `or`: each narrows `step` to
            # the key type its container takes.
            if isinstance(node, dict) and isinstance(step, str) and step in node:
                node = node[step]
                continue
            if isinstance(node, tuple) and isinstance(step, int) and step < len(node):
                node = node[step]
                continue
            # A `missing_key` problem names where the key belongs, so it
            # is allowed to run past the end of what is there. Any other
            # kind must address a node that exists.
            assert problem.kind == "missing_key", (problem.loc, step)
            break


def test_an_unreadable_member_costs_the_entity() -> None:
    # `clevel` is the wrong type, so this blosc cannot be built, and an
    # entity that does not exist has no values to judge. The type problem
    # is what you have to fix first, and the raw JSON is still on the
    # `Opaque` standing in for the codec.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": "default",
        "codecs": (
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": "five",
                    "shuffle": "noshuffle",
                    "blocksize": -1,
                },
            },
        ),
    }
    problems = validate_array_metadata_v3(document)
    assert {problem.loc for problem in problems} == {
        ("codecs", 1, "configuration", "clevel"),
    }


def test_an_optional_member_of_the_wrong_type_is_not_judged_as_absent() -> None:
    # `typesize` could not be read. Building the entity around the hole
    # would have `__post_init__` see it as absent and add a second,
    # contradictory problem at the same location.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": "default",
        "codecs": (
            "bytes",
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "typesize": "four",
                    "blocksize": 0,
                },
            },
        ),
    }
    problems = validate_array_metadata_v3(document)
    assert [(problem.loc, problem.kind) for problem in problems] == [
        (("codecs", 1, "configuration", "typesize"), "invalid_type")
    ]


def test_the_document_writes_back_only_the_fields_it_read() -> None:
    # An absent field is read as an `Opaque` standing in for it; writing
    # it back as `null` would invent a value the document never wrote.
    array, problems = read_array_v3({"shape": (4,)}, CORE_AND_EXTENSIONS)
    assert problems == ()
    assert array.to_json() == {"shape": (4,)}


def test_an_unreadable_member_is_not_judged_by_its_default() -> None:
    # `shuffle` could not be read, so it falls back to `noshuffle`, under
    # which `typesize` means nothing. The absent `typesize` must not be
    # reported as required -- that would be the default talking.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4,)}},
        "chunk_key_encoding": "default",
        "codecs": (
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 5,
                    "shuffle": 7,
                    "blocksize": 0,
                },
            },
        ),
    }
    problems = validate_array_metadata_v3(document)
    assert [problem.loc for problem in problems] == [("codecs", 1, "configuration", "shuffle")]


# Entities whose written form and canonical form differ, or could.
FAITHFUL: dict[str, tuple[type[MetadataEntity], object]] = {
    "rectilinear-expanded": (
        ChunkGridEntity,
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((32, 32, 32),)},
        },
    ),
    "rectilinear-encoded": (
        ChunkGridEntity,
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (((32, 3),),)},
        },
    ),
    "blosc-ignored-typesize": (
        CodecEntity,
        {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "noshuffle",
                "blocksize": 0,
                "typesize": 4,
            },
        },
    ),
    "raw-bytes-padded": (DataTypeEntity, "r008"),
    "scale-offset-scalar": (
        CodecEntity,
        {"name": "scale_offset", "configuration": {"offset": 2, "scale": 0.5}},
    ),
    "struct-nested": (
        DataTypeEntity,
        {
            "name": "struct",
            "configuration": {
                "fields": ({"name": "a", "data_type": "uint8"},),
            },
        },
    ),
}


@pytest.mark.parametrize(("field", "written"), FAITHFUL.values(), ids=list(FAITHFUL))
def test_to_json_writes_back_what_was_read(field: type[MetadataEntity], written: object) -> None:
    # Serialization is not canonicalization. A reader that reads a
    # document and writes it back must not change bytes it was not asked
    # to change -- `canonical()` is where you ask.
    entity, problems = CORE_AND_EXTENSIONS.coerce(field, written)
    assert problems == ()
    assert isinstance(entity, MetadataEntity)
    assert entity.to_json() == written


def test_canonical_is_what_simplifies() -> None:
    encoded, _ = CORE_AND_EXTENSIONS.coerce(
        ChunkGridEntity,
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((32, 32, 32),)},
        },
    )
    assert isinstance(encoded, MetadataEntity)
    assert encoded.canonical().to_json() == {
        "name": "rectilinear",
        "configuration": {"kind": "inline", "chunk_shapes": (((32, 3),),)},
    }
    blosc, _ = CORE_AND_EXTENSIONS.coerce(
        CodecEntity,
        {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 5,
                "shuffle": "noshuffle",
                "blocksize": 0,
                "typesize": 4,
            },
        },
    )
    assert isinstance(blosc, BloscCodec)
    assert "typesize" not in configuration_of(blosc.canonical().to_json())


def test_canonical_reaches_a_contained_entity() -> None:
    shard, _ = CORE_AND_EXTENSIONS.coerce(
        CodecEntity,
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": (4,),
                "codecs": (
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {
                        "name": "blosc",
                        "configuration": {
                            "cname": "zstd",
                            "clevel": 5,
                            "shuffle": "noshuffle",
                            "blocksize": 0,
                            "typesize": 4,
                        },
                    },
                ),
                "index_codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
            },
        },
    )
    assert isinstance(shard, ShardingIndexedCodec)
    inner = entry_at(shard.canonical().to_json(), "configuration", "codecs", 1)
    assert "typesize" not in configuration_of(inner)


def test_error_an_explicit_null_scalar_is_refused() -> None:
    # `null` is a value the document wrote, distinct from absence -- and
    # no data type admits it as a scalar, so the codec cannot be built.
    codec, problems = CORE_AND_EXTENSIONS.coerce(
        CodecEntity, {"name": "scale_offset", "configuration": {"offset": None}}
    )
    assert codec is not None
    assert not isinstance(codec, MetadataEntity)
    assert [problem.loc for problem in problems] == [("configuration", "offset")]


# (an entity whose configuration holds a mutable JSON value)
MUTABLE_MEMBERS: dict[str, tuple[type[MetadataEntity], object]] = {
    "scale-offset-object": (
        CodecEntity,
        {"name": "scale_offset", "configuration": {"offset": {"a": 1}}},
    ),
    "cast-value-scalar-map": (
        CodecEntity,
        {
            "name": "cast_value",
            "configuration": {"data_type": "int8", "scalar_map": {"encode": (("NaN", 0),)}},
        },
    ),
    "struct-fields": (
        DataTypeEntity,
        {"name": "struct", "configuration": {"fields": ({"name": "a", "data_type": "uint8"},)}},
    ),
}


@pytest.mark.parametrize(("field", "written"), MUTABLE_MEMBERS.values(), ids=list(MUTABLE_MEMBERS))
def test_to_json_shares_no_mutable_state_with_the_entity(
    field: type[MetadataEntity], written: object
) -> None:
    # The model layer has this test; the entity layer did not, and handed
    # out its own dict -- so a caller mutating the document it was given
    # mutated a frozen entity.
    entity, problems = CORE_AND_EXTENSIONS.coerce(field, written)
    assert problems == ()
    assert isinstance(entity, MetadataEntity)
    baseline = copy.deepcopy(entity.to_json())
    handed_out = entity.to_json()
    configuration = configuration_of(handed_out)
    assert isinstance(configuration, dict)
    for key in list(configuration):
        value = configuration[key]
        if isinstance(value, dict):
            value["INJECTED"] = "boom"
        else:
            configuration[key] = "clobbered"
    assert entity.to_json() == baseline


def test_a_member_the_entity_does_not_model_is_not_written_back() -> None:
    # `unknown_key` is survivable so that one stray member cannot hide
    # every other finding about its entity -- a concession about
    # reporting, not a promise to carry the member. The entity holds what
    # it models, so writing back drops it.
    entry = {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": "shuffle",
            "typesize": 2,
            "blocksize": 0,
            "typo_key": 1,
        },
    }
    codec, problems = CORE_AND_EXTENSIONS.coerce(CodecEntity, entry)
    assert [(p.loc, p.kind) for p in problems] == [(("configuration", "typo_key"), "unknown_key")]
    assert isinstance(codec, BloscCodec)
    assert "typo_key" not in configuration_of(codec.to_json())


def test_the_fail_fast_reader_refuses_a_member_it_would_drop() -> None:
    # Which is why dropping it is survivable: the reader that does not
    # hand back problems does not hand back the entity either.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,)}},
        "chunk_key_encoding": "default",
        "codecs": ("bytes", {"name": "gzip", "configuration": {"level": 1, "typo_key": 1}}),
    }
    with pytest.raises(MetadataValidationError) as caught:
        ArrayDocumentV3.from_json(cast("Any", document))
    assert (("codecs", 1, "configuration", "typo_key"), "unknown_key") in {
        (problem.loc, problem.kind) for problem in caught.value.problems
    }


# A storage transformer: the one extension point nothing in the package
# models, so the only way to reach it is to register one.
@dataclasses.dataclass(frozen=True)
class AcmeShardCacheOptions(Configuration):
    verbose: bool | UNSET = UNSET


@dataclasses.dataclass(frozen=True)
class AcmeShardCache(StorageTransformerEntity):
    """A third-party storage transformer with a member canonical form drops."""

    configuration: AcmeShardCacheOptions

    identifier: ClassVar[str] = "acme.shard_cache"

    @property
    def verbose(self) -> bool | UNSET:
        return self.configuration.verbose

    def canonical(self) -> Self:
        return self.with_configuration(verbose=UNSET)


def test_the_document_writes_itself_back_and_canonical_reaches_every_point() -> None:
    # `to_json` is faithful, entities included; `canonical` walks every
    # field that holds an entity -- `storage_transformers` among them,
    # which the hand-written walk it replaces never reached.
    scope = CORE_AND_EXTENSIONS.extended_with(AcmeShardCache)
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": (4,),
        "data_type": "uint8",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,)}},
        "chunk_key_encoding": "default",
        "codecs": (
            "bytes",
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "noshuffle",
                    "typesize": 4,
                    "blocksize": 0,
                },
            },
        ),
        "storage_transformers": ({"name": "acme.shard_cache", "configuration": {"verbose": True}},),
        "dimension_names": (None,),
    }
    array = ArrayDocumentV3.from_json(document, context=scope)
    assert array.to_json() == document
    canonical = array.canonical().to_json()
    assert canonical["codecs"] == (
        "bytes",
        {
            "name": "blosc",
            "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "noshuffle", "blocksize": 0},
        },
    )
    assert canonical["storage_transformers"] == ("acme.shard_cache",)
    assert "dimension_names" not in canonical


def test_the_fields_are_the_public_configuration_type() -> None:
    # Each entity module's `*Configuration` TypedDict is the public JSON
    # type of what the entity holds; the fields are what it reads and
    # writes. Nothing else ties the two, so this does: same keys, same
    # requiredness. An entity of no members has no such type.
    for cls in CORE_AND_EXTENSIONS.entities():
        record = field_hints(cls).get("configuration")
        hints = field_hints(record) if isinstance(record, type) else {}
        members = {key for key, annotation in hints.items() if not is_from_name(annotation)}
        required = {key for key in members if not is_optional(hints[key])}
        declared = [
            value
            for name, value in vars(sys.modules[cls.__module__]).items()
            if name.endswith("Configuration") and is_typeddict(value)
        ]
        if len(declared) == 0:
            assert members == set(), cls
            continue
        (configuration,) = declared
        keys = get_type_hints(configuration, include_extras=True)
        assert set(keys) == members, cls
        assert {
            key for key, annotation in keys.items() if not is_not_required(annotation)
        } == required, cls
