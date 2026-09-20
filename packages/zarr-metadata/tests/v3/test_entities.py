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
from typing import Any, cast, get_args, get_type_hints

import pytest

from zarr_metadata.model import MetadataValidationError
from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3._registry import CORE, CORE_AND_EXTENSIONS
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
from zarr_metadata.v3.entity import ArrayDocumentV3, MetadataEntity

# Each registered entity, paired with the TypedDict its constructor
# mirrors. Keyed by `<field>:<identifier>`, because an identifier is only
# unique within its extension point -- `bytes` is both a codec and a data
# type.
# Every registered entity, keyed by `<field>:<identifier>` -- an identifier
# is unique only within its extension point, and `bytes` is both a codec
# and a data type. What each one's configuration is comes off the class:
# `configuration_type` is the only place that says so, and the fields are
# held to it below.
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


@pytest.mark.parametrize("entity", ENTITIES.values(), ids=list(ENTITIES))
def test_the_constructor_mirrors_the_configuration(entity: type[MetadataEntity]) -> None:
    # The one correspondence still written by hand, and so the one that
    # can still drift: the member table and `configuration_required` are
    # now read off `configuration_type`, but the dataclass fields are
    # not. It is also what catches an entity pointing at the wrong
    # TypedDict, since the fields would stop matching.
    #
    # `must_understand` belongs to the object, not the configuration, so
    # it is the one field the two deliberately do not share.
    configuration = entity.configuration_type
    fields = {field.name for field in dataclasses.fields(entity)} - {"must_understand"}
    if configuration is None:
        # `r<N>` keeps its width in its name, so it holds a member that
        # is not a configuration key.
        assert fields == ({"data_type_name"} if entity is RawBytesDataType else set())
        return
    assert fields == set(get_type_hints(configuration))


@pytest.mark.parametrize("entity", ENTITIES.values(), ids=list(ENTITIES))
def test_the_value_routine_takes_the_members_it_will_be_given(
    entity: type[MetadataEntity],
) -> None:
    # `coerce` calls it as `value_problems(**members)`, which no type can
    # check: members is a dict built at run time. So the correspondence
    # is checked here, against the fields rather than the configuration
    # -- `r<N>` holds a member that is not a configuration key, and a
    # `struct` and a `sharding_indexed` annotate a TypedDict of their own
    # because `prepare` has replaced field objects with entities by then.
    # Neither changes which members there are.
    if entity.value_problems is MetadataEntity.value_problems:
        return
    # The annotation is `Unpack[X]`; X is what says which members.
    (members,) = get_args(get_type_hints(entity.value_problems)["members"])
    fields = {field.name for field in dataclasses.fields(entity)} - {"must_understand"}
    assert set(get_type_hints(members)) == fields


def test_every_registered_entity_is_checked_here() -> None:
    registered = {
        f"{field}:{identifier}"
        for field, entities in CORE_AND_EXTENSIONS.tables().items()
        for identifier in entities
    }
    assert registered == set(ENTITIES)


def test_core_is_a_subset_of_core_and_extensions() -> None:
    both = CORE_AND_EXTENSIONS.tables()
    for field, entities in CORE.tables().items():
        assert entities.items() <= both[field].items()


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
    problems = validate_array_metadata_v3(document)  # type: ignore[arg-type]
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
    problems = validate_array_metadata_v3(document)  # type: ignore[arg-type]
    assert len(problems) != 0
    for problem in problems:
        node: object = document
        for step in problem.loc:
            if not isinstance(node, (dict, tuple)) or (isinstance(node, dict) and step not in node):
                # A `missing_key` problem names where the key belongs, so
                # it is allowed to run past the end of what is there. Any
                # other kind must address a node that exists.
                assert problem.kind == "missing_key", (problem.loc, step)
                break
            node = node[step]  # type: ignore[index]


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
    problems = validate_array_metadata_v3(document)  # type: ignore[arg-type]
    assert {problem.loc for problem in problems} == {
        ("codecs", 1, "configuration", "clevel"),
    }


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
    problems = validate_array_metadata_v3(document)  # type: ignore[arg-type]
    assert [problem.loc for problem in problems] == [("codecs", 1, "configuration", "shuffle")]


# Entities whose written form and canonical form differ, or could.
FAITHFUL: dict[str, tuple[str, object]] = {
    "rectilinear-expanded": (
        "chunk_grid",
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((32, 32, 32),)},
        },
    ),
    "rectilinear-encoded": (
        "chunk_grid",
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (((32, 3),),)},
        },
    ),
    "blosc-ignored-typesize": (
        "codecs",
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
    "raw-bytes-padded": ("data_type", "r008"),
    "scale-offset-scalar": (
        "codecs",
        {"name": "scale_offset", "configuration": {"offset": 2, "scale": 0.5}},
    ),
    "struct-nested": (
        "data_type",
        {
            "name": "struct",
            "configuration": {
                "fields": ({"name": "a", "data_type": "uint8"},),
            },
        },
    ),
}


@pytest.mark.parametrize(("field", "written"), FAITHFUL.values(), ids=list(FAITHFUL))
def test_to_json_writes_back_what_was_read(field: str, written: object) -> None:
    # Serialization is not canonicalization. A reader that reads a
    # document and writes it back must not change bytes it was not asked
    # to change -- `canonical()` is where you ask.
    entity, problems = CORE_AND_EXTENSIONS.coerce(field, written)  # type: ignore[arg-type]
    assert problems == ()
    assert isinstance(entity, MetadataEntity)
    assert entity.to_json() == written


def test_canonical_is_what_simplifies() -> None:
    encoded, _ = CORE_AND_EXTENSIONS.coerce(
        "chunk_grid",
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
        "codecs",
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
    assert isinstance(blosc, MetadataEntity)
    assert "typesize" not in blosc.canonical().to_json()["configuration"]  # type: ignore[index]


def test_canonical_reaches_a_contained_entity() -> None:
    shard, _ = CORE_AND_EXTENSIONS.coerce(
        "codecs",
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
    assert isinstance(shard, MetadataEntity)
    inner = shard.canonical().to_json()["configuration"]["codecs"][1]  # type: ignore[index]
    assert "typesize" not in inner["configuration"]  # type: ignore[index]


def test_error_an_explicit_null_scalar_is_refused() -> None:
    # `null` is a value the document wrote, distinct from absence -- and
    # no data type admits it as a scalar, so the codec cannot be built.
    codec, problems = CORE_AND_EXTENSIONS.coerce(
        "codecs", {"name": "scale_offset", "configuration": {"offset": None}}
    )
    assert codec is not None
    assert not isinstance(codec, MetadataEntity)
    assert [problem.loc for problem in problems] == [("configuration", "offset")]


# (an entity whose configuration holds a mutable JSON value)
MUTABLE_MEMBERS: dict[str, tuple[str, object]] = {
    "scale-offset-object": (
        "codecs",
        {"name": "scale_offset", "configuration": {"offset": {"a": 1}}},
    ),
    "cast-value-scalar-map": (
        "codecs",
        {
            "name": "cast_value",
            "configuration": {"data_type": "int8", "scalar_map": {"encode": (("NaN", 0),)}},
        },
    ),
    "struct-fields": (
        "data_type",
        {"name": "struct", "configuration": {"fields": ({"name": "a", "data_type": "uint8"},)}},
    ),
}


@pytest.mark.parametrize(("field", "written"), MUTABLE_MEMBERS.values(), ids=list(MUTABLE_MEMBERS))
def test_to_json_shares_no_mutable_state_with_the_entity(field: str, written: object) -> None:
    # The model layer has this test; the entity layer did not, and handed
    # out its own dict -- so a caller mutating the document it was given
    # mutated a frozen entity.
    entity, problems = CORE_AND_EXTENSIONS.coerce(field, written)  # type: ignore[arg-type]
    assert problems == ()
    assert isinstance(entity, MetadataEntity)
    baseline = copy.deepcopy(entity.to_json())
    handed_out = entity.to_json()
    configuration = handed_out["configuration"]  # type: ignore[index]
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
    codec, problems = CORE_AND_EXTENSIONS.coerce("codecs", entry)
    assert [(p.loc, p.kind) for p in problems] == [(("configuration", "typo_key"), "unknown_key")]
    assert isinstance(codec, MetadataEntity)
    assert "typo_key" not in codec.to_json()["configuration"]  # type: ignore[index,operator]


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
