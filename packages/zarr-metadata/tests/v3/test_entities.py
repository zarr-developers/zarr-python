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
import types
from typing import (
    Any,
    ClassVar,
    NotRequired,
    Self,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import pytest
from hypothesis import given, settings
from typing_extensions import ReadOnly, is_typeddict

from tests.rules.strategies import valid_documents
from zarr_metadata.model import UNSET, MetadataValidationError
from zarr_metadata.rules import validate_array_metadata_v3
from zarr_metadata.v3._common import ZarrV3MetadataFieldJSON
from zarr_metadata.v3._compile import check_for
from zarr_metadata.v3._document import read_array_v3
from zarr_metadata.v3._entity import json_type_of
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

# Every registered entity, keyed by `<field>:<identifier>` -- an identifier
# is unique only within its extension point, and `bytes` is both a codec
# and a data type.
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
    # The member table and `configuration_required` are read off the fields,
    # and the JSON type is named as the base's argument; the fields are the
    # only spelling left that can drift from the public TypedDict -- and a
    # field the TypedDict does not have would be a member no document could
    # write. `must_understand` belongs to the object, not the configuration,
    # so it is the one field the two deliberately do not share.
    fields = {field.name for field in dataclasses.fields(entity)} - {"must_understand"}
    json_type = json_type_of(entity)
    objects = [part for part in _parts(json_type) if is_typeddict(part)]
    if len(objects) == 0:
        # A bare-name type: nothing to configure. `r<N>` keeps its width
        # in its name, so it holds a member that is not a configuration key.
        assert fields == ({"data_type_name"} if entity is RawBytesDataType else set())
        return
    (obj,) = objects
    configuration = get_type_hints(obj, include_extras=True).get("configuration")
    assert configuration is not None, f"{obj!r} has no configuration member"
    while get_origin(configuration) in (NotRequired, ReadOnly):
        (configuration,) = get_args(configuration)
    assert fields == set(get_type_hints(configuration))


def _parts(json_type: object) -> tuple[object, ...]:
    return (
        get_args(json_type) if get_origin(json_type) in (Union, types.UnionType) else (json_type,)
    )


@pytest.mark.parametrize("entity", ENTITIES.values(), ids=list(ENTITIES))
def test_every_entity_names_its_json_type(entity: type[MetadataEntity]) -> None:
    # The default is not wrong, only uninformative; every entity this
    # package models says exactly what it writes.
    assert json_type_of(entity) is not ZarrV3MetadataFieldJSON


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


def _assert_conforms(entity: MetadataEntity) -> None:
    # The one `cast` in `to_json` asserts that what it builds has the
    # entity's named type. This is that assertion, checked: the named
    # type compiled by the package's own compiler, and the output run
    # through it.
    json_type = json_type_of(type(entity))
    check = check_for(json_type)
    assert check is not None, f"{json_type!r} is not a shape the compiler reads"
    assert check(entity.to_json(), ()) == ()


@pytest.mark.parametrize(
    ("entity", "document"),
    [(ENTITIES[key], document) for key, documents in EXAMPLES.items() for document in documents],
    ids=[
        f"{key}:{index}" for key, documents in EXAMPLES.items() for index in range(len(documents))
    ],
)
def test_to_json_conforms_to_the_named_json_type(
    entity: type[MetadataEntity], document: object
) -> None:
    read, problems = entity.coerce(document, CORE_AND_EXTENSIONS)
    assert problems == ()
    assert read is not None
    _assert_conforms(read)


@given(document=valid_documents())
@settings(max_examples=50, deadline=None)
def test_to_json_conforms_across_a_valid_document(document: dict[str, object]) -> None:
    # The top-level entities of documents valid by construction, for the
    # variation the examples fix: permutations, chunk shapes, an index
    # pipeline. A nested entity is some entity's top-level example.
    array, problems = read_array_v3(document, CORE_AND_EXTENSIONS)
    assert problems == ()
    for entity in (array.data_type, array.chunk_grid, array.chunk_key_encoding, *array.codecs):
        assert isinstance(entity, MetadataEntity)
        _assert_conforms(entity)


def test_every_registered_entity_is_checked_here() -> None:
    registered = {
        f"{field}:{identifier}"
        for field, entities in CORE_AND_EXTENSIONS.tables().items()
        for identifier in entities
    }
    assert registered == set(ENTITIES)
    assert set(EXAMPLES) == set(ENTITIES)


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


# A storage transformer: the one extension point nothing in the package
# models, so the only way to reach it is to register one.
@dataclasses.dataclass(frozen=True)
class AcmeShardCache(MetadataEntity):
    """A third-party storage transformer with a member canonical form drops."""

    verbose: bool | UNSET = UNSET

    identifier: ClassVar[str] = "acme.shard_cache"

    def simplified(self) -> Self:
        return dataclasses.replace(self, verbose=UNSET)


def test_the_document_writes_itself_back_and_canonical_reaches_every_point() -> None:
    # `to_json` is faithful, entities included; `canonical` walks every
    # field that holds an entity -- `storage_transformers` among them,
    # which the hand-written walk it replaces never reached.
    scope = CORE_AND_EXTENSIONS.extended_with(
        storage_transformers={AcmeShardCache.identifier: AcmeShardCache}
    )
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
