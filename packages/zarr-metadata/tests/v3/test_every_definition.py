"""Every extension the package defines, read against its definition.

What 2a showed for one field, interpolated: each codec, data type, chunk
grid and chunk key encoding is a definition, its configuration a
TypedDict and its rules a function over it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.data_type.raw import RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    ChunkGridDefinition,
    ChunkKeyEncodingDefinition,
    CodecDefinition,
    DataTypeDefinition,
    Definition,
    EmptyConfiguration,
    ValidationProblem,
    canonicalize,
    resolve,
)

KINDS: dict[str, type[Definition[Any]]] = {
    "codecs": CodecDefinition,
    "data_type": DataTypeDefinition,
    "chunk_grid": ChunkGridDefinition,
    "chunk_key_encoding": ChunkKeyEncodingDefinition,
}

# Fields that read, per extension: at least one per definition in scope.
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
            "configuration": {"data_type": "int8", "scalar_map": {"encode": [["NaN", 0]]}},
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
                "chunk_shape": [4],
                "codecs": ["bytes"],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    "crc32c",
                ],
                "index_location": "start",
            },
        },
    ),
    "codecs:transpose": ({"name": "transpose", "configuration": {"order": [2, 1, 0]}},),
    "codecs:zstd": ({"name": "zstd", "configuration": {"level": 3, "checksum": False}},),
    "chunk_grid:regular": ({"name": "regular", "configuration": {"chunk_shape": [4, 4]}},),
    "chunk_grid:rectilinear": (
        {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": [[32, 32, 32]]},
        },
        {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": [[[32, 3]]]}},
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
                "fields": [
                    {"name": "a", "data_type": "uint8"},
                    {
                        "name": "b",
                        "data_type": {
                            "name": "numpy.datetime64",
                            "configuration": {"unit": "s", "scale_factor": 1},
                        },
                    },
                ]
            },
        },
    ),
    "data_type:string": ("string",),
    "data_type:r<N>": ("r16", "r008"),
}

CASES = [(key, field) for key, fields in EXAMPLES.items() for field in fields]


def _read(key: str, field: object) -> tuple[str, list[tuple[tuple[str | int, ...], str]]]:
    resolved, problems = resolve(field, KINDS[key.split(":")[0]], CORE_AND_EXTENSIONS)
    return resolved.resolution, [(found.loc, found.kind) for found in problems]


def _problems(key: str, field: object) -> list[tuple[tuple[str | int, ...], str]]:
    return _read(key, field)[1]


def test_every_definition_in_scope_has_an_example() -> None:
    names = {definition.name for definition in CORE_AND_EXTENSIONS.definitions()}
    assert names == {key.split(":")[1] for key in EXAMPLES}
    assert set(CORE.definitions()) < set(CORE_AND_EXTENSIONS.definitions())


@pytest.mark.parametrize(
    ("key", "field"), CASES, ids=[f"{k}:{i}" for i, (k, _) in enumerate(CASES)]
)
def test_every_example_reads_and_its_simplest_spelling_is_stable(key: str, field: object) -> None:
    # Read, with nothing wrong; and its simplest spelling reads the same,
    # and is its own simplest spelling.
    assert _read(key, field) == ("read", [])
    kind = KINDS[key.split(":")[0]]
    simplest, problems = canonicalize(field, kind, CORE_AND_EXTENSIONS)
    assert problems == ()
    assert simplest is not None
    assert canonicalize(simplest, kind, CORE_AND_EXTENSIONS) == (simplest, ())


@pytest.mark.parametrize(
    ("field", "simplest"),
    [
        # `noshuffle` ignores `typesize`, so the simplest spelling drops it.
        (
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 1,
                    "shuffle": "noshuffle",
                    "typesize": 4,
                    "blocksize": 0,
                },
            },
            {
                "name": "blosc",
                "configuration": {
                    "cname": "lz4",
                    "clevel": 1,
                    "shuffle": "noshuffle",
                    "blocksize": 0,
                },
            },
        ),
        # Nothing configured: the bare name, and no `must_understand: true`.
        ({"name": "crc32c", "configuration": {}, "must_understand": True}, "crc32c"),
        # Nested fields each in their own simplest spelling.
        (
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [2],
                    "codecs": [{"name": "bytes"}],
                    "index_codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {"name": "crc32c"},
                    ],
                },
            },
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": (2,),
                    "codecs": ("bytes",),
                    "index_codecs": (
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        "crc32c",
                    ),
                },
            },
        ),
        (
            {
                "name": "cast_value",
                "configuration": {"data_type": {"name": "int8", "configuration": {}}},
            },
            {"name": "cast_value", "configuration": {"data_type": "int8"}},
        ),
        # A dimension's runs of equal sizes are run-length encoded.
        (
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": [[32, 32, 32, 16], 8]},
            },
            {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": (((32, 3), 16), 8)},
            },
        ),
    ],
    ids=["blosc-noshuffle", "bare-name", "sharding-nested", "cast-value-target", "rectilinear-rle"],
)
def test_the_simplest_spelling(field: dict[str, Any], simplest: object) -> None:
    kind = ChunkGridDefinition if field["name"] == "rectilinear" else CodecDefinition
    assert canonicalize(field, kind, CORE_AND_EXTENSIONS) == (simplest, ())


def test_a_reader_takes_a_name_over_from_a_family() -> None:
    # The raw-bytes family claims every `r<N>`; a reader's own `r16`,
    # passed later, reads `r16`, and the family keeps the rest.
    mine = DataTypeDefinition(name="r16", configuration=EmptyConfiguration)
    scope = CORE_AND_EXTENSIONS.extended_with(mine)
    assert resolve("r16", DataTypeDefinition, scope)[0].definition is mine
    assert resolve("r8", DataTypeDefinition, scope)[0].definition is RAW_BYTES_DATA_TYPE


def test_a_definition_filed_again_is_asked_first() -> None:
    # Filed again after a reader's `r16`, the raw-bytes family is the one
    # passed last, and reads `r16` once more.
    mine = DataTypeDefinition(name="r16", configuration=EmptyConfiguration)
    scope = CORE_AND_EXTENSIONS.extended_with(mine).extended_with(RAW_BYTES_DATA_TYPE)
    assert resolve("r16", DataTypeDefinition, scope)[0].definition is RAW_BYTES_DATA_TYPE


def test_an_unclaimed_field_keeps_its_own_spelling() -> None:
    assert canonicalize({"name": "zfpy"}, CodecDefinition, CORE) == ({"name": "zfpy"}, ())


@pytest.mark.parametrize(
    ("field", "kind", "found"),
    [
        (
            {"name": "gzip", "configuration": {"level": 12}},
            CodecDefinition,
            [(("configuration", "level"), "invalid_value")],
        ),
        (
            {"name": "bytes", "configuration": {"endain": "big"}},
            CodecDefinition,
            [(("configuration", "endain"), "unknown_key")],
        ),
        (
            {
                "name": "struct",
                "configuration": {"fields": [{"name": "a", "data_type": "int8", "x": 1}]},
            },
            DataTypeDefinition,
            [(("configuration", "fields", 0, "x"), "unknown_key")],
        ),
        (
            {"name": "crc32c", "must_understand": False},
            CodecDefinition,
            [(("must_understand",), "invalid_value")],
        ),
        (
            {"name": "zfpy", "configuration": {}, "extra": 1},
            CodecDefinition,
            [(("extra",), "invalid_value")],
        ),
        (None, CodecDefinition, [((), "invalid_type")]),
    ],
    ids=[
        "refused-value",
        "unknown-key",
        "unknown-key-nested",
        "must-understand-false",
        "stray",
        "null",
    ],
)
def test_error_a_field_with_a_problem_has_no_simplest_spelling(
    field: dict[str, Any] | None, kind: type[Definition[Any]], found: list[object]
) -> None:
    # Whatever the author wrote stays theirs: a simpler spelling would
    # drop the unknown key, the stray member or the `must_understand`.
    simplest, problems = canonicalize(field, kind, CORE_AND_EXTENSIONS)
    assert simplest is None
    assert [(problem.loc, problem.kind) for problem in problems] == found


def test_error_a_canonical_that_does_not_hold_is_the_definitions_fault() -> None:
    # What `canonical` gives is judged again: a level the rules refuse is
    # a fault in the definition, not in the field it was asked about.
    lying = dataclasses.replace(
        GZIP_CODEC, name="acme.lying", canonical=lambda configuration: {"level": -7}
    )
    with pytest.raises(ValueError, match="'acme.lying': its canonical gave"):
        canonicalize(
            {"name": "acme.lying", "configuration": {"level": 1}},
            CodecDefinition,
            CORE_AND_EXTENSIONS.extended_with(lying),
        )


def _one(key: str, configuration: object) -> list[tuple[tuple[str | int, ...], str]]:
    return _problems(key, {"name": key.split(":")[1], "configuration": configuration})


BLOSC = {"cname": "lz4", "clevel": 5, "shuffle": "shuffle", "typesize": 4, "blocksize": 0}


def test_error_blosc_clevel_is_out_of_range() -> None:
    assert _one("codecs:blosc", {**BLOSC, "clevel": 10}) == [
        (("configuration", "clevel"), "invalid_value")
    ]


def test_error_blosc_blocksize_is_negative() -> None:
    assert _one("codecs:blosc", {**BLOSC, "blocksize": -1}) == [
        (("configuration", "blocksize"), "invalid_value")
    ]


def test_error_blosc_typesize_is_missing_while_shuffling() -> None:
    configuration = {key: value for key, value in BLOSC.items() if key != "typesize"}
    assert _one("codecs:blosc", configuration) == [(("configuration", "typesize"), "missing_key")]


def test_error_blosc_typesize_is_not_positive() -> None:
    assert _one("codecs:blosc", {**BLOSC, "typesize": 0}) == [
        (("configuration", "typesize"), "invalid_value")
    ]


def test_error_zstd_level_is_out_of_range() -> None:
    assert _one("codecs:zstd", {"level": 23}) == [(("configuration", "level"), "invalid_value")]


def test_error_transpose_order_is_not_a_permutation() -> None:
    assert _one("codecs:transpose", {"order": [0, 0]}) == [
        (("configuration", "order"), "invalid_value")
    ]


def test_error_scale_offset_scalar_is_null() -> None:
    assert _one("codecs:scale_offset", {"scale": None}) == [
        (("configuration", "scale"), "invalid_value")
    ]


def test_error_sharding_inner_chunk_extent_is_zero() -> None:
    configuration = {"chunk_shape": [0], "codecs": ["bytes"], "index_codecs": ["bytes"]}
    assert _one("codecs:sharding_indexed", configuration) == [
        (("configuration", "chunk_shape", 0), "invalid_value")
    ]


STATIC_SIZE = ("bytes", "cast_value", "crc32c", "scale_offset", "transpose")
DYNAMIC_SIZE = ("blosc", "gzip", "sharding_indexed", "zstd")


def test_every_codec_says_whether_its_size_is_static() -> None:
    # As zarr-python's `is_fixed_size` says: a compressor, and a shard,
    # whose empty inner chunks take no bytes, give out a size that depends
    # on the values.
    sizes = {
        definition.name: definition.size
        for definition in CORE_AND_EXTENSIONS.definitions()
        if isinstance(definition, CodecDefinition)
    }
    assert sizes == dict.fromkeys(STATIC_SIZE, "static") | dict.fromkeys(DYNAMIC_SIZE, "dynamic")


@pytest.mark.parametrize(
    "index_codecs",
    [
        ["bytes"],
        ["bytes", "crc32c"],
        [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
        # A name nothing in scope claims is left unjudged, its size unknown.
        ["bytes", {"name": "acme.lz9"}],
    ],
    ids=["bytes", "bytes-crc32c", "spelled-out", "unclaimed"],
)
def test_a_shard_s_index_codecs_are_of_static_size(index_codecs: list[object]) -> None:
    # The inner chunks' own codecs may be of either size.
    configuration = {
        "chunk_shape": [2],
        "codecs": ["bytes", {"name": "gzip", "configuration": {"level": 5}}],
        "index_codecs": index_codecs,
    }
    assert _one("codecs:sharding_indexed", configuration) == []


@pytest.mark.parametrize(
    "codec",
    [
        {"name": "gzip", "configuration": {"level": 5}},
        {"name": "zstd", "configuration": {"level": 3}},
        {
            "name": "blosc",
            "configuration": {
                "cname": "lz4",
                "clevel": 5,
                "shuffle": "noshuffle",
                "blocksize": 0,
            },
        },
    ],
    ids=["gzip", "zstd", "blosc"],
)
def test_error_a_shard_s_index_codec_is_of_dynamic_size(codec: dict[str, object]) -> None:
    # The spec: "Codecs that produce variable-sized encoded representation,
    # such as compression codecs, MUST NOT be used for index codecs."
    configuration = {"chunk_shape": [2], "codecs": ["bytes"], "index_codecs": ["bytes", codec]}
    assert _one("codecs:sharding_indexed", configuration) == [
        (("configuration", "index_codecs", 1), "invalid_value")
    ]


def test_error_a_codec_size_that_is_neither() -> None:
    with pytest.raises(TypeError, match="size is one of"):
        dataclasses.replace(GZIP_CODEC, size="fixed")


def test_error_a_codec_inside_a_shard_is_judged_where_it_sits() -> None:
    configuration = {
        "chunk_shape": [2],
        "codecs": [{"name": "gzip", "configuration": {"level": 12}}],
        "index_codecs": ["bytes"],
    }
    assert _one("codecs:sharding_indexed", configuration) == [
        (("configuration", "codecs", 0, "configuration", "level"), "invalid_value")
    ]


def test_error_a_cast_target_is_judged_where_it_sits() -> None:
    target = {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 0}}
    assert _one("codecs:cast_value", {"data_type": target}) == [
        (("configuration", "data_type", "configuration", "scale_factor"), "invalid_value")
    ]


def test_error_rectilinear_extent_is_not_positive() -> None:
    configuration = {"kind": "inline", "chunk_shapes": [0, [4, 0], [[4, 0]]]}
    assert _one("chunk_grid:rectilinear", configuration) == [
        (("configuration", "chunk_shapes", 0), "invalid_value"),
        (("configuration", "chunk_shapes", 1, 1), "invalid_value"),
        (("configuration", "chunk_shapes", 2, 0, 1), "invalid_value"),
    ]


def test_error_numpy_time_scale_factor_is_out_of_range() -> None:
    assert _one("data_type:numpy.timedelta64", {"unit": "s", "scale_factor": 2**31}) == [
        (("configuration", "scale_factor"), "invalid_value")
    ]


def test_error_struct_has_no_fields() -> None:
    assert _one("data_type:struct", {"fields": []}) == [
        (("configuration", "fields"), "invalid_value")
    ]


def test_error_struct_field_name_is_empty() -> None:
    fields = [{"name": "", "data_type": "uint8"}]
    assert _one("data_type:struct", {"fields": fields}) == [
        (("configuration", "fields", 0, "name"), "invalid_value")
    ]


def test_error_struct_field_name_is_repeated() -> None:
    fields = [{"name": "a", "data_type": "uint8"}, {"name": "a", "data_type": "int8"}]
    assert _one("data_type:struct", {"fields": fields}) == [
        (("configuration", "fields", 1, "name"), "invalid_value")
    ]


def test_error_struct_field_type_is_judged_where_it_sits() -> None:
    fields = [
        {
            "name": "a",
            "data_type": {
                "name": "numpy.datetime64",
                "configuration": {"unit": "s", "scale_factor": 0},
            },
        }
    ]
    assert _one("data_type:struct", {"fields": fields}) == [
        (
            ("configuration", "fields", 0, "data_type", "configuration", "scale_factor"),
            "invalid_value",
        )
    ]


@pytest.mark.parametrize("name", ["r12", "r0"])
def test_error_a_raw_bytes_width_is_not_a_positive_multiple_of_8(name: str) -> None:
    # Claimed by the family, so reported as its own; the name lands on the field.
    assert _problems("data_type:r<N>", name) == [((), "invalid_value")]


def test_error_a_chunk_key_separator_is_not_one_the_encoding_takes() -> None:
    assert _one("chunk_key_encoding:v2", {"separator": "-"}) == [
        (("configuration", "separator"), "invalid_value")
    ]


def test_a_rule_is_a_function_over_the_typeddict() -> None:
    # The rules are plain functions over JSON: a caller holding one
    # configuration can ask them without a scope or a field around it.
    from zarr_metadata.v3.codec.blosc import BLOSC_CODEC

    _, problems = BLOSC_CODEC.judge({**BLOSC, "clevel": 10})
    assert problems == (
        ValidationProblem(("clevel",), "expected an integer in [0, 9], got 10", "invalid_value"),
    )
