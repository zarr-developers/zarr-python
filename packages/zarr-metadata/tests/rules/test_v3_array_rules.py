"""Tests for the v3 array and group composition rules added with the
rules-layer promotion: chunk grid values/geometry, transpose orders,
sharding pipelines/geometry, and consolidated-entry recursion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from zarr_metadata.model import MetadataValidationError
from zarr_metadata.rules import validate_array_metadata_v3, validate_group_metadata_v3

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr_metadata import ZarrV3ArrayMetadataJSON

BASE: ZarrV3ArrayMetadataJSON = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": (4, 4),
    "data_type": "uint8",
    "fill_value": 0,
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2, 2)}},
    "chunk_key_encoding": "default",
    "codecs": ("bytes",),
}


# The shard index is a uint64 array, so its bytes codec needs an endianness.
_INDEX_BYTES: Mapping[str, object] = {"name": "bytes", "configuration": {"endian": "little"}}


def _shard(**overrides: object) -> Mapping[str, object]:
    """A sharding codec entry; overrides may be deliberately malformed."""
    configuration: dict[str, object] = {
        "chunk_shape": (2, 2),
        "codecs": ("bytes",),
        "index_codecs": (_INDEX_BYTES, "crc32c"),
    }
    configuration.update(overrides)
    return {"name": "sharding_indexed", "configuration": configuration}


# Documents that must be fully valid: the rules judge geometry and values
# without rejecting legitimate spellings of the same constructs.
VALID_CASES: dict[str, Mapping[str, object]] = {
    "regular": BASE,
    "rectilinear-explicit-sums": {
        **BASE,
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((2, 2), (1, 3))},
        },
    },
    "rectilinear-explicit-overflow": {
        **BASE,
        "shape": (6,),
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": ((4, 4, 4),)},
        },
    },
    "rectilinear-rle-and-uniform": {
        **BASE,
        "chunk_grid": {
            "name": "rectilinear",
            "configuration": {"kind": "inline", "chunk_shapes": (((2, 2),), 4)},
        },
    },
    "transpose": {
        **BASE,
        "codecs": ({"name": "transpose", "configuration": {"order": (1, 0)}}, "bytes"),
    },
    "sharding": {**BASE, "codecs": (_shard(),)},
    "nested-sharding": {
        **BASE,
        "codecs": (
            _shard(
                codecs=(
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": (1, 2),
                            "codecs": ("bytes",),
                            "index_codecs": (_INDEX_BYTES,),
                        },
                    },
                )
            ),
        ),
    },
    "nested-struct": {
        **BASE,
        "data_type": {
            "name": "struct",
            "configuration": {
                "fields": (
                    {"name": "id", "data_type": "uint8"},
                    {
                        "name": "point",
                        "data_type": {
                            "name": "struct",
                            "configuration": {"fields": ({"name": "x", "data_type": "int16"},)},
                        },
                    },
                )
            },
        },
        "fill_value": {"id": 1, "point": {"x": -2}},
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    },
    "unknown-grid-passes": {
        **BASE,
        "chunk_grid": {"name": "hilbert", "configuration": {"level": 3}},
    },
    "unknown-codec-inconclusive": {**BASE, "codecs": ({"name": "zfpy"}, "bytes")},
    "scale-offset-scalars-of-the-array-type": {
        **BASE,
        "codecs": ({"name": "scale_offset", "configuration": {"offset": 1, "scale": 2}}, "bytes"),
    },
    "scale-offset-scalars-of-the-type-cast-to": {
        **BASE,
        "data_type": "float32",
        "fill_value": "NaN",
        "codecs": (
            {"name": "cast_value", "configuration": {"data_type": "uint8"}},
            {"name": "scale_offset", "configuration": {"offset": 10}},
            "bytes",
        ),
    },
}


@pytest.mark.parametrize("doc", VALID_CASES.values(), ids=list(VALID_CASES))
def test_valid_documents(doc: Mapping[str, object]) -> None:
    assert validate_array_metadata_v3(doc) == ()


def _sole_problem(doc: Mapping[str, object]) -> tuple[tuple[str | int, ...], str]:
    problems = validate_array_metadata_v3(doc)
    assert len(problems) == 1, [p.message for p in problems]
    return problems[0].loc, problems[0].message


def test_error_regular_chunk_extent_zero() -> None:
    loc, message = _sole_problem(
        {**BASE, "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (0, 2)}}}
    )
    assert loc == ("chunk_grid", "configuration", "chunk_shape", 0)
    assert "expected an integer >= 1" in message


def test_error_rectilinear_rank_mismatch() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": ((2, 2),)},
            },
        }
    )
    assert loc == ("chunk_grid", "configuration", "chunk_shapes")
    assert "2 dimensions" in message


def test_error_rectilinear_sum_mismatch() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": ((3,), (2, 2))},
            },
        }
    )
    assert loc == ("chunk_grid", "configuration", "chunk_shapes", 0)
    assert "sum to 3" in message


def test_error_rectilinear_nonpositive_rle() -> None:
    problems = validate_array_metadata_v3(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": (((0, 4),), 4)},
            },
        }
    )
    assert any("expected an integer >= 1" in p.message for p in problems)


def test_error_transpose_not_a_permutation() -> None:
    loc, message = _sole_problem(
        {**BASE, "codecs": ({"name": "transpose", "configuration": {"order": (5, 5)}}, "bytes")}
    )
    assert loc == ("codecs", 0, "configuration", "order")
    assert "permutation" in message


def test_error_transpose_rank_mismatch() -> None:
    loc, message = _sole_problem(
        {**BASE, "codecs": ({"name": "transpose", "configuration": {"order": (2, 0, 1)}}, "bytes")}
    )
    assert loc == ("codecs", 0, "configuration", "order")
    assert "incoming array has 2 dimensions" in message


def test_error_sharding_inner_pipeline_order() -> None:
    loc, _ = _sole_problem({**BASE, "codecs": (_shard(codecs=("crc32c", "bytes")),)})
    assert loc == ("codecs", 0, "configuration", "codecs", 1)


def test_error_sharding_inner_no_array_bytes() -> None:
    loc, message = _sole_problem({**BASE, "codecs": (_shard(codecs=("crc32c",)),)})
    assert loc == ("codecs", 0, "configuration", "codecs")
    assert "no array->bytes codec" in message


def test_error_sharding_index_codecs_no_array_bytes() -> None:
    loc, message = _sole_problem({**BASE, "codecs": (_shard(index_codecs=("crc32c",)),)})
    assert loc == ("codecs", 0, "configuration", "index_codecs")
    assert "no array->bytes codec" in message


def test_error_sharding_index_codecs_are_variable_sized() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "codecs": (
                _shard(
                    index_codecs=(
                        _INDEX_BYTES,
                        {"name": "gzip", "configuration": {"level": 1}},
                    )
                ),
            ),
        }
    )
    assert loc == ("codecs", 0, "configuration", "index_codecs", 1)
    assert "fixed-size" in message


def test_error_sharding_rank_mismatch() -> None:
    loc, message = _sole_problem({**BASE, "codecs": (_shard(chunk_shape=(2,)),)})
    assert loc == ("codecs", 0, "configuration", "chunk_shape")
    assert "incoming array has 2 dimensions" in message


def test_error_sharding_not_divisible() -> None:
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
            "codecs": (_shard(chunk_shape=(3, 2)),),
        }
    )
    assert loc == ("codecs", 0, "configuration", "chunk_shape", 0)
    assert "does not evenly divide" in message


def test_error_nested_sharding_not_divisible() -> None:
    inner: Mapping[str, object] = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": (2, 3),
            "codecs": ("bytes",),
            "index_codecs": (_INDEX_BYTES,),
        },
    }
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (4, 4)}},
            "codecs": (_shard(codecs=(inner,)),),
        }
    )
    assert loc == ("codecs", 0, "configuration", "codecs", 0, "configuration", "chunk_shape", 1)
    assert "does not evenly divide" in message


def test_error_sharding_inner_chunk_extent_zero() -> None:
    problems = validate_array_metadata_v3({**BASE, "codecs": (_shard(chunk_shape=(0, 2)),)})
    assert any(
        p.loc == ("codecs", 0, "configuration", "chunk_shape", 0)
        and "expected an integer >= 1" in p.message
        for p in problems
    )


def test_error_bytes_requires_endian_for_multibyte_data() -> None:
    loc, message = _sole_problem({**BASE, "data_type": "int32", "codecs": ("bytes",)})
    assert loc == ("codecs", 0, "configuration", "endian")
    assert "required" in message


def test_error_bytes_rejects_variable_length_data_type() -> None:
    # The problem is about the codec, not about a member of its
    # configuration — and this codec is spelled as a bare string, so it has
    # no `configuration` node for a loc to point into.
    loc, message = _sole_problem(
        {**BASE, "data_type": "string", "fill_value": "", "codecs": ("bytes",)}
    )
    assert loc == ("codecs", 0)
    assert "not compatible" in message


def test_error_struct_fields_are_empty() -> None:
    doc = {
        **BASE,
        "data_type": {"name": "struct", "configuration": {"fields": ()}},
        "fill_value": {},
    }
    loc, message = _sole_problem(doc)
    assert loc == ("data_type", "configuration", "fields")
    assert "at least one" in message


def test_error_struct_field_is_variable_length() -> None:
    doc = {
        **BASE,
        "data_type": {
            "name": "struct",
            "configuration": {"fields": ({"name": "label", "data_type": "string"},)},
        },
        "fill_value": {"label": ""},
    }
    problems = validate_array_metadata_v3(doc)
    assert any(
        problem.loc == ("data_type", "configuration", "fields", 0, "data_type")
        and "fixed-size" in problem.message
        for problem in problems
    )


def test_error_struct_fill_is_missing_field() -> None:
    doc = {
        **BASE,
        "data_type": {
            "name": "struct",
            "configuration": {"fields": ({"name": "x", "data_type": "uint8"},)},
        },
        "fill_value": {},
    }
    loc, message = _sole_problem(doc)
    assert loc == ("fill_value", "x")
    assert "missing" in message


def test_error_struct_fill_field_is_invalid() -> None:
    doc = {
        **BASE,
        "data_type": {
            "name": "struct",
            "configuration": {"fields": ({"name": "x", "data_type": "uint8"},)},
        },
        "fill_value": {"x": 300},
    }
    loc, message = _sole_problem(doc)
    assert loc == ("fill_value", "x")
    assert "[0, 255]" in message


def test_error_gzip_level_is_out_of_range() -> None:
    doc = {
        **BASE,
        "codecs": ("bytes", {"name": "gzip", "configuration": {"level": 99}}),
    }
    loc, message = _sole_problem(doc)
    assert loc == ("codecs", 1, "configuration", "level")
    assert "[0, 9]" in message


@pytest.mark.parametrize("data_type_name", ["numpy.datetime64", "numpy.timedelta64"])
def test_error_numpy_time_scale_factor_is_out_of_range(data_type_name: str) -> None:
    doc = {
        **BASE,
        "data_type": {
            "name": data_type_name,
            "configuration": {"unit": "ns", "scale_factor": 0},
        },
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    loc, message = _sole_problem(doc)
    assert loc == ("data_type", "configuration", "scale_factor")
    assert "[1, 2147483647]" in message


@pytest.mark.parametrize("data_type_name", ["numpy.datetime64", "numpy.timedelta64"])
def test_error_numpy_time_fill_is_out_of_range(data_type_name: str) -> None:
    doc = {
        **BASE,
        "data_type": {
            "name": data_type_name,
            "configuration": {"unit": "ns", "scale_factor": 1},
        },
        "fill_value": 2**80,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
    }
    loc, message = _sole_problem(doc)
    assert loc == ("fill_value",)
    assert "64-bit" in message


def test_error_consolidated_child_violates_array_rules() -> None:
    doc: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": {**BASE, "fill_value": 300}},
        },
    }
    problems = validate_group_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [
        (("consolidated_metadata", "metadata", "a", "fill_value"), "invalid_value")
    ]


def test_error_consolidated_nested_group_recursion() -> None:
    child_group: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"b": {**BASE, "fill_value": 300}},
        },
    }
    doc: Mapping[str, object] = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"g": child_group},
        },
    }
    problems = validate_group_metadata_v3(doc)
    assert [p.loc for p in problems] == [
        (
            "consolidated_metadata",
            "metadata",
            "g",
            "consolidated_metadata",
            "metadata",
            "b",
            "fill_value",
        )
    ]


def test_error_group_parse_raises() -> None:
    from zarr_metadata.rules import parse_group_metadata_v3

    with pytest.raises(
        MetadataValidationError, match=r"consolidated_metadata\.metadata\.a\.fill_value"
    ):
        parse_group_metadata_v3(
            {
                "zarr_format": 3,
                "node_type": "group",
                "consolidated_metadata": {
                    "kind": "inline",
                    "must_understand": False,
                    "metadata": {"a": {**BASE, "fill_value": 300}},
                },
            }
        )


# -- unknown configuration members --------------------------------------------
#
# The v3 spec does not say whether an extension's `configuration` is closed
# (zarr-developers/zarr-specs#270, open since 2023). This package takes the
# strict reading, matching most registered extension schemas and most other
# implementations — but reports it as its own `unknown_key` kind, and never
# lets it mask a real finding about the same entity.


def test_unknown_configuration_member_has_its_own_kind() -> None:
    doc = {
        **BASE,
        "codecs": ({"name": "bytes", "configuration": {"endian": "little", "hint": 1}},),
    }
    problems = validate_array_metadata_v3(doc)
    # The location names the offending key, so a consumer can route the
    # problem without parsing the message.
    assert [(p.loc, p.kind) for p in problems] == [
        (("codecs", 0, "configuration", "hint"), "unknown_key")
    ]


def test_error_known_data_type_has_invalid_configuration() -> None:
    doc = {
        **BASE,
        "data_type": {
            "name": "numpy.datetime64",
            "configuration": {"unit": "banana", "scale_factor": 1},
        },
    }
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [
        (("data_type", "configuration", "unit"), "invalid_value")
    ]


def test_error_known_chunk_key_encoding_has_invalid_configuration() -> None:
    doc = {
        **BASE,
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "!"}},
    }
    problems = validate_array_metadata_v3(doc)
    assert [(p.loc, p.kind) for p in problems] == [
        (("chunk_key_encoding", "configuration", "separator"), "invalid_value")
    ]


def test_unknown_member_does_not_mask_a_codec_rule() -> None:
    # Regression: an unrecognized member used to make the whole entity
    # uninterpretable, silently suppressing every other rule about it — so a
    # cosmetic extra key hid a genuine permutation error.
    doc = {
        **BASE,
        "codecs": ({"name": "transpose", "configuration": {"order": (5, 5), "hint": 1}}, "bytes"),
    }
    kinds = {(p.loc, p.kind) for p in validate_array_metadata_v3(doc)}
    assert (("codecs", 0, "configuration", "hint"), "unknown_key") in kinds
    assert (("codecs", 0, "configuration", "order"), "invalid_value") in kinds


def test_unknown_member_does_not_mask_a_chunk_grid_rule() -> None:
    doc = {
        **BASE,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,), "hint": 1}},
    }
    kinds = {(p.loc, p.kind) for p in validate_array_metadata_v3(doc)}
    assert (("chunk_grid", "configuration", "hint"), "unknown_key") in kinds
    assert (("chunk_grid", "configuration", "chunk_shape"), "invalid_value") in kinds


def test_unknown_member_survives_a_round_trip() -> None:
    # Whatever the strict validator says, the package must never silently
    # drop a member it does not model: a writer that knows more than we do
    # must get its bytes back. (zarr-python's own chunk-grid path is lossy
    # here; this asserts we are not.)
    import json

    from zarr_metadata.model import ZarrV3ArrayMetadata

    raw = {
        **BASE,
        "shape": [4, 4],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
        "codecs": [
            {
                "name": "blosc",
                "configuration": {
                    "cname": "zstd",
                    "clevel": 5,
                    "shuffle": "shuffle",
                    "blocksize": 0,
                    "numThreads": 4,
                },
            },
            "bytes",
        ],
    }
    model = ZarrV3ArrayMetadata.from_json(json.loads(json.dumps(raw)))
    emitted = model.to_json()
    codec = cast("Mapping[str, Any]", emitted["codecs"][0])
    assert codec["configuration"]["numThreads"] == 4


# -- gaps found by adversarial review ----------------------------------------


def test_error_scale_offset_does_not_stand_down_later_rules() -> None:
    # A no-op array->array codec used to stop spec propagation, silently
    # switching off every rule after it.
    loc, message = _sole_problem(
        {
            **BASE,
            "data_type": "uint16",
            "codecs": (
                {"name": "scale_offset", "configuration": {"offset": 0, "scale": 1}},
                "bytes",
            ),
        }
    )
    assert loc == ("codecs", 1, "configuration", "endian")
    assert "uint16" in message


def test_error_rank_is_judged_under_a_chunk_grid_with_unknown_extents() -> None:
    # A rectilinear grid gives no single chunk shape, but every chunk still
    # has the array's rank, so a rank-3 transpose over a 1-D array is a fault.
    loc, message = _sole_problem(
        {
            **BASE,
            "chunk_grid": {
                "name": "rectilinear",
                "configuration": {"kind": "inline", "chunk_shapes": ((2, 2), (2, 2))},
            },
            "codecs": ({"name": "transpose", "configuration": {"order": (0, 1, 2)}}, "bytes"),
        }
    )
    assert loc == ("codecs", 0, "configuration", "order")
    assert "2 dimensions" in message


def test_error_an_unusable_member_costs_the_entity() -> None:
    # An entity exists only if its members are readable and its values
    # allowed, so a bad `index_location` means there is no shard to ask
    # about its pipelines. The JSON survives on the `Opaque` that stands
    # in for it; what is gone is the interpretation, and the report that
    # remains is the one that has to be fixed first.
    problems = validate_array_metadata_v3(
        {
            **BASE,
            "codecs": (
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": (2, 2),
                        "codecs": ("bytes", "bytes"),
                        "index_codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                        "index_location": "middle",
                    },
                },
            ),
        }
    )
    assert {problem.loc for problem in problems} == {
        ("codecs", 0, "configuration", "index_location"),
    }


def test_error_a_bare_entity_reports_at_the_entity_not_a_missing_node() -> None:
    # "bytes" has no `configuration` node, so a problem about the codec as a
    # whole must not point into one.
    loc, _ = _sole_problem({**BASE, "data_type": "string", "fill_value": "", "codecs": ("bytes",)})
    assert loc == ("codecs", 0)


def test_error_endian_message_names_the_shard_index_type() -> None:
    # Inside index_codecs the array is the shard index, whose uint64 type
    # appears nowhere in the document; the message has to say so.
    loc, message = _sole_problem(
        {
            **BASE,
            "codecs": (
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": (2, 2),
                        "codecs": ({"name": "bytes", "configuration": {"endian": "little"}},),
                        "index_codecs": ("bytes",),
                    },
                },
            ),
        }
    )
    assert loc == ("codecs", 0, "configuration", "index_codecs", 0, "configuration", "endian")
    assert "uint64" in message


def test_error_a_malformed_must_understand_does_not_suppress_the_entity() -> None:
    # `must_understand` is part of the envelope, not the configuration, so
    # a bad one says nothing about whether the configuration is readable.
    problems = validate_array_metadata_v3(
        {
            **BASE,
            "codecs": (
                {
                    "name": "transpose",
                    "configuration": {"order": (2, 1, 0)},
                    "must_understand": "yes",
                },
                "bytes",
            ),
        }
    )
    assert ("codecs", 0, "configuration", "order") in {problem.loc for problem in problems}


# -- blosc: the one member whose requiredness the spec makes conditional ------


def _blosc_configuration(**overrides: object) -> dict[str, object]:
    return {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "shuffle",
        "typesize": 2,
        "blocksize": 0,
        **overrides,
    }


def _blosc(**overrides: object) -> Mapping[str, object]:
    return {"name": "blosc", "configuration": _blosc_configuration(**overrides)}


def _with_blosc(**overrides: object) -> Mapping[str, object]:
    return {**BASE, "codecs": ("bytes", _blosc(**overrides))}


def test_blosc_accepts_a_document_zarr_python_writes() -> None:
    assert validate_array_metadata_v3(_with_blosc()) == ()


def test_blosc_typesize_may_be_omitted_only_without_shuffling() -> None:
    # "Required unless `shuffle` is `"noshuffle"`, in which case the value
    # is ignored." A TypedDict cannot say that, so the rule does.
    configuration = _blosc_configuration()
    del configuration["typesize"]
    for shuffle, required in (("noshuffle", False), ("shuffle", True), ("bitshuffle", True)):
        codec = {"name": "blosc", "configuration": {**configuration, "shuffle": shuffle}}
        problems = validate_array_metadata_v3({**BASE, "codecs": ("bytes", codec)})
        assert [problem.loc[-1] for problem in problems] == (["typesize"] if required else [])


def test_error_blosc_typesize_is_not_positive() -> None:
    loc, message = _sole_problem(_with_blosc(typesize=0))
    assert loc == ("codecs", 1, "configuration", "typesize")
    assert "positive" in message


def test_error_blosc_clevel_out_of_range() -> None:
    loc, message = _sole_problem(_with_blosc(clevel=99))
    assert loc == ("codecs", 1, "configuration", "clevel")
    assert "[0, 9]" in message


def test_error_blosc_blocksize_is_negative() -> None:
    loc, message = _sole_problem(_with_blosc(blocksize=-1))
    assert loc == ("codecs", 1, "configuration", "blocksize")
    assert "expected an integer >= 0" in message


@pytest.mark.parametrize(
    ("level", "valid"),
    [(0, True), (22, True), (-131072, True), (23, False), (-131073, False), (1000, False)],
)
def test_zstd_level_range(level: int, valid: bool) -> None:
    # "An integer from -131072 to 22"; 0 selects the default level.
    document = {
        **BASE,
        "codecs": ("bytes", {"name": "zstd", "configuration": {"level": level, "checksum": False}}),
    }
    problems = validate_array_metadata_v3(document)
    if valid:
        assert problems == ()
    else:
        assert [problem.loc for problem in problems] == [("codecs", 1, "configuration", "level")]


# -- must_understand: false, wherever a document writes it --------------------

# (a document nesting an ignorable extension point, and where it sits)
IGNORABLE_NESTED: dict[str, tuple[Mapping[str, object], tuple[object, ...]]] = {
    "inner-codec-of-a-shard": (
        {
            **BASE,
            "codecs": (_shard(codecs=({"name": "bytes", "must_understand": False},)),),
        },
        ("codecs", 0, "configuration", "codecs", 0, "must_understand"),
    ),
    "index-codec-of-a-shard": (
        {
            **BASE,
            "codecs": (_shard(index_codecs=({**_INDEX_BYTES, "must_understand": False},)),),
        },
        ("codecs", 0, "configuration", "index_codecs", 0, "must_understand"),
    ),
    "data-type-of-a-struct-field": (
        {
            **BASE,
            "data_type": {
                "name": "struct",
                "configuration": {
                    "fields": (
                        {
                            "name": "a",
                            "data_type": {"name": "uint8", "must_understand": False},
                        },
                    )
                },
            },
            "fill_value": {"a": 0},
        },
        ("data_type", "configuration", "fields", 0, "data_type", "must_understand"),
    ),
}


@pytest.mark.parametrize("case", IGNORABLE_NESTED)
def test_error_a_nested_extension_point_may_not_be_declared_ignorable(case: str) -> None:
    # An extension point is something a reader must understand, and depth
    # does not change that: a codec inside a shard is still a codec. The
    # top-level check would otherwise be a check on where the flag was
    # written rather than on what it says.
    document, loc = IGNORABLE_NESTED[case]
    problems = validate_array_metadata_v3(cast("Any", document))
    assert loc in {problem.loc for problem in problems}


def test_error_a_pipeline_that_is_not_an_array_is_not_judged_as_empty() -> None:
    # Nothing was read, so there is no chain to find an array->bytes
    # codec missing from: the one problem is the shape of the field.
    problems = validate_array_metadata_v3(cast("Any", {**BASE, "codecs": "bytes"}))
    assert [(problem.loc, problem.kind) for problem in problems] == [(("codecs",), "invalid_type")]


@pytest.mark.parametrize(
    ("codecs", "loc"),
    [
        (
            ({"name": "scale_offset", "configuration": {"scale": "0"}}, "bytes"),
            ("codecs", 0, "configuration", "scale"),
        ),
        (
            ({"name": "scale_offset", "configuration": {"offset": -1}}, "bytes"),
            ("codecs", 0, "configuration", "offset"),
        ),
        (
            (
                {"name": "cast_value", "configuration": {"data_type": "uint8"}},
                {"name": "scale_offset", "configuration": {"scale": 0.5}},
                "bytes",
            ),
            ("codecs", 1, "configuration", "scale"),
        ),
    ],
    ids=["string-for-uint8", "out-of-range-for-uint8", "float-for-the-type-cast-to"],
)
def test_error_scale_offset_scalar_is_no_fill_value_of_the_type_it_receives(
    codecs: tuple[object, ...], loc: tuple[str | int, ...]
) -> None:
    # Each scalar is "encoded to JSON using the Zarr V3 fill value encoding
    # for the input array's data type" -- the type that reaches the codec.
    problems = validate_array_metadata_v3({**BASE, "codecs": codecs})
    assert [(p.loc, p.kind) for p in problems] == [(loc, "invalid_value")]


def test_error_scale_offset_data_type_has_no_arithmetic() -> None:
    doc = {**BASE, "data_type": "bool", "fill_value": False, "codecs": ("scale_offset", "bytes")}
    assert _sole_problem(doc) == (
        ("codecs", 0),
        "scale_offset is defined for integer and floating-point data types, not 'bool'",
    )


# (a cast target, whether `out_of_range: "wrap"` is defined for it)
WRAP_TARGETS: dict[str, tuple[object, bool]] = {
    "int32": ("int32", True),
    "uint64": ("uint64", True),
    "bool": ("bool", False),
    "float32": ("float32", False),
    "complex64": ("complex64", False),
    "raw-bytes": ("r8", False),
    "string": ("string", False),
    "numpy-time": (
        {"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 1}},
        False,
    ),
    # Out of scope, so it may be an integral extension type; declining
    # beats guessing.
    "unmodelled": ("mycorp.bigint", True),
}


@pytest.mark.parametrize(("target", "allowed"), WRAP_TARGETS.values(), ids=list(WRAP_TARGETS))
def test_wrap_requires_a_twos_complement_integer_target(target: object, allowed: bool) -> None:
    document = {
        **BASE,
        "codecs": (
            {"name": "cast_value", "configuration": {"data_type": target, "out_of_range": "wrap"}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        ),
    }
    wrap = [
        problem
        for problem in validate_array_metadata_v3(document)
        if problem.loc == ("codecs", 0, "configuration", "out_of_range")
    ]
    assert (len(wrap) == 0) is allowed, wrap


def test_the_wrap_message_names_the_spelling_not_the_family() -> None:
    # `r<N>` is an invented lookup key, not a name any document writes.
    document = {
        **BASE,
        "codecs": (
            {"name": "cast_value", "configuration": {"data_type": "r24", "out_of_range": "wrap"}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        ),
    }
    messages = [problem.message for problem in validate_array_metadata_v3(document)]
    assert any("got 'r24'" in message for message in messages), messages
