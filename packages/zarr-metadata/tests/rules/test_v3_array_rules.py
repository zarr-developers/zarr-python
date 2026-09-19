"""Tests for the v3 array and group composition rules added with the
rules-layer promotion: chunk grid values/geometry, transpose orders,
sharding pipelines/geometry, and consolidated-entry recursion."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    assert "positive chunk extent" in message


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
    assert any("positive [size, count] pair" in p.message for p in problems)


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
        and "positive chunk extent" in p.message
        for p in problems
    )


def test_error_bytes_requires_endian_for_multibyte_data() -> None:
    loc, message = _sole_problem({**BASE, "data_type": "int32", "codecs": ("bytes",)})
    assert loc == ("codecs", 0, "configuration", "endian")
    assert "required" in message


def test_error_bytes_rejects_variable_length_data_type() -> None:
    loc, message = _sole_problem(
        {**BASE, "data_type": "string", "fill_value": "", "codecs": ("bytes",)}
    )
    assert loc == ("codecs", 0, "configuration")
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

    with pytest.raises(MetadataValidationError, match="fill_value invalid"):
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
    assert [(p.loc, p.kind) for p in problems] == [(("codecs", 0, "configuration"), "unknown_key")]


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
    assert (("codecs", 0, "configuration"), "unknown_key") in kinds
    assert (("codecs", 0, "configuration", "order"), "invalid_value") in kinds


def test_unknown_member_does_not_mask_a_chunk_grid_rule() -> None:
    doc = {
        **BASE,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (2,), "hint": 1}},
    }
    kinds = {(p.loc, p.kind) for p in validate_array_metadata_v3(doc)}
    assert (("chunk_grid", "configuration"), "unknown_key") in kinds
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
    codec = emitted["codecs"][0]
    assert codec["configuration"]["numThreads"] == 4
