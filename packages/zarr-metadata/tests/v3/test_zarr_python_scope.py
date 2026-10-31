"""The `ZARR_PYTHON_CONTEXT` scope, and the definitions it adds for what zarr-python writes.

`fixed_length_utf32`, `vlen-utf8` and `vlen-bytes` are registered in
zarr-extensions, so they are in `CORE_AND_EXTENSIONS_CONTEXT`.
`null_terminated_bytes`, `variable_length_bytes` and `raw_bytes` are not,
so only `ZARR_PYTHON_CONTEXT` reads them.
"""

from __future__ import annotations

from typing import Any

import pytest

from zarr_metadata.definition import (
    AcceptedField,
    RefusedField,
    UnclaimedField,
    canonical_fill_value,
    fill_value_problems,
    resolve,
)
from zarr_metadata.model import read_array_metadata_v3
from zarr_metadata.v3.data_type.null_terminated_bytes import NULL_TERMINATED_BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.variable_length_bytes import VARIABLE_LENGTH_BYTES_DATA_TYPE
from zarr_metadata.v3.data_type.zarr_python_raw_bytes import ZARR_PYTHON_RAW_BYTES_DATA_TYPE
from zarr_metadata.v3.definition import (
    CORE_AND_EXTENSIONS_CONTEXT,
    ZARR_PYTHON_CONTEXT,
    DataTypeDefinition,
    ZarrV3Context,
)


def _document(data_type: object, fill_value: object, codecs: list[object]) -> dict[str, Any]:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2],
        "data_type": data_type,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": fill_value,
        "codecs": codecs,
        "attributes": {},
    }


_LITTLE = {"name": "bytes", "configuration": {"endian": "little"}}
_UTF32 = {"name": "fixed_length_utf32", "configuration": {"length_bytes": 12}}
_NULL_TERMINATED = {"name": "null_terminated_bytes", "configuration": {"length_bytes": 4}}
_RAW = {"name": "raw_bytes", "configuration": {"length_bytes": 4}}


# What zarr-python 3.4.2 writes for `<U3`, `S4`, `str`, `VariableLengthBytes()` and `V4` arrays.
@pytest.mark.parametrize(
    ("document", "in_core_and_extensions"),
    [
        (_document(_UTF32, "ab", [_LITTLE]), True),
        (_document("string", "x", [{"name": "vlen-utf8", "configuration": {}}]), True),
        (_document(_NULL_TERMINATED, "YWI=", [{"name": "bytes"}]), False),
        (
            _document(
                "variable_length_bytes", "eA==", [{"name": "vlen-bytes", "configuration": {}}]
            ),
            False,
        ),
        (_document(_RAW, "AAAAAA==", [{"name": "bytes"}]), False),
    ],
    ids=[
        "fixed_length_utf32",
        "string",
        "null_terminated_bytes",
        "variable_length_bytes",
        "raw_bytes",
    ],
)
def test_what_zarr_python_writes_reads_without_a_problem_in_zarr_python(
    document: dict[str, Any], in_core_and_extensions: bool
) -> None:
    # In `ZARR_PYTHON_CONTEXT` every field is read by a definition and has no problem.
    reading = read_array_metadata_v3(document, context=ZARR_PYTHON_CONTEXT)
    assert reading.problems == ()
    assert all(isinstance(field, AcceptedField) for _, field in reading.fields())
    # `CORE_AND_EXTENSIONS_CONTEXT` reads the registered extensions, and leaves
    # zarr-python's own data types unclaimed.
    data_type = read_array_metadata_v3(document, context=CORE_AND_EXTENSIONS_CONTEXT).data_type
    assert isinstance(data_type, AcceptedField) is in_core_and_extensions
    assert isinstance(data_type, UnclaimedField) is not in_core_and_extensions


def test_zarr_python_is_core_and_extensions_with_zarr_python_s_own_data_types() -> None:
    own = CORE_AND_EXTENSIONS_CONTEXT.update(
        data_types=[
            *CORE_AND_EXTENSIONS_CONTEXT.data_types.values(),
            NULL_TERMINATED_BYTES_DATA_TYPE,
            VARIABLE_LENGTH_BYTES_DATA_TYPE,
            ZARR_PYTHON_RAW_BYTES_DATA_TYPE,
        ]
    )
    assert own == ZARR_PYTHON_CONTEXT


@pytest.mark.parametrize(
    ("data_type", "value", "canonical"),
    [
        (_UTF32, "", ""),
        (_UTF32, "abc", "abc"),
        # Trailing U+0000 code points are padding, so they are the same value.
        (_UTF32, "a\x00\x00", "a"),
        # A value is padded with null bytes, so trailing ones are the same value.
        (_NULL_TERMINATED, "QQAA", "QQ=="),
        (_NULL_TERMINATED, "QR==", "QQ=="),
        (_NULL_TERMINATED, "", ""),
        # The padding does not count against the size: b"A\x00\x00" fits in 2 bytes.
        ({"name": "null_terminated_bytes", "configuration": {"length_bytes": 2}}, "QQAA", "QQ=="),
        ("variable_length_bytes", "QR==", "QQ=="),
        # zarr-python pads a shorter value with null bytes, and writes it padded.
        (_RAW, "QQ==", "QQAAAA=="),
        (_RAW, "QQAAAA==", "QQAAAA=="),
        ("variable_length_bytes", "QQAA", "QQAA"),
    ],
)
def test_the_canonical_fill_value(data_type: object, value: str, canonical: str) -> None:
    resolved, problems = resolve(data_type, DataTypeDefinition, ZARR_PYTHON_CONTEXT)
    assert problems == ()
    assert fill_value_problems(resolved, value) == ()
    assert canonical_fill_value(resolved, value) == canonical


@pytest.mark.parametrize(
    "data_type",
    [
        {"name": "fixed_length_utf32", "configuration": {"length_bytes": 0}},
        {"name": "fixed_length_utf32", "configuration": {"length_bytes": 6}},
        {"name": "null_terminated_bytes", "configuration": {"length_bytes": 0}},
        {"name": "raw_bytes", "configuration": {"length_bytes": 0}},
    ],
    ids=["utf32-zero", "utf32-not-a-multiple-of-4", "null-terminated-zero", "raw-zero"],
)
def test_error_a_size_that_holds_no_whole_value(data_type: object) -> None:
    resolved, problems = resolve(data_type, DataTypeDefinition, ZARR_PYTHON_CONTEXT)
    assert isinstance(resolved, RefusedField)
    assert [(p.loc, p.kind) for p in problems] == [
        (("configuration", "length_bytes"), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("data_type", "value"),
    [(_UTF32, "abcd"), (_NULL_TERMINATED, "YWJjZGU="), (_RAW, "YWJjZGU=")],
    ids=["utf32-four-code-points", "null-terminated-five-bytes", "raw-five-bytes"],
)
def test_error_a_fill_value_longer_than_the_type(data_type: object, value: str) -> None:
    resolved, _ = resolve(data_type, DataTypeDefinition, ZARR_PYTHON_CONTEXT)
    assert [(p.loc, p.kind) for p in fill_value_problems(resolved, value)] == [
        ((), "invalid_value")
    ]


@pytest.mark.parametrize("data_type", [_NULL_TERMINATED, "variable_length_bytes", _RAW])
def test_error_a_fill_value_that_is_not_base64(data_type: object) -> None:
    resolved, _ = resolve(data_type, DataTypeDefinition, ZARR_PYTHON_CONTEXT)
    assert [(p.loc, p.kind) for p in fill_value_problems(resolved, "YWI")] == [
        ((), "invalid_value")
    ]


@pytest.mark.parametrize(
    ("data_type", "codec"),
    [
        (_UTF32, "vlen-utf8"),
        (_UTF32, "vlen-bytes"),
        ("bytes", "vlen-utf8"),
        ("variable_length_bytes", "vlen-utf8"),
        ("string", "vlen-bytes"),
    ],
)
def test_error_a_variable_length_codec_for_a_data_type_it_does_not_encode(
    data_type: object, codec: str
) -> None:
    document = _document(data_type, "", [codec])
    reading = read_array_metadata_v3(document, context=ZARR_PYTHON_CONTEXT)
    assert [(p.loc, p.kind) for p in reading.problems] == [(("codecs", 0), "invalid_value")]


def test_error_a_bytes_codec_without_an_endian_for_fixed_length_utf32() -> None:
    reading = read_array_metadata_v3(_document(_UTF32, "", ["bytes"]), context=ZARR_PYTHON_CONTEXT)
    assert [(p.loc, p.kind) for p in reading.problems] == [
        (("codecs", 0, "configuration"), "missing_key")
    ]


def test_an_empty_scope_reads_none_of_them() -> None:
    for name in (
        "fixed_length_utf32",
        "null_terminated_bytes",
        "variable_length_bytes",
        "raw_bytes",
    ):
        assert ZarrV3Context().claimant(DataTypeDefinition, name) is None
