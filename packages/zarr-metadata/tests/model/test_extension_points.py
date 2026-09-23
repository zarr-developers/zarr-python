"""The model's v3 validators read each extension point through its definition.

A document's structure is the model's own judgment; what a codec, data
type, chunk grid or chunk key encoding holds in its configuration is its
definition's, found in a scope (`CORE_AND_EXTENSIONS` unless a caller
passes another) and reported at the field's place in the document. A
name nothing in the scope claims is left unjudged.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from zarr_metadata._json import arrays_to_tuples
from zarr_metadata.model import (
    MetadataValidationError,
    ZarrV3ArrayMetadata,
    ZarrV3GroupMetadata,
    is_array_metadata_v3,
    parse_array_metadata_v3,
    validate_array_metadata_v3,
    validate_group_metadata_v3,
)
from zarr_metadata.v3.codec.gzip import GZIP_CODEC
from zarr_metadata.v3.definition import CORE, CORE_AND_EXTENSIONS, CodecDefinition, Context

BYTES = {"name": "bytes", "configuration": {"endian": "little"}}


def _document(**fields: object) -> dict[str, Any]:
    """A default v3 array document with `fields` in it, arrays as tuples as a reader's are."""
    document = {**ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json(), **fields}
    return cast("dict[str, Any]", arrays_to_tuples(document))


LENIENT_GZIP = CodecDefinition(
    name="gzip",
    configuration=GZIP_CODEC.configuration,
    kind="bytes_bytes",
    size="dynamic",
    rules=lambda configuration: [],
)
"""A reader's own gzip, which takes any level: a scope can grow, and substitute."""


@pytest.mark.parametrize(
    ("document", "context"),
    [
        (_document(), CORE_AND_EXTENSIONS),
        (
            _document(codecs=[BYTES, {"name": "gzip", "configuration": {"level": 5}}]),
            CORE_AND_EXTENSIONS,
        ),
        (
            _document(codecs=[BYTES, {"name": "acme.lz9", "configuration": {"anything": 1}}]),
            CORE_AND_EXTENSIONS,
        ),
        (_document(codecs=[BYTES, {"name": "zstd", "configuration": {"level": 99}}]), CORE),
        (
            _document(codecs=[BYTES, {"name": "gzip", "configuration": {"level": 99}}]),
            CORE.extended_with(LENIENT_GZIP),
        ),
        (dict(ZarrV3ArrayMetadata.create_default(shape=(0, 4)).to_json()), CORE_AND_EXTENSIONS),
    ],
    ids=[
        "default",
        "a-codec-its-definition-allows",
        "a-name-nothing-claims",
        "an-extension-outside-the-scope",
        "a-reader-s-own-definition",
        "a-zero-length-array",
    ],
)
def test_a_document_reads_through_the_definitions_in_its_scope(
    document: dict[str, Any], context: Context
) -> None:
    # What the scope holds judges; what it does not is left to the reader.
    # The model reads in the same scope as the validators.
    assert validate_array_metadata_v3(document, context=context) == ()
    assert is_array_metadata_v3(document, context=context)
    assert parse_array_metadata_v3(document, context=context) is not None
    assert ZarrV3ArrayMetadata.from_json(document, context=context).to_json() is not None


@pytest.mark.parametrize(
    ("fields", "loc", "kind"),
    [
        (
            {"codecs": [BYTES, {"name": "gzip", "configuration": {"level": 99}}]},
            ("codecs", 1, "configuration", "level"),
            "invalid_value",
        ),
        (
            {"codecs": [BYTES, {"name": "gzip", "configuration": {"level": 5, "clevel": 9}}]},
            ("codecs", 1, "configuration", "clevel"),
            "unknown_key",
        ),
        (
            {
                "data_type": {
                    "name": "numpy.datetime64",
                    "configuration": {"unit": "s", "scale_factor": 0},
                }
            },
            ("data_type", "configuration", "scale_factor"),
            "invalid_value",
        ),
        (
            {"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [-1]}}},
            ("chunk_grid", "configuration", "chunk_shape", 0),
            "invalid_value",
        ),
        (
            {
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": [2],
                            "codecs": [BYTES, {"name": "gzip", "configuration": {"level": 99}}],
                            "index_codecs": [BYTES],
                        },
                    }
                ]
            },
            ("codecs", 0, "configuration", "codecs", 1, "configuration", "level"),
            "invalid_value",
        ),
    ],
    ids=["codec", "unknown-key-in-a-configuration", "data-type", "chunk-grid", "nested-in-a-shard"],
)
def test_error_a_configuration_its_definition_refuses(
    fields: dict[str, Any], loc: tuple[str | int, ...], kind: str
) -> None:
    # Located at the field's place in the document; the document is
    # refused by every judging entry point, and by the model reading it.
    document = _document(**fields)
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(document)] == [
        (loc, kind)
    ]
    assert not is_array_metadata_v3(document)
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadata.from_json(document)


def test_error_an_array_in_a_group_s_consolidated_metadata() -> None:
    array = _document(codecs=[BYTES, {"name": "gzip", "configuration": {"level": 99}}])
    group = {
        **ZarrV3GroupMetadata.create_default().to_json(),
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": array},
        },
    }
    assert [(problem.loc, problem.kind) for problem in validate_group_metadata_v3(group)] == [
        (
            ("consolidated_metadata", "metadata", "a", "codecs", 1, "configuration", "level"),
            "invalid_value",
        )
    ]


def test_an_array_in_a_group_s_consolidated_metadata_reads_in_the_group_s_scope() -> None:
    array = _document(codecs=[BYTES, {"name": "gzip", "configuration": {"level": 99}}])
    group = {
        **ZarrV3GroupMetadata.create_default().to_json(),
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {"a": array},
        },
    }
    lenient = CORE.extended_with(LENIENT_GZIP)
    assert validate_group_metadata_v3(group, context=lenient) == ()
    assert ZarrV3GroupMetadata.from_json(group, context=lenient).to_json() is not None


def test_error_an_envelope_is_judged_once() -> None:
    # `resolve` judges the envelope and reads the configuration: one
    # report, and the configuration still judged.
    document = _document(
        codecs=[BYTES, {"name": "gzip", "configuration": {"level": 5}, "must_understand": False}]
    )
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(document)] == [
        (("codecs", 1, "must_understand"), "invalid_value")
    ]


def test_error_a_field_that_is_not_json_is_not_read() -> None:
    # Not JSON is the first verdict: the NaN is reported, and the
    # configuration beside it is not read.
    document = _document(
        codecs=[
            BYTES,
            {"name": "gzip", "configuration": {"level": 99}, "must_understand": math.nan},
        ]
    )
    assert [(problem.loc, problem.kind) for problem in validate_array_metadata_v3(document)] == [
        (("codecs", 1, "must_understand"), "invalid_value")
    ]
