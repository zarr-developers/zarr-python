"""The model's v3 validators read each extension point through its definition.

The model checks a document's structure itself. The configuration of a
codec, data type, chunk grid or chunk key encoding is checked by its
definition. The definition is found in a scope (`CORE_AND_EXTENSIONS`
unless a caller passes another), and problems are reported at the field's
location in the document. A name that no definition in the scope claims
is not checked.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import pytest
from typing_extensions import TypedDict

from zarr_metadata._json import arrays_to_tuples
from zarr_metadata.model import (
    ZarrV3ArrayMetadata,
    ZarrV3ConsolidatedMetadata,
    read_array_metadata_v3,
    read_consolidated_metadata_v3,
)
from zarr_metadata.problems import MetadataValidationError
from zarr_metadata.v3.definition import (
    CORE,
    CORE_AND_EXTENSIONS,
    CodecDefinition,
)

if TYPE_CHECKING:
    from zarr_metadata.v3.definition import ZarrV3Context

BYTES = {"name": "bytes", "configuration": {"endian": "little"}}


def _document(**fields: object) -> dict[str, Any]:
    """Return a default v3 array document with `fields` added, with arrays as tuples as a reader holds them."""
    document = {**ZarrV3ArrayMetadata.create_default(shape=(4,)).to_json(), **fields}
    return cast("dict[str, Any]", arrays_to_tuples(document))


class LenientGzipConfiguration(TypedDict, closed=True):
    """A gzip configuration whose `level` is any integer. The bound on `level` is part of the configuration type, so accepting any level needs a separate type."""

    level: int


LENIENT_GZIP = CodecDefinition(
    name="gzip",
    configuration=LenientGzipConfiguration,
    kind="bytes_bytes",
    size="dynamic",
)
"""A reader's own gzip definition, which accepts any level. A scope can be extended, and an extension can replace a definition."""

BAD_GZIP_ARRAY = _document(codecs=[BYTES, {"name": "gzip", "configuration": {"level": 99}}])
"""An array whose gzip level the core `gzip` definition refuses and `LENIENT_GZIP` accepts."""


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
            CORE.update(codecs={**CORE.codecs, "gzip": LENIENT_GZIP}),
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
    document: dict[str, Any], context: ZarrV3Context
) -> None:
    # Definitions in the scope check their fields. Fields with no definition
    # in the scope are left to the reader. The model reads in the same scope
    # as the reader, and its output reads back in that scope as the same
    # model.
    reading = read_array_metadata_v3(document, context=context)
    assert reading.problems == ()
    model = ZarrV3ArrayMetadata.from_json(document, context=context)
    assert reading.metadata == model
    assert ZarrV3ArrayMetadata.from_key_value(model.to_key_value(), context=context) == model


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
    # The problem is located at the field's position in the document. The
    # reader reports it, and the model refuses the document.
    document = _document(**fields)
    assert [
        (problem.loc, problem.kind) for problem in read_array_metadata_v3(document).problems
    ] == [(loc, kind)]
    with pytest.raises(MetadataValidationError):
        ZarrV3ArrayMetadata.from_json(document)


def test_error_an_array_in_consolidated_metadata() -> None:
    """A codec configuration that the codec's definition refuses, in an array listed in consolidated metadata, is reported at its location in the member."""
    member = {"kind": "inline", "must_understand": False, "metadata": {"a": BAD_GZIP_ARRAY}}
    assert [
        (problem.loc, problem.kind) for problem in read_consolidated_metadata_v3(member).problems
    ] == [(("metadata", "a", "codecs", 1, "configuration", "level"), "invalid_value")]


def test_an_array_in_consolidated_metadata_reads_in_the_scope_it_is_read_in() -> None:
    """An array listed in consolidated metadata is read in the scope the member is read in: a lenient `gzip` accepts the level the core `gzip` refuses, and the model round-trips in that scope."""
    member = {"kind": "inline", "must_understand": False, "metadata": {"a": BAD_GZIP_ARRAY}}
    lenient = CORE.update(codecs={**CORE.codecs, "gzip": LENIENT_GZIP})
    reading = read_consolidated_metadata_v3(member, context=lenient)
    assert reading.problems == ()
    assert reading.metadata is not None
    assert (
        ZarrV3ConsolidatedMetadata(reading.metadata.to_json(), context=lenient) == reading.metadata
    )


def test_error_an_envelope_is_judged_once() -> None:
    # `resolve` checks the envelope and reads the configuration. The
    # envelope problem is reported once, and the configuration is still
    # checked.
    document = _document(
        codecs=[BYTES, {"name": "gzip", "configuration": {"level": 5}, "must_understand": False}]
    )
    assert [
        (problem.loc, problem.kind) for problem in read_array_metadata_v3(document).problems
    ] == [(("codecs", 1, "must_understand"), "invalid_value")]


def test_error_a_field_that_is_not_json_is_not_read() -> None:
    # The JSON check runs first. The NaN is reported, and the configuration
    # next to it is not read.
    document = _document(
        codecs=[
            BYTES,
            {"name": "gzip", "configuration": {"level": 99}, "must_understand": math.nan},
        ]
    )
    assert [
        (problem.loc, problem.kind) for problem in read_array_metadata_v3(document).problems
    ] == [(("codecs", 1, "must_understand"), "invalid_value")]
